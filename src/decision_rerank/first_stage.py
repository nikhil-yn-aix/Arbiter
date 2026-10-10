import time

import bm25s
import numpy as np
import Stemmer
from sentence_transformers import SentenceTransformer

from .datasets import load_dataset
from .device import peak_gb, release, reset_peak, sync
from .retrieval import QUERY_PREFIX
from .settings import MODELS_DIR, load_config


def first_stage_clock(name: str, qids: list[str], device: str) -> dict[str, float]:
    cfg = load_config().retrieval
    dataset = load_dataset(name)
    doc_ids = list(dataset.corpus)
    stemmer = Stemmer.Stemmer("english")
    tokens = bm25s.tokenize([dataset.corpus[d] for d in doc_ids], stopwords="en", stemmer=stemmer, show_progress=False)
    index = bm25s.BM25(k1=cfg.bm25_k1, b=cfg.bm25_b)
    index.index(tokens, show_progress=False)
    reset_peak(device)
    model = SentenceTransformer(str(MODELS_DIR / "bge-small"), device=device)
    if device == "cuda":
        model.half()
    docs = model.encode(
        [dataset.corpus[d] for d in doc_ids],
        batch_size=128,
        normalize_embeddings=True,
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    depth = min(cfg.depth, len(doc_ids))
    samples = []

    def one(qid: str) -> None:
        text = dataset.queries[qid]
        query = bm25s.tokenize([text], stopwords="en", stemmer=stemmer, show_progress=False)
        index.retrieve(query, k=depth, show_progress=False)
        vector = model.encode(
            [QUERY_PREFIX + text], normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False
        )
        (vector @ docs.T).topk(depth, dim=1)

    for qid in qids[:3]:
        one(qid)
    sync(device)
    start = time.perf_counter()
    for qid in qids:
        begin = time.perf_counter()
        one(qid)
        sync(device)
        samples.append((time.perf_counter() - begin) * 1000)
    total = time.perf_counter() - start
    result = {
        "p50": float(np.percentile(samples, 50)),
        "p95": float(np.percentile(samples, 95)),
        "throughput": len(qids) / total,
        "vram": peak_gb(device),
    }
    release(device)
    return result
