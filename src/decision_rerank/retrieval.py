import json

import bm25s
import numpy as np
import Stemmer
from sentence_transformers import SentenceTransformer

from .datasets import Dataset, load_dataset
from .device import release, resolve_device
from .settings import CACHE_DIR, MODELS_DIR, ensure_dirs, load_config

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def reciprocal_rank_fusion(rankings: list[list[int]], k: int) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for position, doc in enumerate(ranking):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + position + 1)
    return fused


def lexical_ranking(dataset: Dataset, doc_ids: list[str], depth: int) -> dict[str, list[int]]:
    cfg = load_config().retrieval
    stemmer = Stemmer.Stemmer("english")
    tokens = bm25s.tokenize([dataset.corpus[d] for d in doc_ids], stopwords="en", stemmer=stemmer, show_progress=False)
    index = bm25s.BM25(k1=cfg.bm25_k1, b=cfg.bm25_b)
    index.index(tokens, show_progress=False)
    qids = list(dataset.queries)
    query_tokens = bm25s.tokenize(
        [dataset.queries[q] for q in qids], stopwords="en", stemmer=stemmer, show_progress=False
    )
    hits, _ = index.retrieve(query_tokens, k=min(depth, len(doc_ids)), show_progress=False)
    return {qid: hits[i].tolist() for i, qid in enumerate(qids)}


def dense_ranking(dataset: Dataset, doc_ids: list[str], depth: int, device: str) -> dict[str, list[int]]:
    model = SentenceTransformer(str(MODELS_DIR / "bge-small"), device=device)
    model.max_seq_length = 512
    if device == "cuda":
        model.half()
    docs = model.encode(
        [dataset.corpus[d] for d in doc_ids],
        batch_size=128,
        normalize_embeddings=True,
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    qids = list(dataset.queries)
    queries = model.encode(
        [QUERY_PREFIX + dataset.queries[q] for q in qids],
        batch_size=128,
        normalize_embeddings=True,
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    sims = queries @ docs.T
    top = sims.topk(min(depth, len(doc_ids)), dim=1).indices.cpu().numpy()
    del model, docs, queries, sims
    release(device)
    return {qid: top[i].tolist() for i, qid in enumerate(qids)}


def fuse_candidates(dataset: Dataset, depth: int, device: str) -> dict[str, list[str]]:
    cfg = load_config().retrieval
    doc_ids = list(dataset.corpus)
    lexical = lexical_ranking(dataset, doc_ids, depth)
    dense = dense_ranking(dataset, doc_ids, depth, device)
    candidates = {}
    for qid in dataset.queries:
        fused = reciprocal_rank_fusion([lexical[qid], dense[qid]], cfg.rrf_k)
        order = sorted(fused, key=lambda d: (-fused[d], d))[:depth]
        candidates[qid] = [doc_ids[i] for i in order]
    return candidates


def build_candidates(name: str, device: str | None = None) -> dict[str, list[str]]:
    ensure_dirs()
    path = CACHE_DIR / f"{name}_candidates.json"
    if path.exists():
        return json.loads(path.read_text())
    candidates = fuse_candidates(load_dataset(name), load_config().retrieval.depth, resolve_device(device))
    path.write_text(json.dumps(candidates))
    return candidates


def recall_at(candidates: dict[str, list[str]], dataset: Dataset, depth: int) -> float:
    values = []
    for qid, ranked in candidates.items():
        relevant = {d for d, g in dataset.qrels[qid].items() if g > 0}
        if relevant:
            values.append(len(relevant & set(ranked[:depth])) / len(relevant))
    return float(np.mean(values))
