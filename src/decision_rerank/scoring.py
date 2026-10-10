import time
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from .adapters import Reranker
from .datasets import Dataset
from .device import peak_gb, reset_peak, sync
from .settings import CACHE_DIR, ensure_dirs

CHUNK = 4096


@dataclass
class CandidateScores:
    doc_ids: list[str]
    score: np.ndarray
    prob: np.ndarray


def score_path(tag: str, dataset: str):
    ensure_dirs()
    directory = CACHE_DIR / "scores"
    directory.mkdir(exist_ok=True)
    return directory / f"{tag}__{dataset}.npz"


def save_scores(path, scores: dict[str, CandidateScores]) -> None:
    payload = {}
    for qid, entry in scores.items():
        payload[f"{qid}|score"] = entry.score
        payload[f"{qid}|prob"] = entry.prob
        payload[f"{qid}|docs"] = np.array(entry.doc_ids)
    np.savez(path, **payload)


def load_scores(path) -> dict[str, CandidateScores]:
    out: dict[str, CandidateScores] = {}
    with np.load(path) as bundle:
        for key in bundle.files:
            qid, field = key.rsplit("|", 1)
            if field == "docs":
                out[qid] = CandidateScores([str(d) for d in bundle[key]], bundle[f"{qid}|score"], bundle[f"{qid}|prob"])
    return out


def score_candidates(
    reranker: Reranker,
    dataset: Dataset,
    candidates: dict[str, list[str]],
    depth: int,
    max_tokens: int | None = None,
    label: str = "",
) -> dict[str, CandidateScores]:
    pairs = []
    owners = []
    for qid, docs in candidates.items():
        for doc in docs[:depth]:
            pairs.append((dataset.queries[qid], dataset.corpus[doc]))
            owners.append(qid)
    score = np.zeros(len(pairs))
    prob = np.zeros(len(pairs))
    for start in tqdm(range(0, len(pairs), CHUNK), desc=label, leave=False):
        result = reranker.score_pairs(pairs[start : start + CHUNK], max_tokens=max_tokens)
        score[start : start + CHUNK] = result.score
        prob[start : start + CHUNK] = result.prob
    out: dict[str, CandidateScores] = {}
    cursor = 0
    for qid, docs in candidates.items():
        n = len(docs[:depth])
        out[qid] = CandidateScores(docs[:depth], score[cursor : cursor + n], prob[cursor : cursor + n])
        cursor += n
    return out


def time_queries(
    reranker: Reranker,
    dataset: Dataset,
    candidates: dict[str, list[str]],
    qids: list[str],
    size: int,
    warmup: int,
) -> dict[str, float]:
    device = reranker.device
    for qid in qids[:warmup]:
        reranker.score_pairs([(dataset.queries[qid], dataset.corpus[d]) for d in candidates[qid][:size]])
    sync(device)
    reset_peak(device)
    samples = []
    start_all = time.perf_counter()
    for qid in qids:
        pairs = [(dataset.queries[qid], dataset.corpus[d]) for d in candidates[qid][:size]]
        sync(device)
        begin = time.perf_counter()
        reranker.score_pairs(pairs)
        sync(device)
        samples.append((time.perf_counter() - begin) * 1000)
    total = time.perf_counter() - start_all
    return {
        "p50": float(np.percentile(samples, 50)),
        "p95": float(np.percentile(samples, 95)),
        "throughput": len(qids) / total,
        "vram": peak_gb(device),
    }
