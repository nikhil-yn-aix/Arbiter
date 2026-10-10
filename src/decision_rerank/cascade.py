import argparse
import json
import time

import numpy as np
import pandas as pd

from . import checkpoint
from .adapters import build_reranker, dispose
from .benchmark import timing_sample
from .datasets import load_dataset
from .device import resolve_device, sync
from .metrics import macro_bootstrap, ndcg_at, rank_by
from .retrieval import build_candidates
from .scoring import load_scores, score_path
from .settings import RESULTS_DIR, ensure_dirs, load_config


def cascade_ranking(first, second, pool: int, keep: int) -> list[str]:
    head = rank_by(first.doc_ids[:pool], first.score[:pool])
    chosen = head[:keep]
    lookup = {d: s for d, s in zip(second.doc_ids, second.score, strict=True)}
    refined = rank_by(chosen, np.array([lookup[d] for d in chosen]))
    return refined + head[keep:]


def quality(first_id: str, second_id: str, keep: int) -> dict[str, np.ndarray]:
    config = load_config()
    pool = config.cascade.pool
    cutoff = config.evaluation.cutoff
    out = {}
    for name in config.datasets:
        dataset = load_dataset(name)
        first = load_scores(score_path(first_id, name))
        second = load_scores(score_path(second_id, name))
        qids = sorted(dataset.queries)
        out[name] = np.array(
            [ndcg_at(cascade_ranking(first[q], second[q], pool, keep), dataset.qrels[q], cutoff) for q in qids]
        )
    return out


def reference_quality(model_id: str) -> dict[str, np.ndarray]:
    config = load_config()
    out = {}
    for name in config.datasets:
        dataset = load_dataset(name)
        scores = load_scores(score_path(model_id, name))
        out[name] = np.array(
            [
                ndcg_at(
                    rank_by(scores[q].doc_ids[: config.cascade.pool], scores[q].score[: config.cascade.pool]),
                    dataset.qrels[q],
                    config.evaluation.cutoff,
                )
                for q in sorted(dataset.queries)
            ]
        )
    return out


def retrieval_quality() -> dict[str, np.ndarray]:
    config = load_config()
    out = {}
    for name in config.datasets:
        dataset = load_dataset(name)
        candidates = build_candidates(name)
        out[name] = np.array(
            [
                ndcg_at(candidates[q][: config.cascade.pool], dataset.qrels[q], config.evaluation.cutoff)
                for q in sorted(dataset.queries)
            ]
        )
    return out


def macro(groups: dict[str, np.ndarray]) -> float:
    return float(np.mean([g.mean() for g in groups.values()]))


def interval(groups: dict[str, np.ndarray]) -> tuple[float, float]:
    config = load_config()
    _, low, high = macro_bootstrap(list(groups.values()), config.evaluation.bootstrap_resamples, config.seed)
    return low, high


def timed_pass(first, second, device: str, ks: list[int], first_cache: dict) -> dict[str, dict[int, list[float]]]:
    config = load_config()
    pool = config.cascade.pool
    cached = {}
    for name in config.datasets:
        dataset = load_dataset(name)
        candidates = build_candidates(name)
        sample = timing_sample(dataset, config.cascade.timing_queries + config.timing.warmup_queries, config.seed)
        cached[name] = (dataset, candidates, sample)
    first_ms: list[float] = []
    second_ms: dict[int, list[float]] = {k: [] for k in ks}
    total_ms: dict[int, list[float]] = {k: [] for k in ks}
    for dataset, candidates, sample in cached.values():
        for position, qid in enumerate(sample):
            counting = position >= config.timing.warmup_queries
            docs = candidates[qid][:pool]
            query = dataset.queries[qid]
            pairs = [(query, dataset.corpus[d]) for d in docs]
            key = (dataset.name, qid)
            if key not in first_cache:
                sync(device)
                begin = time.perf_counter()
                result = first.score_pairs(pairs)
                sync(device)
                first_cache[key] = ((time.perf_counter() - begin) * 1000, rank_by(docs, result.score))
            first_cost, head = first_cache[key]
            if counting:
                first_ms.append(first_cost)
            for k in ks:
                chosen = [(query, dataset.corpus[d]) for d in head[:k]] if k < pool else pairs
                sync(device)
                begin = time.perf_counter()
                second.score_pairs(chosen)
                sync(device)
                cost = (time.perf_counter() - begin) * 1000
                if counting:
                    second_ms[k].append(cost)
                    total_ms[k].append(first_cost + cost)
    return {"first": {0: first_ms}, "second": second_ms, "total": total_ms}


def run(device: str) -> None:
    config = load_config()
    ensure_dirs()
    cfg = config.cascade
    baseline = retrieval_quality()
    base_ndcg = macro(baseline)
    detail = {"no_reranker": {"ndcg": base_ndcg}, "configs": []}
    rows = []
    alone_cache = {}
    for first_id in cfg.decision:
        first = None
        first_cache: dict = {}
        for second_id in cfg.refiner:
            key = f"{first_id}__{second_id}"
            done = checkpoint.load("cascade", key)
            if done is not None:
                rows.extend(done["rows"])
                detail["configs"].extend(done["configs"])
                continue
            if first is None:
                first = build_reranker(first_id, device)
            first_rows, first_configs = len(rows), len(detail["configs"])
            second = build_reranker(second_id, device)
            clock = timed_pass(first, second, device, cfg.top_k, first_cache)
            dispose(second)
            first_alone = reference_quality(first_id)
            second_alone = alone_cache.setdefault(second_id, reference_quality(second_id))
            second_full = float(np.percentile(clock["second"][cfg.pool], 50))
            first_full = float(np.percentile(clock["first"][0], 50))
            for k in cfg.top_k:
                groups = quality(first_id, second_id, k)
                low, high = interval(groups)
                total = clock["total"][k]
                row = {
                    "decision_model": first_id,
                    "cross_encoder": second_id,
                    "top_k": k,
                    "cascade_ndcg": macro(groups),
                    "ndcg_ci_low": low,
                    "ndcg_ci_high": high,
                    "decision_ndcg": macro(first_alone),
                    "cross_encoder_ndcg": macro(second_alone),
                    "no_reranker_ndcg": base_ndcg,
                    "ce_calls_saved_pct": 100.0 * (1 - k / cfg.pool),
                    "latency_p50_ms": float(np.percentile(total, 50)),
                    "latency_p95_ms": float(np.percentile(total, 95)),
                    "decision_alone_latency_ms": first_full,
                    "cross_encoder_alone_latency_ms": second_full,
                    "speedup_vs_cross_encoder": second_full / float(np.percentile(total, 50)),
                }
                rows.append(row)
                detail["configs"].append({**row, "per_dataset_ndcg": {n: float(g.mean()) for n, g in groups.items()}})
                print(
                    f"{first_id} -> {second_id} k={k} ndcg={row['cascade_ndcg']:.4f} p50={row['latency_p50_ms']:.1f}ms",
                    flush=True,
                )
            checkpoint.save("cascade", key, {"rows": rows[first_rows:], "configs": detail["configs"][first_configs:]})
        if first is not None:
            dispose(first)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "cascade_summary.csv", index=False)
    (RESULTS_DIR / "cascade.json").write_text(json.dumps(detail, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run(resolve_device(args.device))


if __name__ == "__main__":
    main()
