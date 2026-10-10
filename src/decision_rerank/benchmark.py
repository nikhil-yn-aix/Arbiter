import argparse
import json
import random

import numpy as np
import pandas as pd

from .adapters import build_reranker, dispose
from .datasets import Dataset, load_dataset
from .device import resolve_device
from .metrics import kendall, paired_bootstrap, query_metrics, rank_by, top_overlap
from .retrieval import build_candidates
from .scoring import (
    CandidateScores,
    load_scores,
    save_scores,
    score_candidates,
    score_path,
    time_queries,
)
from .settings import CACHE_DIR, RESULTS_DIR, ensure_dirs, load_config


def timing_sample(dataset: Dataset, count: int, seed: int) -> list[str]:
    qids = sorted(dataset.queries)
    random.Random(seed).shuffle(qids)
    return qids[:count]


def rerank_order(entry: CandidateScores, size: int) -> list[str]:
    return rank_by(entry.doc_ids[:size], entry.score[:size])


def evaluate_ranking(dataset: Dataset, rankings: dict[str, list[str]], threshold: int, cutoff: int) -> dict[str, dict]:
    return {qid: query_metrics(ranked, dataset.qrels[qid], cutoff, threshold) for qid, ranked in rankings.items()}


def mean_of(per_query: dict[str, dict], key: str) -> float:
    return float(np.mean([m[key] for m in per_query.values()]))


def partial_path(model_id: str):
    directory = CACHE_DIR / "benchmark"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{model_id}.json"


def run_model(model_id: str, device: str) -> dict:
    config = load_config()
    sizes = config.retrieval.candidate_sizes
    depth = config.retrieval.depth
    timing = config.timing
    reranker = build_reranker(model_id, device)
    record = {"rows": [], "per_query": {}}
    try:
        for name, spec in config.datasets.items():
            dataset = load_dataset(name)
            candidates = build_candidates(name, device)
            path = score_path(model_id, name)
            if path.exists():
                scores = load_scores(path)
            else:
                scores = score_candidates(reranker, dataset, candidates, depth, label=f"{model_id}/{name}")
                save_scores(path, scores)
            sample = timing_sample(dataset, timing.queries_per_cell + timing.warmup_queries, config.seed)
            record["per_query"][name] = {}
            for size in sizes:
                rankings = {qid: rerank_order(scores[qid], size) for qid in dataset.queries}
                per_query = evaluate_ranking(dataset, rankings, spec.binary_threshold, config.evaluation.cutoff)
                clock = time_queries(reranker, dataset, candidates, sample, size, timing.warmup_queries)
                record["per_query"][name][str(size)] = {q: m["ndcg"] for q, m in per_query.items()}
                record["rows"].append(
                    {
                        "model": model_id,
                        "dataset": name,
                        "candidate_size": size,
                        "queries": len(dataset.queries),
                        "ndcg_at_10": mean_of(per_query, "ndcg"),
                        "recall_at_10": mean_of(per_query, "recall"),
                        "mrr_at_10": mean_of(per_query, "mrr"),
                        "latency_p50_ms": clock["p50"],
                        "latency_p95_ms": clock["p95"],
                        "throughput_qps": clock["throughput"],
                        "vram_peak_gb": clock["vram"],
                    }
                )
                print(
                    f"{model_id:11s} {name:13s} k={size:3d} ndcg={record['rows'][-1]['ndcg_at_10']:.4f} "
                    f"p50={clock['p50']:.1f}ms",
                    flush=True,
                )
    finally:
        dispose(reranker)
    return record


def score_stage(model_ids: list[str], device: str) -> None:
    config = load_config()
    depth = config.retrieval.depth
    for model_id in model_ids:
        pending = [name for name in config.datasets if not score_path(model_id, name).exists()]
        if not pending:
            continue
        reranker = build_reranker(model_id, device)
        try:
            for name in pending:
                dataset = load_dataset(name)
                candidates = build_candidates(name, device)
                scores = score_candidates(reranker, dataset, candidates, depth, label=f"{model_id}/{name}")
                save_scores(score_path(model_id, name), scores)
                print(f"scored {model_id} {name}", flush=True)
        finally:
            dispose(reranker)


def baseline_record(device: str) -> dict:
    from .first_stage import first_stage_clock

    config = load_config()
    record = {"rows": [], "per_query": {}}
    for name, spec in config.datasets.items():
        dataset = load_dataset(name)
        candidates = build_candidates(name, device)
        clock = first_stage_clock(name, timing_sample(dataset, config.timing.queries_per_cell, config.seed), device)
        record["per_query"][name] = {}
        for size in config.retrieval.candidate_sizes:
            rankings = {qid: candidates[qid][:size] for qid in dataset.queries}
            per_query = evaluate_ranking(dataset, rankings, spec.binary_threshold, config.evaluation.cutoff)
            record["per_query"][name][str(size)] = {q: m["ndcg"] for q, m in per_query.items()}
            record["rows"].append(
                {
                    "model": config.evaluation.baseline,
                    "dataset": name,
                    "candidate_size": size,
                    "queries": len(dataset.queries),
                    "ndcg_at_10": mean_of(per_query, "ndcg"),
                    "recall_at_10": mean_of(per_query, "recall"),
                    "mrr_at_10": mean_of(per_query, "mrr"),
                    "latency_p50_ms": clock["p50"],
                    "latency_p95_ms": clock["p95"],
                    "throughput_qps": clock["throughput"],
                    "vram_peak_gb": clock["vram"],
                }
            )
    return record


def agreement(model_ids: list[str]) -> dict[tuple[str, str, int], dict[str, float]]:
    config = load_config()
    reference = config.evaluation.reference_model
    out = {}
    for name in config.datasets:
        ref_path = score_path(reference, name)
        if not ref_path.exists():
            continue
        ref = load_scores(ref_path)
        for model_id in model_ids:
            path = score_path(model_id, name)
            if not path.exists():
                continue
            own = load_scores(path)
            for size in config.retrieval.candidate_sizes:
                taus, overlaps = [], []
                for qid in ref:
                    taus.append(kendall(own[qid].score[:size], ref[qid].score[:size]))
                    k = min(config.evaluation.cutoff, size)
                    overlaps.append(top_overlap(rerank_order(own[qid], size), rerank_order(ref[qid], size), k))
                out[(model_id, name, size)] = {
                    "kendall_tau_vs_reference": float(np.mean(taus)),
                    "top10_overlap_vs_reference": float(np.mean(overlaps)),
                }
    return out


def assemble(records: dict[str, dict]) -> None:
    config = load_config()
    ensure_dirs()
    model_ids = [m for m in records if m != config.evaluation.baseline]
    links = agreement(model_ids)
    rows = []
    for record in records.values():
        for row in record["rows"]:
            extra = links.get((row["model"], row["dataset"], row["candidate_size"]), {})
            rows.append(
                {
                    **row,
                    **{
                        "kendall_tau_vs_reference": extra.get("kendall_tau_vs_reference", float("nan")),
                        "top10_overlap_vs_reference": extra.get("top10_overlap_vs_reference", float("nan")),
                    },
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_DIR / "benchmark_summary.csv", index=False)
    (RESULTS_DIR / "benchmark.json").write_text(json.dumps({m: r["per_query"] for m, r in records.items()}))
    reference = config.evaluation.reference_model
    bootstrap = {}
    if reference in records:
        for model_id, record in records.items():
            if model_id == reference:
                continue
            for name, sizes in record["per_query"].items():
                for size, values in sizes.items():
                    base = records[reference]["per_query"][name][size]
                    qids = sorted(values)
                    a = np.array([values[q] for q in qids])
                    b = np.array([base[q] for q in qids])
                    bootstrap[f"{model_id}|{name}|{size}"] = paired_bootstrap(
                        a, b, config.evaluation.bootstrap_resamples, config.seed
                    )
    (RESULTS_DIR / "benchmark_bootstrap.json").write_text(json.dumps(bootstrap, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--stage", choices=["score", "report"], default="report")
    args = parser.parse_args()
    config = load_config()
    device = resolve_device(args.device)
    order = [config.evaluation.reference_model] + [m for m in config.models if m != config.evaluation.reference_model]
    chosen = [m for m in order if args.models is None or m in args.models]
    if args.stage == "score":
        score_stage(chosen, device)
        return
    records = {}
    for model_id in chosen:
        path = partial_path(model_id)
        if path.exists() and not args.fresh:
            records[model_id] = json.loads(path.read_text())
            continue
        records[model_id] = run_model(model_id, device)
        path.write_text(json.dumps(records[model_id]))
    baseline = config.evaluation.baseline
    base_path = partial_path(baseline)
    if base_path.exists() and not args.fresh:
        records[baseline] = json.loads(base_path.read_text())
    else:
        records[baseline] = baseline_record(device)
        base_path.write_text(json.dumps(records[baseline]))
    assemble(records)


if __name__ == "__main__":
    main()
