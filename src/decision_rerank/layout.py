import argparse
import json

import numpy as np

from . import checkpoint
from .adapters import build_reranker, dispose
from .datasets import load_dataset
from .device import resolve_device
from .metrics import ndcg_at, rank_by
from .prompts import sample_ids
from .retrieval import build_candidates
from .scoring import CandidateScores
from .settings import RESULTS_DIR, ensure_dirs, load_config

LAYOUTS = ["state", "question"]


def per_query(reranker, dataset, candidates, qids, depth: int, cutoff: int) -> np.ndarray:
    pairs = [(dataset.queries[q], dataset.corpus[d]) for q in qids for d in candidates[q][:depth]]
    result = reranker.score_pairs(pairs)
    cursor = 0
    values = []
    for qid in qids:
        docs = candidates[qid][:depth]
        entry = CandidateScores(docs, result.score[cursor : cursor + len(docs)], result.prob[cursor : cursor + len(docs)])
        cursor += len(docs)
        values.append(ndcg_at(rank_by(docs, entry.score), dataset.qrels[qid], cutoff))
    return np.array(values)


def paired(a: np.ndarray, b: np.ndarray, draws: int, seed: int) -> tuple[float, float, float]:
    diff = b - a
    rng = np.random.default_rng(seed)
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(draws)]
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run(device: str) -> dict:
    config = load_config()
    cfg = config.prompts
    ensure_dirs()
    prepared = {}
    for name in cfg.datasets:
        dataset = load_dataset(name)
        candidates = build_candidates(name)
        prepared[name] = (dataset, candidates, sample_ids(list(dataset.queries), cfg.queries, config.seed))
    report = {}
    for model_id, spec in config.models.items():
        if not spec.is_decision:
            continue
        done = checkpoint.load("layout", model_id)
        if done is not None:
            report[model_id] = done
            continue
        reranker = build_reranker(model_id, device)
        scores = {layout: {} for layout in LAYOUTS}
        for layout in LAYOUTS:
            reranker.layout = layout
            for name, (dataset, candidates, qids) in prepared.items():
                scores[layout][name] = per_query(reranker, dataset, candidates, qids, cfg.candidates, config.evaluation.cutoff)
        dispose(reranker)
        entry = {"layouts": {}, "difference": {}}
        for layout in LAYOUTS:
            per = {d: float(a.mean()) for d, a in scores[layout].items()}
            entry["layouts"][layout] = {"per_dataset": per, "ndcg_at_10": float(np.mean(list(per.values())))}
        for name in prepared:
            mean, low, high = paired(scores["state"][name], scores["question"][name], 10000, config.seed)
            entry["difference"][name] = {"mean": mean, "ci_low": low, "ci_high": high, "queries": len(prepared[name][2])}
        entry["difference"]["mean"] = float(np.mean([v["mean"] for v in entry["difference"].values()]))
        report[model_id] = entry
        checkpoint.save("layout", model_id, entry)
        print(model_id, {k: round(v["ndcg_at_10"], 3) for k, v in entry["layouts"].items()}, flush=True)
    (RESULTS_DIR / "layout_check.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    run(resolve_device(parser.parse_args().device))


if __name__ == "__main__":
    main()
