import argparse
import json
import random

import numpy as np

from . import checkpoint
from .adapters import build_reranker, dispose
from .adapters.cross import QWEN_INSTRUCTION
from .adapters.decision import RUBRICS
from .datasets import load_dataset
from .device import resolve_device
from .metrics import ndcg_at, rank_by
from .retrieval import build_candidates
from .scoring import CandidateScores
from .settings import RESULTS_DIR, ensure_dirs, load_config

TASKS = {
    "scifact": "Given a scientific claim, retrieve abstracts that support or refute the claim",
    "nfcorpus": "Given a medical or nutrition question, retrieve relevant scientific articles",
}
JINA_INSTRUCTION = "Judge whether the document answers the query."


def variant_plan(model_id: str) -> tuple[str, list[str]]:
    spec = load_config().models[model_id]
    if spec.is_decision:
        return "rubric", list(RUBRICS)
    if spec.backend == "qwen":
        return "instruction", ["default", "none", "task"]
    if spec.backend == "jina":
        return "instruction", ["plain", "instructed"]
    return "none", ["plain"]


def apply_variant(reranker, model_id: str, variant: str, dataset: str) -> None:
    spec = load_config().models[model_id]
    if spec.is_decision:
        reranker.use_rubric(variant)
    elif spec.backend == "qwen":
        reranker.instruction = {"default": QWEN_INSTRUCTION, "none": None, "task": TASKS[dataset]}[variant]
    elif spec.backend == "jina":
        reranker.instruction = JINA_INSTRUCTION if variant == "instructed" else None


def sample_ids(qids: list[str], count: int, seed: int) -> list[str]:
    ordered = sorted(qids)
    random.Random(seed).shuffle(ordered)
    return sorted(ordered[:count])


def spread(table: dict[str, dict[str, np.ndarray]], draws: int, seed: int) -> tuple[float, float, float]:
    names = list(table)
    datasets = list(next(iter(table.values())))
    if len(names) < 2:
        return 0.0, 0.0, 0.0
    sizes = {d: len(table[names[0]][d]) for d in datasets}
    rng = np.random.default_rng(seed)

    def deviation(index: dict[str, np.ndarray]) -> float:
        means = [np.mean([table[v][d][index[d]].mean() for d in datasets]) for v in names]
        return float(np.std(means, ddof=1))

    point = deviation({d: np.arange(sizes[d]) for d in datasets})
    samples = [deviation({d: rng.integers(0, sizes[d], sizes[d]) for d in datasets}) for _ in range(draws)]
    return point, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def run(device: str) -> dict:
    config = load_config()
    cfg = config.prompts
    ensure_dirs()
    cutoff = config.evaluation.cutoff
    prepared = {}
    for name in cfg.datasets:
        dataset = load_dataset(name)
        candidates = build_candidates(name)
        qids = sample_ids(list(dataset.queries), cfg.queries, config.seed)
        prepared[name] = (dataset, candidates, qids)
    report = {}
    for model_id in config.models:
        done = checkpoint.load("prompts", model_id)
        if done is not None:
            report[model_id] = done
            continue
        kind, variants = variant_plan(model_id)
        reranker = build_reranker(model_id, device)
        table: dict[str, dict[str, np.ndarray]] = {}
        for variant in variants:
            table[variant] = {}
            for name, (dataset, candidates, qids) in prepared.items():
                apply_variant(reranker, model_id, variant, name)
                pairs = [(dataset.queries[q], dataset.corpus[d]) for q in qids for d in candidates[q][: cfg.candidates]]
                result = reranker.score_pairs(pairs)
                cursor = 0
                values = []
                for qid in qids:
                    docs = candidates[qid][: cfg.candidates]
                    entry = CandidateScores(
                        docs, result.score[cursor : cursor + len(docs)], result.prob[cursor : cursor + len(docs)]
                    )
                    cursor += len(docs)
                    values.append(ndcg_at(rank_by(docs, entry.score), dataset.qrels[qid], cutoff))
                table[variant][name] = np.array(values)
        dispose(reranker)
        std, low, high = spread(table, 1000, config.seed)
        means = {v: float(np.mean([a.mean() for a in per.values()])) for v, per in table.items()}
        entry = {
            "variant_type": kind,
            "variants": {
                v: {"ndcg_at_10": means[v], "per_dataset": {d: float(a.mean()) for d, a in per.items()}}
                for v, per in table.items()
            },
            "mean_ndcg": float(np.mean(list(means.values()))),
            "std_ndcg": std,
            "std_ci_low": low,
            "std_ci_high": high,
        }
        report[model_id] = entry
        checkpoint.save("prompts", model_id, entry)
        print(
            f"{model_id:11s} variants={len(variants)} std={std:.4f} { {v: round(m, 3) for v, m in means.items()} }",
            flush=True,
        )
    (RESULTS_DIR / "prompt_sensitivity.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run(resolve_device(args.device))


if __name__ == "__main__":
    main()
