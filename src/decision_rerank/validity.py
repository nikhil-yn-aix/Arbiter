import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import checkpoint
from .adapters import build_reranker, dispose
from .calibration import collect
from .datasets import load_dataset
from .device import resolve_device
from .layout import per_query
from .metrics import calibration, ndcg_at
from .prompts import sample_ids
from .retrieval import build_candidates
from .scoring import load_scores, score_path
from .settings import RESULTS_DIR, ensure_dirs, load_config

PRECISION_DATASET = "scifact"
RANDOM_DRAWS = 200


def random_order(config) -> dict:
    out = {}
    for name in config.datasets:
        dataset = load_dataset(name)
        candidates = build_candidates(name)
        rng = np.random.default_rng(config.seed)
        values = []
        for qid in dataset.queries:
            docs = candidates[qid][: config.cascade.pool]
            for _ in range(RANDOM_DRAWS):
                order = [docs[i] for i in rng.permutation(len(docs))]
                values.append(ndcg_at(order, dataset.qrels[qid], config.evaluation.cutoff))
        out[name] = float(np.mean(values))
    out["mean"] = float(np.mean(list(out.values())))
    return out


def precision(device: str) -> dict:
    config = load_config()
    cfg = config.prompts
    dataset = load_dataset(PRECISION_DATASET)
    candidates = build_candidates(PRECISION_DATASET)
    qids = sample_ids(list(dataset.queries), cfg.queries, config.seed)
    report = {}
    for model_id, spec in config.models.items():
        if not spec.is_decision:
            continue
        done = checkpoint.load("validity", model_id)
        if done is not None:
            report[model_id] = done
            continue
        reranker = build_reranker(model_id, device)
        pairs = [(dataset.queries[q], dataset.corpus[d]) for q in qids for d in candidates[q][: cfg.candidates]]
        default = reranker.score_pairs(pairs)
        values = {"default": per_query(reranker, dataset, candidates, qids, cfg.candidates, config.evaluation.cutoff)}
        reranker.agent.amp_enabled = False
        full = reranker.score_pairs(pairs)
        values["full"] = per_query(reranker, dataset, candidates, qids, cfg.candidates, config.evaluation.cutoff)
        dispose(reranker)
        entry = {
            "default_ndcg": float(values["default"].mean()),
            "full_ndcg": float(values["full"].mean()),
            "mean_abs_score_difference": float(np.abs(default.score - full.score).mean()),
            "queries": len(qids),
        }
        report[model_id] = entry
        checkpoint.save("validity", model_id, entry)
        print(model_id, entry, flush=True)
    return report


def pairs_by_query(model_id: str, names: list[str]):
    config = load_config()
    groups, probs, labels = [], [], []
    for name in names:
        dataset = load_dataset(name)
        threshold = config.datasets[name].binary_threshold
        for qid, entry in load_scores(score_path(model_id, name)).items():
            grades = dataset.qrels[qid]
            probs.append(entry.prob)
            labels.append(np.array([1.0 if grades.get(d, 0) >= threshold else 0.0 for d in entry.doc_ids]))
            groups.append(np.full(len(entry.doc_ids), len(groups)))
    return np.concatenate(groups), np.concatenate(probs), np.concatenate(labels)


def recalibrated(model_id: str, names: list[str], folds: int = 5) -> dict:
    groups, probs, labels = pairs_by_query(model_id, names)
    logit = np.log(np.clip(probs, 1e-6, 1 - 1e-6) / (1 - np.clip(probs, 1e-6, 1 - 1e-6)))[:, None]
    fold = groups % folds
    adjusted = np.zeros_like(probs)
    for k in range(folds):
        train = fold != k
        fit = LogisticRegression(C=1e6).fit(logit[train], labels[train])
        adjusted[~train] = fit.predict_proba(logit[~train])[:, 1]
    before = calibration(probs, labels, 10)
    after = calibration(adjusted, labels, 10)
    return {"ece_before": before["ece"], "ece_after": after["ece"], "brier_before": before["brier"], "brier_after": after["brier"]}


def run(device: str) -> dict:
    config = load_config()
    ensure_dirs()
    names = list(config.datasets)
    report = {
        "random_order": random_order(config),
        "precision": precision(device),
        "recalibration": {m: recalibrated(m, names) for m in config.models},
    }
    (RESULTS_DIR / "validity_check.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    run(resolve_device(parser.parse_args().device))


if __name__ == "__main__":
    main()
