import json

import numpy as np

from .datasets import load_dataset
from .metrics import calibration
from .scoring import load_scores, score_path
from .settings import RESULTS_DIR, ensure_dirs, load_config


def collect(model_id: str, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    config = load_config()
    probs, labels = [], []
    for name in names:
        dataset = load_dataset(name)
        scores = load_scores(score_path(model_id, name))
        threshold = config.datasets[name].binary_threshold
        for qid, entry in scores.items():
            grades = dataset.qrels[qid]
            probs.append(entry.prob)
            labels.append(np.array([1.0 if grades.get(d, 0) >= threshold else 0.0 for d in entry.doc_ids]))
    return np.concatenate(probs), np.concatenate(labels)


def run() -> dict:
    config = load_config()
    ensure_dirs()
    names = list(config.datasets)
    report = {}
    for model_id in config.models:
        probs, labels = collect(model_id, names)
        report[model_id] = {**calibration(probs, labels, 10), "pairs": int(len(probs))}
        report[model_id]["per_dataset"] = {}
        for name in names:
            p, y = collect(model_id, [name])
            summary = calibration(p, y, 10)
            report[model_id]["per_dataset"][name] = {k: summary[k] for k in ("ece", "brier", "slope", "intercept")}
        print(f"{model_id:11s} ece={report[model_id]['ece']:.4f} brier={report[model_id]['brier']:.4f}", flush=True)
    (RESULTS_DIR / "calibration.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    run()


if __name__ == "__main__":
    main()
