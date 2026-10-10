import math

import numpy as np
from scipy.stats import kendalltau


def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at(ranked: list[str], grades: dict[str, int], k: int) -> float:
    ideal = dcg(sorted((g for g in grades.values() if g > 0), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return dcg([max(grades.get(d, 0), 0) for d in ranked[:k]]) / ideal


def recall_at(ranked: list[str], grades: dict[str, int], k: int, threshold: int) -> float:
    relevant = {d for d, g in grades.items() if g >= threshold}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def mrr_at(ranked: list[str], grades: dict[str, int], k: int, threshold: int) -> float:
    for position, doc in enumerate(ranked[:k]):
        if grades.get(doc, 0) >= threshold:
            return 1.0 / (position + 1)
    return 0.0


def rank_by(doc_ids: list[str], scores: np.ndarray) -> list[str]:
    order = np.lexsort((np.arange(len(scores)), -np.asarray(scores)))
    return [doc_ids[i] for i in order]


def query_metrics(ranked: list[str], grades: dict[str, int], k: int, threshold: int) -> dict[str, float]:
    return {
        "ndcg": ndcg_at(ranked, grades, k),
        "recall": recall_at(ranked, grades, k, threshold),
        "mrr": mrr_at(ranked, grades, k, threshold),
    }


def kendall(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    value = kendalltau(a, b).statistic
    return 0.0 if value is None or math.isnan(value) else float(value)


def top_overlap(a: list[str], b: list[str], k: int) -> float:
    return len(set(a[:k]) & set(b[:k])) / k


def bootstrap_mean(values: np.ndarray, resamples: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.array([values[rng.integers(0, n, n)].mean() for _ in range(resamples)])
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, resamples: int, seed: int) -> dict[str, float]:
    diff = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, (resamples, n))
    means = diff[idx].mean(axis=1)
    mean = float(diff.mean())
    return {
        "mean_diff": mean,
        "ci_low": float(min(np.percentile(means, 2.5), mean)),
        "ci_high": float(max(np.percentile(means, 97.5), mean)),
        "p_two_sided": float(min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean()))),
    }


def wilson(successes: float, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def calibration(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> dict:
    probs = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    labels = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0, 1, bins + 1)
    which = np.minimum(np.digitize(probs, edges[1:-1]), bins - 1)
    rows = []
    ece = 0.0
    for b in range(bins):
        mask = which == b
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin": b,
                    "count": 0,
                    "confidence": float((edges[b] + edges[b + 1]) / 2),
                    "accuracy": None,
                    "low": None,
                    "high": None,
                }
            )
            continue
        conf = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        low, high = wilson(labels[mask].sum(), count)
        rows.append({"bin": b, "count": count, "confidence": conf, "accuracy": acc, "low": low, "high": high})
        ece += count / len(probs) * abs(acc - conf)
    filled = [r for r in rows if r["count"]]
    if len(filled) >= 2:
        weights = np.array([r["count"] for r in filled], dtype=np.float64)
        x = np.array([r["confidence"] for r in filled])
        y = np.array([r["accuracy"] for r in filled])
        slope, intercept = np.polyfit(x, y, 1, w=np.sqrt(weights)) if np.ptp(x) > 0 else (0.0, float(y.mean()))
    else:
        slope, intercept = 0.0, float(labels.mean())
    return {
        "ece": float(ece),
        "brier": float(np.mean((probs - labels) ** 2)),
        "slope": float(slope),
        "intercept": float(intercept),
        "positives": float(labels.mean()),
        "bins": rows,
    }


def macro_bootstrap(groups: list[np.ndarray], resamples: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    draws = np.zeros(resamples)
    for values in groups:
        idx = rng.integers(0, len(values), (resamples, len(values)))
        draws += values[idx].mean(axis=1)
    draws /= len(groups)
    point = float(np.mean([g.mean() for g in groups]))
    return point, float(min(np.percentile(draws, 2.5), point)), float(max(np.percentile(draws, 97.5), point))
