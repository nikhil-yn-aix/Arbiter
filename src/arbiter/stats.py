import numpy as np
import pandas as pd
from scipy import stats as sps

METRIC = {"heldout": "mrr10", "dl": "ndcg10", "dev": "mrr10"}


def metric_for(set_name: str) -> str:
    return METRIC.get(set_name, "ndcg10")


def long_values(pq: pd.DataFrame) -> pd.DataFrame:
    """one row per (run, set, qid) with the set's primary metric; dl19+dl20 pooled as 'dl'."""
    df = pq.copy()
    dl = df[df.set.isin(["dl19", "dl20"])].assign(qid=lambda d: d.set + ":" + d.qid, set="dl")
    df = pd.concat([df, dl], ignore_index=True)
    df["value"] = [getattr(r, metric_for(r.set)) for r in df[["set", "mrr10", "ndcg10"]].itertuples(index=False)]
    return df


def seed_mean(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["arm", "budget", "set", "qid"], as_index=False).agg(value=("value", "mean"), seeds=("seed", "nunique"))


def _paired(sm: pd.DataFrame, budget: int, set_name: str) -> pd.DataFrame:
    sub = sm[(sm.budget == budget) & (sm.set == set_name)]
    return sub.pivot(index="qid", columns="arm", values="value").dropna(axis=1, how="all").dropna()


def bootstrap(x: np.ndarray, reps: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(reps, len(x)))
    return x[idx].mean(axis=1)


def _boot_summary(x: np.ndarray, reps: int, alpha: float) -> dict:
    b = bootstrap(x, reps)
    p = 2 * min((b <= 0).mean(), (b >= 0).mean())
    lo, hi = np.quantile(b, [alpha / 2, 1 - alpha / 2])
    lo90, hi90 = np.quantile(b, [alpha, 1 - alpha])
    return {"mean": float(x.mean()), "ci": [float(lo), float(hi)], "ci90": [float(lo90), float(hi90)], "p": float(min(1.0, p)), "n_queries": int(len(x))}


def class_effect(sm: pd.DataFrame, budget: int, set_name: str, arms: list[str], base: str, reps: int, alpha: float) -> dict | None:
    t = _paired(sm, budget, set_name)
    arms = [a for a in arms if a in t]
    if base not in t or len(arms) < 2:
        return None
    deltas = t[arms].sub(t[base], axis=0)
    per_arm = deltas.mean(axis=0)
    k = len(arms)
    m, se = float(per_arm.mean()), float(per_arm.std(ddof=1) / np.sqrt(k))
    tq, tq90 = sps.t.ppf(1 - alpha / 2, k - 1), sps.t.ppf(1 - alpha, k - 1)
    re = {
        "mean": m,
        "se": se,
        "ci": [m - tq * se, m + tq * se],
        "ci90": [m - tq90 * se, m + tq90 * se],
        "p": float(2 * sps.t.sf(abs(m / se), k - 1)) if se > 0 else 0.0,
        "k": k,
    }
    fe = _boot_summary(deltas.mean(axis=1).to_numpy(), reps, alpha)
    arm_fe = {a: _boot_summary(deltas[a].to_numpy(), reps, alpha) for a in arms}
    return {"re": re, "fe": fe, "per_arm": arm_fe, "arms": arms}


def contrast(sm: pd.DataFrame, budget: int, set_name: str, group: list[str], other: str, reps: int, alpha: float) -> dict | None:
    t = _paired(sm, budget, set_name)
    group = [a for a in group if a in t]
    if other not in t or not group:
        return None
    return _boot_summary((t[group].mean(axis=1) - t[other]).to_numpy(), reps, alpha)


def holm(p: dict[str, float]) -> dict[str, float]:
    keys = sorted(p, key=p.get)
    out, running = {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (len(keys) - i) * p[k]))
        out[k] = running
    return out


def tost(ci90: list[float], margin: float) -> bool:
    return -margin < ci90[0] and ci90[1] < margin


def bad_runs(run_means: pd.DataFrame, drop: float) -> pd.DataFrame:
    med = run_means.groupby(["arm", "budget"])["value"].transform("median")
    return run_means.assign(bad=run_means.value < med - drop)


def fisher(bad: pd.DataFrame, group: list[str], base: str) -> dict:
    g = bad[bad.arm.isin(group)]
    b = bad[bad.arm == base]
    table = [[int(g.bad.sum()), int((~g.bad).sum())], [int(b.bad.sum()), int((~b.bad).sum())]]
    res = sps.fisher_exact(table)
    return {"table": table, "odds_ratio": float(res.statistic) if np.isfinite(res.statistic) else None, "p": float(res.pvalue)}


def savings(curve: pd.Series, target: float) -> float | None:
    """budget at which the curve (indexed by budget) first reaches target, interpolated in log2 budget."""
    xs, ys = np.log2(curve.index.to_numpy(dtype=float)), curve.to_numpy()
    if target <= ys[0]:
        return float(curve.index[0])
    for i in range(1, len(xs)):
        if ys[i] >= target:
            f = (target - ys[i - 1]) / (ys[i] - ys[i - 1])
            return float(2 ** (xs[i - 1] + f * (xs[i] - xs[i - 1])))
    return None


def outcome(re_small: dict, re_large: dict, margin: float, alpha: float) -> str:
    adv = re_small["p"] < alpha and re_small["mean"] > 0
    persists = re_large["p"] < alpha and re_large["mean"] > 0
    if adv and persists:
        return "persists"
    if adv and tost(re_large["ci90"], margin) and re_small["mean"] > re_large["mean"]:
        return "fades"
    if tost(re_small["ci90"], margin):
        return "null"
    if adv:
        return "advantage (large budget inconclusive)"
    return "inconclusive"
