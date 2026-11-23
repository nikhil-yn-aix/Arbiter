import json

import numpy as np
import pandas as pd

from arbiter import stats as S
from arbiter.config import Study, lr_tag
from arbiter.data import eval_sets
from arbiter.io import Status, read_json, write_json, write_parquet
from arbiter.jobs import SetCache, all_runs, grid_jobs, main_jobs, run_dir, selection
from arbiter.metrics import per_query, rank


def confirmatory_ready(study: Study) -> list[str]:
    missing = []
    sel = selection(study)
    for j in grid_jobs(study):
        st = Status(run_dir(study, j)).read()
        if j["lr"] in (sel["lr"][j["arm"]], sel["shared_lr"]) and not st.get("sealed_scored"):
            missing.append(run_dir(study, j).name)
    for j in main_jobs(study):
        st = Status(run_dir(study, j)).read()
        if not (st["state"] == "done" or (st["state"] == "failed" and st.get("final"))):
            missing.append(run_dir(study, j).name)
    return missing


def collect(study: Study, confirmatory: bool, log) -> pd.DataFrame:
    if confirmatory:
        missing = confirmatory_ready(study)
        if missing:
            raise RuntimeError(f"{len(missing)} confirmatory runs not finished, e.g. {missing[:3]}")
    sel = read_json(study.path("runs") / "selection.json", {"lr": {}, "shared_lr": None})
    cache = SetCache(study)
    sets = eval_sets(study, beir=True)
    rows = []
    for r in all_runs(study):
        if r.get("state") != "done":
            continue
        d = study.path("runs") / r["stage"] / r["run_id"]
        protocol = r.get("protocol", "tuned")
        tuned = (r["stage"] == "main" and protocol == "tuned") or (r["stage"] == "grid" and r["lr"] == sel["lr"].get(r["arm"]))
        shared = (r["stage"] == "grid" and r["lr"] == sel.get("shared_lr")) or protocol == "shared"
        for name in sets:
            if name != "dev" and not confirmatory:
                continue
            f = d / "scores" / ("dev.parquet" if name == "dev" else f"sealed/{name}.parquet")
            if not f.exists():
                continue
            cands, qrels = cache[name]
            sc = pd.read_parquet(f)
            if not (sc.qid.to_numpy() == cands.qid.to_numpy()).all():
                raise ValueError(f"{f}: candidate order mismatch")
            pq = per_query(rank(cands, sc.score.to_numpy()), qrels)
            rows.append(
                pq.assign(
                    run_id=r["run_id"],
                    stage=r["stage"],
                    arm=r["arm"],
                    budget=r.get("budget"),
                    lr=r.get("lr"),
                    seed=r.get("seed"),
                    pooling=r.get("pooling") or "mean",
                    tuned=tuned,
                    shared=shared,
                    set=name,
                )
            )
    for name in sets:
        if name != "dev" and not confirmatory:
            continue
        cands, qrels = cache[name]
        pq = per_query(rank(cands, -cands.bm25_rank.to_numpy(dtype=float)), qrels)
        rows.append(pq.assign(run_id="bm25", stage="baseline", arm="BM25", budget=None, lr=None, seed=0, pooling="-", tuned=False, shared=False, set=name))
    out = pd.concat(rows, ignore_index=True)
    write_parquet(study.path("results") / "per_query.parquet", out)
    runs = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in all_runs(study)])
    write_parquet(study.path("results") / "runs.parquet", runs)
    log("collected", rows=len(out), runs=len(runs), confirmatory=confirmatory)
    return out


def _fmt(x, d=3):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def write_table(df: pd.DataFrame, path, caption: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False)
    cols = " ".join("l" if i == 0 else "r" for i in range(len(df.columns)))
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("±", r"$\pm$").replace("Δ", r"$\Delta$")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        " & ".join(esc(c) for c in df.columns) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(esc(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.with_suffix(".tex").write_text("\n".join(lines) + "\n")


MIN_LOGS = 10


def training_failures(study: Study, run_ids) -> dict:
    """runs of at least 100 steps whose mean loss over the last tenth stays above 80% of the uniform-ranking loss ln(group)."""
    chance = np.log(study.train["group"])
    out = {}
    for rid in run_ids:
        for f in study.path("runs").glob(f"*/{rid}/train.jsonl"):
            loss = [json.loads(x)["loss"] for x in f.read_text().splitlines() if x.strip()]
            tail = float(np.mean(loss[-max(1, len(loss) // 10) :])) if len(loss) >= MIN_LOGS else None
            if tail is not None and tail > 0.8 * chance:
                out[rid] = round(tail, 3)
    return out


def exploratory(study: Study, tuned: pd.DataFrame, failed: set, bad: set, reps: int, alpha: float) -> dict:
    """post-unsealing sensitivity checks at the small primary budget."""
    a, dec = study.analysis, study.decision_arms
    b = a["primary_budgets"][0]
    controls = [c for c, v in study.arms.items() if v["kind"] == "control"]
    base_bad = {r for r in bad if r.startswith("B__") and f"__q{b}__" in r}
    views = {
        "excl_failed": tuned[~tuned.run_id.isin(failed)],
        "excl_failed_and_base_bad": tuned[~tuned.run_id.isin(failed | base_bad)],
        "seed0": tuned[(tuned.seed == 0) & ~tuned.run_id.isin(failed)],
        "seeds12": tuned[(tuned.seed > 0) & ~tuned.run_id.isin(failed)],
    }
    out = {"budget": b, "base_bad_excluded": sorted(base_bad)}
    for name, view in views.items():
        vsm = S.seed_mean(view)
        for s in a["primary_sets"]:
            e = S.class_effect(vsm, b, s, dec, "B", reps, alpha)
            if e:
                out[f"{name}/class/{s}"] = e["re"]
            if name == "excl_failed":
                for ctl in controls:
                    c = S.contrast(vsm, b, s, [ctl], "B", reps, alpha)
                    if c:
                        out[f"{name}/{ctl}_minus_B/{s}"] = c
    return out


def probe_rank(probes: pd.DataFrame, sm: pd.DataFrame, budget: int) -> dict:
    """spearman between best-layer frozen probe gain and fine-tuned held-out gain over all arms, exact permutation p."""
    from itertools import permutations

    best = probes.groupby("arm")["dev_mrr10"].max()
    ft = sm[(sm.set == "heldout") & (sm.budget == budget)].groupby("arm")["value"].mean()
    arms = [x for x in best.index if x in ft.index]
    x = (best[arms] - best["B"]).rank().to_numpy()
    y = (ft[arms] - ft["B"]).rank().to_numpy()
    rho = lambda yy: float(np.corrcoef(x, yy)[0, 1])
    obs = rho(y)
    perms = [rho(y[list(p)]) for p in permutations(range(len(arms)))]
    return {"arms": arms, "rho": obs, "p_exact": float(np.mean([abs(r) >= abs(obs) - 1e-12 for r in perms]))}


def analyze(study: Study, log) -> dict:
    a = study.analysis
    reps, alpha = a["bootstrap"], a["alpha"]
    pq = pd.read_parquet(study.path("results") / "per_query.parquet")
    lv = S.long_values(pq)
    tuned = lv[lv.tuned & (lv.pooling == "mean")]
    sm = S.seed_mean(tuned)
    dec = study.decision_arms
    out = {"primary": {}, "secondary": {}, "descriptive": {}}
    budgets = sorted(tuned.budget.dropna().unique().astype(int))
    effects = {}
    for s in ["heldout", "dl", "dev"] + [f"beir-{b}" for b in study.eval["beir"]]:
        for b in budgets:
            e = S.class_effect(sm, b, s, dec, "B", reps, alpha)
            if e:
                effects[(s, b)] = e
    pb = a["primary_budgets"]
    raw_p = {f"{b}": max(effects[(s, b)]["re"]["p"] for s in a["primary_sets"]) for b in pb if all((s, b) in effects for s in a["primary_sets"])}
    adj = S.holm(raw_p)
    for b in pb:
        if f"{b}" not in adj:
            continue
        out["primary"][f"class_delta_{b}"] = {
            "sets": {s: effects[(s, b)] for s in a["primary_sets"]},
            "p_intersection_union": raw_p[f"{b}"],
            "p_holm": adj[f"{b}"],
            "agree_positive": all(effects[(s, b)]["re"]["mean"] > 0 for s in a["primary_sets"]),
            "tost_heldout": S.tost(effects[("heldout", b)]["re"]["ci90"], a["equivalence_margin"]["heldout"]),
        }
    if all(("heldout", b) in effects for b in pb):
        out["primary"]["outcome"] = S.outcome(effects[("heldout", pb[0])]["re"], effects[("heldout", pb[1])]["re"], a["equivalence_margin"]["heldout"], alpha)
    spec = {}
    for ctl in [c for c, v in study.arms.items() if v["kind"] == "control"]:
        for b in budgets:
            for s in a["primary_sets"]:
                c = S.contrast(sm, b, s, dec, ctl, reps, alpha)
                if c:
                    spec[f"{ctl}_{b}_{s}"] = c
    p1k = {k: v["p"] for k, v in spec.items() if f"_{pb[0]}_heldout" in k}
    for k, v in S.holm(p1k).items():
        spec[k]["p_holm"] = v
    out["secondary"]["specificity"] = spec
    runs = tuned.groupby(["run_id", "arm", "budget", "seed", "set"], as_index=False)["value"].mean()
    bad = S.bad_runs(runs[runs.set == "heldout"], a["bad_run_drop"])
    out["secondary"]["stability"] = {
        "fisher": S.fisher(bad, dec, "B"),
        "bad_runs": bad[bad.bad].run_id.tolist(),
        "seed_sd": runs[runs.set == "heldout"].groupby(["arm", "budget"])["value"].std().dropna().round(4).to_dict(),
    }
    out["secondary"]["stability"]["training_failures"] = training_failures(study, runs.run_id.unique())
    failed = set(out["secondary"]["stability"]["training_failures"])
    if failed:
        fsm = S.seed_mean(tuned[~tuned.run_id.isin(failed)])
        fx = {"excluded": sorted(failed)}
        for b in budgets:
            for s in a["primary_sets"]:
                e = S.class_effect(fsm, b, s, dec, "B", reps, alpha)
                if e:
                    fx[f"{s}@{b}"] = {"re": e["re"], "fe": e["fe"], "per_arm": {k: {"mean": v["mean"], "ci": v["ci"]} for k, v in e["per_arm"].items()}}
                for ctl in [c for c, v in study.arms.items() if v["kind"] == "control"]:
                    c = S.contrast(fsm, b, s, dec, ctl, reps, alpha)
                    if c:
                        fx[f"vs_{ctl}_{s}@{b}"] = c
        if all(f"heldout@{b}" in fx for b in pb):
            fx["outcome"] = S.outcome(fx[f"heldout@{pb[0]}"]["re"], fx[f"heldout@{pb[1]}"]["re"], a["equivalence_margin"]["heldout"], alpha)
        out["secondary"]["failure_excluded"] = fx
        out["secondary"]["exploratory"] = exploratory(study, tuned, failed, set(out["secondary"]["stability"]["bad_runs"]), reps, alpha)
    out["secondary"]["stability"]["seed_sd"] = {f"{k[0]}@{k[1]}": v for k, v in out["secondary"]["stability"]["seed_sd"].items()}
    curve = sm[(sm.arm == "B") & (sm.set == "heldout")].groupby("budget")["value"].mean().sort_index()
    if ("heldout", pb[0]) in effects:
        target = float(sm[(sm.arm.isin(dec)) & (sm.set == "heldout") & (sm.budget == pb[0])].groupby("arm")["value"].mean().mean())
        sv = S.savings(curve, target)
        out["descriptive"]["query_equivalent_savings"] = {"target": target, "base_budget_needed": sv, "ratio": sv / pb[0] if sv else None}
    out["descriptive"]["effects"] = {
        f"{s}@{b}": {"re": e["re"], "fe": e["fe"], "per_arm": {k: {"mean": v["mean"], "ci": v["ci"]} for k, v in e["per_arm"].items()}}
        for (s, b), e in effects.items()
    }
    shared = lv[lv.shared & (lv.pooling == "mean")]
    if not shared.empty:
        ssm = S.seed_mean(shared)
        out["secondary"]["protocol"] = {
            s: {
                "tuned": S.class_effect(sm[sm.budget == pb[0]], pb[0], s, dec, "B", reps, alpha)["re"],
                "shared": (S.class_effect(ssm, pb[0], s, dec, "B", reps, alpha) or {}).get("re"),
            }
            for s in a["primary_sets"]
        }
    swap = lv[(lv.stage == "main") & lv.shared & ~lv.tuned & (lv.pooling == "mean")]
    if not swap.empty:
        keys = set(zip(swap.arm, swap.budget))
        keep = tuned[[k not in keys for k in zip(tuned.arm, tuned.budget)]]
        alt = S.seed_mean(pd.concat([keep, swap]))
        sens = {"replaced": sorted(f"{arm}@{int(b)}" for arm, b in keys)}
        for b in sorted({int(b) for _, b in keys}):
            for s in a["primary_sets"]:
                e = S.class_effect(alt, b, s, dec, "B", reps, alpha)
                if e:
                    sens[f"{s}@{b}"] = {"re": e["re"], "fe": e["fe"], "per_arm": {k: {"mean": v["mean"], "ci": v["ci"]} for k, v in e["per_arm"].items()}}
        if ("heldout", pb[0]) in effects and f"heldout@{pb[1]}" in sens:
            sens["outcome"] = S.outcome(effects[("heldout", pb[0])]["re"], sens[f"heldout@{pb[1]}"]["re"], a["equivalence_margin"]["heldout"], alpha)
        out["secondary"]["lr_sensitivity"] = sens
    out["descriptive"]["without_D4"] = {
        f"{s}@{b}": S.class_effect(sm, b, s, [d for d in dec if d != "D4"], "B", reps, alpha)["re"] for b in pb for s in a["primary_sets"] if (s, b) in effects
    }
    probes_path = study.path("runs") / "extra" / "probes.parquet"
    if probes_path.exists():
        out["descriptive"]["probe_vs_finetuned"] = probe_rank(pd.read_parquet(probes_path), sm, pb[0])
    write_json(study.path("results") / "stats.json", out)
    tables(study, lv, sm, effects, out)
    log("analyzed", outcome=out["primary"].get("outcome"))
    return out


def tables(study: Study, lv: pd.DataFrame, sm: pd.DataFrame, effects: dict, out: dict) -> None:
    tdir = study.path("results") / "tables"
    label = {a: s["label"] for a, s in study.arms.items()}
    ex = study.path("runs") / "extra"
    if (ex / "drift.parquet").exists():
        drift = pd.read_parquet(ex / "drift.parquet")
        cos = pd.read_parquet(ex / "cosine.parquet")
        rows = []
        for arm, spec in study.arms.items():
            if arm == "B":
                continue
            c = cos[(cos.a == arm) & (cos.b != arm)]
            top = c.loc[c.cosine.idxmax()] if len(c) else None
            rows.append(
                {
                    "arm": arm,
                    "checkpoint": spec["label"],
                    "role": spec["kind"],
                    "card-reported data": spec.get("card", "–"),
                    "rel. drift": _fmt(float(drift[(drift.arm == arm) & (drift.layer == "all")].rel_drift.iloc[0]) * 100, 2) + "%",
                    "closest arm (cos)": f"{label[top.b]} ({top.cosine:.2f})" if top is not None else "–",
                }
            )
        write_table(
            pd.DataFrame(rows),
            tdir / "t1_inventory",
            "checkpoints. drift is relative l2 distance of encoder weights from ModernBERT-large; cos is the cosine between drift vectors.",
        )
    runs = lv[lv.tuned & (lv.pooling == "mean")].groupby(["run_id", "arm", "budget", "seed", "set"], as_index=False)["value"].mean()
    rows = []
    for (arm, b), g in runs.groupby(["arm", "budget"]):
        row = {"arm": label.get(arm, arm), "budget": int(b), "seeds": int(g.seed.nunique())}
        for s in ["heldout", "dl"]:
            v = g[g.set == s].value
            row[s] = f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else _fmt(v.mean() if len(v) else None)
        rows.append(row)
    t2 = pd.DataFrame(rows).sort_values(["budget", "arm"])
    for (s, b), e in sorted(effects.items(), key=lambda x: (x[0][1], x[0][0])):
        if s in ("heldout", "dl"):
            re = e["re"]
            t2 = pd.concat(
                [
                    t2,
                    pd.DataFrame(
                        [
                            {
                                "arm": f"Δ class, {e['re']['k']} ckpts ({s})",
                                "budget": b,
                                "seeds": "–",
                                s: f"{re['mean']:+.3f} [{re['ci'][0]:+.3f}, {re['ci'][1]:+.3f}] p={re['p']:.3f}",
                            }
                        ]
                    ),
                ]
            )
    write_table(
        t2.fillna("–"),
        tdir / "t2_main",
        "main results. mean ± sd over seeds; Δ class is the random-effects mean over decision checkpoints with 95% t interval.",
    )
    spec = out["secondary"].get("specificity", {})
    t3 = pd.DataFrame(
        [
            {
                "contrast": f"decision − {label[k.split('_')[0]]}",
                "budget": int(k.split("_")[1]),
                "set": k.split("_")[2],
                "Δ": f"{v['mean']:+.3f}",
                "95% CI": f"[{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]",
                "p (holm)": _fmt(v.get("p_holm", v["p"])),
            }
            for k, v in spec.items()
        ]
    )
    if len(t3):
        write_table(t3, tdir / "t3_specificity", "decision checkpoints versus controls (paired bootstrap over queries).")
    beir = lv[lv.set.str.startswith("beir-") & (lv.pooling == "mean")]
    if len(beir):
        t4 = beir.groupby(["set", "arm", "budget"], dropna=False)["value"].mean().unstack("set").reset_index()
        t4["arm"] = t4["arm"].map(lambda a: label.get(a, a))
        write_table(t4.round(3).fillna("–"), tdir / "t4_beir", "zero-shot transfer (exploratory), ndcg@10, bm25 top-30.")
    sel = read_json(study.path("runs") / "selection.json")
    if sel:
        g = pd.DataFrame(sel["grid"])
        t5 = g.pivot(index="arm", columns="lr", values="dev_mrr10").round(4).reset_index()
        t5.columns = ["arm"] + [f"lr {lr_tag(c)}" for c in t5.columns[1:]]
        t5["selected"] = t5["arm"].map(lambda a: lr_tag(sel["lr"][a]))
        t5["arm"] = t5["arm"].map(label)
        write_table(t5, tdir / "t5_lr_grid", "learning-rate grid at 1k queries (dev mrr@10, seed 0).")
    rs = pd.read_parquet(study.path("results") / "runs.parquet")
    if "train_seconds" in rs:
        t6 = (
            rs.groupby("stage")
            .agg(runs=("run_id", "count"), gpu_hours=("train_seconds", lambda x: x.sum() / 3600), pairs_per_s=("pairs_per_s", "mean"))
            .reset_index()
            .round(2)
        )
        write_table(t6, tdir / "t6_compute", "training compute by stage (scoring time excluded).")
    sec = out["secondary"]
    variants = [("preregistered", {f"{k[0]}@{k[1]}": v for k, v in effects.items()}), ("excl. failed runs", sec.get("failure_excluded", {}))]
    variants.append(("laya 16k at base lr", sec.get("lr_sensitivity", {})))
    rows = []
    for name, src in variants:
        for b in study.analysis["primary_budgets"]:
            row = {"analysis": name, "budget": b}
            for st in study.analysis["primary_sets"]:
                e = src.get(f"{st}@{b}")
                row[st] = f"{e['re']['mean']:+.3f} [{e['re']['ci'][0]:+.3f}, {e['re']['ci'][1]:+.3f}] (k={e['re']['k']})" if e else "–"
            if any(row[st] != "–" for st in study.analysis["primary_sets"]):
                rows.append(row)
    write_table(
        pd.DataFrame(rows),
        tdir / "t7_sensitivity",
        "class effect over decision checkpoints (random effects, 95% t interval) under the preregistered analysis and two sensitivity analyses.",
    )
