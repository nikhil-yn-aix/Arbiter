import numpy as np
import pandas as pd

from arbiter import stats as S
from arbiter.config import Study
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
        tuned = r["stage"] == "main" or (r["stage"] == "grid" and r["lr"] == sel["lr"].get(r["arm"]))
        shared = r["stage"] == "grid" and r["lr"] == sel.get("shared_lr")
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
        t5.columns = ["arm"] + [f"lr {c:.0e}" for c in t5.columns[1:]]
        t5["selected"] = t5["arm"].map(lambda a: f"{sel['lr'][a]:.0e}")
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
