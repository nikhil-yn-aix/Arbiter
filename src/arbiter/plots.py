import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator

from arbiter import stats as S
from arbiter.config import Study
from arbiter.io import write_json

FULL, HALF = 6.5, 3.2
INK, MUTED, FAINT = "#1a1a1a", "#6b6b6b", "#d9d9d9"
COLOR = {"B": INK, "D1": "#0072B2", "D2": "#E69F00", "D3": "#009E73", "D4": "#D55E00", "D5": "#CC79A7", "N": "#8a8a8a", "R": "#b3b3b3"}
MARKER = {"B": "o", "D1": "s", "D2": "^", "D3": "D", "D4": "v", "D5": "P", "N": "X", "R": "h"}
DASH = {"N": (0, (4, 2)), "R": (0, (1, 1.6))}
BUDGETS = [250, 1000, 4000, 16000]
SET_LABEL = {"heldout": "held-out MS MARCO, MRR@10", "dl": "TREC DL19+20, nDCG@10"}


def _short(b) -> str:
    b = int(b)
    return f"{b // 1000}k" if b >= 1000 and b % 1000 == 0 else str(b)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#9a9a9a",
            "axes.linewidth": 0.6,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#ececec",
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.3,
            "lines.markersize": 3.6,
            "legend.frameon": False,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "figure.dpi": 150,
            "axes.titlesize": 8,
        }
    )


def _budget_axis(ax) -> None:
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(BUDGETS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["250", "1k", "4k", "16k"])
    ax.set_xlim(190, 21000)
    ax.set_xlabel("training queries")


def _panel(ax, letter: str) -> None:
    ax.text(0.0, 1.06, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="left", va="bottom", color=INK)


def layout_issues(fig) -> list[str]:
    """pairs of overlapping text or legend boxes after drawing."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = []
    for ax in fig.axes:
        x0, x1 = sorted(ax.get_xlim())
        y0, y1 = sorted(ax.get_ylim())
        ticks = [t for t in ax.get_xticklabels() if x0 <= t.get_position()[0] <= x1] + [t for t in ax.get_yticklabels() if y0 <= t.get_position()[1] <= y1]
        for t in ticks + [ax.xaxis.label, ax.yaxis.label, ax.title] + ax.texts:
            if t.get_visible() and t.get_text():
                boxes.append((f"text:{t.get_text()[:20]}", t.get_window_extent(r)))
        if ax.get_legend():
            boxes.append(("legend", ax.get_legend().get_window_extent(r)))
    for lg in fig.legends:
        boxes.append(("figlegend", lg.get_window_extent(r)))
        for ax in fig.axes:
            if lg.get_window_extent(r).overlaps(ax.get_window_extent(r)):
                boxes.append(("figlegend-axes", ax.get_window_extent(r)))
    issues = [n for n, _ in boxes if n == "figlegend-axes"]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if a.x0 < b.x1 - 1 and b.x0 < a.x1 - 1 and a.y0 < b.y1 - 1 and b.y0 < a.y1 - 1:
                issues.append(f"{boxes[i][0]} x {boxes[j][0]}")
    return issues


def save(fig, path) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    issues = layout_issues(fig)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return issues


def _fig_legend(fig, handles, ncol: int, y: float = 1.02) -> None:
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, handlelength=2.2, columnspacing=1.2, borderaxespad=0.2)


def run_means(lv: pd.DataFrame) -> pd.DataFrame:
    return lv.groupby(["run_id", "arm", "budget", "seed", "set"], as_index=False)["value"].mean()


def learning_curves(study: Study, lv: pd.DataFrame, refs: pd.DataFrame):
    rm = run_means(lv[lv.tuned & (lv.pooling == "mean")])
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.35))
    handles = {}
    for ax, s, letter in zip(axes, ["heldout", "dl"], "ab"):
        sub = rm[rm.set == s]
        for arm in study.arms:
            g = sub[sub.arm == arm].groupby("budget")["value"]
            if g.ngroups == 0:
                continue
            m, lo, hi = g.mean(), g.min(), g.max()
            lw = 1.7 if arm == "B" else 1.2
            (h,) = ax.plot(
                m.index,
                m.values,
                color=COLOR[arm],
                marker=MARKER[arm],
                lw=lw,
                ls=DASH.get(arm, "-"),
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=study.arms[arm]["label"],
                zorder=3 if arm == "B" else 2,
            )
            ax.fill_between(m.index, lo.values, hi.values, color=COLOR[arm], alpha=0.12, lw=0, zorder=1)
            handles[arm] = h
        for name, ls in [("BM25", (0, (5, 3))), ("minilm", (0, (1, 2))), ("bge", (0, (3, 1, 1, 1)))]:
            if name == "BM25" and s == "heldout":
                continue
            v = refs[(refs.arm == name) & (refs.set == s)]["value"]
            if len(v):
                (h,) = ax.plot(
                    [190, 21000],
                    [v.mean()] * 2,
                    color="#a0a0a0",
                    lw=0.8,
                    ls=ls,
                    zorder=0,
                    label={"BM25": "BM25 order", "minilm": study.references["minilm"]["label"], "bge": study.references["bge"]["label"]}[name],
                )
                handles[name] = h
        _budget_axis(ax)
        ax.set_ylabel(SET_LABEL[s])
        _panel(ax, letter)
    fig.tight_layout(w_pad=2.0)
    _fig_legend(fig, list(handles.values()), ncol=6)
    return fig


def forest(study: Study, stats: dict):
    eff = stats["descriptive"]["effects"]
    margin = study.analysis["equivalence_margin"]
    dec = study.decision_arms
    budgets = sorted({int(k.split("@")[1]) for k in eff if k.startswith("heldout@")})
    fig, axes = plt.subplots(2, len(budgets), figsize=(FULL, 3.3), sharey=True, squeeze=False)
    ylab = [study.arms[a]["label"] for a in dec] + ["decision class"]
    for r, s in enumerate(["heldout", "dl"]):
        cis = [x for b in budgets if f"{s}@{b}" in eff for v in [*eff[f"{s}@{b}"]["per_arm"].values(), eff[f"{s}@{b}"]["re"]] for x in v["ci"]]
        lim = max([abs(x) for x in cis] + [margin[s] * 1.5])
        for c, b in enumerate(budgets):
            ax = axes[r, c]
            ax.grid(False)
            ax.axvspan(-margin[s], margin[s], color="#f1f1f1", lw=0, zorder=0)
            ax.axvline(0, color="#9a9a9a", lw=0.6, zorder=1)
            e = eff.get(f"{s}@{b}")
            if e:
                for i, a in enumerate(dec):
                    if a in e["per_arm"]:
                        v = e["per_arm"][a]
                        ax.plot(v["ci"], [i, i], color=COLOR[a], lw=1.1, solid_capstyle="round")
                        ax.plot(v["mean"], i, marker=MARKER[a], color=COLOR[a], markeredgecolor="white", markeredgewidth=0.6, ms=4.2)
                re = e["re"]
                k = len(dec)
                ax.plot(re["ci"], [k, k], color=INK, lw=1.6, solid_capstyle="round")
                ax.plot(re["mean"], k, marker="D", color=INK, markeredgecolor="white", ms=4.6)
            ax.set_xlim(-lim * 1.08, lim * 1.08)
            ax.set_yticks(range(len(ylab)))
            ax.set_yticklabels(ylab)
            ax.set_ylim(len(ylab) - 0.4, -0.6)
            ax.tick_params(axis="x", labelsize=6.5)
            ax.locator_params(axis="x", nbins=4)
            if r == 0:
                ax.set_title(f"{_short(b)} queries", color=INK)
            if c == 0:
                ax.set_ylabel(r"$\Delta$ " + ("MRR@10" if s == "heldout" else "nDCG@10") + (" (held-out)" if s == "heldout" else " (DL)"))
    for ax in axes[1]:
        ax.set_xlabel("difference from ModernBERT-large")
    fig.tight_layout(w_pad=0.8, h_pad=1.2)
    return fig


def specificity(study: Study, lv: pd.DataFrame, reps: int, alpha: float):
    sm = S.seed_mean(lv[lv.tuned & (lv.pooling == "mean")])
    dec = study.decision_arms
    controls = [a for a, v in study.arms.items() if v["kind"] == "control"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.1), sharey=False)
    groups = [("decision class", dec, INK, "D")] + [(study.arms[c]["label"], [c], COLOR[c], MARKER[c]) for c in controls]
    budgets = sorted(b for b in sm.budget.unique() if all((sm[sm.budget == b].arm == c).any() for c in controls))
    handles = []
    for ax, s, letter in zip(axes, ["heldout", "dl"], "ab"):
        ax.axhline(0, color="#9a9a9a", lw=0.6)
        for gi, (name, arms, col, mk) in enumerate(groups):
            xs, ms, los, his = [], [], [], []
            for bi, b in enumerate(budgets):
                c = S.contrast(sm, b, s, arms, "B", reps, alpha)
                if c:
                    xs.append(bi + (gi - 1) * 0.16)
                    ms.append(c["mean"])
                    los.append(c["ci"][0])
                    his.append(c["ci"][1])
            for x, lo, hi in zip(xs, los, his):
                ax.plot([x, x], [lo, hi], color=col, lw=1.1, solid_capstyle="round")
            h = ax.plot(xs, ms, ls="none", marker=mk, color=col, markeredgecolor="white", markeredgewidth=0.6, ms=4.6, label=name)[0]
            if letter == "a":
                handles.append(h)
        ax.set_xticks(range(len(budgets)))
        ax.set_xticklabels([f"{_short(b)} queries" for b in budgets])
        ax.set_xlim(-0.6, len(budgets) - 0.4)
        ax.set_ylabel(r"$\Delta$ vs ModernBERT-large" + ("\n(MRR@10, held-out)" if s == "heldout" else "\n(nDCG@10, DL)"))
        _panel(ax, letter)
    fig.tight_layout(w_pad=2.0)
    _fig_legend(fig, handles, ncol=len(handles))
    return fig


def probes_fig(study: Study, probes: pd.DataFrame, stats: dict):
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    handles = []
    for arm in study.arms:
        g = probes[probes.arm == arm].sort_values("layer")
        h = ax.plot(
            g.layer,
            g.dev_mrr10,
            color=COLOR[arm],
            marker=MARKER[arm],
            ls=DASH.get(arm, "-"),
            lw=1.7 if arm == "B" else 1.1,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=study.arms[arm]["label"],
        )[0]
        handles.append(h)
    ax.set_xlabel("encoder layer (frozen, before fine-tuning)")
    ax.set_ylabel("linear probe dev MRR@10")
    _panel(ax, "a")
    ax = axes[1]
    eff = stats["descriptive"]["effects"].get("heldout@1000", {}).get("per_arm", {})
    best = probes.groupby("arm")["dev_mrr10"].max()
    base_best = best.get("B", np.nan)
    pts = [(a, best[a] - base_best, eff[a]["mean"]) for a in eff if a in best]
    for a, x, y in pts:
        ax.plot(x, y, marker=MARKER[a], color=COLOR[a], ls="none", markeredgecolor="white", ms=5)
    ax.axhline(0, color="#9a9a9a", lw=0.6)
    ax.axvline(0, color="#9a9a9a", lw=0.6)
    ax.set_xlabel(r"$\Delta$ best-layer probe vs base")
    ax.set_ylabel(r"$\Delta$ MRR@10 after fine-tuning (1k)")
    if len(pts) >= 3:
        rho = pd.Series([p[1] for p in pts]).corr(pd.Series([p[2] for p in pts]), method="spearman")
        ax.text(0.98, 0.04, rf"Spearman $\rho$ = {rho:.2f} (n = {len(pts)})", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=MUTED)
    _panel(ax, "b")
    fig.tight_layout(w_pad=2.0)
    _fig_legend(fig, handles, ncol=8)
    return fig


def convergence_fig(study: Study, conv: pd.DataFrame):
    conv = conv[conv.layer != "other"].assign(layer=lambda d: d.layer.astype(int), kept=lambda d: d.finetuned / d.initial)
    budgets = sorted(conv.budget.unique())
    fig, axes = plt.subplots(1, len(budgets), figsize=(FULL, 2.2), sharey=True, squeeze=False)
    handles = {}
    for ax, b, letter in zip(axes[0], budgets, "abcd"):
        ax.axhline(1.0, color="#9a9a9a", lw=0.6)
        for arm in study.arms:
            g = conv[(conv.arm == arm) & (conv.budget == b)].sort_values("layer")
            if g.empty:
                continue
            handles[arm] = ax.plot(g.layer, g.kept, color=COLOR[arm], ls=DASH.get(arm, "-"), lw=1.2, label=study.arms[arm]["label"])[0]
        ax.set_xlabel("encoder layer")
        ax.set_title(f"{_short(b)} training queries", color=INK)
        _panel(ax, letter)
    axes[0][0].set_ylabel("distance to fine-tuned base\n/ distance before fine-tuning")
    fig.tight_layout(w_pad=1.5)
    _fig_legend(fig, list(handles.values()), ncol=len(handles))
    return fig


def lr_grid_fig(study: Study, sel: dict):
    g = pd.DataFrame(sel["grid"])
    fig, ax = plt.subplots(figsize=(HALF, 2.2))
    handles = []
    for arm in study.arms:
        d = g[g.arm == arm].sort_values("lr")
        handles.append(
            ax.plot(
                d.lr,
                d.dev_mrr10,
                color=COLOR[arm],
                marker=MARKER[arm],
                ls=DASH.get(arm, "-"),
                markeredgecolor="white",
                markeredgewidth=0.5,
                lw=1.7 if arm == "B" else 1.1,
                label=study.arms[arm]["label"],
            )[0]
        )
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(sorted(g.lr.unique())))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{x:.0e}".replace("e-0", "e-") for x in sorted(g.lr.unique())])
    ax.set_xlabel("peak learning rate")
    ax.set_ylabel("dev MRR@10 (1k queries, seed 0)")
    fig.tight_layout()
    _fig_legend(fig, handles, ncol=4)
    return fig


def loss_fig(study: Study, runs_dir):
    fig, ax = plt.subplots(figsize=(HALF, 2.2))
    handles = {}
    for f in sorted(runs_dir.glob("main/*__q16000__*__s0/train.jsonl")):
        arm = f.parent.name.split("__")[0]
        d = pd.read_json(f, lines=True)
        handles[arm] = ax.plot(d.step, d.loss.rolling(10, min_periods=1).mean(), color=COLOR[arm], lw=1.0, label=study.arms[arm]["label"])[0]
    ax.set_xlabel("optimizer step (16k queries)")
    ax.set_ylabel("training loss (listwise CE)")
    fig.tight_layout()
    _fig_legend(fig, list(handles.values()), ncol=3)
    return fig


def render_all(study: Study, log) -> dict:
    style()
    res, figs = study.path("results"), study.path("figures")
    pq = pd.read_parquet(res / "per_query.parquet")
    lv = S.long_values(pq)
    stats = json.loads((res / "stats.json").read_text())
    refs = lv[lv.stage.isin(["baseline", "references"])]
    ex = study.path("runs") / "extra"
    report = {}
    a = study.analysis
    todo = {
        "f1_learning_curves": lambda: learning_curves(study, lv, refs),
        "f2_forest": lambda: forest(study, stats),
        "f3_specificity": lambda: specificity(study, lv, a["bootstrap"], a["alpha"]),
    }
    if (ex / "probes.parquet").exists():
        todo["f4_probes"] = lambda: probes_fig(study, pd.read_parquet(ex / "probes.parquet"), stats)
    if (ex / "convergence.parquet").exists():
        todo["f5_convergence"] = lambda: convergence_fig(study, pd.read_parquet(ex / "convergence.parquet"))
    sel_path = study.path("runs") / "selection.json"
    if sel_path.exists():
        todo["fa1_lr_grid"] = lambda: lr_grid_fig(study, json.loads(sel_path.read_text()))
    todo["fa2_loss"] = lambda: loss_fig(study, study.path("runs"))
    for name, make in todo.items():
        issues = save(make(), figs / name)
        report[name] = issues
        log("figure", name=name, issues=len(issues))
    write_json(figs / "layout_report.json", report)
    return report
