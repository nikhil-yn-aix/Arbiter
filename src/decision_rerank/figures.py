import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from . import style
from .settings import RESULTS_DIR, load_config


def read_json(name: str) -> dict:
    return json.loads((RESULTS_DIR / name).read_text())


def pareto_front(points: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    front = []
    best = -1.0
    for item in sorted(points, key=lambda p: (p[1], -p[2])):
        if item[2] > best:
            front.append(item)
            best = item[2]
    return front


def plain_ms(value: float, _=None) -> str:
    return f"{value:,.0f}"


def quality_latency(summary: pd.DataFrame, size: int = 50):
    config = load_config()
    baseline = config.evaluation.baseline
    fig, axes = plt.subplots(2, 2, figsize=(style.DOUBLE, 5.3), sharex=True)
    low = summary["latency_p50_ms"].min() * 0.7
    high = summary[summary["candidate_size"] == size]["latency_p50_ms"].max() * 1.6
    for ax, name, tag in zip(axes.flat, config.datasets, "abcd", strict=False):
        part = summary[(summary["dataset"] == name) & (summary["candidate_size"] == size)]
        style.panel_tag(ax, tag, config.datasets[name].label)
        base = float(part[part["model"] == baseline]["ndcg_at_10"].iloc[0])
        ax.axhline(base, color=style.BASELINE, lw=1.2, zorder=1)
        ax.text(
            high * 0.97,
            base,
            "hybrid retrieval",
            color=style.BASELINE,
            fontsize=6.6,
            va="top",
            ha="right",
            fontweight="bold",
        )
        ax.axvline(100, color=style.GRID, lw=1.0, zorder=0)
        points = []
        for model_id in config.models:
            row = part[part["model"] == model_id].iloc[0]
            points.append((model_id, float(row["latency_p50_ms"]), float(row["ndcg_at_10"])))
        front = pareto_front(points)
        ax.plot([p[1] for p in front], [p[2] for p in front], color=style.MUTED, lw=1.0, ls=(0, (3, 2)), zorder=2)
        for model_id, x, y in points:
            ax.scatter(
                x,
                y,
                s=62,
                marker=style.marker_of(model_id),
                color=style.MODEL_COLOR[model_id],
                edgecolors="white",
                linewidths=0.8,
                zorder=4,
            )
        best_model, best_x, best_y = max(points, key=lambda p: p[2])
        ax.annotate(
            f"{style.label(best_model)}  {best_y:.3f}",
            xy=(best_x, best_y),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
            color=style.INK,
            annotation_clip=False,
        )
        ax.set_xscale("log")
        ax.set_xlim(low, high)
        ax.xaxis.set_major_locator(FixedLocator([10, 100, 1000, 10000]))
        ax.xaxis.set_major_formatter(FuncFormatter(plain_ms))
        ax.xaxis.set_minor_formatter(NullFormatter())
        values = [p[2] for p in points] + [base]
        pad = (max(values) - min(values)) * 0.1 + 0.01
        ax.set_ylim(min(values) - pad, max(values) + pad * 2.2)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    for ax in axes[1]:
        ax.set_xlabel("Latency per query (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("nDCG@10")
    handles = [
        Line2D(
            [],
            [],
            marker=style.MODEL_MARKER[m],
            color=style.MODEL_COLOR[m],
            ls="",
            markersize=6.5,
            markeredgecolor="white",
            label=style.label(m),
        )
        for m in config.models
    ] + [
        Line2D([], [], color=style.INK, ls=(0, (3, 2)), lw=1.0, label="Pareto front"),
        Line2D([], [], color=style.BASELINE, lw=1.2, label="hybrid retrieval"),
    ]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=5, columnspacing=1.6, handletextpad=0.5
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.98), w_pad=2.0, h_pad=1.4)
    return fig


def candidate_scaling(summary: pd.DataFrame):
    config = load_config()
    sizes = config.retrieval.candidate_sizes
    fig, axes = plt.subplots(1, 4, figsize=(style.DOUBLE, 2.7), sharex=True)
    positions = list(range(len(sizes)))
    for ax, name, tag in zip(axes, config.datasets, "abcd", strict=False):
        part = summary[summary["dataset"] == name]
        style.panel_tag(ax, tag, config.datasets[name].label)
        for model_id in config.models:
            rows = part[part["model"] == model_id].sort_values("candidate_size")
            ax.plot(
                positions,
                rows["ndcg_at_10"],
                color=style.MODEL_COLOR[model_id],
                lw=1.2,
                marker=style.marker_of(model_id),
                markersize=4.5,
                markeredgecolor="white",
                markeredgewidth=0.5,
            )
        base = part[part["model"] == config.evaluation.baseline].sort_values("candidate_size")
        ax.plot(positions, base["ndcg_at_10"], color=style.BASELINE, lw=1.6, zorder=0)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_xlabel("Candidates")
    axes[0].set_ylabel("nDCG@10")
    handles = [
        Line2D(
            [],
            [],
            marker=style.MODEL_MARKER[m],
            color=style.MODEL_COLOR[m],
            lw=1.2,
            markersize=5.5,
            markeredgecolor="white",
            label=style.label(m),
        )
        for m in config.models
    ] + [Line2D([], [], color=style.BASELINE, lw=1.6, label="hybrid retrieval")]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=9,
        columnspacing=1.0,
        handletextpad=0.4,
        fontsize=6.6,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.4)
    return fig


def cascade_curves(summary: pd.DataFrame):
    config = load_config()
    ce_colour = {"bge-v2-m3": style.CROSS, "qwen3-06b": style.CASCADE, "jina-v2": style.SKY}
    ce_marker = {"bge-v2-m3": "s", "qwen3-06b": "^", "jina-v2": "D"}
    top_k = sorted(summary["top_k"].unique())
    decisions = config.cascade.decision
    fig, axes = plt.subplots(3, 2, figsize=(style.DOUBLE, 6.6), sharex=True)
    for row, model_id in enumerate(decisions):
        part = summary[summary["decision_model"] == model_id]
        for col, (metric, ylabel) in enumerate([("cascade_ndcg", "nDCG@10"), ("latency_p50_ms", "Latency (ms)")]):
            ax = axes[row, col]
            style.panel_tag(ax, "abcdef"[row * 2 + col], f"{style.label(model_id)}: {ylabel.split(' (')[0]}")
            if col == 0:
                ax.axhline(part["no_reranker_ndcg"].iloc[0], color=style.BASELINE, lw=1.0)
                ax.axhline(part["decision_ndcg"].iloc[0], color=style.DECISION, lw=1.0, ls=(0, (4, 2)))
            else:
                ax.axhline(part["decision_alone_latency_ms"].iloc[0], color=style.DECISION, lw=1.0, ls=(0, (4, 2)))
            for ce in config.cascade.refiner:
                rows = part[part["cross_encoder"] == ce].sort_values("ce_calls_saved_pct")
                colour = ce_colour[ce]
                ax.plot(
                    rows["ce_calls_saved_pct"],
                    rows[metric],
                    color=colour,
                    lw=1.3,
                    marker=ce_marker[ce],
                    markersize=4.5,
                    markerfacecolor="white",
                    markeredgewidth=1.1,
                    zorder=3,
                )
                if col == 0:
                    ax.axhline(rows["cross_encoder_ndcg"].iloc[0], color=colour, lw=0.9, ls=":")
                    eligible = rows[rows["ce_calls_saved_pct"] >= 50]
                    if len(eligible):
                        best = eligible.sort_values("cascade_ndcg").iloc[-1]
                        ax.scatter(
                            best["ce_calls_saved_pct"],
                            best["cascade_ndcg"],
                            marker="*",
                            s=120,
                            color=style.HIGHLIGHT,
                            edgecolors=style.INK,
                            linewidths=0.5,
                            zorder=5,
                        )
                else:
                    ax.axhline(rows["cross_encoder_alone_latency_ms"].iloc[0], color=colour, lw=0.9, ls=":")
            if col == 1:
                ax.set_yscale("log")
                ticks = [
                    t
                    for t in (5, 10, 20, 50, 100, 200, 500, 1000, 2000)
                    if ax.get_ylim()[0] * 0.8 <= t <= ax.get_ylim()[1] * 1.25
                ]
                ax.yaxis.set_major_locator(FixedLocator(ticks))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xlim(-3, 93)
            ticks = [100 - 2 * k for k in sorted(top_k, reverse=True)]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t}%\nK={(100 - t) // 2}" for t in ticks])
            if col == 0:
                ax.set_ylabel("nDCG@10")
            else:
                ax.set_ylabel("Latency (ms)")
    for ax in axes[-1]:
        ax.set_xlabel("Cross-encoder calls saved")
    handles = [
        Line2D(
            [],
            [],
            color=ce_colour[ce],
            marker=ce_marker[ce],
            markerfacecolor="white",
            markersize=5,
            label=style.label(ce),
        )
        for ce in config.cascade.refiner
    ] + [
        Line2D([], [], color=style.DECISION, ls=(0, (4, 2)), label="decision model"),
        Line2D([], [], color=style.MUTED, ls=":", label="cross-encoder alone"),
        Line2D([], [], color=style.BASELINE, label="hybrid retrieval"),
        Line2D(
            [],
            [],
            marker="*",
            color=style.HIGHLIGHT,
            markeredgecolor=style.INK,
            ls="",
            markersize=9,
            label="best at ≥50% saved",
        ),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 0.93), title="Refiner", alignment="left")
    fig.tight_layout(w_pad=2.2, h_pad=1.2)
    return fig


def long_context(data: dict):
    config = load_config()
    lengths = config.longcontext.lengths
    colours = {"laya-multi": style.DECISION, "qwen3-06b": style.CASCADE, "jina-v2": style.SKY, "bge-v2-m3": style.CROSS}
    fig, (left, right) = plt.subplots(1, 2, figsize=(style.DOUBLE, 3.0), gridspec_kw={"width_ratios": [1.55, 1]})
    style.panel_tag(left, "a", "nDCG@10 by passage length")
    style.panel_tag(right, "b", "Gain from full passage")
    labels = []
    for model_id, entry in data.items():
        colour = colours[model_id]
        marker = style.marker_of(model_id)
        full = [entry["by_length"][str(L)]["full"] for L in lengths]
        short = [entry["by_length"][str(L)]["truncated_512"] for L in lengths]
        y_full = np.array([f["ndcg_at_10"] for f in full])
        left.plot(
            lengths, y_full, color=colour, marker=marker, markersize=4.5, markerfacecolor=colour, lw=1.5, zorder=3
        )
        if entry["native_context"] > 512:
            left.plot(
                lengths,
                [s["ndcg_at_10"] for s in short],
                color=colour,
                lw=1.1,
                ls=(0, (4, 2)),
                marker=marker,
                markersize=4,
                markerfacecolor="white",
                markeredgewidth=1.0,
                alpha=0.9,
                zorder=3,
            )
            gain = np.array([f["ndcg_at_10"] - s["ndcg_at_10"] for f, s in zip(full, short, strict=True)])
            right.plot(lengths, gain, color=colour, marker=marker, markersize=4.5, lw=1.4)
            labels.append((lengths[-1], gain[-1], f"{entry['label']}  {gain[-1]:+.3f}", colour, right))
        labels.append((lengths[-1], y_full[-1], entry["label"], colour, left))
    for ax in (left, right):
        ax.set_xscale("log", base=2)
        ax.set_xticks(lengths)
        ax.set_xticklabels([f"{L:,}" for L in lengths])
        ax.minorticks_off()
        ax.set_xlabel("Passage tokens")
        ax.set_xlim(lengths[0] * 0.85, lengths[-1] * (3.2 if ax is left else 6.5))
    right.axhline(0, color=style.MUTED, lw=0.8)
    left.set_ylabel("nDCG@10")
    right.set_ylabel("Δ nDCG@10")
    for ax in (left, right):
        items = [(x, y, t, c) for x, y, t, c, a in labels if a is ax]
        style.spread_labels(ax, items, gap_px=40, dx_px=14)
    handles = [
        Line2D([], [], color=style.MUTED, lw=1.5, label="full"),
        Line2D(
            [],
            [],
            color=style.MUTED,
            lw=1.1,
            ls=(0, (4, 2)),
            marker="o",
            markerfacecolor="white",
            markersize=4,
            label="cut at 512",
        ),
    ]
    left.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2)
    fig.tight_layout(w_pad=2.5)
    return fig


def reliability(data: dict, height: float = 5.5):
    order = ["laya-en", "laya-multi", "laya-td", "bge-v2-m3", "bge-base", "qwen3-06b", "jina-v2", "minilm"]
    spacing = {"hspace": 0.62, "wspace": 0.28, "top": 0.94, "bottom": 0.09}
    fig, axes = plt.subplots(2, 4, figsize=(style.DOUBLE, height), gridspec_kw=spacing)
    tags = "abcdefgh"
    for index, model_id in enumerate(order):
        row, col = index // 4, index % 4
        ax = axes[row, col]
        hist = ax.inset_axes([0, -0.3, 1, 0.19], sharex=ax)
        entry = data[model_id]
        colour = style.MODEL_COLOR[model_id]
        bins = [b for b in entry["bins"] if b["count"]]
        x = np.array([b["confidence"] for b in bins])
        y = np.array([b["accuracy"] for b in bins])
        ax.plot([0, 1], [0, 1], color=style.MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.fill_between(x, [b["low"] for b in bins], [b["high"] for b in bins], color=colour, alpha=0.2, lw=0)
        ax.plot(
            x,
            y,
            color=colour,
            lw=1.4,
            marker=style.marker_of(model_id),
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.5,
            zorder=3,
        )
        style.panel_tag(ax, tags[index], style.label(model_id))
        ax.text(
            0.04,
            0.95,
            f"ECE {entry['ece']:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=7.2,
            color=style.INK,
            fontweight="bold",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        plt.setp(ax.get_xticklabels(), visible=False)
        if col:
            plt.setp(ax.get_yticklabels(), visible=False)
        centres = [b["confidence"] for b in entry["bins"]]
        hist.bar(centres, [max(b["count"], 0.8) for b in entry["bins"]], width=0.085, color=style.MUTED, alpha=0.75)
        hist.set_yscale("log")
        hist.set_ylim(1, 1e6)
        hist.set_yticks([])
        hist.grid(False)
        hist.spines["left"].set_visible(False)
        hist.set_xticks([0, 0.5, 1])
        if row == 1:
            hist.set_xlabel("Predicted")
        else:
            plt.setp(hist.get_xticklabels(), visible=False)
        if col == 0:
            ax.set_ylabel("Observed", fontsize=7)
            hist.set_ylabel("n", fontsize=6, rotation=0, ha="right", va="center", labelpad=4)
    return fig


def prompt_sensitivity(data: dict):
    ordered = sorted(data, key=lambda m: (data[m]["std_ndcg"], m))
    fig, ax = plt.subplots(figsize=(style.DOUBLE * 0.62, 3.1))
    y = np.arange(len(ordered))[::-1]
    config = load_config()
    stds = np.array([data[m]["std_ndcg"] for m in ordered])
    low = np.array([data[m]["std_ndcg"] - data[m]["std_ci_low"] for m in ordered])
    high = np.array([data[m]["std_ci_high"] - data[m]["std_ndcg"] for m in ordered])
    colours = [style.DECISION if config.models[m].is_decision else style.CROSS for m in ordered]
    ax.barh(y, stds, color=colours, height=0.58, zorder=3)
    ax.errorbar(
        stds,
        y,
        xerr=[np.clip(low, 0, None), np.clip(high, 0, None)],
        fmt="none",
        ecolor=style.INK,
        elinewidth=0.8,
        capsize=2,
        zorder=4,
    )
    ax.axvline(float(np.median(stds)), color=style.HIGHLIGHT, lw=1.1, ls=(0, (4, 2)), zorder=2)
    top = max((stds + high).max(), 1e-3)
    for pos, model_id in zip(y, ordered, strict=True):
        note = f"{data[model_id]['std_ndcg']:.3f}"
        ax.text(top * 1.06, pos, note, va="center", fontsize=6.4, color=style.INK)
    ax.set_yticks(y)
    ax.set_yticklabels([style.label(m) for m in ordered])
    ax.set_xlim(0, top * 1.22)
    ax.set_xlabel("Std of nDCG@10 across prompts")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, labelsize=7.5, colors=style.INK)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=style.DECISION, label="Decision"),
        plt.Rectangle((0, 0), 1, 1, color=style.CROSS, label="Cross-encoder"),
        Line2D([], [], color=style.HIGHLIGHT, ls=(0, (4, 2)), label="median"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    return fig


def build_all() -> list[str]:
    style.apply()
    made = []
    summary = pd.read_csv(RESULTS_DIR / "benchmark_summary.csv")
    style.save(quality_latency(summary), "quality_latency")
    made.append("quality_latency")
    style.save(candidate_scaling(summary), "candidate_scaling")
    made.append("candidate_scaling")
    cascade = pd.read_csv(RESULTS_DIR / "cascade_summary.csv")
    style.save(cascade_curves(cascade), "cascade")
    made.append("cascade")
    style.save(long_context(read_json("longcontext.json")), "long_context")
    made.append("long_context")
    style.save(reliability(read_json("calibration.json")), "reliability")
    made.append("reliability")
    style.save(prompt_sensitivity(read_json("prompt_sensitivity.json")), "prompt_sensitivity")
    made.append("prompt_sensitivity")
    return made


def main() -> None:
    for name in build_all():
        print("wrote", name)


if __name__ == "__main__":
    main()
