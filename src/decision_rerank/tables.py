import json

import pandas as pd

from .settings import RESULTS_DIR, TABLES_DIR, ensure_dirs, load_config


def read_json(name: str) -> dict:
    return json.loads((RESULTS_DIR / name).read_text())


def emphasise(values: list[float], fmt: str, higher_better: bool = True) -> list[str]:
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=higher_better)
    first = order[0]
    second = order[1] if len(order) > 1 else None
    out = []
    for i, v in enumerate(values):
        text = fmt.format(v)
        if i == first:
            text = rf"\textbf{{{text}}}"
        elif i == second:
            text = rf"\underline{{{text}}}"
        out.append(text)
    return out


def write(name: str, header: list[str], rows: list[list[str]], spec: str, caption: str, tag: str) -> None:
    ensure_dirs()
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:{tag}}}",
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TABLES_DIR / f"{name}.tex").write_text("\n".join(lines))


def main_results() -> None:
    config = load_config()
    summary = pd.read_csv(RESULTS_DIR / "benchmark_summary.csv")
    size = 50
    part = summary[summary["candidate_size"] == size]
    models = list(config.models)
    columns = list(config.datasets)
    cells = {}
    for name in columns:
        sub = part[part["dataset"] == name].set_index("model")
        cells[name] = emphasise([float(sub.loc[m, "ndcg_at_10"]) for m in models], "{:.3f}")
    rows = []
    for i, model_id in enumerate(models):
        spec = config.models[model_id]
        sub = part[part["model"] == model_id]
        latency = float(sub["latency_p50_ms"].mean())
        vram = float(sub["vram_peak_gb"].max())
        params = f"{spec.params_m / 1000:.1f}B" if spec.params_m >= 1000 else f"{spec.params_m}M"
        rows.append(
            [
                spec.label,
                "Decision" if spec.is_decision else "Cross-enc.",
                params,
                *[cells[name][i] for name in columns],
                f"{latency:.0f}",
                f"{vram:.1f}",
            ]
        )
    base = part[part["model"] == config.evaluation.baseline]
    rows.append(
        [
            "Hybrid retrieval",
            "None",
            "--",
            *[f"{float(base[base['dataset'] == n]['ndcg_at_10'].iloc[0]):.3f}" for n in columns],
            f"{float(base['latency_p50_ms'].mean()):.0f}",
            f"{float(base['vram_peak_gb'].max()):.1f}",
        ]
    )
    header = ["Model", "Type", "Params", *[config.datasets[n].label for n in columns], "p50 (ms)", "VRAM (GB)"]
    write(
        "main_results",
        header,
        rows,
        "llr" + "r" * len(columns) + "rr",
        "nDCG@10 reranking the top 50 hybrid-retrieval candidates. Latency is the median per-query rerank time "
        "averaged over datasets; VRAM is the peak allocation. Best in bold, second best underlined.",
        "main",
    )


def cascade_table() -> None:
    config = load_config()
    frame = pd.read_csv(RESULTS_DIR / "cascade_summary.csv")
    eligible = frame[frame["ce_calls_saved_pct"] >= 50]
    rows = []
    for (first, second), group in eligible.groupby(["decision_model", "cross_encoder"], sort=False):
        best = group.sort_values(["cascade_ndcg", "top_k"], ascending=[False, True]).iloc[0]
        delta = best["cascade_ndcg"] - best["cross_encoder_ndcg"]
        rows.append(
            [
                config.models[first].label,
                config.models[second].label,
                str(int(best["top_k"])),
                f"{best['cascade_ndcg']:.3f}",
                f"{best['ce_calls_saved_pct']:.0f}\\%",
                f"{best['speedup_vs_cross_encoder']:.2f}$\\times$",
                f"{delta:+.3f}",
            ]
        )
    write(
        "cascade",
        ["Decision", "Cross-encoder", "K", "nDCG@10", "CE saved", "Speedup", "vs. CE alone"],
        rows,
        "llrrrrr",
        "Best cascade per pair with at least 50\\% of cross-encoder calls saved. Speedup is relative to the "
        "cross-encoder reranking all 50 candidates.",
        "cascade",
    )


def calibration_table() -> None:
    config = load_config()
    cal = read_json("calibration.json")
    stab = read_json("stability.json")
    models = list(config.models)
    ece = emphasise([cal[m]["ece"] for m in models], "{:.3f}", higher_better=False)
    brier = emphasise([cal[m]["brier"] for m in models], "{:.3f}", higher_better=False)
    typos = emphasise([stab[m]["typos"]["kendall_tau_perturb"] for m in models], "{:.3f}")
    syn = emphasise([stab[m]["synonyms"]["kendall_tau_perturb"] for m in models], "{:.3f}")
    para = emphasise([stab[m]["paraphrase"]["kendall_tau_perturb"] for m in models], "{:.3f}")
    rows = [
        [config.models[m].label, ece[i], brier[i], f"{cal[m]['slope']:.2f}", typos[i], syn[i], para[i]]
        for i, m in enumerate(models)
    ]
    write(
        "calibration",
        ["Model", "ECE", "Brier", "Slope", r"$\tau$ typos", r"$\tau$ synonyms", r"$\tau$ paraphrase"],
        rows,
        "lrrrrrr",
        "Calibration of the relevance probability over all benchmark pairs (lower ECE and Brier are better, slope "
        "near 1 is ideal) and rank stability under query perturbation (Kendall $\\tau$ between clean and perturbed "
        "scores, higher is more stable).",
        "calibration",
    )


def long_context_table() -> None:
    config = load_config()
    data = read_json("longcontext.json")
    rows = []
    for entry in data.values():
        full, short = entry["full"], entry["truncated_512"]
        rows.append(
            [
                entry["label"],
                f"{entry['native_context']:,}",
                f"{short['ndcg_at_10']:.3f}",
                f"{full['ndcg_at_10']:.3f}",
                f"{full['ndcg_at_10'] - short['ndcg_at_10']:+.3f}",
                f"{full['latency_p50_ms']:.0f}",
            ]
        )
    top = max(config.longcontext.lengths)
    write(
        "long_context",
        ["Model", "Context", "nDCG@512", "nDCG@full", r"$\Delta$", "p50 (ms)"],
        rows,
        "lrrrrr",
        f"LoCoMo retrieval with {top:,}-token passages. nDCG@10 with each passage cut to 512 tokens versus read up "
        "to the model limit; latency is per query for the full-length condition.",
        "longcontext",
    )


def prompt_table() -> None:
    config = load_config()
    data = read_json("prompt_sensitivity.json")
    ordered = sorted(data, key=lambda m: (data[m]["std_ndcg"], m))
    rows = []
    for rank, model_id in enumerate(ordered, 1):
        entry = data[model_id]
        rows.append(
            [
                config.models[model_id].label,
                entry["variant_type"].capitalize(),
                str(len(entry["variants"])),
                f"{entry['mean_ndcg']:.3f}",
                f"{entry['std_ndcg']:.4f}",
                str(rank),
            ]
        )
    write(
        "prompt_sensitivity",
        ["Model", "Variation", "Variants", "nDCG mean", "nDCG std", "Rank"],
        rows,
        "llrrrr",
        "Sensitivity of nDCG@10 to prompt or rubric wording. Rank 1 is the most robust. Models without a prompt "
        "have a single variant, so their deviation is zero by construction.",
        "prompts",
    )


def main() -> None:
    main_results()
    cascade_table()
    calibration_table()
    long_context_table()
    prompt_table()
    for path in sorted(TABLES_DIR.glob("*.tex")):
        print("wrote", path.name)


if __name__ == "__main__":
    main()
