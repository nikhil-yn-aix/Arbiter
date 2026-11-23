import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arbiter import plots


def _lv(study):
    rng = np.random.default_rng(0)
    rows = []
    for arm in study.arms:
        for b in plots.BUDGETS:
            if arm in ("N", "R") and b > 1000:
                continue
            for seed in (0, 1):
                for s in ("heldout", "dl"):
                    for q in range(60):
                        rows.append(
                            {
                                "run_id": f"{arm}{b}{seed}",
                                "arm": arm,
                                "budget": b,
                                "seed": seed,
                                "set": s,
                                "qid": f"q{q}",
                                "value": 0.3 + 0.05 * np.log2(b / 250) + rng.normal(0, 0.1),
                                "tuned": True,
                                "pooling": "mean",
                                "stage": "main",
                            }
                        )
    return pd.DataFrame(rows)


def test_figures_render_without_overlaps(study, tmp_path):
    plots.style()
    lv = _lv(study)
    refs = pd.DataFrame([{"arm": "BM25", "set": "dl", "value": 0.48}, {"arm": "minilm", "set": "dl", "value": 0.7}])
    fig = plots.learning_curves(study, lv, refs)
    assert plots.save(fig, tmp_path / "f1") == []
    fig = plots.specificity(study, lv, 200, 0.05)
    assert plots.save(fig, tmp_path / "f3") == []
    assert (tmp_path / "f1.pdf").exists()


def test_layout_checker_flags_overlap():
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.text(0.5, 0.5, "overlapping text", transform=ax.transAxes)
    ax.text(0.5, 0.5, "overlapping text", transform=ax.transAxes)
    assert plots.layout_issues(fig)
    plt.close(fig)
