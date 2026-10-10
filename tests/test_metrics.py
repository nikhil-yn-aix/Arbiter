import numpy as np
import pytest

from decision_rerank import metrics


def test_ndcg_perfect_and_empty():
    grades = {"a": 3, "b": 1}
    assert metrics.ndcg_at(["a", "b", "c"], grades, 10) == 1.0
    assert metrics.ndcg_at(["c"], grades, 10) == 0.0
    assert metrics.ndcg_at(["a"], {}, 10) == 0.0


def test_recall_and_mrr():
    grades = {"a": 2, "b": 2, "c": 1}
    assert metrics.recall_at(["a", "x", "b"], grades, 10, 2) == 1.0
    assert metrics.recall_at(["a", "x"], grades, 10, 2) == 0.5
    assert metrics.mrr_at(["x", "y", "b"], grades, 10, 2) == 1 / 3


def test_rank_by_is_stable_on_ties():
    assert metrics.rank_by(["a", "b", "c"], np.array([1.0, 1.0, 2.0])) == ["c", "a", "b"]


def test_kendall_handles_constant_scores():
    assert metrics.kendall(np.ones(5), np.arange(5.0)) == 0.0
    assert metrics.kendall(np.arange(5.0), np.arange(5.0)) == pytest.approx(1.0)


def test_calibration_perfect_and_worst():
    labels = np.array([0, 0, 1, 1] * 25, dtype=float)
    perfect = metrics.calibration(labels, labels)
    assert perfect["ece"] == 0.0 and perfect["brier"] == 0.0
    worst = metrics.calibration(1 - labels, labels)
    assert worst["ece"] == 1.0 and worst["brier"] == 1.0
    assert len(worst["bins"]) == 10


def test_paired_bootstrap_interval_contains_mean():
    a = np.array([0.5, 0.6, 0.7, 0.8, 0.4])
    b = np.array([0.4, 0.6, 0.5, 0.7, 0.4])
    out = metrics.paired_bootstrap(a, b, 500, 1)
    assert out["ci_low"] <= out["mean_diff"] <= out["ci_high"]


def test_wilson_bounds():
    low, high = metrics.wilson(5, 10)
    assert 0 <= low < 0.5 < high <= 1
