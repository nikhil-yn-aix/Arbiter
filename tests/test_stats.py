import numpy as np
import pandas as pd
import pytest

from arbiter import stats as S


def _sm(effect, arms=("B", "D1", "D2", "D3", "D4", "D5"), n=400, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.2, 0.8, n)
    rows = []
    for a in arms:
        shift = 0.0 if a == "B" else effect + rng.normal(0, 0.003)
        v = np.clip(base + shift + rng.normal(0, 0.05, n), 0, 1)
        rows += [{"arm": a, "budget": 1000, "set": "heldout", "qid": f"q{i}", "value": x, "seeds": 3} for i, x in enumerate(v)]
    return pd.DataFrame(rows)


def test_class_effect_detects_shift():
    e = S.class_effect(_sm(0.03), 1000, "heldout", ["D1", "D2", "D3", "D4", "D5"], "B", 2000, 0.05)
    assert e["re"]["ci"][0] > 0 and e["fe"]["ci"][0] > 0
    assert e["re"]["mean"] == pytest.approx(0.03, abs=0.01)


def test_null_is_equivalent_not_significant():
    e = S.class_effect(_sm(0.0), 1000, "heldout", ["D1", "D2", "D3", "D4", "D5"], "B", 2000, 0.05)
    assert e["re"]["p"] > 0.05
    assert S.tost(e["re"]["ci90"], 0.015)


def test_holm_monotone_and_capped():
    adj = S.holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adj["a"] == pytest.approx(0.03) and adj["b"] == pytest.approx(0.08) and adj["c"] == 0.5
    assert S.holm({"a": 0.6, "b": 0.7})["a"] == 1.0


def test_bootstrap_coverage():
    rng = np.random.default_rng(1)
    hits = 0
    for _ in range(200):
        x = rng.normal(0.1, 1, 200)
        lo, hi = S._boot_summary(x, 1000, 0.05)["ci"]
        hits += lo < 0.1 < hi
    assert 0.9 < hits / 200 <= 1.0


def test_savings_interpolates_in_log_budget():
    curve = pd.Series([0.4, 0.5, 0.6], index=[250, 1000, 4000])
    assert S.savings(curve, 0.55) == pytest.approx(2000)
    assert S.savings(curve, 0.7) is None
