import pandas as pd

from decision_rerank.settings import RESULTS_DIR, load_config


def test_cascade_grid_is_complete():
    config = load_config()
    frame = pd.read_csv(RESULTS_DIR / "cascade_summary.csv")
    assert len(frame) == len(config.cascade.decision) * len(config.cascade.refiner) * len(config.cascade.top_k)
    assert set(frame["decision_model"]) == set(config.cascade.decision)
    assert set(frame["cross_encoder"]) == set(config.cascade.refiner)
    assert set(frame["top_k"]) == set(config.cascade.top_k)
    assert frame["cascade_ndcg"].between(0, 1).all()
    assert (frame["ndcg_ci_low"] <= frame["cascade_ndcg"]).all()
    assert (frame["cascade_ndcg"] <= frame["ndcg_ci_high"]).all()


def test_full_pool_cascade_matches_cross_encoder_alone_up_to_ties():
    frame = pd.read_csv(RESULTS_DIR / "cascade_summary.csv")
    full = frame[frame["top_k"] == load_config().cascade.pool]
    assert ((full["cascade_ndcg"] - full["cross_encoder_ndcg"]).abs() < 2e-3).all()


def test_calibration(load):
    report = load("calibration.json")
    assert set(report) == set(load_config().models)
    for entry in report.values():
        assert 0 <= entry["ece"] <= 1
        assert 0 <= entry["brier"] <= 1
        assert len(entry["bins"]) == 10


def test_stability(load):
    report = load("stability.json")
    assert set(report) == set(load_config().models)
    for entry in report.values():
        assert -1 <= entry["kendall_tau_perturb"] <= 1
        for kind in load_config().stability.kinds:
            assert -1 <= entry[kind]["kendall_tau_perturb"] <= 1


def test_long_context(load):
    report = load("longcontext.json")
    assert set(report) == set(load_config().longcontext.models)
    for entry in report.values():
        assert 0 <= entry["full"]["ndcg_at_10"] <= 1
        assert 0 <= entry["truncated_512"]["ndcg_at_10"] <= 1
        assert set(entry["by_length"]) == {str(n) for n in load_config().longcontext.lengths}


def test_prompt_sensitivity(load):
    report = load("prompt_sensitivity.json")
    assert set(report) == set(load_config().models)
    for entry in report.values():
        assert len(entry["variants"]) >= 1
        assert entry["std_ndcg"] >= 0
