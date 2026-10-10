from decision_rerank.settings import load_config


def test_every_cell_present(benchmark):
    config = load_config()
    expected = (len(config.models) + 1) * len(config.datasets) * len(config.retrieval.candidate_sizes)
    assert len(benchmark) == expected
    assert not benchmark[["ndcg_at_10", "latency_p50_ms", "vram_peak_gb"]].isna().any().any()


def test_metric_ranges(benchmark):
    assert benchmark["ndcg_at_10"].between(0, 1).all()
    assert benchmark["recall_at_10"].between(0, 1).all()
    assert benchmark["mrr_at_10"].between(0, 1).all()
    assert benchmark["latency_p50_ms"].gt(0).all()
    assert (benchmark["latency_p95_ms"] >= benchmark["latency_p50_ms"]).all()
    assert benchmark["throughput_qps"].gt(0).all()


def test_models_use_gpu_memory_when_present(benchmark):
    import torch

    if torch.cuda.is_available():
        models = benchmark[benchmark["model"] != load_config().evaluation.baseline]
        assert models["vram_peak_gb"].gt(0).all()


def test_bootstrap_intervals(load):
    intervals = load("benchmark_bootstrap.json")
    assert intervals
    for entry in intervals.values():
        assert entry["ci_low"] <= entry["mean_diff"] <= entry["ci_high"]
        assert 0 <= entry["p_two_sided"] <= 1
