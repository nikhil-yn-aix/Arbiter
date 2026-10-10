import json

from decision_rerank.settings import DATA_DIR, MODELS_DIR, load_config


def test_every_model_is_on_disk():
    config = load_config()
    for model_id in config.models:
        folder = MODELS_DIR / model_id
        assert (folder / "config.json").exists(), model_id
        assert any(folder.glob("*.safetensors")), model_id


def test_model_manifest_covers_models():
    manifest = json.loads((MODELS_DIR / "manifest.json").read_text())
    assert set(load_config().models) <= set(manifest)


def test_datasets_have_expected_sizes():
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    assert manifest["scifact"]["test_queries"] == 300
    assert manifest["nfcorpus"]["test_queries"] == 323
    assert manifest["trec-dl-2019"]["judged_queries"] == 43
    assert manifest["trec-dl-2020"]["judged_queries"] == 54
    assert manifest["locomo"]["conversations"] == 10


def test_retrieval_returns_full_depth():
    from decision_rerank.retrieval import build_candidates

    config = load_config()
    for name in config.datasets:
        candidates = build_candidates(name)
        assert all(len(ranked) == config.retrieval.depth for ranked in candidates.values())
