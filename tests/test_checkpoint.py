import pytest

from decision_rerank import checkpoint, settings


@pytest.fixture()
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(checkpoint, "CACHE_DIR", tmp_path)
    return tmp_path


def test_round_trip(cache):
    assert checkpoint.load("unit", "a") is None
    checkpoint.save("unit", "a", {"value": 3})
    assert checkpoint.load("unit", "a") == {"value": 3}


def test_save_leaves_no_partial_files(cache):
    checkpoint.save("unit", "b", [1, 2, 3])
    assert not list((cache / "unit").glob("*.tmp"))


def test_overwrite_replaces_value(cache):
    checkpoint.save("unit", "c", 1)
    checkpoint.save("unit", "c", 2)
    assert checkpoint.load("unit", "c") == 2
