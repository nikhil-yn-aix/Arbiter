import os

import pytest

from arbiter.io import Status, read_json, run_lock, write_json


def test_atomic_json_leaves_no_partial_file(tmp_path, monkeypatch):
    p = tmp_path / "x.json"
    write_json(p, {"a": 1})

    def boom(*_):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        write_json(p, {"a": 2})
    assert read_json(p) == {"a": 1}


def test_status_transitions(tmp_path):
    s = Status(tmp_path)
    assert s.state == "pending"
    s.update(state="running", attempts=1)
    s.update(state="done", dev_mrr10=0.5)
    assert s.read()["attempts"] == 1 and s.state == "done"


def test_lock_rejects_live_owner_and_recovers_dead(tmp_path):
    (tmp_path / "lock").write_text("1")
    with pytest.raises(RuntimeError):
        with run_lock(tmp_path):
            pass
    (tmp_path / "lock").write_text("999999")
    with run_lock(tmp_path):
        assert (tmp_path / "lock").read_text() == str(os.getpid())
    assert not (tmp_path / "lock").exists()


def test_lr_tag_keeps_short_ids_and_disambiguates():
    from arbiter.config import lr_tag

    assert [lr_tag(x) for x in (1e-5, 2e-5, 8e-5, 1.6e-4)] == ["1e-05", "2e-05", "8e-05", "1.6e-04"]
