import json
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from arbiter.config import ROOT


def _replace(tmp: Path, dst: Path) -> None:
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, dst)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    _replace(tmp, path)


def read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    _replace(tmp, path)


def save_atomic(path: Path, saver) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    saver(tmp)
    _replace(tmp, path)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Status:
    """persistent per-run state machine: pending -> running -> done | failed."""

    def __init__(self, run_dir: Path):
        self.path = run_dir / "status.json"

    def read(self) -> dict:
        return read_json(self.path, {"state": "pending", "attempts": 0})

    def update(self, **kw) -> dict:
        s = self.read()
        s.update(kw, updated=time.time())
        write_json(self.path, s)
        return s

    @property
    def state(self) -> str:
        return self.read()["state"]


@contextmanager
def run_lock(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / "lock"
    if lock.exists():
        pid = int(lock.read_text() or 0)
        if pid and pid != os.getpid() and pid_alive(pid):
            raise RuntimeError(f"{run_dir.name} is locked by live pid {pid}")
    lock.write_text(str(os.getpid()))
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


class EventLog:
    def __init__(self, logs: Path, name: str):
        logs.mkdir(parents=True, exist_ok=True)
        self.path = logs / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}.jsonl"

    def __call__(self, event: str, **kw) -> None:
        rec = {"t": time.strftime("%H:%M:%S"), "event": event, **kw}
        line = json.dumps(rec, default=str)
        with open(self.path, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)


def _cmd(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, cwd=ROOT, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def git_state() -> dict:
    return {
        "sha": _cmd(["git", "rev-parse", "HEAD"]),
        "dirty": bool(_cmd(["git", "status", "--porcelain", "--", "src", "configs", "pyproject.toml", "uv.lock"])),
    }


def environment() -> dict:
    import torch
    import transformers

    gpu = _cmd(["nvidia-smi", "--query-gpu=name,driver_version,power.limit", "--format=csv,noheader"])
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "gpu": gpu,
        "host": platform.node(),
        **git_state(),
    }
