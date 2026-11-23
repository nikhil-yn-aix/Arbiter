import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "configs" / "study.yaml"


def _freeze(x):
    if isinstance(x, dict):
        return MappingProxyType({k: _freeze(v) for k, v in x.items()})
    if isinstance(x, list):
        return tuple(_freeze(v) for v in x)
    return x


def lr_tag(lr: float) -> str:
    short = f"{lr:.0e}"
    return short if float(short) == lr else f"{lr:.1e}"


@dataclass(frozen=True)
class Study:
    raw: dict
    paths: MappingProxyType
    backbone: str
    arms: MappingProxyType
    references: MappingProxyType
    train: MappingProxyType
    eval: MappingProxyType
    grid: MappingProxyType
    main: tuple
    extra: MappingProxyType
    analysis: MappingProxyType

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True).encode()).hexdigest()[:16]

    def path(self, key: str) -> Path:
        return ROOT / self.paths[key]

    def arm_dir(self, arm: str) -> Path:
        return self.path("models") / self.arms[arm]["dir"]

    @property
    def decision_arms(self) -> list[str]:
        return [a for a, s in self.arms.items() if s["kind"] == "decision"]


def load(path: Path | str = DEFAULT) -> Study:
    raw = yaml.safe_load(Path(path).read_text())
    for arm, spec in raw["arms"].items():
        if spec["kind"] not in {"base", "decision", "control"}:
            raise ValueError(f"unknown kind for {arm}")
        if "dir" not in spec and "noise_of" not in spec:
            raise ValueError(f"{arm} needs dir or noise_of")
    tr = raw["train"]
    if tr["queries_per_step"] % tr["micro_queries"]:
        raise ValueError("queries_per_step must be a multiple of micro_queries")
    fields = {k: _freeze(raw[k]) for k in Study.__dataclass_fields__ if k != "raw"}
    return Study(raw=raw, **fields)
