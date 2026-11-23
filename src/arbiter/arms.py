import re

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file

from arbiter.config import Study
from arbiter.io import save_atomic

ANCHOR = "layers.0.attn.Wqkv.weight"


def encoder_state(path) -> dict[str, torch.Tensor]:
    """encoder tensors of a modernbert-style checkpoint, keys relative to the encoder root."""
    sd = load_file(str(path))
    hits = [k for k in sd if k.endswith(ANCHOR)]
    if len(hits) != 1:
        raise ValueError(f"{path}: expected one {ANCHOR}, found {len(hits)}")
    prefix = hits[0][: -len(ANCHOR)]
    return {k[len(prefix) :]: v.float() for k, v in sd.items() if k.startswith(prefix)}


def noise_like(base: dict, ref: dict, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k in sorted(base):
        d = torch.randn(base[k].shape, generator=g)
        out[k] = base[k] + d * ((ref[k] - base[k]).norm() / d.norm())
    return out


def noise_path(study: Study, arm: str):
    return study.path("cache") / "arms" / f"{arm}.safetensors"


def arm_state(study: Study, arm: str) -> dict[str, torch.Tensor]:
    spec = study.arms[arm]
    if "noise_of" in spec:
        return encoder_state(noise_path(study, arm))
    return encoder_state(study.arm_dir(arm) / "model.safetensors")


def build_noise_arms(study: Study, log) -> None:
    for arm, spec in study.arms.items():
        if "noise_of" not in spec or noise_path(study, arm).exists():
            continue
        base = arm_state(study, "B")
        ref = arm_state(study, spec["noise_of"])
        sd = noise_like(base, ref, spec["noise_seed"])
        save_atomic(noise_path(study, arm), lambda p, sd=sd: save_file({k: v.contiguous() for k, v in sd.items()}, str(p)))
        log("noise_arm", arm=arm, of=spec["noise_of"])


def _layer(key: str) -> str:
    m = re.match(r"layers\.(\d+)\.", key)
    return m.group(1) if m else "embeddings" if key.startswith("embeddings") else "final_norm"


def drift_table(study: Study) -> tuple[pd.DataFrame, pd.DataFrame]:
    """per-layer relative drift of each arm from the base, and cosine between arms' drift vectors."""
    base = arm_state(study, "B")
    keys = sorted(base)
    deltas, rows = {}, []
    base_sq = {k: float(base[k].double().pow(2).sum()) for k in keys}
    for arm in study.arms:
        if arm == "B":
            continue
        sd = arm_state(study, arm)
        d = {k: (sd[k].double() - base[k].double()) for k in keys}
        deltas[arm] = d
        per = {}
        for k in keys:
            a = per.setdefault(_layer(k), [0.0, 0.0])
            a[0] += float(d[k].pow(2).sum())
            a[1] += base_sq[k]
        tot = sum(v[0] for v in per.values()) / sum(v[1] for v in per.values())
        rows.append({"arm": arm, "layer": "all", "rel_drift": float(np.sqrt(tot))})
        rows += [{"arm": arm, "layer": lay, "rel_drift": float(np.sqrt(a / b))} for lay, (a, b) in per.items()]
    arms = list(deltas)
    cos = []
    for i in arms:
        for j in arms:
            dot = sum(float((deltas[i][k] * deltas[j][k]).sum()) for k in keys)
            ni = sum(float(deltas[i][k].pow(2).sum()) for k in keys)
            nj = sum(float(deltas[j][k].pow(2).sum()) for k in keys)
            cos.append({"a": i, "b": j, "cosine": dot / np.sqrt(ni * nj)})
    return pd.DataFrame(rows), pd.DataFrame(cos)
