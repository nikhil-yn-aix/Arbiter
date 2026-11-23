import shutil
import time

import numpy as np
import torch
from scipy.stats import spearmanr

from arbiter.config import Study
from arbiter.data import load_pool
from arbiter.io import write_json
from arbiter.jobs import SetCache
from arbiter.model import build, tokenizer
from arbiter.score import score
from arbiter.train import Interrupted, StopFlag, train


class _StopAt(StopFlag):
    def __init__(self, step_calls: int):
        super().__init__()
        self.calls, self.limit = 0, step_calls

    @property
    def set(self):
        self.calls += 1
        return self.calls > self.limit

    @set.setter
    def set(self, v):
        pass


def run(study: Study, log) -> dict:
    """gpu checks before the study: arms load, interrupt/resume, scoring precision, throughput."""
    out = {}
    tok = tokenizer(study)
    pool = load_pool(study)
    for arm in study.arms:
        _, n = build(study, arm, 0)
        out[f"load_{arm}"] = n
    log("arms_load", **{k: v for k, v in out.items()})
    root = study.path("runs") / "_smoke"
    shutil.rmtree(root, ignore_errors=True)
    job = {"arm": "D1", "budget": 96, "lr": 4e-5, "seed": 0}
    with StopFlag() as stop:
        ref, _ = train(study, job, root / "straight", tok, pool, log, stop)
    try:
        train(study, job, root / "resumed", tok, pool, log, _StopAt(5))
    except Interrupted:
        pass
    with StopFlag() as stop:
        res, _ = train(study, job, root / "resumed", tok, pool, log, stop)
    diff = max((a - b).abs().max().item() for a, b in zip(ref.state_dict().values(), res.state_dict().values()))
    out["resume_max_abs_diff"] = diff
    log("resume_check", max_abs_diff=diff)
    cands, _ = SetCache(study)["dl19"]
    cands = cands[cands.qid.isin(cands.qid.unique()[:10])]
    e = study.eval
    s32 = score(ref.float(), tok, cands, e["max_len"], e["max_tokens"])
    s16 = score(ref.half(), tok, cands, e["max_len"], e["max_tokens"])
    out["fp16_spearman"] = float(spearmanr(s32, s16).statistic)
    out["fp16_max_abs"] = float(np.abs(s32 - s16).max())
    log("precision", spearman=out["fp16_spearman"], max_abs=out["fp16_max_abs"])
    big, _ = SetCache(study)["heldout"]
    big = big.iloc[:6000]
    torch.cuda.synchronize()
    t = time.time()
    score(ref, tok, big, e["max_len"], e["max_tokens"])
    torch.cuda.synchronize()
    out["infer_pairs_per_s"] = len(big) / (time.time() - t)
    with StopFlag() as stop:
        _, st = train(study, {**job, "budget": 400, "arm": "B"}, root / "bench", tok, pool, log, stop)
    out["train_pairs_per_s"] = st["pairs_per_s"]
    out["peak_mem_gb"] = st.get("peak_mem_gb")
    log("throughput", infer=round(out["infer_pairs_per_s"]), train=round(out["train_pairs_per_s"]), peak_gb=round(out["peak_mem_gb"], 2))
    shutil.rmtree(root, ignore_errors=True)
    ok = diff < 1e-3 and out["fp16_spearman"] > 0.999
    out["ok"] = ok
    write_json(study.path("runs") / "smoke.json", out)
    if not ok:
        raise RuntimeError(f"smoke checks failed: {out}")
    return out
