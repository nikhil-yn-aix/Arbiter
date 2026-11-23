import json
import math
import signal
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from arbiter.config import Study
from arbiter.data import build_groups
from arbiter.io import save_atomic
from arbiter.model import build
from arbiter.score import encode_pairs, pad


class Interrupted(Exception):
    pass


class NonFinite(Exception):
    pass


class StopFlag:
    """turns sigint/sigterm into a request to checkpoint at the next step boundary."""

    def __init__(self):
        self.set = False
        self._old = {}

    def __enter__(self):
        for s in (signal.SIGINT, signal.SIGTERM):
            self._old[s] = signal.signal(s, self._handle)
        return self

    def _handle(self, *_):
        self.set = True

    def __exit__(self, *exc):
        for s, h in self._old.items():
            signal.signal(s, h)


def schedule(steps: int, warmup: float):
    w = max(1, int(warmup * steps))
    return lambda s: min((s + 1) / w, max(0.0, (steps - s) / max(1, steps - w)))


def tokenize_groups(tok, groups, max_len: int) -> list[list[list[int]]]:
    flat = [p for g in groups for p in g]
    ids = encode_pairs(tok, [q for q, _ in flat], [d for _, d in flat], max_len)
    n = len(groups[0])
    return [ids[i * n : (i + 1) * n] for i in range(len(groups))]


def micro_batches(step_groups: list, micro: int) -> list[list]:
    order = sorted(range(len(step_groups)), key=lambda i: max(len(x) for x in step_groups[i]))
    return [[step_groups[i] for i in order[j : j + micro]] for j in range(0, len(order), micro)]


def step_loss(model, batch: list, pad_id: int, n_total: int, device) -> float:
    flat = [x for g in batch for x in g]
    x, m = pad(flat, pad_id, device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(input_ids=x, attention_mask=m).logits.float().view(len(batch), -1)
    loss = F.cross_entropy(logits, torch.zeros(len(batch), dtype=torch.long, device=device), reduction="sum")
    if not torch.isfinite(loss):
        raise NonFinite(f"loss {loss.item()}")
    (loss / n_total).backward()
    return loss.item()


def _trim_log(path: Path, step: int) -> None:
    if not path.exists():
        return
    keep = [ln for ln in path.read_text().splitlines() if ln and json.loads(ln)["step"] <= step]
    path.write_text("".join(ln + "\n" for ln in keep))


def train(study: Study, job: dict, run_dir: Path, tok, pool, log, stop: StopFlag, device=None, micro: int | None = None):
    """train one job to completion, resuming from run_dir/ckpt.pt when present."""
    tr = study.train
    device = device or torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model, _ = build(study, job["arm"], job["seed"], job.get("pooling"))
    model.to(device).train()
    groups = tokenize_groups(tok, build_groups(pool, job["budget"], job["seed"], tr["group"]), tr["max_len"])
    qps = tr["queries_per_step"]
    micro = micro or tr["micro_queries"]
    steps = math.ceil(len(groups) / qps)
    opt = torch.optim.AdamW(model.parameters(), lr=job["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule(steps, tr["warmup"]))
    ckpt = run_dir / "ckpt.pt"
    run_dir.mkdir(parents=True, exist_ok=True)
    logf = run_dir / "train.jsonl"
    start = 0
    if ckpt.exists():
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start = st["step"]
        _trim_log(logf, start)
        log("resume", run=run_dir.name, step=start, of=steps)

    def save(step: int) -> None:
        state = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}
        save_atomic(ckpt, lambda p: torch.save(state, p))

    t0, last_ckpt, pairs, acc = time.time(), time.time(), 0, []
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    with open(logf, "a") as fl:
        for i in range(start, steps):
            sg = groups[i * qps : (i + 1) * qps]
            total = 0.0
            for mb in micro_batches(sg, micro):
                total += step_loss(model, mb, tok.pad_token_id, len(sg), device)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tr["clip"]).item()
            if not math.isfinite(gnorm):
                raise NonFinite(f"grad norm {gnorm} at step {i}")
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            pairs += sum(len(g) for g in sg)
            acc.append(total / len(sg))
            if (i + 1) % 10 == 0 or i + 1 == steps:
                rate = pairs / (time.time() - t0)
                fl.write(json.dumps({"step": i + 1, "loss": sum(acc) / len(acc), "lr": sched.get_last_lr()[0], "grad_norm": gnorm, "pairs_per_s": rate}) + "\n")
                fl.flush()
                acc = []
            due = (i + 1) % tr["ckpt_every_steps"] == 0 or time.time() - last_ckpt > 60 * tr["ckpt_every_minutes"]
            if stop.set or (due and i + 1 < steps):
                save(i + 1)
                last_ckpt = time.time()
                if stop.set:
                    raise Interrupted(f"stopped at step {i + 1}")
    secs = time.time() - t0
    stats = {"steps": steps, "train_seconds": secs, "pairs_per_s": pairs / secs if secs else None, "resumed_from": start, "micro_queries": micro}
    if device.type == "cuda":
        stats["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    return model, stats
