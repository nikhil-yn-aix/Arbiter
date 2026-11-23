import gc
import time
import traceback
from pathlib import Path

import pandas as pd
import torch
from safetensors.torch import save_file

from arbiter.config import Study, lr_tag
from arbiter.data import eval_sets, load_pool, load_set
from arbiter.io import Status, read_json, run_lock, save_atomic, write_json, write_parquet
from arbiter.metrics import per_query, rank
from arbiter.model import load_trained, tokenizer
from arbiter.score import score
from arbiter.train import Interrupted, NonFinite, StopFlag, train

SEALED = ("heldout", "dl19", "dl20")


def run_id(job: dict) -> str:
    rid = f"{job['arm']}__q{job['budget']}__lr{lr_tag(job['lr'])}__s{job['seed']}"
    return rid + (f"__{job['pooling']}" if job.get("pooling") else "")


def run_dir(study: Study, job: dict) -> Path:
    return study.path("runs") / job["stage"] / run_id(job)


def _arms(study: Study, spec) -> list[str]:
    return list(study.arms) if spec == "all" else list(spec)


def grid_jobs(study: Study) -> list[dict]:
    g = study.grid
    return [
        {"stage": "grid", "arm": a, "budget": g["budget"], "lr": lr, "seed": g["seed"], "sets": ["dev"], "keep": True} for a in study.arms for lr in g["lrs"]
    ]


def selection(study: Study) -> dict:
    sel = read_json(study.path("runs") / "selection.json")
    if sel is None:
        raise RuntimeError("run `arbiter select` first")
    return sel


def main_jobs(study: Study) -> list[dict]:
    sel = selection(study)
    jobs = []
    for block in study.main:
        protocol = block.get("protocol", "tuned")
        for a in _arms(study, block["arms"]):
            lr = sel["shared_lr"] if protocol == "shared" else sel["lr"][a]
            for s in block["seeds"]:
                sets = eval_sets(study, beir=s in block.get("beir_seeds", ()))
                keep = s in block.get("keep_seeds", ())
                jobs.append({"stage": "main", "arm": a, "budget": block["budget"], "lr": lr, "seed": s, "sets": sets, "keep": keep, "protocol": protocol})
    return jobs


def extra_jobs(study: Study) -> list[dict]:
    lr = selection(study)["lr"]
    c = study.extra["cls_check"]
    return [
        {"stage": "extra", "arm": a, "budget": c["budget"], "lr": lr[a], "seed": c["seed"], "pooling": "cls", "sets": eval_sets(study), "keep": False}
        for a in c["arms"]
    ]


class SetCache:
    def __init__(self, study: Study):
        self.study, self.sets = study, {}

    def __getitem__(self, name):
        if name not in self.sets:
            self.sets[name] = load_set(self.study, name)
        return self.sets[name]


def score_sets(study: Study, model, tok, sets: list[str], cache: SetCache, out: Path) -> dict:
    e = study.eval
    summary = {}
    for name in sets:
        cands, qrels = cache[name]
        t = time.time()
        s = score(model, tok, cands, e["max_len"], e["max_tokens"])
        dest = out / "scores" / (f"{name}.parquet" if name == "dev" else f"sealed/{name}.parquet")
        write_parquet(dest, cands[["qid", "docid"]].assign(score=s))
        summary[f"{name}_seconds"] = time.time() - t
        if name == "dev":
            summary["dev_mrr10"] = float(per_query(rank(cands, s), qrels)["mrr10"].mean())
    return summary


def free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def run_job(study: Study, job: dict, tok, pool, cache: SetCache, log, stop: StopFlag) -> str:
    d = run_dir(study, job)
    status = Status(d)
    if status.state == "done":
        return "done"
    with run_lock(d):
        write_json(d / "config.json", {**job, "run_id": run_id(job), "config_digest": study.digest})
        st = status.read()
        status.update(state="running", attempts=st.get("attempts", 0) + 1, started=time.time())
        micro = st.get("micro_queries")
        log("start", run=run_id(job), stage=job["stage"])
        while True:
            try:
                model, stats = train(study, job, d, tok, pool, log, stop, micro=micro)
                break
            except torch.cuda.OutOfMemoryError:
                free()
                micro = (micro or study.train["micro_queries"]) // 2
                if micro < 1:
                    status.update(state="failed", final=True, reason="oom")
                    log("failed", run=run_id(job), reason="oom")
                    return "failed"
                status.update(micro_queries=micro)
                log("oom_retry", run=run_id(job), micro=micro)
            except Interrupted as e:
                status.update(state="pending", reason=str(e))
                log("interrupted", run=run_id(job), reason=str(e))
                raise
            except NonFinite as e:
                status.update(state="failed", final=True, reason=f"nonfinite: {e}")
                log("failed", run=run_id(job), reason=str(e))
                free()
                return "failed"
            except Exception:
                status.update(state="failed", reason=traceback.format_exc()[-2000:])
                raise
        model.half()
        stats.update(score_sets(study, model, tok, job["sets"], cache, d))
        if job["keep"]:
            sd = {k: v.contiguous() for k, v in model.state_dict().items()}
            save_atomic(d / "model.safetensors", lambda p: save_file(sd, str(p)))
        (d / "ckpt.pt").unlink(missing_ok=True)
        status.update(state="done", finished=time.time(), **stats)
        log("done", run=run_id(job), **{k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()})
        del model
        free()
        return "done"


def run_stage(study: Study, jobs: list[dict], log) -> dict:
    tok = tokenizer(study)
    pool = load_pool(study)
    cache = SetCache(study)
    counts = {"done": 0, "failed": 0}
    with StopFlag() as stop:
        for i, job in enumerate(jobs):
            st = Status(run_dir(study, job)).read()
            if st["state"] == "done" or (st["state"] == "failed" and st.get("final")):
                counts[st["state"]] += 1
                continue
            log("job", index=i + 1, of=len(jobs), run=run_id(job))
            counts[run_job(study, job, tok, pool, cache, log, stop)] += 1
    return counts


def select(study: Study, log) -> dict:
    """lr per arm from dev mrr@10 of the grid; then score sealed sets for winners and base-lr runs."""
    jobs = grid_jobs(study)
    rows = []
    for j in jobs:
        s = Status(run_dir(study, j)).read()
        if s["state"] != "done":
            raise RuntimeError(f"grid run {run_id(j)} is {s['state']}")
        rows.append({"arm": j["arm"], "lr": j["lr"], "dev_mrr10": s["dev_mrr10"]})
    df = pd.DataFrame(rows).sort_values(["arm", "dev_mrr10", "lr"], ascending=[True, False, True])
    best = df.groupby("arm").head(1).set_index("arm")["lr"].to_dict()
    sel = {"lr": best, "shared_lr": best["B"], "grid": rows}
    write_json(study.path("runs") / "selection.json", sel)
    log("selection", **{a: lr_tag(v) for a, v in best.items()})
    tok = tokenizer(study)
    cache = SetCache(study)
    for j in jobs:
        d = run_dir(study, j)
        keep = j["lr"] == best[j["arm"]]
        needed = keep or j["lr"] == sel["shared_lr"]
        st = Status(d).read()
        if needed and not st.get("sealed_scored"):
            model = load_trained(study, d / "model.safetensors").half().cuda()
            stats = score_sets(study, model, tok, [s for s in eval_sets(study) if s != "dev"], cache, d)
            Status(d).update(sealed_scored=True, **stats)
            del model
            free()
            log("scored", run=run_id(j))
        if not keep:
            (d / "model.safetensors").unlink(missing_ok=True)
    return sel


def all_runs(study: Study) -> list[dict]:
    out = []
    for cfg in sorted(study.path("runs").glob("*/*/config.json")):
        job = read_json(cfg)
        out.append({**job, **{k: v for k, v in Status(cfg.parent).read().items() if k != "sets"}, "dir": str(cfg.parent)})
    return out
