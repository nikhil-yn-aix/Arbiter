import argparse
import collections
import sys

from arbiter import config
from arbiter.io import EventLog, Status, environment, write_json


def _study(args):
    return config.load(args.config)


def _guard(study, log, allow_dirty: bool) -> None:
    env = environment()
    if env["dirty"] and not allow_dirty:
        sys.exit("refusing to run: commit src/, configs/ and the lockfile first (or pass --allow-dirty)")
    write_json(study.path("runs") / "manifest.json", {**env, "config_digest": study.digest})
    log("manifest", sha=env["sha"][:10], gpu=env["gpu"], digest=study.digest)


def cmd_prepare(args):
    from arbiter.arms import build_noise_arms
    from arbiter.data import prepare

    study = _study(args)
    log = EventLog(study.path("logs"), "prepare")
    prepare(study, log)
    build_noise_arms(study, log)


def cmd_run(args):
    from arbiter import extra, jobs

    study = _study(args)
    log = EventLog(study.path("logs"), f"run-{args.stage}")
    _guard(study, log, args.allow_dirty)
    stages = ["grid", "select", "main", "extra"] if args.stage == "all" else [args.stage]
    for st in stages:
        log("stage", name=st)
        if st == "grid":
            log("stage_done", name=st, **jobs.run_stage(study, jobs.grid_jobs(study), log))
        elif st == "select":
            jobs.select(study, log)
        elif st == "main":
            log("stage_done", name=st, **jobs.run_stage(study, jobs.main_jobs(study), log))
        elif st == "extra":
            log("stage_done", name="cls", **jobs.run_stage(study, jobs.extra_jobs(study), log))
            extra.references(study, log)
            extra.probes(study, log)
            extra.drift(study, log)
            extra.convergence(study, log)
    log("finished", stages=stages)


def cmd_status(args):
    from arbiter import jobs

    study = _study(args)
    groups = {"grid": jobs.grid_jobs(study)}
    try:
        groups["main"] = jobs.main_jobs(study)
        groups["extra"] = jobs.extra_jobs(study)
    except RuntimeError:
        pass
    for name, js in groups.items():
        c = collections.Counter(Status(jobs.run_dir(study, j)).state for j in js)
        print(f"{name:6s} {len(js):3d} jobs  " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))


def cmd_collect(args):
    from arbiter.report import collect

    study = _study(args)
    collect(study, args.confirmatory, EventLog(study.path("logs"), "collect"))


def cmd_analyze(args):
    from arbiter.report import analyze

    study = _study(args)
    analyze(study, EventLog(study.path("logs"), "analyze"))


def cmd_plot(args):
    from arbiter.plots import render_all

    study = _study(args)
    render_all(study, EventLog(study.path("logs"), "plot"))


def cmd_smoke(args):
    from arbiter.smoke import run

    study = _study(args)
    run(study, EventLog(study.path("logs"), "smoke"))


def cmd_rerank(args):
    import pandas as pd
    import torch

    from arbiter import jobs
    from arbiter.model import build, load_trained, tokenizer
    from arbiter.score import score

    study = _study(args)
    lines = [x.strip() for x in open(args.file) if x.strip()]
    query, docs = lines[0], lines[1:]
    if args.untrained:
        model, _ = build(study, "B", 0)
        what = "ModernBERT-large + fresh head, no reranker training"
    else:
        job = jobs.grid_jobs(study)[0]
        model = load_trained(study, jobs.run_dir(study, job) / "model.safetensors")
        what = f"fine-tuned on {job['budget']} queries"
    model.to("cuda" if torch.cuda.is_available() else "cpu").half()
    cands = pd.DataFrame({"query": query, "text": docs})
    s = score(model, tokenizer(study), cands, study.eval["max_len"], study.eval["max_tokens"])
    print(f"\nmodel: {what}\nquery: {query}\n")
    for r, i in enumerate(s.argsort()[::-1], 1):
        print(f"  {r}. [{s[i]:+6.2f}]  {docs[i][:96]}")
    print()


def cmd_summary(args):
    from arbiter.io import read_json

    study = _study(args)
    st = read_json(study.path("results") / "stats.json")
    eff = st["descriptive"]["effects"]
    fx = st["secondary"]["failure_excluded"]
    ex = st["secondary"]["exploratory"]
    f = lambda e: f"{e['mean']:+.3f}  [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]"
    print("\ndecision checkpoints minus plain ModernBERT-large (mean of 5, 95% interval)\n")
    print(f"{'':34s}{'held-out MRR@10':>28s}{'TREC DL nDCG@10':>30s}")
    for b in (250, 1000, 4000, 16000):
        print(f"  all runs, {b:>5d} queries         {f(eff[f'heldout@{b}']['re']):>28s}{f(eff[f'dl@{b}']['re']):>30s}")
    print(f"  preregistered outcome: {st['primary']['outcome']}\n")
    for b in (1000, 4000, 16000):
        print(f"  collapsed runs out, {b:>5d}       {f(fx[f'heldout@{b}']['re']):>28s}{f(fx[f'dl@{b}']['re']):>30s}")
    print(f"  exploratory outcome:   {fx['outcome']}\n")
    print(f"  NLI checkpoint minus base, 1000     {f(ex['excl_failed/N_minus_B/heldout']):>22s}")
    print(f"  random noise minus base,   1000     {f(ex['excl_failed/R_minus_B/heldout']):>22s}")
    print(f"  collapsed runs: {len(fx['excluded'])}   frozen probe vs fine-tuned gain: rho = {st['descriptive']['probe_vs_finetuned']['rho']:.2f}\n")


def main(argv=None):
    p = argparse.ArgumentParser(prog="arbiter")
    p.add_argument("--config", default=config.DEFAULT)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare").set_defaults(fn=cmd_prepare)
    r = sub.add_parser("run")
    r.add_argument("stage", choices=["grid", "select", "main", "extra", "all"])
    r.add_argument("--allow-dirty", action="store_true")
    r.set_defaults(fn=cmd_run)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    c = sub.add_parser("collect")
    c.add_argument("--confirmatory", action="store_true")
    c.set_defaults(fn=cmd_collect)
    sub.add_parser("analyze").set_defaults(fn=cmd_analyze)
    sub.add_parser("plot").set_defaults(fn=cmd_plot)
    sub.add_parser("smoke").set_defaults(fn=cmd_smoke)
    rr = sub.add_parser("rerank")
    rr.add_argument("file", help="first line: query; other lines: passages")
    rr.add_argument("--untrained", action="store_true")
    rr.set_defaults(fn=cmd_rerank)
    sub.add_parser("summary").set_defaults(fn=cmd_summary)
    args = p.parse_args(argv)
    from arbiter.train import Interrupted

    try:
        args.fn(args)
    except Interrupted as e:
        sys.exit(f"interrupted ({e}); checkpoint saved, run the same command again to resume")


if __name__ == "__main__":
    main()
