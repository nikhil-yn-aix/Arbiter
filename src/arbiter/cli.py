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
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
