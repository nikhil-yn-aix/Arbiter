import argparse
import time

from .adapters import build_reranker, dispose
from .device import resolve_device
from .settings import load_config

QUERY = "What causes the seasons on Earth?"
RELEVANT = "The seasons are caused by the tilt of Earth's rotational axis relative to its orbit around the Sun."
UNRELATED = "The stock exchange closed higher on Tuesday after a strong earnings report from several banks."


def check(model_id: str, device: str) -> dict:
    start = time.perf_counter()
    reranker = build_reranker(model_id, device)
    loaded = time.perf_counter() - start
    begin = time.perf_counter()
    result = reranker.score_pairs([(QUERY, RELEVANT), (QUERY, UNRELATED)])
    scored = time.perf_counter() - begin
    dispose(reranker)
    return {
        "model": model_id,
        "ok": bool(result.score[0] > result.score[1]),
        "relevant": float(result.score[0]),
        "unrelated": float(result.score[1]),
        "prob_relevant": float(result.prob[0]),
        "load_s": round(loaded, 2),
        "score_s": round(scored, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()
    device = resolve_device(args.device)
    chosen = args.models or list(load_config().models)
    failures = 0
    print(f"device: {device}")
    for model_id in chosen:
        row = check(model_id, device)
        failures += not row["ok"]
        print(row, flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
