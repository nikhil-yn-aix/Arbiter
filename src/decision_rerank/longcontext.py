import argparse
import json
import time

import numpy as np
from transformers import AutoTokenizer

from .adapters import build_reranker, dispose
from .datasets import Dataset, load_locomo
from .device import resolve_device, sync
from .metrics import bootstrap_mean, query_metrics, rank_by
from .retrieval import fuse_candidates
from .settings import CACHE_DIR, MODELS_DIR, RESULTS_DIR, ensure_dirs, load_config

TRUNCATED = 512


def build_level(length: int, counter, device: str) -> tuple[Dataset, dict[str, list[str]], float, float]:
    cfg = load_config().longcontext
    corpus = load_locomo(length, counter, cfg.sample_queries, load_config().seed)
    dataset = Dataset(f"locomo-{length}", corpus.queries, {p.pid: p.text for p in corpus.passages}, corpus.qrels)
    candidates = fuse_candidates(dataset, cfg.candidates, device)
    reach = np.mean(
        [len({d for d in dataset.qrels[q]} & set(candidates[q])) / len(dataset.qrels[q]) for q in dataset.queries]
    )
    tokens = float(np.mean([counter(p.text) for p in corpus.passages]))
    return dataset, candidates, tokens, float(reach)


def evaluate(reranker, dataset: Dataset, candidates: dict[str, list[str]], max_tokens: int | None) -> dict:
    cutoff = load_config().evaluation.cutoff
    values, latency = [], []
    for qid in sorted(dataset.queries):
        docs = candidates[qid]
        pairs = [(dataset.queries[qid], dataset.corpus[d]) for d in docs]
        sync(reranker.device)
        begin = time.perf_counter()
        result = reranker.score_pairs(pairs, max_tokens=max_tokens)
        sync(reranker.device)
        latency.append((time.perf_counter() - begin) * 1000)
        ranked = rank_by(docs, result.score)
        values.append(query_metrics(ranked, dataset.qrels[qid], cutoff, 1))
    ndcg = np.array([v["ndcg"] for v in values])
    mean, low, high = bootstrap_mean(ndcg, 2000, load_config().seed)
    return {
        "ndcg_at_10": mean,
        "ndcg_ci_low": low,
        "ndcg_ci_high": high,
        "mrr_at_10": float(np.mean([v["mrr"] for v in values])),
        "recall_at_10": float(np.mean([v["recall"] for v in values])),
        "latency_p50_ms": float(np.percentile(latency[1:] or latency, 50)),
        "queries": len(values),
    }


def run(device: str) -> dict:
    config = load_config()
    cfg = config.longcontext
    ensure_dirs()
    tokenizer = AutoTokenizer.from_pretrained(MODELS_DIR / "qwen3-06b")

    def counter(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    levels = {}
    for length in cfg.lengths:
        levels[length] = build_level(length, counter, device)
        print(
            f"length {length}: passages={len(levels[length][0].corpus)} "
            f"mean_tokens={levels[length][2]:.0f} first_stage_reach={levels[length][3]:.3f}",
            flush=True,
        )
    report = {}
    cache_dir = CACHE_DIR / "longcontext"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for model_id in cfg.models:
        spec = config.models[model_id]
        entry = {"native_context": spec.context, "label": spec.label, "by_length": {}}
        reranker = None
        for length, (dataset, candidates, tokens, reach) in levels.items():
            saved = cache_dir / f"{model_id}__{length}.json"
            if saved.exists():
                entry["by_length"][str(length)] = json.loads(saved.read_text())
                continue
            if reranker is None:
                reranker = build_reranker(model_id, device)
            full = evaluate(reranker, dataset, candidates, None)
            short = dict(full) if spec.context <= TRUNCATED else evaluate(reranker, dataset, candidates, TRUNCATED)
            level = {"mean_passage_tokens": tokens, "first_stage_reach": reach, "full": full, "truncated_512": short}
            saved.write_text(json.dumps(level))
            entry["by_length"][str(length)] = level
            print(
                f"{model_id:11s} L={length:5d} full={full['ndcg_at_10']:.3f} trunc={short['ndcg_at_10']:.3f}",
                flush=True,
            )
        if reranker is not None:
            dispose(reranker)
        longest = entry["by_length"][str(max(cfg.lengths))]
        entry["full"] = longest["full"]
        entry["truncated_512"] = longest["truncated_512"]
        report[model_id] = entry
    (RESULTS_DIR / "longcontext.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run(resolve_device(args.device))


if __name__ == "__main__":
    main()
