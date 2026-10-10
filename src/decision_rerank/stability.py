import argparse
import json
import random

import numpy as np

from . import checkpoint
from .adapters import build_reranker, dispose
from .datasets import load_dataset
from .device import resolve_device
from .metrics import kendall, ndcg_at, rank_by, top_overlap
from .perturb import build_perturbations
from .retrieval import build_candidates
from .scoring import load_scores, score_path
from .settings import RESULTS_DIR, ensure_dirs, load_config


def sample_queries(qids: list[str], count: int, seed: int) -> list[str]:
    ordered = sorted(qids)
    random.Random(seed).shuffle(ordered)
    return sorted(ordered[:count])


def summarise(taus, overlaps, variances, clean_ndcg, noisy_ndcg) -> dict:
    return {
        "kendall_tau_perturb": float(np.mean(taus)),
        "kendall_tau_std": float(np.std(taus)),
        "top10_overlap": float(np.mean(overlaps)),
        "rank_variance": float(np.mean(variances)),
        "ndcg_clean": float(np.mean(clean_ndcg)),
        "ndcg_perturbed": float(np.mean(noisy_ndcg)),
    }


def run(device: str) -> dict:
    config = load_config()
    cfg = config.stability
    ensure_dirs()
    dataset = load_dataset(cfg.dataset)
    candidates = build_candidates(cfg.dataset)
    qids = sample_queries(list(dataset.queries), cfg.queries, config.seed)
    perturbed = build_perturbations({q: dataset.queries[q] for q in qids}, cfg.perturbations, config.seed, device)
    threshold = config.datasets[cfg.dataset].binary_threshold
    cutoff = config.evaluation.cutoff
    report = {}
    for model_id in config.models:
        done = checkpoint.load("stability", model_id)
        if done is not None:
            report[model_id] = done
            continue
        clean = load_scores(score_path(model_id, cfg.dataset))
        reranker = build_reranker(model_id, device)
        pairs, owners = [], []
        for qid in qids:
            docs = candidates[qid][: cfg.candidates]
            for kind in cfg.kinds:
                for version, text in enumerate(perturbed[qid][kind][: cfg.perturbations]):
                    for doc in docs:
                        pairs.append((text, dataset.corpus[doc]))
                    owners.append((qid, kind, version, len(docs)))
        result = reranker.score_pairs(pairs)
        dispose(reranker)
        buckets = {kind: ([], [], {}, [], []) for kind in cfg.kinds}
        cursor = 0
        for qid, kind, _, size in owners:
            docs = candidates[qid][: cfg.candidates]
            noisy = result.score[cursor : cursor + size]
            cursor += size
            base = clean[qid].score[:size]
            taus, overlaps, positions, clean_ndcg, noisy_ndcg = buckets[kind]
            taus.append(kendall(base, noisy))
            base_rank = rank_by(docs, base)
            noisy_rank = rank_by(docs, noisy)
            overlaps.append(top_overlap(base_rank, noisy_rank, cutoff))
            clean_ndcg.append(ndcg_at(base_rank, dataset.qrels[qid], cutoff))
            noisy_ndcg.append(ndcg_at(noisy_rank, dataset.qrels[qid], cutoff))
            positions.setdefault(qid, []).append({d: i for i, d in enumerate(noisy_rank)})
        entry = {}
        all_taus, all_overlaps, all_var, all_clean, all_noisy = [], [], [], [], []
        for kind, (taus, overlaps, positions, clean_ndcg, noisy_ndcg) in buckets.items():
            variances = [
                float(np.mean([np.var([p[d] for p in versions]) for d in versions[0]]))
                for versions in positions.values()
            ]
            entry[kind] = summarise(taus, overlaps, variances, clean_ndcg, noisy_ndcg)
            all_taus += taus
            all_overlaps += overlaps
            all_var += variances
            all_clean += clean_ndcg
            all_noisy += noisy_ndcg
        entry.update(summarise(all_taus, all_overlaps, all_var, all_clean, all_noisy))
        entry["threshold"] = threshold
        report[model_id] = entry
        checkpoint.save("stability", model_id, entry)
        print(
            f"{model_id:11s} tau={entry['kendall_tau_perturb']:.3f} "
            f"typos={entry['typos']['kendall_tau_perturb']:.3f} "
            f"syn={entry['synonyms']['kendall_tau_perturb']:.3f} "
            f"para={entry['paraphrase']['kendall_tau_perturb']:.3f}",
            flush=True,
        )
    (RESULTS_DIR / "stability.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run(resolve_device(args.device))


if __name__ == "__main__":
    main()
