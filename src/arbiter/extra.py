import re

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from arbiter.arms import drift_table
from arbiter.config import Study
from arbiter.data import build_groups, eval_sets, load_pool
from arbiter.io import Status, write_json, write_parquet
from arbiter.jobs import SetCache, free, grid_jobs, main_jobs, run_dir, score_sets, selection
from arbiter.metrics import per_query, rank
from arbiter.model import build, tokenizer
from arbiter.score import encode_pairs, pad, token_batches


def references(study: Study, log) -> None:
    cache = SetCache(study)
    for name, spec in study.references.items():
        d = study.path("runs") / "references" / name
        if Status(d).state == "done":
            continue
        path = study.path("models") / spec["dir"]
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=torch.float16).cuda()
        write_json(d / "config.json", {"stage": "references", "arm": name, "label": spec["label"], "budget": None, "seed": 0, "lr": None, "run_id": name})
        stats = score_sets(study, model, tok, eval_sets(study, beir=True), cache, d)
        Status(d).update(state="done", **stats)
        log("reference", name=name, dev_mrr10=round(stats["dev_mrr10"], 4))
        del model
        free()


@torch.inference_mode()
def _features(encoder, tok, pairs: list[tuple[str, str]], layers: list[int], max_len: int, max_tokens: int) -> dict[int, np.ndarray]:
    ids = encode_pairs(tok, [q for q, _ in pairs], [d for _, d in pairs], max_len)
    lengths = np.array([len(x) for x in ids])
    out = {lay: np.zeros((len(ids), encoder.config.hidden_size), dtype=np.float32) for lay in layers}
    for b in token_batches(lengths, max_tokens):
        x, m = pad([ids[i] for i in b], tok.pad_token_id, "cuda")
        hs = encoder(input_ids=x, attention_mask=m, output_hidden_states=True).hidden_states
        w = m.unsqueeze(-1).float()
        for lay in layers:
            out[lay][b] = ((hs[lay].float() * w).sum(1) / w.sum(1)).cpu().numpy()
    return out


def probes(study: Study, log) -> pd.DataFrame:
    dest = study.path("runs") / "extra" / "probes.parquet"
    if dest.exists():
        return pd.read_parquet(dest)
    tok = tokenizer(study)
    groups = build_groups(load_pool(study), study.extra["probe_queries"], 0, study.train["group"])
    train_pairs = [p for g in groups for p in g]
    y = np.array([1 if i % len(g) == 0 else 0 for g in groups for i in range(len(g))])
    cands, qrels = SetCache(study)["dev"]
    test_pairs = list(zip(cands["query"], cands["text"]))
    layers = list(study.extra["probe_layers"])
    rows = []
    for arm in study.arms:
        model, _ = build(study, arm, 0)
        enc = model.model.half().cuda().eval()
        e = study.eval
        ftr = _features(enc, tok, train_pairs, layers, study.train["max_len"], e["max_tokens"])
        fte = _features(enc, tok, test_pairs, layers, e["max_len"], e["max_tokens"])
        for lay in layers:
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
            clf.fit(ftr[lay], y)
            s = clf.decision_function(fte[lay])
            mrr = float(per_query(rank(cands, s), qrels)["mrr10"].mean())
            rows.append({"arm": arm, "layer": lay, "dev_mrr10": mrr})
        log("probe", arm=arm, best=round(max(r["dev_mrr10"] for r in rows if r["arm"] == arm), 4))
        del model, enc
        free()
    df = pd.DataFrame(rows)
    write_parquet(dest, df)
    return df


def _layer(key: str) -> str:
    m = re.search(r"layers\.(\d+)\.", key)
    return m.group(1) if m else "other"


def _encoder(sd: dict) -> dict:
    return {k[len("model.") :]: v.double() for k, v in sd.items() if k.startswith("model.")}


def convergence(study: Study, log) -> pd.DataFrame:
    """per-layer distance to the base before fine-tuning and after fine-tuning at the same budget and seed 0."""
    from arbiter.arms import arm_state

    sel = selection(study)
    kept = {}
    for j in grid_jobs(study) + main_jobs(study):
        p = run_dir(study, j) / "model.safetensors"
        if j["seed"] == 0 and j["lr"] == sel["lr"][j["arm"]] and p.exists():
            kept[(j["arm"], j["budget"])] = p
    init_b = {k: v.double() for k, v in arm_state(study, "B").items()}
    rows = []
    for budget in sorted({b for _, b in kept}):
        if ("B", budget) not in kept:
            continue
        ft_b = _encoder(load_file(str(kept[("B", budget)])))
        for arm in study.arms:
            if arm == "B" or (arm, budget) not in kept:
                continue
            ft_a = _encoder(load_file(str(kept[(arm, budget)])))
            init_a = {k: v.double() for k, v in arm_state(study, arm).items()}
            acc = {}
            for k in init_b:
                a = acc.setdefault(_layer(k), np.zeros(3))
                a += [float((init_a[k] - init_b[k]).pow(2).sum()), float((ft_a[k] - ft_b[k]).pow(2).sum()), float(init_b[k].pow(2).sum())]
            for lay, (di, df_, n) in acc.items():
                rows.append({"arm": arm, "budget": budget, "layer": lay, "initial": np.sqrt(di / n), "finetuned": np.sqrt(df_ / n)})
        log("convergence", budget=budget)
    df = pd.DataFrame(rows)
    write_parquet(study.path("runs") / "extra" / "convergence.parquet", df)
    return df


def drift(study: Study, log) -> None:
    rel, cos = drift_table(study)
    write_parquet(study.path("runs") / "extra" / "drift.parquet", rel)
    write_parquet(study.path("runs") / "extra" / "cosine.parquet", cos)
    log("drift", arms=int(rel.arm.nunique()))
