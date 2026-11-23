import gzip
import json
import random
import re
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd
import Stemmer

from arbiter.config import Study
from arbiter.io import write_json, write_parquet

TRIPLETS = "msmarco/msmarco-bm25/triplet-50/train-00000-of-00018.parquet"
V11 = "msmarco/ms_marco/v1.1/validation-00000-of-00001.parquet"
DL_YEARS = {"dl19": "trec-dl-2019", "dl20": "trec-dl-2020"}
NEG = [f"negative_{k}" for k in range(1, 51)]
MIN_NEG = 30


def norm(q: str) -> str:
    return re.sub(r"\W+", " ", q.lower()).strip()


def _bm25(texts: list[str]):
    stem = Stemmer.Stemmer("english")
    idx = bm25s.BM25(k1=1.2, b=0.75)
    idx.index(bm25s.tokenize(texts, stopwords="en", stemmer=stem, show_progress=False), show_progress=False)
    return idx, lambda qs, ids=True: bm25s.tokenize(qs, stopwords="en", stemmer=stem, return_ids=ids, show_progress=False)


def _dl_queries(study: Study) -> set[str]:
    out = set()
    for d in DL_YEARS.values():
        with gzip.open(study.path("data") / d / "top1000.tsv.gz", "rt") as f:
            out.update(norm(line.split("\t")[2]) for line in f)
    return out


def split_msmarco(study: Study) -> dict[str, pd.DataFrame]:
    data = study.path("data")
    df = pd.read_parquet(data / TRIPLETS)
    v11 = {norm(q) for q in pd.read_parquet(data / V11, columns=["query"])["query"]}
    banned = v11 | _dl_queries(study)
    key = df["query"].map(norm)
    df = df[~key.isin(banned)].assign(_k=key[~key.isin(banned)]).drop_duplicates("_k").drop(columns="_k")
    negs = [list(dict.fromkeys(n for n in row[1:] if n != row[0])) for row in df[["positive", *NEG]].itertuples(index=False)]
    df = pd.DataFrame({"query": df["query"].to_numpy(), "positive": df["positive"].to_numpy(), "negatives": negs})
    df = df[df.negatives.map(len) >= MIN_NEG].reset_index(drop=True)
    perm = np.random.RandomState(study.train["sample_seed"]).permutation(len(df))
    e = study.eval
    test = df.iloc[perm[-e["test_queries"] :]]
    dev = df.iloc[perm[-(e["test_queries"] + e["dev_queries"]) : -e["test_queries"]]]
    pool = df.iloc[perm[: -(e["test_queries"] + e["dev_queries"])]]
    return {"pool": pool.reset_index(drop=True), "dev": dev.reset_index(drop=True), "test": test.reset_index(drop=True)}


def heldout_set(df: pd.DataFrame, n_neg: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, qrels = [], []
    rng = random.Random(seed)
    for i, r in enumerate(df.itertuples(index=False)):
        qid = f"m{i}"
        docs = [("pos", r.positive)] + [(f"neg{k}", n) for k, n in enumerate(r.negatives[:n_neg], 1)]
        order = list(range(len(docs)))
        rng.shuffle(order)
        for rank, j in enumerate(order):
            rows.append((qid, r.query, docs[j][0], docs[j][1], rank))
        qrels.append((qid, "pos", 1))
    cols = ["qid", "query", "docid", "text", "bm25_rank"]
    return pd.DataFrame(rows, columns=cols), pd.DataFrame(qrels, columns=["qid", "docid", "rel"])


def dl_set(study: Study, year: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = study.path("data") / DL_YEARS[year]
    top = pd.read_csv(root / "top1000.tsv.gz", sep="\t", names=["qid", "docid", "query", "text"], dtype=str, quoting=3)
    qrels = pd.read_csv(root / "qrels.txt", sep=r"\s+", names=["qid", "q0", "docid", "rel"], dtype={"qid": str, "docid": str})
    qrels = qrels[["qid", "docid", "rel"]]
    judged = set(qrels.loc[qrels.rel > 0, "qid"])
    passages = top.drop_duplicates("docid")[["docid", "text"]].reset_index(drop=True)
    pos = {d: i for i, d in enumerate(passages.docid)}
    idx, tok = _bm25(passages.text.tolist())
    top = top[top.qid.isin(judged)]
    out = []
    for _, g in top.groupby("qid", sort=True):
        terms = [t for t in tok([g["query"].iloc[0]], ids=False)[0] if t in idx.vocab_dict]
        s = idx.get_scores(terms)[[pos[d] for d in g.docid]]
        g = g.assign(bm25=s).sort_values(["bm25", "docid"], ascending=[False, True]).head(study.eval["dl_depth"])
        out.append(g.assign(bm25_rank=np.arange(len(g))))
    cands = pd.concat(out)[["qid", "query", "docid", "text", "bm25_rank"]].reset_index(drop=True)
    return cands, qrels[qrels.qid.isin(judged)].reset_index(drop=True)


def _beir_text(r: dict) -> str:
    title = (r.get("title") or "").strip()
    return f"{title}. {r['text']}" if title else r["text"]


def beir_set(study: Study, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = study.path("data") / name
    corpus = {}
    with open(root / "corpus.jsonl") as f:
        for line in f:
            r = json.loads(line)
            corpus[str(r["_id"])] = _beir_text(r)
    with open(root / "queries.jsonl") as f:
        queries = {str(r["_id"]): r["text"] for r in map(json.loads, f)}
    qrels = pd.read_csv(root / "qrels" / "test.tsv", sep="\t", dtype={"query-id": str, "corpus-id": str})
    qrels.columns = ["qid", "docid", "rel"]
    qids = sorted(set(qrels.loc[qrels.rel > 0, "qid"]) & set(queries))
    random.Random(study.eval["query_seed"]).shuffle(qids)
    qids = sorted(qids[: study.eval["beir_cap"]])
    ids = list(corpus)
    idx, tok = _bm25([corpus[d] for d in ids])
    hits, _ = idx.retrieve(tok([queries[q] for q in qids]), k=study.eval["beir_depth"], show_progress=False)
    rows = [(q, queries[q], ids[h], corpus[ids[h]], r) for q, hs in zip(qids, hits) for r, h in enumerate(hs)]
    cands = pd.DataFrame(rows, columns=["qid", "query", "docid", "text", "bm25_rank"])
    return cands, qrels[qrels.qid.isin(qids)].reset_index(drop=True)


def eval_sets(study: Study, beir: bool = False) -> list[str]:
    base = ["dev", "heldout", "dl19", "dl20"]
    return base + [f"beir-{b}" for b in study.eval["beir"]] if beir else base


def load_set(study: Study, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = study.path("cache") / "sets"
    return pd.read_parquet(d / f"{name}.cands.parquet"), pd.read_parquet(d / f"{name}.qrels.parquet")


def load_pool(study: Study) -> pd.DataFrame:
    return pd.read_parquet(study.path("cache") / "msmarco" / "pool.parquet")


def build_groups(pool: pd.DataFrame, budget: int, seed: int, group: int) -> list[list[tuple[str, str]]]:
    if budget > len(pool):
        raise ValueError(f"budget {budget} exceeds pool {len(pool)}")
    rng = random.Random(seed)
    groups = []
    for r in pool.iloc[:budget].itertuples(index=False):
        picks = rng.sample(range(len(r.negatives)), group - 1)
        groups.append([(r.query, r.positive)] + [(r.query, r.negatives[k]) for k in picks])
    rng.shuffle(groups)
    return groups


def prepare(study: Study, log) -> dict:
    cache = study.path("cache")
    split = split_msmarco(study)
    for k, v in split.items():
        write_parquet(cache / "msmarco" / f"{k}.parquet", v)
    log("msmarco", **{k: len(v) for k, v in split.items()})
    sets = {
        "dev": heldout_set(split["dev"], study.eval["heldout_negatives"], study.eval["query_seed"]),
        "heldout": heldout_set(split["test"], study.eval["heldout_negatives"], study.eval["query_seed"] + 1),
    }
    for y in DL_YEARS:
        sets[y] = dl_set(study, y)
    for b in study.eval["beir"]:
        sets[f"beir-{b}"] = beir_set(study, b)
    summary = {}
    for name, (cands, qrels) in sets.items():
        write_parquet(cache / "sets" / f"{name}.cands.parquet", cands)
        write_parquet(cache / "sets" / f"{name}.qrels.parquet", qrels)
        summary[name] = {"queries": int(cands.qid.nunique()), "pairs": len(cands)}
        log("set", name=name, **summary[name])
    write_json(cache / "summary.json", summary)
    return summary


def cache_ready(study: Study) -> bool:
    return (study.path("cache") / "summary.json").exists()


def set_path(study: Study, name: str) -> Path:
    return study.path("cache") / "sets" / f"{name}.cands.parquet"
