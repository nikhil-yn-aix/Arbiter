import numpy as np
import pandas as pd


def rank(cands: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """order candidates per query by score, ties broken by the first-stage rank."""
    df = cands[["qid", "docid", "bm25_rank"]].assign(score=np.asarray(scores, dtype=np.float64))
    df = df.sort_values(["qid", "score", "bm25_rank"], ascending=[True, False, True], kind="mergesort")
    return df.assign(pos=df.groupby("qid").cumcount())


def per_query(ranked: pd.DataFrame, qrels: pd.DataFrame, k: int = 10, recall_k: int = 100, recall_rel: int = 1) -> pd.DataFrame:
    """ndcg@k (linear gains, ideal from all judged documents), mrr@k (rel>=1) and recall@k per query."""
    rel = {(q, d): r for q, d, r in qrels[["qid", "docid", "rel"]].itertuples(index=False)}
    ideal = qrels[qrels.rel > 0].groupby("qid")["rel"].apply(lambda s: np.sort(s.to_numpy())[::-1])
    nrel = qrels[qrels.rel >= recall_rel].groupby("qid").size()
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    rows = []
    for qid, g in ranked.groupby("qid", sort=True):
        if qid not in ideal.index:
            continue
        gains = np.array([max(rel.get((qid, d), 0), 0) for d in g.docid], dtype=float)
        top = gains[:k]
        idcg = float((ideal[qid][:k] * disc[: len(ideal[qid][:k])]).sum())
        ndcg = float((top * disc[: len(top)]).sum()) / idcg
        hit = np.flatnonzero(top > 0)
        mrr = 1.0 / (hit[0] + 1) if len(hit) else 0.0
        n = int(nrel.get(qid, 0))
        recall = float((gains[:recall_k] >= recall_rel).sum()) / n if n else np.nan
        rows.append((qid, ndcg, mrr, recall))
    return pd.DataFrame(rows, columns=["qid", "ndcg10", "mrr10", "recall"])
