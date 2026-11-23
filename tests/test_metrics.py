import ir_measures
import numpy as np
import pandas as pd
import pytest
from ir_measures import RR, nDCG

from arbiter.metrics import per_query, rank


def cands(rows):
    return pd.DataFrame(rows, columns=["qid", "docid", "bm25_rank"]).assign(query="q", text="t")


def test_ndcg_hand_computed(toy_qrels):
    c = cands([("a", "d2", 0), ("a", "d1", 1), ("a", "x", 2)])
    pq = per_query(rank(c, [3.0, 2.0, 1.0]), toy_qrels).set_index("qid")
    dcg = 1 / np.log2(2) + 3 / np.log2(3)
    idcg = 3 / np.log2(2) + 1 / np.log2(3)
    assert pq.loc["a", "ndcg10"] == pytest.approx(dcg / idcg)
    assert pq.loc["a", "mrr10"] == 1.0


def test_unretrieved_relevant_counts_in_ideal(toy_qrels):
    c = cands([("b", "zz", 0)])
    pq = per_query(rank(c, [1.0]), toy_qrels).set_index("qid")
    assert pq.loc["b", "ndcg10"] == 0.0 and pq.loc["b", "mrr10"] == 0.0


def test_ties_follow_first_stage_rank(toy_qrels):
    c = cands([("a", "x", 0), ("a", "d1", 1)])
    r = rank(c, [1.0, 1.0])
    assert r.docid.tolist() == ["x", "d1"]


def test_matches_ir_measures_on_random_runs():
    rng = np.random.default_rng(0)
    rows, qrels = [], []
    for q in range(30):
        for d in range(40):
            rows.append((f"q{q}", f"d{d}", d))
            if rng.random() < 0.15:
                qrels.append((f"q{q}", f"d{d}", int(rng.integers(1, 4))))
        qrels.append((f"q{q}", "unretrieved", 2))
    c = cands(rows)
    q = pd.DataFrame(qrels, columns=["qid", "docid", "rel"])
    s = rng.normal(size=len(c))
    ours = per_query(rank(c, s), q)
    run = [ir_measures.ScoredDoc(a, b, float(v)) for (a, b, _), v in zip(rows, s)]
    ref = ir_measures.calc_aggregate([nDCG @ 10, RR @ 10], [ir_measures.Qrel(*x) for x in qrels], run)
    assert ours.ndcg10.mean() == pytest.approx(ref[nDCG @ 10], abs=1e-9)
    assert ours.mrr10.mean() == pytest.approx(ref[RR @ 10], abs=1e-9)
