import pandas as pd
import pytest

from arbiter import config
from arbiter.data import build_groups, cache_ready, load_pool, load_set, norm

pytestmark = pytest.mark.skipif(not cache_ready(config.load()), reason="run arbiter prepare first")


def test_no_query_leakage(study):
    pool = {norm(q) for q in load_pool(study)["query"]}
    for name in ["dev", "heldout", "dl19", "dl20"]:
        c, _ = load_set(study, name)
        assert not pool & {norm(q) for q in c["query"].unique()}, name
    dev = {norm(q) for q in load_set(study, "dev")[0]["query"]}
    test = {norm(q) for q in load_set(study, "heldout")[0]["query"]}
    assert not dev & test


def test_v11_validation_excluded(study):
    v11 = {norm(q) for q in pd.read_parquet(study.path("data") / "msmarco/ms_marco/v1.1/validation-00000-of-00001.parquet", columns=["query"])["query"]}
    assert not v11 & {norm(q) for q in load_pool(study)["query"]}


def test_budgets_are_nested_and_deterministic(study):
    pool = load_pool(study)
    small = {g[0] for g in build_groups(pool, 250, 0, 8)}
    big = {g[0] for g in build_groups(pool, 1000, 0, 8)}
    assert small <= big
    assert build_groups(pool, 250, 3, 8) == build_groups(pool, 250, 3, 8)
    assert build_groups(pool, 250, 3, 8) != build_groups(pool, 250, 4, 8)


def test_group_shape(study):
    g = build_groups(load_pool(study), 50, 0, 8)
    assert all(len(x) == 8 and len({q for q, _ in x}) == 1 for x in g)
    assert all(x[0][1] not in {d for _, d in x[1:]} for x in g)


def test_heldout_has_one_positive_per_query(study):
    c, q = load_set(study, "heldout")
    assert q.groupby("qid").size().eq(1).all()
    assert c.groupby("qid").size().eq(study.eval["heldout_negatives"] + 1).all()
