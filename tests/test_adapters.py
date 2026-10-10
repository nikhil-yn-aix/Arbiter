import pytest

from decision_rerank.adapters import build_reranker, dispose
from decision_rerank.device import resolve_device

QUERY = "What causes the seasons on Earth?"
RELEVANT = "The seasons are caused by the tilt of Earth's rotational axis relative to its orbit around the Sun."
UNRELATED = "The stock exchange closed higher on Tuesday after a strong earnings report from several banks."


@pytest.mark.parametrize("model_id", ["minilm", "laya-td", "bge-base"])
def test_small_models_rank_relevant_first_on_cpu(model_id):
    reranker = build_reranker(model_id, "cpu")
    result = reranker.score_pairs([(QUERY, RELEVANT), (QUERY, UNRELATED)])
    dispose(reranker)
    assert result.score[0] > result.score[1]
    assert 0 <= result.prob.min() and result.prob.max() <= 1


def test_device_resolution_falls_back():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cpu", "cuda"}
