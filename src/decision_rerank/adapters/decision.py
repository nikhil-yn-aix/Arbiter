import warnings

import laya
import numpy as np

from ..settings import load_config, model_dir
from .base import Pair, Reranker, Scored, length_order, restore

RUBRICS = {
    "four-level": {
        "type": "score",
        "instructions": "How well does the passage answer or support the query?",
        "criteria": [
            "irrelevant to the query",
            "on the same topic but does not help answer the query",
            "partially answers or supports the query",
            "fully answers or supports the query",
        ],
    },
    "three-level": {
        "type": "score",
        "instructions": "How useful is the passage for the query?",
        "criteria": ["not useful", "somewhat useful", "very useful"],
    },
    "yes-no": {
        "type": "noul",
        "instructions": "Does the passage answer or support the query?",
    },
}


def render_state(query: str, passage: str) -> str:
    return f"Query: {query}\nPassage: {passage}"


class DecisionReranker(Reranker):
    def __init__(self, model_id: str, device: str, rubric: str = "four-level"):
        super().__init__(model_id, device)
        self.spec = load_config().models[model_id]
        self.rubric = rubric
        self.layout = "state"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.agent = laya.load(str(model_dir(model_id)), device=device)
        self.questions = {"rank": RUBRICS[rubric]}

    def use_rubric(self, rubric: str) -> None:
        self.rubric = rubric
        self.questions = {"rank": RUBRICS[rubric]}

    def extract(self, result: dict) -> tuple[float, float]:
        answer = result["answers"]["rank"]
        if answer["type"] == "noul":
            return float(answer["noul"]), float(answer["noul"])
        levels = [answer["probabilities"][str(i)] for i in range(len(answer["probabilities"]))]
        expected = sum(i * p for i, p in enumerate(levels)) / (len(levels) - 1)
        return float(expected), float(sum(levels[len(levels) // 2 :]))

    def score_in_question(self, pairs: list[Pair], limit: int, size: int) -> Scored:
        values = np.zeros((len(pairs), 2))
        groups: dict[str, list[int]] = {}
        for i, (query, _) in enumerate(pairs):
            groups.setdefault(query, []).append(i)
        for query, members in groups.items():
            text = RUBRICS[self.rubric]["instructions"].replace("the query?", f"this query: {query}")
            rubric = {**RUBRICS[self.rubric], "instructions": text}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = self.agent.predict_batch(
                    [pairs[i][1] for i in members], {"rank": rubric}, batch_size=size, max_len=limit
                )
            for i, result in zip(members, results, strict=True):
                values[i] = self.extract(result)
        return Scored(values[:, 0], values[:, 1])

    def score_pairs(self, pairs: list[Pair], max_tokens: int | None = None, batch_size: int | None = None) -> Scored:
        if not pairs:
            return Scored(np.zeros(0), np.zeros(0))
        if self.layout == "question":
            limit = min(max_tokens or self.spec.context, self.spec.context)
            return self.score_in_question(pairs, limit, max(1, min(batch_size or self.spec.batch, 16384 // limit)))
        order = length_order(pairs)
        states = [render_state(*pairs[i]) for i in order]
        limit = min(max_tokens or self.spec.context, self.spec.context)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            size = max(1, min(batch_size or self.spec.batch, 16384 // limit))
            results = self.agent.predict_batch(states, self.questions, batch_size=size, max_len=limit)
        values = np.array([self.extract(r) for r in results], dtype=np.float64)
        return Scored(restore(order, values[:, 0]), restore(order, values[:, 1]))

    def close(self) -> None:
        del self.agent
