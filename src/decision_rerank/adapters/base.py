from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

Pair = tuple[str, str]


@dataclass
class Scored:
    score: np.ndarray
    prob: np.ndarray


class Reranker(ABC):
    def __init__(self, model_id: str, device: str):
        self.model_id = model_id
        self.device = device

    @abstractmethod
    def score_pairs(
        self, pairs: list[Pair], max_tokens: int | None = None, batch_size: int | None = None
    ) -> Scored: ...

    def close(self) -> None:
        pass


def length_order(pairs: list[Pair]) -> np.ndarray:
    return np.argsort([len(q) + len(d) for q, d in pairs], kind="stable")


def restore(order: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values)
    out[order] = values
    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
