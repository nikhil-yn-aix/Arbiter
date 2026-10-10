from ..device import release, resolve_device
from ..settings import load_config
from .base import Reranker, Scored
from .cross import EncoderReranker, JinaReranker, QwenReranker
from .decision import DecisionReranker


def build_reranker(model_id: str, device: str | None = None, **options) -> Reranker:
    spec = load_config().models[model_id]
    where = resolve_device(device)
    if spec.is_decision:
        return DecisionReranker(model_id, where, **options)
    if spec.backend == "qwen":
        return QwenReranker(model_id, where, **options)
    if spec.backend == "jina":
        return JinaReranker(model_id, where, **options)
    return EncoderReranker(model_id, where, **options)


def dispose(reranker: Reranker) -> None:
    device = reranker.device
    reranker.close()
    release(device)


__all__ = ["Reranker", "Scored", "build_reranker", "dispose"]
