import torch

from .settings import load_config


def resolve_device(requested: str | None = None) -> str:
    choice = (requested or load_config().device).lower()
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return choice


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def reset_peak(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_gb(device: str) -> float:
    return torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0


def release(device: str) -> None:
    import gc

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
