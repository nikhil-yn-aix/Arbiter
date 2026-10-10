import json

from .settings import CACHE_DIR


def path(namespace: str, key: str):
    directory = CACHE_DIR / namespace
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{key}.json"


def load(namespace: str, key: str):
    target = path(namespace, key)
    return json.loads(target.read_text()) if target.exists() else None


def save(namespace: str, key: str, value) -> None:
    target = path(namespace, key)
    partial = target.with_suffix(".tmp")
    partial.write_text(json.dumps(value))
    partial.rename(target)
