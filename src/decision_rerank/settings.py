from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
CACHE_DIR = ROOT / ".cache"
TABLES_DIR = ROOT / "tables"


class Retrieval(BaseModel):
    bm25_k1: float
    bm25_b: float
    depth: int
    rrf_k: int
    dense_model: str
    candidate_sizes: list[int]


class Evaluation(BaseModel):
    cutoff: int
    bootstrap_resamples: int
    reference_model: str
    baseline: str


class Timing(BaseModel):
    warmup_queries: int
    queries_per_cell: int


class DatasetSpec(BaseModel):
    label: str
    source: str
    binary_threshold: int
    year: int | None = None


class LongContext(BaseModel):
    sample_queries: int
    candidates: int
    lengths: list[int]
    models: list[str]


class Cascade(BaseModel):
    pool: int
    timing_queries: int
    top_k: list[int]
    decision: list[str]
    refiner: list[str]


class Stability(BaseModel):
    dataset: str
    queries: int
    candidates: int
    perturbations: int
    kinds: list[str]


class Prompts(BaseModel):
    datasets: list[str]
    queries: int
    candidates: int


class ModelSpec(BaseModel):
    kind: str
    label: str
    family: str
    repo: str
    allow: list[str]
    params_m: int
    context: int
    batch: int
    backend: str | None = None

    @property
    def is_decision(self) -> bool:
        return self.kind == "decision"


class SupportSpec(BaseModel):
    repo: str
    allow: list[str]


class Config(BaseModel):
    seed: int
    device: str
    retrieval: Retrieval
    evaluation: Evaluation
    timing: Timing
    datasets: dict[str, DatasetSpec]
    longcontext: LongContext
    cascade: Cascade
    stability: Stability
    prompts: Prompts
    models: dict[str, ModelSpec]
    support_models: dict[str, SupportSpec]


@lru_cache(maxsize=1)
def load_config() -> Config:
    with open(ROOT / "CONFIG.yaml") as handle:
        return Config(**yaml.safe_load(handle))


def model_dir(model_id: str) -> Path:
    return MODELS_DIR / model_id


def ensure_dirs() -> None:
    for path in (RESULTS_DIR, PLOTS_DIR, CACHE_DIR, TABLES_DIR):
        path.mkdir(parents=True, exist_ok=True)
