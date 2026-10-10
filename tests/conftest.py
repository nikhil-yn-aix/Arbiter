import json

import pandas as pd
import pytest

from decision_rerank.settings import RESULTS_DIR


@pytest.fixture(scope="session")
def benchmark():
    return pd.read_csv(RESULTS_DIR / "benchmark_summary.csv")


@pytest.fixture(scope="session")
def load():
    def read(name):
        return json.loads((RESULTS_DIR / name).read_text())

    return read
