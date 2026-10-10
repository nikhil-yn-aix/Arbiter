import xml.etree.ElementTree as ET

from PIL import Image

from decision_rerank.settings import PLOTS_DIR, TABLES_DIR

FIGURES = ["quality_latency", "candidate_scaling", "cascade", "long_context", "reliability", "prompt_sensitivity"]
TABLES = ["main_results", "cascade", "calibration", "long_context", "prompt_sensitivity"]


def test_figures_exist_in_both_formats():
    for name in FIGURES:
        assert (PLOTS_DIR / f"{name}.png").exists(), name
        assert (PLOTS_DIR / f"{name}.svg").exists(), name


def test_figure_resolution_and_validity():
    for name in FIGURES:
        with Image.open(PLOTS_DIR / f"{name}.png") as image:
            assert image.info["dpi"][0] >= 299
            assert image.width >= 900
        ET.parse(PLOTS_DIR / f"{name}.svg")


def test_tables_exist_and_are_complete():
    for name in TABLES:
        text = (TABLES_DIR / f"{name}.tex").read_text()
        assert r"\begin{tabular}" in text and r"\end{tabular}" in text
        assert "nan" not in text.lower().replace("nan0", "")
