import matplotlib as mpl
import matplotlib.pyplot as plt

from .settings import load_config

INK = "#1f2933"
MUTED = "#5f6b76"
GRID = "#d9dde1"
SURFACE = "#ffffff"

DECISION = "#0072B2"
CROSS = "#D55E00"
BASELINE = "#009E73"
CASCADE = "#CC79A7"
HIGHLIGHT = "#E69F00"
SKY = "#56B4E9"

FAMILY_MARKER = {"laya": "o", "bge": "s", "qwen": "^", "jina": "D", "minilm": "v"}

MODEL_COLOR = {
    "laya-en": DECISION,
    "laya-multi": DECISION,
    "laya-td": DECISION,
    "bge-v2-m3": CROSS,
    "bge-base": HIGHLIGHT,
    "qwen3-06b": CASCADE,
    "jina-v2": SKY,
    "minilm": "#6b7280",
}
MODEL_MARKER = {
    "laya-en": "o",
    "laya-multi": "s",
    "laya-td": "^",
    "bge-v2-m3": "D",
    "bge-base": "P",
    "qwen3-06b": "X",
    "jina-v2": "v",
    "minilm": "h",
}
MODEL_FILL = {m: True for m in MODEL_COLOR}
SINGLE = 3.5
DOUBLE = 7.16
SIZE_AREA = {10: 12, 20: 24, 50: 42, 100: 68}


def apply() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 8,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.7,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "text.color": INK,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def label(model_id: str) -> str:
    return load_config().models[model_id].label


def marker_of(model_id: str) -> str:
    return MODEL_MARKER[model_id]


def scatter_style(model_id: str) -> dict:
    colour = MODEL_COLOR[model_id]
    filled = MODEL_FILL[model_id]
    return {
        "marker": marker_of(model_id),
        "facecolors": colour if filled else "white",
        "edgecolors": colour,
        "linewidths": 1.1,
    }


def panel_tag(ax, tag: str, title: str) -> None:
    ax.set_title(f"({tag}) {title}", loc="left")


def spread_labels(ax, items: list[tuple[float, float, str, str]], gap_px: float = 14.0, dx_px: float = 9.0) -> None:
    fig = ax.figure
    fig.canvas.draw()
    box = ax.get_window_extent()
    px = [ax.transData.transform((x, y))[0] for x, y, _, _ in items]
    anchor = [ax.transData.transform((x, y))[1] for x, y, _, _ in items]
    ys = list(anchor)
    low, high = box.y0 + gap_px / 2, box.y1 - gap_px / 2
    for _ in range(80):
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if abs(px[i] - px[j]) >= 160:
                    continue
                gap = ys[j] - ys[i]
                if abs(gap) >= gap_px:
                    continue
                direction = 1.0 if gap > 0 or (gap == 0 and anchor[j] >= anchor[i]) else -1.0
                push = (gap_px - abs(gap)) / 2 + 0.01
                ys[i] -= direction * push
                ys[j] += direction * push
        ys = [min(max(y, low), high) for y in ys]
    for i, (x, y, text, colour) in enumerate(items):
        target = ax.transData.inverted().transform((px[i] + dx_px, ys[i]))
        moved = abs(ys[i] - anchor[i]) > 4
        ax.annotate(
            text,
            xy=(x, y),
            xytext=target,
            textcoords="data",
            fontsize=6.3,
            color=INK,
            va="center",
            ha="left",
            arrowprops={"arrowstyle": "-", "color": colour, "lw": 0.5, "alpha": 0.8, "shrinkA": 0, "shrinkB": 4}
            if moved
            else None,
            annotation_clip=False,
        )


def save(fig, name: str) -> None:
    from .settings import PLOTS_DIR, ensure_dirs

    ensure_dirs()
    fig.savefig(PLOTS_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(PLOTS_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
