"""Shared chart helpers and the comparison chart, following the repository plotting rules (16:9, PNG).

Figures are built with `matplotlib.figure.Figure` directly, so no GUI backend is needed.
"""

from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure

from vitfinetune import constants as C
from vitfinetune.errors import ResultsError
from vitfinetune.training import RunResult

TITLE_SIZE = 20
LABEL_SIZE = 14
TICK_SIZE = 12
ANNOTATION_SIZE = 12
HEADROOM = 1.15
PERCENT = 100.0


def apply_style() -> None:
    """Set the shared rcParams for every chart in the repository."""
    mpl.rcParams.update({
        "axes.titlesize": LABEL_SIZE + 2, "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
    })


def save_png(fig: Figure, path: Path) -> Path:
    """Write a figure as PNG at the shared DPI and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=C.FIG_DPI, format="png")
    return path


def require(results: list) -> None:
    """Raise ResultsError if there is nothing to plot."""
    if not results:
        raise ResultsError("no results to plot; run the experiment first")


def _final_acc(result: RunResult) -> float:
    return (result.final_test_acc or 0.0) * PERCENT


class TransferPlotter:
    """Draws the comparison chart: accuracy, training time, peak VRAM and trainable params."""

    PANELS = (
        ("Final test accuracy (%)", lambda r: _final_acc(r), "{:.1f}%"),
        ("Training time (min)", lambda r: r.train_seconds / 60.0, "{:.1f}"),
        ("Peak VRAM (MiB)", lambda r: r.peak_vram_mib or 0.0, "{:,.0f}"),
        ("Trainable params (M)", lambda r: r.trainable_params / 1e6, "{:.2f}"),
    )

    def plot(self, results: list[RunResult], path: Path) -> Path:
        """One panel per metric, one bar per method."""
        require(results)
        apply_style()
        fig = Figure(figsize=C.FIG_SIZE, layout="constrained")
        axes = fig.subplots(1, len(self.PANELS))
        colors = C.PALETTE[: len(results)]
        for ax, (title, getter, fmt) in zip(axes, self.PANELS):
            values = [getter(r) for r in results]
            bars = ax.bar(range(len(results)), values, color=colors, width=0.7)
            for bar, value in zip(bars, values):
                ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 5), textcoords="offset points", ha="center", fontsize=ANNOTATION_SIZE)
            ax.set_xticks(range(len(results)), [r.label for r in results], rotation=20, ha="right")
            ax.set_ylim(0, max(values) * HEADROOM if max(values) > 0 else 1)
            ax.set_title(title)
        fig.suptitle("ViT-B/16 on Flowers-102 (1,020 training images): three ways to train",
                     fontsize=TITLE_SIZE)
        return save_png(fig, path)
