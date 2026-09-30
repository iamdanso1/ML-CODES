"""
Thesis figure theme.

One place that decides what every chart in the study looks like, so the figures
read as one system in print: a fixed categorical hue order (never cycled), a
single-hue sequential ramp, a blue/red diverging ramp with a neutral midpoint,
hairline chrome and thin marks. ``save_fig`` writes a 300 dpi PNG plus a vector
PDF for the thesis document.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------------------
# Palette (validated: adjacent CVD dE 9.1, normal-vision dE 19.6 on #fcfcfb)
# --------------------------------------------------------------------------

CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

INK = {
    "surface": "#fcfcfb",
    "plane": "#f9f9f7",
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

_BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]

#: Single-hue sequential ramp for magnitude (heatmaps, matrices).
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", _BLUE_RAMP)

#: Diverging ramp: blue <-> red with a neutral grey midpoint (never a hue).
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a3a3", "#e34948", "#8f1f1f"]
)

#: Stable model -> hue assignment. Colour follows the entity, never its rank,
#: so a model keeps its hue in every figure regardless of how it scores.
MODEL_COLORS = {
    "Actual": INK["primary"],
    "Seasonal Naive": INK["muted"],
    "Naive": INK["muted"],
    "SARIMA": CATEGORICAL[0],
    "SARIMAX": CATEGORICAL[0],
    "Random Forest": CATEGORICAL[1],
    "XGBoost": CATEGORICAL[2],
    "LSTM": CATEGORICAL[6],
    "Ensemble": CATEGORICAL[4],
}

FIG_DIR = Path("outputs/figures")


def setup_theme(base_size: int = 11) -> None:
    """Install the thesis look as matplotlib defaults."""
    mpl.rcParams.update(
        {
            "figure.facecolor": INK["surface"],
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.facecolor": INK["surface"],
            "savefig.bbox": "tight",
            "axes.facecolor": INK["surface"],
            "axes.edgecolor": INK["axis"],
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK["secondary"],
            "axes.labelsize": base_size,
            "axes.titlesize": base_size + 2,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK["primary"],
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
            "grid.color": INK["grid"],
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",          # never dashed: dashes read as "projection"
            "xtick.color": INK["muted"],
            "ytick.color": INK["muted"],
            "xtick.labelsize": base_size - 1,
            "ytick.labelsize": base_size - 1,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": base_size - 1,
            "legend.labelcolor": INK["secondary"],
            "lines.linewidth": 2.0,         # thin marks
            "lines.markersize": 4.5,
            "lines.solid_capstyle": "round",
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": base_size,
            "text.color": INK["primary"],
            "figure.autolayout": False,
        }
    )


def titled(ax, title: str, subtitle: str = None) -> None:
    """
    Left-aligned title with an optional explanatory subtitle underneath.

    The title is pushed up far enough to leave the subtitle its own line --
    matplotlib has no notion of a two-part title, so the padding has to be
    opened up by hand or the two overlap.
    """
    if subtitle:
        ax.set_title(title, loc="left", pad=28)
        ax.text(
            0.0, 1.012, subtitle, transform=ax.transAxes,
            fontsize=mpl.rcParams["font.size"] - 1.5, color=INK["secondary"],
            ha="left", va="bottom",
        )
    else:
        ax.set_title(title, loc="left")


def label_end(ax, x, y, text, color, dx=6, fontsize=None) -> None:
    """Direct-label a series at its endpoint instead of leaning on colour alone."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, 0),
        textcoords="offset points",
        color=color,
        fontsize=fontsize or mpl.rcParams["font.size"] - 1.5,
        va="center",
        ha="left",
        fontweight="medium",
    )


def label_ends(ax, items, dx: int = 8, min_gap: float = 15.0, fontsize=None) -> None:
    """
    Direct-label several series at their endpoints without letting the labels
    collide.

    Converging series -- which is exactly what related fuel prices do -- would
    otherwise stack three labels on the same few pixels and render an unreadable
    blob. Labels are separated in *display* space, so the minimum gap holds
    whatever the axis scale is, and a hairline leader is drawn for any label
    that had to move off its true position.

    ``items`` is a sequence of ``(x, y, text, color)``.
    """
    items = [it for it in items if np.isfinite(it[1])]
    if not items:
        return
    fig = ax.figure
    fig.canvas.draw()
    x0 = ax.get_xlim()[0]
    to_px = lambda yv: ax.transData.transform((x0, yv))[1]
    to_data = lambda pv: ax.transData.inverted().transform((0, pv))[1]

    order = sorted(range(len(items)), key=lambda i: to_px(items[i][1]))
    pxs = [to_px(items[i][1]) for i in order]

    for k in range(1, len(pxs)):                      # push apart upwards
        pxs[k] = max(pxs[k], pxs[k - 1] + min_gap)
    top = ax.transData.transform((x0, ax.get_ylim()[1]))[1]
    overflow = pxs[-1] - top
    if overflow > 0:                                  # then pull the block back down
        pxs = [p - overflow for p in pxs]

    size = fontsize or mpl.rcParams["font.size"] - 1.5
    for k, i in enumerate(order):
        x, yv, text, color = items[i]
        y_lab = to_data(pxs[k])
        ax.annotate(text, xy=(x, y_lab), xytext=(dx, 0), textcoords="offset points",
                    color=color, fontsize=size, va="center", ha="left", fontweight="medium",
                    annotation_clip=False)
        if abs(pxs[k] - to_px(yv)) > 2:
            # The label has been displaced, so mark where the series really ends.
            ax.plot([x], [yv], marker="o", markersize=3.5, color=color,
                    markeredgecolor=INK["surface"], markeredgewidth=1.0, zorder=6)


def shade_span(ax, start, end, label=None, color=None, alpha=0.07) -> None:
    """Wash a date range (used for the hold-out window and policy breaks)."""
    color = color or INK["muted"]
    ax.axvspan(start, end, color=color, alpha=alpha, zorder=0, linewidth=0)
    if label:
        ax.text(
            start, 0.97, "  " + label, transform=ax.get_xaxis_transform(),
            color=INK["secondary"], fontsize=mpl.rcParams["font.size"] - 2,
            va="top", ha="left",
        )


def save_fig(fig, name: str, fig_dir=None) -> Path:
    """Write a 300 dpi PNG and a vector PDF, both thesis-ready."""
    fig_dir = Path(fig_dir or FIG_DIR)
    fig_dir.mkdir(parents=True, exist_ok=True)
    png = fig_dir / ("%s.png" % name)
    fig.savefig(png, dpi=300)
    fig.savefig(fig_dir / ("%s.pdf" % name))
    return png


def metric_table_style(df, lower_is_better=("RMSE", "MAE", "MAPE (%)", "sMAPE (%)", "MASE", "Theil U")):
    """Style a metrics frame: one blue ramp per column, best value in bold."""
    sty = df.style
    for col in df.columns:
        if df[col].dtype.kind not in "fi":
            continue
        asc = col in lower_is_better
        sty = sty.background_gradient(
            cmap=SEQ_BLUE if asc else SEQ_BLUE.reversed(), subset=[col], vmin=None, vmax=None
        )
    return sty.format(precision=4)


def annotate_source(fig, text: str) -> None:
    fig.text(
        0.0, -0.02, text, fontsize=mpl.rcParams["font.size"] - 2,
        color=INK["muted"], ha="left", va="top",
    )


def nice_ylim(ax, values, pad=0.08) -> None:
    v = np.asarray([x for x in np.ravel(values) if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return
    lo, hi = v.min(), v.max()
    span = hi - lo or abs(hi) or 1.0
    ax.set_ylim(lo - pad * span, hi + pad * span)
