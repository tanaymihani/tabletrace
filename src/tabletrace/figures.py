"""Figures for the README: error by stage, F1 by document, risk-coverage, OCR fine-tuning, failure crops."""

from __future__ import annotations

from pathlib import Path as FsPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order: blue, orange, aqua, yellow

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": INK_2,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def _style(ax, grid_axis="x"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)


# Colour follows the entity: every chart that shows routes uses the same slot for the same route.
ROUTE_COLOR = {"Text layer": SERIES[0], "OCR: rendered pages": SERIES[1], "OCR: real scans": SERIES[2]}


def _hbar(ax, y, width, height, color, radius_px=4.0):
    """A horizontal bar, square at the baseline with a rounded data end.

    The radius is set in pixels, so call this after the axis limits are final.
    """
    if width <= 0:
        return
    (ax0, ay0), (ax1, ay1) = ax.transData.transform([(0, 0), (1, 1)])
    rx = min(radius_px / max(abs(ax1 - ax0), 1e-9), width / 2)
    ry = min(radius_px / max(abs(ay1 - ay0), 1e-9), height / 2)
    k = 0.5523  # cubic Bezier approximation of a quarter circle
    xl, xr, yb, yt = 0.0, width, y - height / 2, y + height / 2
    verts = [(xl, yb), (xr - rx, yb), (xr - rx + k * rx, yb), (xr, yb + ry - k * ry), (xr, yb + ry),
             (xr, yt - ry), (xr, yt - ry + k * ry), (xr - rx + k * rx, yt), (xr - rx, yt), (xl, yt), (xl, yb)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, linewidth=0))


def errors_by_stage(stage_counts: dict[str, int], out: FsPath, title: str) -> None:
    """Misses per pipeline stage, one bar series per route group."""
    order = ["Text layer", "OCR: rendered pages", "OCR: real scans"]
    present = {k.split(" | ")[0] for k in stage_counts}
    groups = [g for g in order if g in present]
    stages = ["table detection", "OCR character error", "table structure error", "header mapping error",
              "unit or number parsing error"]
    stages = [s for s in stages if any(stage_counts.get(f"{g} | {s}", 0) for g in groups)]
    fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(stages) * max(1, len(groups)) + 1.3))
    h = 0.8 / max(1, len(groups))
    xmax = max(stage_counts.values()) if stage_counts else 1
    ax.set_xlim(0, xmax * 1.12)
    ax.set_ylim(len(stages) - 0.5, -0.5)
    for gi, g in enumerate(groups):
        for si, st in enumerate(stages):
            v = stage_counts.get(f"{g} | {st}", 0)
            y = si + (gi - (len(groups) - 1) / 2) * (h + 0.04)
            _hbar(ax, y, v, h, ROUTE_COLOR[g])
            if v:
                ax.text(v + xmax * 0.01, y, str(v), va="center", ha="left", color=INK_2, fontsize=9)
    ax.set_yticks(range(len(stages)), [st[0].upper() + st[1:] for st in stages])
    ax.set_xlabel("Missed labels")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    _style(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ROUTE_COLOR[g]) for g in groups]
    ax.legend(handles, groups, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def f1_by_document(rows: list[tuple[str, str, float, float]], out: FsPath, title: str) -> None:
    """rows: (document, group, strict F1, lenient F1). Bars show strict F1; a tick marks lenient F1."""
    order = ["Text layer", "OCR: rendered pages", "OCR: real scans"]
    groups = [g for g in order if any(r[1] == g for r in rows)]
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * len(rows) + 1.6))
    ax.set_xlim(0, 1.1)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    for i, (_doc, g, f1, lenient) in enumerate(rows):
        _hbar(ax, i, f1, 0.62, ROUTE_COLOR[g])
        if lenient > f1 + 0.005:
            ax.plot([lenient, lenient], [i - 0.31, i + 0.31], color=INK_2, linewidth=1.2)
        # the label names the bar; it moves past the tick only when the tick would run through it
        at = lenient + 0.012 if 0 < lenient - f1 < 0.09 else f1 + 0.012
        ax.text(at, i, f"{f1:.2f}", va="center", ha="left", color=INK_2, fontsize=9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("F1 (bar: strict; tick: lenient, which forgives an OCR slip in the sample name)")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    _style(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ROUTE_COLOR[g]) for g in groups]
    ax.legend(handles, groups, loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=len(groups), fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def risk_coverage(curves: dict, alpha: float, threshold_cov: dict[str, float], out: FsPath, title: str) -> None:
    """Error rate among auto-accepted values as the acceptance threshold loosens."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for i, name in enumerate(("calibration", "test")):
        c = curves[name]
        ax.plot(c["coverage"], c["error"], color=SERIES[i], linewidth=2, solid_capstyle="round", label=f"{name.capitalize()} papers")
        cov = threshold_cov.get(name)
        if cov is not None and cov > 0:
            k = min(range(len(c["coverage"])), key=lambda j: abs(c["coverage"][j] - cov))
            ax.plot([c["coverage"][k]], [c["error"][k]], "o", color=SERIES[i], markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    ax.axhline(alpha, color=MUTED, linewidth=1)
    ax.text(0.005, alpha + 0.008, f"target error {alpha:.0%}", color=INK_2, fontsize=9, va="bottom")
    ax.set_xlabel("Share of values accepted without review")
    ax.set_ylabel("Error rate among accepted values")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.35, max(max(curves[n]["error"]) for n in curves) * 1.05))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    _style(ax, grid_axis="y")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def paired_bars(labels: list[str], a: list[float], b: list[float], names: tuple[str, str], out: FsPath, title: str,
                xlabel: str, fmt: str = "{:.2f}", xmax: float | None = None) -> None:
    """Two series per row (before and after), for the OCR fine-tuning comparison."""
    fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(labels) + 1.3))
    h = 0.36
    top = xmax or max(a + b) * 1.15
    ax.set_xlim(0, top)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    for i, (va, vb) in enumerate(zip(a, b)):
        for j, v in enumerate((va, vb)):
            y = i + (j - 0.5) * (h + 0.04)
            _hbar(ax, y, v, h, SERIES[j])
            ax.text(v + top * 0.01, y, fmt.format(v), va="center", ha="left", color=INK_2, fontsize=9)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    _style(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[i]) for i in range(2)]
    ax.legend(handles, names, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def failure_crop(pdf: FsPath, page: int, bbox, out: FsPath, caption: str, pad: float = 30.0, above: float = 75.0,
                 dpi: int = 250) -> None:
    """A crop of the page around a cell (with the headers above it), the cell outlined, for the README."""
    from PIL import Image, ImageDraw, ImageFont

    from .render import render_page

    img = render_page(pdf, page, dpi=dpi)
    s = dpi / 72.0
    x0, top, x1, bottom = bbox
    box = (max(0, int((x0 - pad * 3) * s)), max(0, int((top - above) * s)),
           min(img.width, int((x1 + pad * 3) * s)), min(img.height, int((bottom + pad) * s)))
    crop = img.crop(box).convert("RGB")
    d = ImageDraw.Draw(crop)
    d.rectangle([(x0 * s - box[0] - 3, top * s - box[1] - 3), (x1 * s - box[0] + 3, bottom * s - box[1] + 3)],
                outline=(235, 104, 52), width=4)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except OSError:
        font = ImageFont.load_default()
    canvas = Image.new("RGB", (crop.width, crop.height + 52), (252, 252, 251))
    canvas.paste(crop, (0, 0))
    ImageDraw.Draw(canvas).text((10, crop.height + 14), caption, fill=(82, 81, 78), font=font)
    canvas.save(out)
