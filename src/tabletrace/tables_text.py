"""Text-layer words and pdfplumber tables.

Two problems with real text layers are handled here:

* Ghost text. Several MDPI PDFs embed a whole page of the author's manuscript
  as a Form XObject so a vector figure can be reused; the renderer clips it to
  the figure, but pdfminer ignores the clip and returns every character,
  including a second copy of the table. Characters inside embedded figures are
  dropped (tables are never drawn inside those figures in this corpus).
* Subscripts. "Nb1.5TaTiZr0.5" is typeset as five runs on two baselines. Words
  are rebuilt from characters so that subscripts join the line they belong to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pdfplumber
from pdfminer.layout import LTChar, LTFigure

from .grid import BBox, Word, assign_words, cluster_rows

# Measured on this corpus: letter gaps sit within +-0.8 pt (kerning), word gaps are
# 0.25-0.3 x font size, and subscripts touch their base glyph. 0.15 x size splits them cleanly.
SPACE_GAP = 0.15  # gap above this fraction of the font size starts a new word


@dataclass
class TextLayerInfo:
    n_chars: int  # visible characters
    n_ghost: int  # characters dropped because they were clipped inside embedded figures
    full_page_image: bool  # one raster image covers most of the page (a scan)
    text_coverage: float  # fraction of the page area covered by character boxes


def _figure_char_keys(page: pdfplumber.page.Page) -> set[tuple[float, float, str]]:
    keys: set[tuple[float, float, str]] = set()

    def walk(obj, inside: bool) -> None:
        if isinstance(obj, LTChar):
            if inside:
                keys.add((round(obj.x0, 2), round(obj.y0, 2), obj.get_text()))
            return
        if hasattr(obj, "__iter__"):
            for child in obj:
                walk(child, inside or isinstance(obj, LTFigure))

    walk(page.layout, False)
    return keys


def visible_page(page: pdfplumber.page.Page) -> tuple[pdfplumber.page.Page, int]:
    """Return the page without characters that sit inside embedded figures."""
    keys = _figure_char_keys(page)
    if not keys:
        return page, 0
    height = page.height

    def keep(obj: dict) -> bool:
        if obj.get("object_type") != "char":
            return True
        return (round(obj["x0"], 2), round(height - obj["bottom"], 2), obj["text"]) not in keys

    filtered = page.filter(keep)
    return filtered, len(page.chars) - len(filtered.chars)


# Switch for the ablation in the write-up: with the filter off, the text layer is read the way a
# plain pdfplumber pipeline reads it, hidden duplicate text included.
GHOST_FILTER = True


def _load(pdf_path: str, page_no: int):
    return _load_cached(pdf_path, page_no, GHOST_FILTER)


@lru_cache(maxsize=64)
def _load_cached(pdf_path: str, page_no: int, ghost_filter: bool):
    pdf = pdfplumber.open(pdf_path)
    page = pdf.pages[page_no - 1]
    if not ghost_filter:
        return pdf, page, page, 0
    vis, n_ghost = visible_page(page)
    return pdf, page, vis, n_ghost


def text_layer_info(pdf_path: Path, page_no: int) -> TextLayerInfo:
    _, page, vis, n_ghost = _load(str(pdf_path), page_no)
    area = float(page.width * page.height)
    img_area = max((float((i["x1"] - i["x0"]) * (i["bottom"] - i["top"])) for i in page.images), default=0.0)
    chars = [c for c in vis.chars if c["text"].strip()]
    char_area = sum(max(c["x1"] - c["x0"], 0) * max(c["bottom"] - c["top"], 0) for c in chars)
    return TextLayerInfo(
        n_chars=len(chars),
        n_ghost=n_ghost,
        full_page_image=img_area > 0.8 * area,
        text_coverage=char_area / area if area else 0.0,
    )


def text_words(pdf_path: Path, page_no: int, bbox: BBox | None = None, pad: float = 3.0) -> list[Word]:
    """Words from the visible text layer, rebuilt from characters.

    Pass the table box: lines are built from the characters inside it only, so
    a paragraph in the neighbouring column can never capture a table row that
    happens to sit at the same height.
    """
    _, _, vis, _ = _load(str(pdf_path), page_no)
    chars = [c for c in vis.chars if c["text"] and c.get("upright", True)]
    if bbox is not None:
        x0, top, x1, bottom = bbox
        chars = [c for c in chars if x0 - pad <= (c["x0"] + c["x1"]) / 2 <= x1 + pad
                 and top - pad <= (c["top"] + c["bottom"]) / 2 <= bottom + pad]
    return words_from_chars(chars)


def words_from_chars(chars: list[dict]) -> list[Word]:
    """Group characters into words, keeping sub- and superscripts attached to their line."""
    items = [
        Word(c["text"], float(c["x0"]), float(c["top"]), float(c["x1"]), float(c["bottom"]), 1.0, float(c.get("size", 0)))
        for c in chars
    ]
    # lines: characters that overlap vertically; small glyphs join the line they overlap most.
    # Each line tracks the vertical extent of its full-size ("base") glyphs only, so a
    # subscript can attach to a line without stretching it into the next one.
    items.sort(key=lambda w: (w.cy, w.x0))
    lines: list[dict] = []
    for ch in items:
        if ch.text.isspace():
            continue
        best, best_ov = None, 0.0
        for line in lines[-8:]:
            # compare against the line's reference band (its first full-size glyph), which never
            # grows; a growing band lets a label centred between two header lines chain them together
            ov = min(ch.bottom, line["bottom"]) - max(ch.top, line["top"])
            frac = ov / max(min(ch.bottom - ch.top, line["bottom"] - line["top"]), 1e-3)
            if frac > best_ov:
                best, best_ov = line, frac
        small = best is not None and ch.size < 0.8 * best["size"]
        if best is not None and best_ov > (0.3 if small else 0.5):
            best["chars"].append(ch)
            if ch.size > best["size"] * 1.25:  # a larger glyph defines the band better
                best["top"], best["bottom"], best["size"] = ch.top, ch.bottom, ch.size
        else:
            lines.append({"chars": [ch], "top": ch.top, "bottom": ch.bottom, "size": ch.size})
    # keep explicit spaces: they separate words even when glyphs are tightly set
    spaces = [c for c in items if c.text.isspace()]
    words: list[Word] = []
    for info in lines:
        line = info["chars"]
        for sp in spaces:
            if sp.cy >= info["top"] and sp.cy <= info["bottom"]:
                line.append(sp)
        line.sort(key=lambda w: w.x0)
        base_size = info["size"] or 1.0
        cur: list[Word] = []
        for ch in line:
            if ch.text.isspace():
                if cur:
                    words.append(_merge(cur))
                    cur = []
                continue
            if cur:
                gap = ch.x0 - cur[-1].x1
                if gap > SPACE_GAP * base_size:
                    words.append(_merge(cur))
                    cur = []
            cur.append(ch)
        if cur:
            words.append(_merge(cur))
    return words


def _merge(chars: list[Word]) -> Word:
    return Word(
        "".join(c.text for c in chars),
        min(c.x0 for c in chars),
        min(c.top for c in chars),
        max(c.x1 for c in chars),
        max(c.bottom for c in chars),
        1.0,
        max(c.size for c in chars),
    )


def plumber_structure(pdf_path: Path, page_no: int, bbox: BBox) -> tuple[list, list] | None:
    """Rows and columns that pdfplumber's text strategy finds inside a table region.

    Returns (row bands, column bands) or None if pdfplumber finds no table. The
    cells are refilled with our own words so subscripts and ghost filtering carry over.
    """
    _, _, vis, _ = _load(str(pdf_path), page_no)
    x0, top, x1, bottom = bbox
    pad = 2.0
    crop_box = (max(0, x0 - pad), max(0, top - pad), min(vis.width, x1 + pad), min(vis.height, bottom + pad))
    region = vis.crop(crop_box, strict=False)
    settings = {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
        "text_x_tolerance": 1.5,
        "text_y_tolerance": 2.5,
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "intersection_tolerance": 4,
        "min_words_vertical": 2,
        "min_words_horizontal": 1,
    }
    tables = region.find_tables(table_settings=settings)
    if not tables:
        return None
    table = max(tables, key=lambda t: (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1]))
    col_edges: set[float] = set()
    row_edges: set[float] = set()
    for c in table.cells:
        col_edges.update((round(c[0], 1), round(c[2], 1)))
        row_edges.update((round(c[1], 1), round(c[3], 1)))
    xs, ys = sorted(col_edges), sorted(row_edges)
    cols = [(a, b) for a, b in zip(xs, xs[1:]) if b - a > 1.0]
    rows = [(a, b) for a, b in zip(ys, ys[1:]) if b - a > 1.0]
    return rows, cols


def geometric_structure(words: list[Word], bbox: BBox) -> tuple[list, list]:
    """Rows from line clustering and columns from whitespace gaps (a model-free baseline)."""
    from .grid import cluster_columns, words_in

    inside = words_in(words, bbox, pad=1.0)
    rows = cluster_rows(inside)
    # columns: project only body lines (lines holding a number) so wide spanning headers
    # do not bridge the gaps between columns; keep every word of those lines, stub included
    body = []
    for top, bottom in rows:
        line = [w for w in inside if top <= w.cy <= bottom]
        if any(re.search(r"\d", w.text) for w in line):
            body.extend(line)
    body = body or inside
    cols = cluster_columns(body, bbox[0], bbox[2], min_gap=6.0)
    # widen columns to tile the table so header words between bands still land somewhere
    tiled = []
    for i, (a, b) in enumerate(cols):
        left = bbox[0] if i == 0 else (cols[i - 1][1] + a) / 2
        right = bbox[2] if i == len(cols) - 1 else (b + cols[i + 1][0]) / 2
        tiled.append((left, right))
    return rows, tiled


def build_cells(words: list[Word], rows: list, cols: list, bbox: BBox):
    return assign_words(words, rows, cols, bbox)
