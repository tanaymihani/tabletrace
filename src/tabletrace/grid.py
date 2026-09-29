"""Geometry shared by both routes: words, cells, table grids, word-to-cell assignment.

All coordinates are PDF points with the origin at the top-left of the page,
the convention pdfplumber uses. OCR words are converted into this space.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BBox = tuple[float, float, float, float]  # x0, top, x1, bottom


@dataclass
class Word:
    text: str
    x0: float
    top: float
    x1: float
    bottom: float
    conf: float = 1.0  # OCR confidence in [0, 1]; 1.0 for text-layer words
    size: float = 0.0  # font size (text layer) or line height (OCR)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def bbox(self) -> BBox:
        return (self.x0, self.top, self.x1, self.bottom)


@dataclass
class Cell:
    row: int
    col: int
    bbox: BBox
    words: list[Word] = field(default_factory=list)
    spanning: bool = False  # the cell text was propagated from a spanning header

    @property
    def text(self) -> str:
        return join_words(self.words)

    @property
    def conf(self) -> float:
        if not self.words:
            return 1.0
        return float(np.mean([w.conf for w in self.words]))


@dataclass
class TableGrid:
    paper_id: str
    page: int
    bbox: BBox
    cells: list[list[Cell]]
    source: str  # text_layer | ocr
    method: str  # how rows and columns were found
    caption: str = ""
    table_index: int = -1  # printed table number, -1 if no caption was linked
    detection_score: float = 1.0
    header_rows: int | None = None  # set by the structure model when it predicts column headers

    @property
    def n_rows(self) -> int:
        return len(self.cells)

    @property
    def n_cols(self) -> int:
        return len(self.cells[0]) if self.cells else 0

    def texts(self) -> list[list[str]]:
        return [[c.text for c in row] for row in self.cells]

    def to_markdown(self) -> str:
        rows = self.texts()
        if not rows:
            return ""
        lines = ["| " + " | ".join(r) + " |" for r in rows]
        lines.insert(1, "|" + "---|" * len(rows[0]))
        return "\n".join(lines)


def join_words(words: list[Word]) -> str:
    """Join words in reading order: lines top to bottom, words left to right."""
    if not words:
        return ""
    lines: list[list[Word]] = []
    for w in sorted(words, key=lambda w: (w.cy, w.x0)):
        for line in lines:
            ref = line[0]
            overlap = min(w.bottom, ref.bottom) - max(w.top, ref.top)
            if overlap > 0.4 * min(w.bottom - w.top, ref.bottom - ref.top):
                line.append(w)
                break
        else:
            lines.append([w])
    out = []
    for line in lines:
        line.sort(key=lambda w: w.x0)
        out.append(" ".join(w.text for w in line))
    return " ".join(out)


def assign_words(words: list[Word], rows: list[tuple[float, float]], cols: list[tuple[float, float]], bbox: BBox) -> list[list[Cell]]:
    """Build cells as row x column intersections and drop each word into the cell holding its center.

    rows are (top, bottom) and cols are (x0, x1) bands inside the table box.
    A word whose center falls between bands goes to the nearest band.
    """
    cells = [[Cell(r, c, (cx0, rt, cx1, rb)) for c, (cx0, cx1) in enumerate(cols)] for r, (rt, rb) in enumerate(rows)]
    if not rows or not cols:
        return cells
    x0, top, x1, bottom = bbox
    for w in words:
        if not (x0 - 2 <= w.cx <= x1 + 2 and top - 2 <= w.cy <= bottom + 2):
            continue
        r = _nearest_band(w.cy, rows)
        c = _nearest_band(w.cx, cols)
        cells[r][c].words.append(w)
    return cells


def _nearest_band(v: float, bands: list[tuple[float, float]]) -> int:
    best, best_d = 0, float("inf")
    for i, (a, b) in enumerate(bands):
        if a <= v <= b:
            return i
        d = min(abs(v - a), abs(v - b))
        if d < best_d:
            best, best_d = i, d
    return best


def words_in(words: list[Word], bbox: BBox, pad: float = 0.0) -> list[Word]:
    x0, top, x1, bottom = bbox
    return [w for w in words if x0 - pad <= w.cx <= x1 + pad and top - pad <= w.cy <= bottom + pad]


def cluster_rows(words: list[Word], min_overlap: float = 0.35) -> list[tuple[float, float]]:
    """Group words into text lines by vertical overlap; returns (top, bottom) bands.

    Subscripts and superscripts overlap the line they belong to, so they join it
    instead of forming a line of their own.
    """
    bands: list[list[float]] = []
    for w in sorted(words, key=lambda w: w.cy):
        h = max(w.bottom - w.top, 1e-3)
        for band in bands:
            overlap = min(w.bottom, band[1]) - max(w.top, band[0])
            if overlap > min_overlap * min(h, band[1] - band[0]):
                band[0], band[1] = min(band[0], w.top), max(band[1], w.bottom)
                break
        else:
            bands.append([w.top, w.bottom])
    bands.sort()
    # merge bands that ended up overlapping after growth
    merged: list[list[float]] = []
    for b in bands:
        if merged and b[0] < merged[-1][1] - 0.3 * min(b[1] - b[0], merged[-1][1] - merged[-1][0]):
            merged[-1][1] = max(merged[-1][1], b[1])
        else:
            merged.append(list(b))
    return [(a, b) for a, b in merged]


def cluster_columns(words: list[Word], x0: float, x1: float, min_gap: float = 4.0) -> list[tuple[float, float]]:
    """Find column bands from gaps in the horizontal projection of word boxes."""
    if not words:
        return []
    res = 0.5
    n = int((x1 - x0) / res) + 2
    occupied = np.zeros(n, dtype=bool)
    for w in words:
        a = max(0, int((w.x0 - x0) / res))
        b = min(n - 1, int((w.x1 - x0) / res) + 1)
        occupied[a:b] = True
    cols = []
    i = 0
    while i < n:
        if occupied[i]:
            j = i
            while j < n and occupied[j]:
                j += 1
            cols.append([x0 + i * res, x0 + j * res])
            i = j
        else:
            i += 1
    # close small gaps (spaces inside a cell)
    merged: list[list[float]] = []
    for c in cols:
        if merged and c[0] - merged[-1][1] < min_gap:
            merged[-1][1] = c[1]
        else:
            merged.append(c)
    return [(a, b) for a, b in merged]


def _text_lines(words: list[Word]) -> list[list[Word]]:
    lines: list[list[Word]] = []
    for w in sorted(words, key=lambda w: w.cy):
        if lines:
            ref = lines[-1][-1]
            h = min(ref.bottom - ref.top, w.bottom - w.top) or 1.0
            if min(w.bottom, ref.bottom) - max(w.top, ref.top) > 0.3 * h:
                lines[-1].append(w)
                continue
        lines.append([w])
    return lines


def split_merged_rows(cells: list[list[Cell]], min_cells: int = 2) -> list[list[Cell]]:
    """Split grid rows that hold two or more lines of numbers.

    Table Transformer sometimes gives two data rows one band, so cells read
    '590 800' and the label cell '1 2'. A row is split along its text lines when
    at least `min_cells` cells hold two or more lines that each read as a number;
    wrapped label lines left without values are folded back later by the
    interpreter, so multi-line alloy descriptions survive.
    """
    from .normalize import is_missing, parse_number

    def numeric(ws: list[Word]) -> bool:
        t = join_words(ws)
        return is_missing(t) or parse_number(t) is not None

    out: list[list[Cell]] = []
    for row in cells:
        multi = sum(1 for c in row if len(ls := _text_lines(c.words)) >= 2 and all(numeric(x) for x in ls))
        bands = cluster_rows([w for c in row for w in c.words]) if multi >= min_cells else []
        if len(bands) < 2:
            out.append(row)
            continue
        for top, bottom in bands:
            out.append([Cell(c.row, c.col, (c.bbox[0], top, c.bbox[2], bottom),
                             [w for w in c.words if top - 0.5 <= w.cy <= bottom + 0.5], c.spanning) for c in row])
    for r, row in enumerate(out):
        for c in row:
            c.row = r
    return out
