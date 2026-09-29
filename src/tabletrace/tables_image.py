"""Table Transformer (TATR) detection and structure recognition, plus word-to-cell assignment.

Detection finds table boxes on a rendered page. Structure recognition predicts
rows, columns, the column-header block and spanning cells inside a table crop.
Cells are row x column intersections; each word goes to the cell containing
its center, and spanning-cell text is copied to every cell it covers.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache

import torch
from PIL import Image

from .grid import BBox, Cell, Word, assign_words

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.getLogger("transformers").setLevel(logging.ERROR)

DETECTION_MODEL = "microsoft/table-transformer-detection"
STRUCTURE_MODEL = "microsoft/table-transformer-structure-recognition-v1.1-all"
STRUCTURE_MODEL_V1 = "microsoft/table-transformer-structure-recognition"


@lru_cache(maxsize=4)
def _load(name: str):
    from transformers import AutoImageProcessor, TableTransformerConfig, TableTransformerForObjectDetection
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    # Match the reference TATR inference: longest side 800 px for detection, 1000 px for structure.
    edge = 800 if name == DETECTION_MODEL else 1000
    processor = AutoImageProcessor.from_pretrained(name, size={"shortest_edge": edge, "longest_edge": edge})
    # The v1.1 checkpoints ship "dilation": null, which newer transformers releases reject.
    config_dict, _ = TableTransformerConfig.get_config_dict(name)
    if config_dict.get("dilation") is None:
        config_dict["dilation"] = False
    config = TableTransformerConfig.from_dict(config_dict)
    model = TableTransformerForObjectDetection.from_pretrained(name, config=config).eval()
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    return processor, model


def _predict(name: str, img: Image.Image, threshold: float):
    processor, model = _load(name)
    inputs = processor(images=img.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    res = processor.post_process_object_detection(out, threshold=threshold, target_sizes=[img.size[::-1]])[0]
    labels = model.config.id2label
    return [
        (labels[int(lbl)], float(score), tuple(float(v) for v in box))
        for score, lbl, box in zip(res["scores"], res["labels"], res["boxes"])
    ]


@dataclass
class DetectedTable:
    bbox: BBox  # PDF points
    score: float
    rotated: bool = False


def detect_tables(page_img: Image.Image, dpi: int, threshold: float = 0.8) -> list[DetectedTable]:
    s = 72.0 / dpi
    out = []
    for label, score, (x0, y0, x1, y1) in _predict(DETECTION_MODEL, page_img, threshold):
        out.append(DetectedTable((x0 * s, y0 * s, x1 * s, y1 * s), score, rotated=(label == "table rotated")))
    # drop boxes mostly contained in a higher-scoring box
    out.sort(key=lambda t: -t.score)
    kept: list[DetectedTable] = []
    for t in out:
        if all(_overlap_frac(t.bbox, k.bbox) < 0.6 for k in kept):
            kept.append(t)
    return sorted(kept, key=lambda t: (t.bbox[1], t.bbox[0]))


@dataclass
class Structure:
    rows: list[tuple[float, float]]
    cols: list[tuple[float, float]]
    header_bbox: BBox | None = None
    spanning: list[tuple[BBox, float]] = field(default_factory=list)  # (box, score)
    raw: list = field(default_factory=list)


def recognize_structure(page_img: Image.Image, dpi: int, bbox: BBox, model: str = STRUCTURE_MODEL,
                        pad_pts: float = 8.0, threshold: float = 0.5) -> Structure:
    s = dpi / 72.0
    x0, top, x1, bottom = bbox
    box = (max(0, int((x0 - pad_pts) * s)), max(0, int((top - pad_pts) * s)),
           min(page_img.width, int((x1 + pad_pts) * s)), min(page_img.height, int((bottom + pad_pts) * s)))
    crop = page_img.crop(box)
    preds = _predict(model, crop, threshold)
    ox, oy = box[0] / s, box[1] / s

    def to_pts(b):
        return (ox + b[0] / s, oy + b[1] / s, ox + b[2] / s, oy + b[3] / s)

    rows = [(to_pts(b), sc) for lbl, sc, b in preds if lbl == "table row"]
    cols = [(to_pts(b), sc) for lbl, sc, b in preds if lbl == "table column"]
    headers = [(to_pts(b), sc) for lbl, sc, b in preds if lbl == "table column header"]
    spans = [(to_pts(b), sc) for lbl, sc, b in preds if lbl == "table spanning cell"]
    row_bands = _tile([(b[1], b[3], sc) for b, sc in rows], top, bottom)
    col_bands = _tile([(b[0], b[2], sc) for b, sc in cols], x0, x1)
    header = max(headers, key=lambda h: h[1])[0] if headers else None
    return Structure(row_bands, col_bands, header, spans, preds)


def _tile(bands: list[tuple[float, float, float]], lo: float, hi: float) -> list[tuple[float, float]]:
    """Non-maximum suppression on 1-D bands, then boundaries at midpoints so bands tile [lo, hi]."""
    bands = sorted(bands, key=lambda b: -b[2])
    kept: list[tuple[float, float, float]] = []
    for a, b, sc in bands:
        ok = True
        for ka, kb, _ in kept:
            ov = min(b, kb) - max(a, ka)
            if ov > 0.5 * min(b - a, kb - ka):
                ok = False
                break
        if ok:
            kept.append((a, b, sc))
    kept.sort(key=lambda b: (b[0] + b[1]) / 2)
    if not kept:
        return []
    # the boundary between neighbours is the midpoint of the gap (or of the overlap)
    bounds = [(kept[i][1] + kept[i + 1][0]) / 2 for i in range(len(kept) - 1)]
    out = []
    for i, (a, b, _) in enumerate(kept):
        left = min(lo, a) if i == 0 else bounds[i - 1]
        right = max(hi, b) if i == len(kept) - 1 else bounds[i]
        out.append((left, right))
    return out


def cells_from_structure(words: list[Word], st: Structure, bbox: BBox) -> tuple[list[list[Cell]], int | None]:
    """Assign words to the predicted grid and give spanning header cells their full text."""
    cells = assign_words(words, st.rows, st.cols, bbox)
    apply_spanning(cells, st, lambda span: [w for w in words if span[0] <= w.cx <= span[2] and span[1] <= w.cy <= span[3]])
    return cells, header_row_count(st)


def header_row_count(st: Structure) -> int | None:
    if st.header_bbox is None or not st.rows:
        return None
    hb = st.header_bbox
    return sum(1 for (a, b) in st.rows if hb[1] - 2 <= (a + b) / 2 <= hb[3] + 2)


def apply_spanning(cells: list[list[Cell]], st: Structure, words_for) -> None:
    """Copy each header spanning cell's text to every grid cell it covers.

    Only spans inside the column-header block are used: in the body, TATR's
    spanning cells are mostly merged row labels, and copying values across
    them would invent data. Spans are applied best-first so overlapping
    predictions cannot stack their text, and a span's text comes from the words
    inside the span itself (`words_for`), never from concatenated sub-cells.
    """
    if st.header_bbox is not None:
        header_bottom = st.header_bbox[3]
    elif st.rows:
        header_bottom = st.rows[0][0] + 0.35 * (st.rows[-1][1] - st.rows[0][0])
    else:
        return
    taken: set[tuple[int, int]] = set()
    for span, _score in sorted(st.spanning, key=lambda s: -s[1]):
        if span[3] > header_bottom + 3:
            continue
        covered = [c for row in cells for c in row
                   if span[0] - 1 <= (c.bbox[0] + c.bbox[2]) / 2 <= span[2] + 1
                   and span[1] - 1 <= (c.bbox[1] + c.bbox[3]) / 2 <= span[3] + 1
                   and (c.row, c.col) not in taken]
        if len(covered) < 2:
            continue
        span_words = words_for(span)
        if not span_words:
            continue
        for c in covered:
            c.words = list(span_words)
            c.spanning = True
            taken.add((c.row, c.col))


def ruled_header_spans(page_img: Image.Image, dpi: int, st: Structure, bbox: BBox) -> list[tuple[BBox, float]]:
    """Header cells read off the ruling lines, for tables drawn with vertical rules.

    A vertical rule crosses a header row band from edge to edge and carries on into
    the rules above or below it. The stem of a capital letter can darken most of a
    tight band too, but its dark run ends inside the line of text, so a column only
    counts as a rule where its unbroken dark run is longer than the band itself.
    The intervals between consecutive rules are the header cells of that row,
    however many grid columns they cover. Returns [] for headers without internal
    vertical rules (booktabs-style or merely boxed tables).
    """
    import numpy as np

    if st.header_bbox is None or not st.rows or not st.cols:
        return []
    s = dpi / 72.0
    x0, x1 = bbox[0], bbox[2]
    hb = st.header_bbox
    bands = [(a, b) for a, b in st.rows if hb[1] - 2 <= (a + b) / 2 <= hb[3] + 2]
    if not bands:
        return []
    y_top = max(int((min(a for a, _ in bands) - 6) * s), 0)
    y_bot = min(int((max(b for _, b in bands) + 6) * s), page_img.height)
    c0, c1 = max(int((x0 - 3) * s), 0), min(int((x1 + 3) * s), page_img.width)
    dark = np.asarray(page_img.convert("L"), dtype=np.uint8)[y_top:y_bot, c0:c1] < 140
    # close one-pixel gaps so a slightly broken scanned rule still reads as one run
    closed = dark.copy()
    closed[1:-1] |= dark[:-2] & dark[2:]
    n = closed.shape[0]
    up = np.zeros(closed.shape, dtype=np.int32)
    down = np.zeros(closed.shape, dtype=np.int32)
    for y in range(n):
        up[y] = (up[y - 1] + 1) * closed[y] if y else closed[y]
    for y in range(n - 1, -1, -1):
        down[y] = (down[y + 1] + 1) * closed[y] if y < n - 1 else closed[y]
    through = np.where(closed, up + down - 1, 0)

    spans: list[tuple[BBox, float]] = []
    any_rules = False
    for top, bottom in bands:
        r0, r1 = int(top * s) - y_top, int(bottom * s) - y_top
        if r1 - r0 < 3:
            continue
        rule_px = np.where(through[(r0 + r1) // 2] >= (r1 - r0) + 3)[0]
        if rule_px.size == 0:
            continue
        rules, start = [], rule_px[0]
        for a, b in zip(rule_px, rule_px[1:]):
            if b - a > 2:
                rules.append((start + a) / 2)
                start = b
        rules.append((start + rule_px[-1]) / 2)
        # the table's own border is not a rule between header cells
        inner = sorted(x for x in (c0 / s + r / s for r in rules) if x0 + 4 < x < x1 - 4)
        if not inner:
            continue
        xs = [x0, *inner, x1]
        any_rules = True
        for a, b in zip(xs, xs[1:]):
            covered = [c for c in st.cols if a - 1 <= (c[0] + c[1]) / 2 <= b + 1]
            if len(covered) >= 2:
                spans.append(((a, top, b, bottom), 1.0))
    return spans if any_rules else []


def _overlap_frac(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
    return ix * iy / area
