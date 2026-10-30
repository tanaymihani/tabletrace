"""Per-value confidence from the evidence the pipeline already has.

The score is a product of factors in [0, 1], so any single strong warning
(an OCR word Tesseract was unsure about, a value the other route read
differently, a yield strength above its own UTS) pulls the value down on its
own. It is not a probability; calibrate.py turns it into a review threshold
with a stated error rate.
"""

from __future__ import annotations

from collections import defaultdict

from .evaluate import material_similarity, value_error
from .schema import ExtractedValue, PropertyKind

# Typical ranges for the alloys in scope. Tighter than the hard physical limits
# in schema.py: a value outside them is legal but unusual enough to double-check.
TYPICAL = {
    PropertyKind.vickers_hardness: (30.0, 1300.0),
    PropertyKind.yield_strength: (20.0, 3500.0),
    PropertyKind.ultimate_strength: (20.0, 4000.0),
    PropertyKind.elongation: (0.0, 80.0),
}

WEIGHTS = {
    "range_fail": 0.6,
    "yield_above_uts": 0.5,
    "agree": 1.0,
    "disagree_text": 0.85,  # OCR contradicting a clean text layer is weak evidence
    "disagree_ocr": 0.55,  # a second OCR-quality reading contradicting ours is strong evidence
    "no_counterpart_text": 0.97,
    "no_counterpart_ocr": 0.85,
    "no_second_route": 0.9,
    "unit_disagree": 0.5,
    "repaired": 0.85,
    "caption_property": 0.95,
}


def _key(v: ExtractedValue) -> tuple:
    return (v.paper_id, v.page, v.property.value)


def route_agreement(values: list[ExtractedValue], secondary: list[ExtractedValue]) -> dict[int, float | None]:
    """For each primary value: 1.0 if the second route read the same cell the same way,
    0.0 if it read a different number, None if it produced no counterpart."""
    index = defaultdict(list)
    for s in secondary:
        index[_key(s)].append(s)
    out: dict[int, float | None] = {}
    for i, v in enumerate(values):
        best, best_sim = None, 0.0
        for s in index.get(_key(v), []):
            if v.test_temperature_c is not None and s.test_temperature_c is not None \
                    and abs(v.test_temperature_c - s.test_temperature_c) > 3:
                continue
            same_cell = v.bbox is not None and s.bbox is not None and _iou(v.bbox, s.bbox) > 0.3
            sim = 1.0 if same_cell else material_similarity(v.material, s.material)
            if sim > best_sim:
                best, best_sim = s, sim
        if best is None or best_sim < 0.8:
            out[i] = None
        else:
            out[i] = 1.0 if value_error(best.value, v.value) <= 0.01 else 0.0
    return out


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def cross_field_flags(values: list[ExtractedValue]) -> set[int]:
    """Indices of yield strengths above the UTS of the same sample and test (a column mix-up)."""
    groups = defaultdict(dict)
    for i, v in enumerate(values):
        k = (v.paper_id, v.page, v.table_index, v.material, v.test_temperature_c, v.condition)
        groups[k].setdefault(v.property, []).append(i)
    bad = set()
    for props in groups.values():
        ys, uts = props.get(PropertyKind.yield_strength, []), props.get(PropertyKind.ultimate_strength, [])
        for i in ys:
            for j in uts:
                if values[i].value > values[j].value * 1.02:
                    bad.update((i, j))
    return bad


def score_values(values: list[ExtractedValue], secondary: list[ExtractedValue], has_second_route: set[str],
                 third: list[ExtractedValue] | None = None, third_mode: str = "penalize") -> None:
    """Fill value.confidence and value.features in place.

    has_second_route: paper ids for which a second route ran on the same pages.
    third: values read independently by a vision-language model from the page image (off by default).
    third_mode "penalize": a disagreement or a missing counterpart costs what it costs for a second
    OCR reading. "boost": agreement closes half the gap to 1, and nothing else changes. Neither
    beat the score without the VLM on the calibration papers (reports/baselines.md).
    """
    agreement = route_agreement(values, secondary)
    vlm = route_agreement(values, third) if third is not None else {}
    yield_flags = cross_field_flags(values)
    for i, v in enumerate(values):
        f = v.features
        c = float(f.get("ocr_conf", 1.0))
        c *= float(f.get("header_score", 1.0))
        lo, hi = TYPICAL[v.property]
        if v.property is PropertyKind.elongation and v.test_temperature_c is not None and v.test_temperature_c >= 200:
            hi = 250.0
        in_range = lo <= v.value <= hi
        f["range_ok"] = float(in_range and i not in yield_flags)
        if not in_range:
            c *= WEIGHTS["range_fail"]
        if i in yield_flags:
            c *= WEIGHTS["yield_above_uts"]
        text_primary = v.source == "text_layer"
        a = agreement[i]
        if v.paper_id not in has_second_route:
            c *= WEIGHTS["no_second_route"]
            f["agreement"] = -1.0
        elif a is None:
            c *= WEIGHTS["no_counterpart_text" if text_primary else "no_counterpart_ocr"]
            f["agreement"] = 0.5
        elif a == 0.0:
            c *= WEIGHTS["disagree_text" if text_primary else "disagree_ocr"]
            f["agreement"] = 0.0
        else:
            f["agreement"] = 1.0
        if third is not None:
            b = vlm[i]
            f["vlm_agreement"] = 0.5 if b is None else b
            if third_mode == "penalize" and b is None:
                c *= WEIGHTS["no_counterpart_ocr"]
            elif third_mode == "penalize" and b == 0.0:
                c *= WEIGHTS["disagree_ocr"]
        if f.get("unit_agreement") == 0.0:
            c *= WEIGHTS["unit_disagree"]
        if f.get("repaired"):
            c *= WEIGHTS["repaired"]
        if f.get("caption_property"):
            c *= WEIGHTS["caption_property"]
        if third is not None and third_mode == "boost" and vlm[i] == 1.0:
            c += (1.0 - c) / 2
        v.confidence = max(0.0, min(1.0, c))
