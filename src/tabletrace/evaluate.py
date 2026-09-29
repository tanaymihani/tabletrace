"""Scoring: Hungarian matching against the labels, P/R/F1, error taxonomy, CER/WER."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import jiwer
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment

from .grid import Word, join_words
from .normalize import clean_text
from .paths import GROUND_TRUTH

STRICT, PARTIAL, IMPOSSIBLE = 0.0, 0.5, 1e6
VALUE_TOL, VALUE_TOL_LOOSE = 0.01, 0.05
MATERIAL_FUZZY = 0.85  # "Ti-25Nb" vs "Ti-25Nb-25V" scores 0.80, and those are different alloys
TEMP_TOL_C = 3.0


def load_ground_truth(path=GROUND_TRUTH) -> pd.DataFrame:
    gt = pd.read_csv(path, keep_default_na=False, na_values=[""])
    gt["value"] = gt["value"].astype(float)
    gt["test_temperature_c"] = pd.to_numeric(gt["test_temperature_c"], errors="coerce")
    return gt


def material_key(s: str) -> str:
    """Normalize a sample label for comparison: case, spacing, punctuation, citations."""
    s = clean_text(str(s)).lower()
    s = re.sub(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]", "", s)
    s = re.sub(r"[\s_,;:()\[\]*†‡'\"-]", "", s)
    return s


def material_similarity(a: str, b: str) -> float:
    ka, kb = material_key(a), material_key(b)
    if not ka or not kb:
        return 0.0
    if ka == kb:
        return 1.0
    return fuzz.ratio(ka, kb) / 100.0


def value_error(pred: float, gold: float) -> float:
    if gold == 0:
        return abs(pred - gold)
    return abs(pred - gold) / abs(gold)


def pair_cost(p: dict, g: dict) -> float:
    """Cost tiers from the spec: 0 exact, 0.5 partial, impossible otherwise (plus a tiny tie-breaker)."""
    if p["property"] != g["property"]:
        return IMPOSSIBLE
    tp, tg = p.get("test_temperature_c"), g.get("test_temperature_c")
    if tp is not None and tg is not None and not (isinstance(tp, float) and math.isnan(tp)) \
            and not (isinstance(tg, float) and math.isnan(tg)) and abs(tp - tg) > TEMP_TOL_C:
        return IMPOSSIBLE
    e = value_error(float(p["value"]), float(g["value"]))
    sim = material_similarity(p["material"], g["material"])
    tie = 0.01 * min(e / VALUE_TOL_LOOSE, 1.0) + 0.01 * (1.0 - sim)
    if sim == 1.0 and e <= VALUE_TOL:
        return STRICT + tie
    if (sim >= MATERIAL_FUZZY and e <= VALUE_TOL) or (sim == 1.0 and e <= VALUE_TOL_LOOSE):
        return PARTIAL + tie
    return IMPOSSIBLE


@dataclass
class MatchResult:
    pairs: list[tuple[int, int, float]]  # (pred index, gt index, cost)
    pred_status: list[str]  # correct | partial | false_positive
    gt_status: list[str]  # found | partial | missed


def match(preds: list[dict], gts: list[dict]) -> MatchResult:
    """One-to-one assignment per paper with the Hungarian algorithm."""
    pred_status = ["false_positive"] * len(preds)
    gt_status = ["missed"] * len(gts)
    pairs: list[tuple[int, int, float]] = []
    by_paper_p: dict[str, list[int]] = {}
    by_paper_g: dict[str, list[int]] = {}
    for i, p in enumerate(preds):
        by_paper_p.setdefault(p["paper_id"], []).append(i)
    for j, g in enumerate(gts):
        by_paper_g.setdefault(g["paper_id"], []).append(j)
    for paper, pi in by_paper_p.items():
        gj = by_paper_g.get(paper, [])
        if not gj:
            continue
        cost = np.full((len(pi), len(gj)), IMPOSSIBLE)
        for a, i in enumerate(pi):
            for b, j in enumerate(gj):
                cost[a, b] = pair_cost(preds[i], gts[j])
        rows, cols = linear_sum_assignment(cost)
        for a, b in zip(rows, cols):
            c = cost[a, b]
            if c >= IMPOSSIBLE:
                continue
            i, j = pi[a], gj[b]
            pairs.append((i, j, float(c)))
            if c < PARTIAL:
                pred_status[i], gt_status[j] = "correct", "found"
            else:
                pred_status[i], gt_status[j] = "partial", "partial"
    return MatchResult(pairs, pred_status, gt_status)


def prf(n_correct: int, n_pred: int, n_gold: int) -> dict:
    p = n_correct / n_pred if n_pred else 0.0
    r = n_correct / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "n_pred": n_pred, "n_gold": n_gold, "n_correct": n_correct}


def score(preds: list[dict], gts: list[dict], group_of=None) -> dict:
    """Strict and lenient P/R/F1, overall and per group (group_of maps a record to its group)."""
    m = match(preds, gts)
    out = {"strict": prf(sum(s == "correct" for s in m.pred_status), len(preds), len(gts)),
           "lenient": prf(sum(s in ("correct", "partial") for s in m.pred_status), len(preds), len(gts))}
    if group_of is not None:
        groups = sorted({group_of(x) for x in preds} | {group_of(x) for x in gts})
        out["groups"] = {}
        for grp in groups:
            pi = [i for i, p in enumerate(preds) if group_of(p) == grp]
            gj = [j for j, g in enumerate(gts) if group_of(g) == grp]
            out["groups"][grp] = {
                "strict": prf(sum(m.pred_status[i] == "correct" for i in pi), len(pi), len(gj)),
                "lenient": prf(sum(m.pred_status[i] in ("correct", "partial") for i in pi), len(pi), len(gj)),
            }
    out["match"] = m
    return out


# --------------------------------------------------------------------------- error taxonomy

STAGES = ["table detection", "OCR character error", "table structure error", "header mapping error",
          "unit or number parsing error"]
_UNIT_FACTORS = [6.894757, 1 / 6.894757, 1000.0, 0.001, 9.80665, 1 / 9.80665, 100.0, 0.01, 10.0, 0.1]


def _temps_compatible(p: dict, g: dict) -> bool:
    tp, tg = p.get("test_temperature_c"), g.get("test_temperature_c")
    if tp is None or tg is None or (isinstance(tp, float) and math.isnan(tp)) or (isinstance(tg, float) and math.isnan(tg)):
        return True
    return abs(tp - tg) <= TEMP_TOL_C


def _digits(s: str) -> str:
    return re.sub(r"[^0-9]", "", clean_text(str(s)))


@dataclass
class MissAnalysis:
    stage: str
    detail: str
    evidence: dict = field(default_factory=dict)


def classify_miss(g: dict, preds: list[dict], quarantine: list[dict], tables: list[dict]) -> MissAnalysis:
    """Assign a missed label to the pipeline stage that most plausibly lost it.

    Evidence is checked from the most to the least specific, and predictions from
    the label's own row (same property, similar sample name) are consulted before
    predictions from other rows, so a misread 820 is not blamed on the 815 below it:
      1. a quarantined cell with the label's digits      -> unit or number parsing
      2. same row, value one or two digit edits away      -> OCR character error
         same row, value off by a unit factor             -> unit or number parsing
      3. same value and row, different property           -> header mapping
      4. same value, sample name misread                  -> OCR (OCR pages) or structure
      0. the label's table was never detected             -> table detection
      5. table found but no property header recognized    -> header mapping
    Everything else is a structure error: the cell never reached the reader.
    """
    page_tables = [t for t in tables if t["paper_id"] == g["paper_id"] and t["page"] == g["page"]]
    linked = {t["table_index"] for t in page_tables}
    if not page_tables or (int(g["table_index"]) not in linked and -1 not in linked):
        # checked first: a value that happens to reappear in another table on the page says nothing
        return MissAnalysis("table detection", f"Table {g['table_index']} was not detected on page {g['page']}")
    # candidate readings come from the label's own table when captions identify it
    own = int(g["table_index"]) if int(g["table_index"]) in linked else None
    same_page = [p for p in preds if p["paper_id"] == g["paper_id"] and p["page"] == g["page"]
                 and (own is None or p.get("table_index") in (own, -1))]
    gv = float(g["value"])
    graw = _digits(str(g["raw_text"]).split("/")[-1])
    for q in quarantine:
        if q["paper_id"] == g["paper_id"] and q["page"] == g["page"] and graw and _digits(q["raw_text"]) == graw:
            return MissAnalysis("unit or number parsing error", f"quarantined: {q['reason']}", q)
    same_row = sorted((p for p in same_page if p["property"] == g["property"] and _temps_compatible(p, g)
                       and material_similarity(p["material"], g["material"]) >= MATERIAL_FUZZY),
                      key=lambda p: -material_similarity(p["material"], g["material"]))
    right_value = [p for p in same_row if value_error(p["value"], gv) <= VALUE_TOL]
    if right_value:
        p = right_value[0]
        if material_key(p["material"]) != material_key(g["material"]):
            stage = "OCR character error" if p.get("source") == "ocr" else "table structure error"
            return MissAnalysis(stage, f"sample name read as {p['material']!r}", p)
        return MissAnalysis("table structure error", "duplicate reading claimed by another label", p)
    for p in same_row:
        pv = float(p["value"])
        if gv and any(abs(pv * f - gv) / abs(gv) <= VALUE_TOL for f in _UNIT_FACTORS):
            return MissAnalysis("unit or number parsing error", f"off by a unit factor ({pv:g} vs {gv:g})", p)
        if Levenshtein.distance(_digits(p["raw_text"]), graw) <= 2 and p.get("source") == "ocr":
            return MissAnalysis("OCR character error", f"value read as {p['raw_text']!r}", p)
    for p in same_page:
        if p["property"] != g["property"] and value_error(p["value"], gv) <= VALUE_TOL \
                and material_similarity(p["material"], g["material"]) >= 0.8:
            return MissAnalysis("header mapping error", f"read as {p['property']}", p)
    same_value = sorted((p for p in same_page if p["property"] == g["property"] and value_error(p["value"], gv) <= VALUE_TOL),
                        key=lambda p: -material_similarity(p["material"], g["material"]))
    for p in same_value:
        sim = material_similarity(p["material"], g["material"])
        if sim >= 0.5:
            stage = "OCR character error" if p.get("source") == "ocr" else "table structure error"
            return MissAnalysis(stage, f"sample name read as {p['material']!r}", p)
    own_tables = [t for t in page_tables if own is None or t["table_index"] == own]
    table_box = {"bbox": own_tables[0]["bbox"], "raw_text": "", "material": ""} if own_tables else {}
    if all(t["orientation"] in ("none", "") for t in own_tables):
        return MissAnalysis("header mapping error", "table found but no property header recognized", table_box)
    if same_value:
        return MissAnalysis("table structure error", f"value attached to {same_value[0]['material']!r}", same_value[0])
    if not any(p["property"] == g["property"] for p in same_page):
        return MissAnalysis("header mapping error", f"no {g['property'].replace('_', ' ')} values read from the table",
                            table_box)
    close = [p for p in same_page if p["property"] == g["property"] and p.get("source") == "ocr"
             and Levenshtein.distance(_digits(p["raw_text"]), graw) <= 1]
    if close and same_row:
        return MissAnalysis("OCR character error", f"nearest cell read as {close[0]['raw_text']!r}", close[0])
    return MissAnalysis("table structure error", "cell not read (merged, split, or outside the grid)")


# --------------------------------------------------------------------------- OCR quality

def linearize(words: list[Word]) -> str:
    """Words in geometric reading order: lines top to bottom, left to right within a line."""
    return clean_text(join_words(words))


def cer_wer(reference: str, hypothesis: str) -> tuple[float, float]:
    ref = re.sub(r"\s+", " ", clean_text(reference)).strip()
    hyp = re.sub(r"\s+", " ", clean_text(hypothesis)).strip()
    if not ref:
        return float("nan"), float("nan")
    return jiwer.cer(ref, hyp or "#"), jiwer.wer(ref, hyp or "#")
