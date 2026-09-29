"""Cross-check the labels against LitXAlloy (Radical AI, LitXBench, 2026).

LitXAlloy stores its ground truth as Python source. This script downloads the
files for the journal papers in papers.csv from a pinned LitXBench commit and
reads them with `ast`, so nothing from the download is executed. It writes
  labels/litxalloy_reference.csv   the LitXAlloy values for our four properties
  reports/label_qa.md              agreement between the two label sets

LitXBench code is MIT licensed and the benchmark data is CC BY 4.0.

Usage: python scripts/build_litxalloy_reference.py
"""

from __future__ import annotations

import ast
import csv
import sys
from pathlib import Path

import requests
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tabletrace.paths import GROUND_TRUTH, LABELS, REPORTS, load_papers  # noqa: E402

COMMIT = "ac93f650059563efa334318cf8eedcb41368db15"
BASE = f"https://raw.githubusercontent.com/Radical-AI/litxbench/{COMMIT}/src/litxbench/litxalloy/extractions"

KIND_MAP = {
    "vickers_hardness": ("vickers_hardness", ""),
    "yield_strength_tension": ("yield_strength", "tension"),
    "yield_strength_compression": ("yield_strength", "compression"),
    "ultimate_tensile_strength": ("ultimate_strength", "tension"),
    "ultimate_compressive_strength": ("ultimate_strength", "compression"),
    "fracture_strain_tension": ("elongation", "tension"),
    "fracture_strain_compression": ("elongation", "compression"),
    "ultimate_strain_tension": ("elongation", "tension"),
    "ultimate_strain_compression": ("elongation", "compression"),
}
# LitXBench unit names -> factor to our canonical unit
UNIT_FACTOR = {"MegaPascal": 1.0, "GigaPascal": 1000.0, "HV": 1.0, "percent": 1.0}


def _kw(call: ast.Call, name: str):
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def _literal(node):
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except ValueError:
        return ast.unparse(node)


def _name(node) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _to_float(v) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().lstrip("~<>≈ ")
        try:
            return float(s)
        except ValueError:
            return None
    return None


def parse_extraction(source: str) -> list[dict]:
    """Collect every Measurement of a mapped kind with the material it belongs to."""
    out = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _name(node.func) == "Material"):
            continue
        process = _literal(_kw(node, "process")) or ""
        comp = ""
        meas = _kw(node, "measurements")
        for sub in ast.walk(meas) if meas is not None else []:
            if isinstance(sub, ast.Call) and _name(sub.func) == "CompMeasurement" and sub.args and not comp:
                comp = str(_literal(sub.args[0]))
        for sub in ast.walk(meas) if meas is not None else []:
            if not (isinstance(sub, ast.Call) and _name(sub.func) == "Measurement"):
                continue
            kind = _name(_kw(sub, "kind"))
            if kind not in KIND_MAP:
                continue
            prop, loading = KIND_MAP[kind]
            unit = _name(_kw(sub, "unit")) or ""
            raw_value = _literal(_kw(sub, "value"))
            value = _to_float(raw_value)
            factor = UNIT_FACTOR.get(unit)
            if prop == "vickers_hardness" and unit in ("MegaPascal", "GigaPascal") and factor:
                factor /= 9.80665  # hardness reported as a pressure: 1 HV = 1 kgf/mm2 = 9.80665 MPa
            temp = _kw(sub, "temperature")
            temp_value = _literal(_kw(temp, "value")) if isinstance(temp, ast.Call) else None
            out.append(
                {
                    "composition": comp,
                    "process": process,
                    "litx_kind": kind,
                    "property": prop,
                    "loading": loading,
                    "raw_value": raw_value,
                    "value": None if value is None or factor is None else round(value * factor, 4),
                    "unit": unit,
                    "uncertainty": _literal(_kw(sub, "uncertainty")),
                    "test_temperature_c": temp_value,
                    "source": (_literal(_kw(sub, "source")) or "")[:120],
                }
            )
    return out


def doi_to_filename(identifier: str) -> str:
    doi = identifier.removeprefix("doi:")
    return "doi_" + doi.replace("/", "__").replace(".", "_").replace("-", "_") + ".py"


def match(ours: list[dict], theirs: list[dict], tol: float = 0.01):
    """One-to-one matching on (property, value within tol); material names use different conventions."""
    if not ours or not theirs:
        return []
    big = 1e6
    cost = [[big] * len(theirs) for _ in ours]
    for i, a in enumerate(ours):
        for j, b in enumerate(theirs):
            if a["property"] != b["property"] or b["value"] is None:
                continue
            va, vb = float(a["value"]), float(b["value"])
            err = abs(va - vb) / max(abs(vb), 1e-9) if vb else abs(va - vb)
            if err <= tol:
                cost[i][j] = err
    rows, cols = linear_sum_assignment(cost)
    return [(i, j) for i, j in zip(rows, cols) if cost[i][j] < big]


def main() -> int:
    with GROUND_TRUTH.open(newline="", encoding="utf-8") as f:
        labels = list(csv.DictReader(f))
    ref_rows, qa_lines = [], []
    totals = {"litx": 0, "litx_table": 0, "ours": 0, "matched": 0, "litx_table_matched": 0}
    for paper in load_papers():
        if paper.doc_type != "journal_article":
            continue
        url = f"{BASE}/{doi_to_filename(paper.identifier)}"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        theirs = [m for m in parse_extraction(r.text) if m["value"] is not None]
        for m in theirs:
            ref_rows.append({"paper_id": paper.paper_id, **m})
        ours = [row for row in labels if row["paper_id"] == paper.paper_id]
        pairs = match(ours, theirs)
        matched_theirs = {j for _, j in pairs}
        from_table = [j for j, m in enumerate(theirs) if "table" in m["source"].lower()]
        totals["litx"] += len(theirs)
        totals["litx_table"] += len(from_table)
        totals["ours"] += len(ours)
        totals["matched"] += len(pairs)
        totals["litx_table_matched"] += len([j for j in from_table if j in matched_theirs])
        qa_lines.append(
            f"| {paper.paper_id} | {len(ours)} | {len(theirs)} | {len(pairs)} | "
            f"{len([j for j in from_table if j in matched_theirs])}/{len(from_table)} |"
        )
        unmatched = [m for j, m in enumerate(theirs) if j not in matched_theirs]
        for m in unmatched:
            qa_lines.append(
                f"|  | not in our labels: {m['property']} {m['raw_value']} {m['unit']} "
                f"({m['composition']}; source: {m['source'][:60] or 'n/a'}) | | | |"
            )

    LABELS.mkdir(exist_ok=True)
    ref_path = LABELS / "litxalloy_reference.csv"
    with ref_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ref_rows[0].keys()))
        w.writeheader()
        w.writerows(ref_rows)

    REPORTS.mkdir(exist_ok=True)
    qa = [
        "# Label QA against LitXAlloy",
        "",
        "LitXAlloy (LitXBench, Radical AI, 2026) curates measurements from the same journal papers,",
        f"including values reported in the text. Commit `{COMMIT[:10]}`. Matching is one-to-one on",
        "property and value within 1%, because the two label sets name materials differently",
        "(printed row label here, composition plus processing history there).",
        "",
        "| paper | our labels | LitXAlloy values (4 properties) | matched | LitXAlloy table-sourced values matched |",
        "|---|---|---|---|---|",
        *qa_lines,
        "",
        f"Totals: {totals['ours']} labels here, {totals['litx']} LitXAlloy values, {totals['matched']} matched; "
        f"{totals['litx_table_matched']} of {totals['litx_table']} LitXAlloy values whose source cites a table are in our labels.",
    ]
    (REPORTS / "label_qa.md").write_text("\n".join(qa) + "\n", encoding="utf-8")
    print("\n".join(qa))
    print(f"\nwrote {ref_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
