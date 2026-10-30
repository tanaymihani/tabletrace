"""Cross-check the labels against the MPEA dataset (Borg et al. 2020, Scientific Data 7:430).

The MPEA dataset compiles mechanical properties of multi-principal element alloys from the
literature, and it covers all seven journal papers used here. This downloads the CSV from a
pinned commit of CitrineInformatics/MPEA_dataset and matches its values to the labels one-to-one
on property and value (within 1%), the same rule as the LitXAlloy check, because the two use
different sample names (a formula there, the printed label here). Writes reports/label_qa_mpea.md.

Usage: python scripts/check_mpea.py
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import requests
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tabletrace.paths import GROUND_TRUTH, REPORTS, load_papers  # noqa: E402

COMMIT = "d31e2510cccc3fb9e79d43ad62652f64ef8c2c5e"
URL = f"https://raw.githubusercontent.com/CitrineInformatics/MPEA_dataset/{COMMIT}/MPEA_dataset.csv"
COLUMNS = {"vickers_hardness": ["PROPERTY: HV"], "yield_strength": ["PROPERTY: YS (MPa)"],
           "ultimate_strength": ["PROPERTY: UTS (MPa)"],
           "elongation": ["PROPERTY: Elongation (%)", "PROPERTY: Elongation plastic (%)"]}


def close(a: float, b: float, tol: float = 0.01) -> bool:
    return a == b or (b != 0 and abs(a - b) / abs(b) <= tol)


def main() -> int:
    rows = list(csv.DictReader(io.StringIO(requests.get(URL, timeout=60).content.decode("utf-8", errors="replace"))))
    with GROUND_TRUTH.open(newline="", encoding="utf-8") as f:
        labels = list(csv.DictReader(f))
    lines, total, total_matched = [], 0, 0
    for paper in load_papers():
        if paper.doc_type != "journal_article":
            continue
        doi = paper.identifier.removeprefix("doi:").lower()
        theirs = []
        for r in rows:
            if doi not in r["REFERENCE: doi"].lower():
                continue
            for prop, cols in COLUMNS.items():
                for c in cols:
                    try:
                        theirs.append((prop, float(r[c]), r["FORMULA"], r["PROPERTY: Type of test"]))
                    except (KeyError, ValueError):
                        pass
        ours = [(lab["property"], float(lab["value"])) for lab in labels if lab["paper_id"] == paper.paper_id]
        cost = [[0.0 if (pa == pb and close(va, vb)) else 1.0 for pb, vb, *_ in theirs] for pa, va in ours]
        matched: set[int] = set()
        if ours and theirs:
            ri, ci = linear_sum_assignment(cost)
            matched = {j for i, j in zip(ri, ci) if cost[i][j] == 0.0}
        total += len(theirs)
        total_matched += len(matched)
        lines.append(f"| {paper.paper_id} | {len(ours)} | {len(theirs)} | {len(matched)} |")
        for j, (prop, value, formula, test) in enumerate(theirs):
            if j not in matched:
                lines.append(f"|  | not in the labels: {prop} {value:g} ({formula}, test {test or 'n/a'}) | | |")
    out = [
        "# Label QA against the MPEA dataset",
        "",
        "The MPEA dataset (Borg et al. 2020, *Scientific Data* 7:430) compiles mechanical properties of",
        f"multi-principal element alloys from the literature. Commit `{COMMIT[:10]}` of",
        "CitrineInformatics/MPEA_dataset. Matching is one-to-one on property and value within 1%.",
        "",
        "| paper | labels | MPEA values (4 properties) | matched |",
        "|---|---|---|---|",
        *lines,
        "",
        f"Totals: {total_matched} of {total} MPEA values match a label.",
    ]
    (REPORTS / "label_qa_mpea.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
