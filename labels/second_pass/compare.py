"""Second labeling pass: compare 50 re-read values with the labels.

sample.csv lists 50 labels drawn at random (seed 20261030) without their values: sample name,
property, temperature and condition only. readings.csv has each value as read again from a fresh
render of the page, before looking at ground_truth.csv. This prints the agreement and a
Clopper-Pearson bound on the transcription error rate.

Usage: python labels/second_pass/compare.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from scipy.stats import beta

HERE = Path(__file__).resolve().parent


def printed_number(raw: str) -> float:
    return float(re.sub(r"[^\d.]", "", raw.split("±")[0].replace(",", "")))


def main() -> int:
    labels = {r["label_id"]: r for r in csv.DictReader((HERE.parent / "ground_truth.csv").open(encoding="utf-8"))}
    reads = list(csv.DictReader((HERE / "readings.csv").open(encoding="utf-8")))
    wrong = [r for r in reads if printed_number(labels[r["label_id"]]["raw_text"]) != float(r["reading"])]
    for r in wrong:
        print(f"disagree  {r['label_id']}: label {labels[r['label_id']]['raw_text']!r}, second pass {r['reading']!r}")
    n, k = len(reads), len(wrong)
    upper = beta.ppf(0.95, k + 1, n - k)
    print(f"{n - k} of {n} agree; one-sided 95% upper bound on the disagreement rate: {upper:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
