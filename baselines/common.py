"""Shared helpers for the baseline runners.

Standard library only, so every tool's environment can import it. Each runner
reads the labeled pages listed in papers.csv (as single-page PDFs and 200 DPI
renders made by prepare_pages.py) and writes what the tool returned to
baselines/raw/<tool>/<paper_id>_p<page>.json. Scoring happens separately, in
the main environment (python -m tabletrace baselines).

A page file looks like this. A table carries whichever of rows, html, markdown
or latex the tool produced, and page-level tools put their whole output in text:

    {"tool": "docling", "version": "2.55.1", "paper_id": "chen2014", "page": 7,
     "seconds": 3.1, "settings": {...},
     "tables": [{"rows": [["Alloy", "HV"], ["Al0.25NbTiMoV", "4.97"]],
                 "caption": "Table 3. ...", "bbox": [x0, top, x1, bottom]}],
     "text": null}

Coordinates are PDF points from the top-left of the page.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / "outputs" / "baselines" / "pages"
RAW = ROOT / "baselines" / "raw"


def labeled_pages() -> list[tuple[str, str, int]]:
    """(paper_id, route, page) for every labeled page, in papers.csv order."""
    out = []
    with (ROOT / "papers.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pages = sorted({int(item.split("@")[1]) for item in r["labeled_tables"].split(";")})
            out.extend((r["paper_id"], r["route"], p) for p in pages)
    return out


def input_pdf(paper_id: str, route: str) -> Path:
    """The PDF TableTrace sees: the image-only papers have no text layer."""
    if route == "image_only":
        return ROOT / "data" / "image_only" / f"{paper_id}.pdf"
    return ROOT / "data" / "raw" / f"{paper_id}.pdf"


def page_pdf(paper_id: str, page: int) -> Path:
    return PAGES / f"{paper_id}_p{page}.pdf"


def page_png(paper_id: str, page: int) -> Path:
    return PAGES / f"{paper_id}_p{page}.png"


def result_path(tool: str, paper_id: str, page: int) -> Path:
    return RAW / tool / f"{paper_id}_p{page}.json"


def write_result(tool: str, version: str, paper_id: str, page: int, seconds: float, tables: list[dict],
                 text: str | None = None, settings: dict | None = None) -> Path:
    out = result_path(tool, paper_id, page)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"tool": tool, "version": version, "paper_id": paper_id, "page": page, "seconds": round(seconds, 2),
           "settings": settings or {}, "tables": tables, "text": text}
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return out
