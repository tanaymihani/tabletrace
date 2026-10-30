"""pdfplumber's own table finder: find_tables / extract with the default (ruling-line) settings.

Runs in the main environment. Reads the same single-page PDFs as the other tools,
hidden text included, since that's what pdfplumber returns out of the box.

Usage: python baselines/run_pdfplumber.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pdfplumber

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import labeled_pages, page_pdf, write_result  # noqa: E402

STRATEGIES = {
    "pdfplumber": {},  # defaults: cells from ruling lines
    "pdfplumber_text": {"vertical_strategy": "text", "horizontal_strategy": "text"},  # alignment of words
}


def main() -> int:
    for tool, settings in STRATEGIES.items():
        for paper_id, _route, page in labeled_pages():
            t0 = time.time()
            with pdfplumber.open(page_pdf(paper_id, page)) as pdf:
                tables = []
                for t in pdf.pages[0].find_tables(table_settings=settings):
                    rows = [[c if c is not None else "" for c in row] for row in t.extract()]
                    tables.append({"rows": rows, "bbox": [round(v, 2) for v in t.bbox], "caption": None})
            write_result(tool, pdfplumber.__version__, paper_id, page, time.time() - t0, tables,
                         settings=settings or {"table_settings": "default"})
            print(f"{tool:16s} {paper_id:18s} p{page:<3d} {len(tables)} tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
