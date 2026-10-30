"""Camelot on the labeled pages: lattice (ruling lines) and stream (whitespace), default options.

Runs in its own environment (baselines/requirements/camelot.txt):
    uv venv .venvs/camelot --python 3.11
    uv pip install --python .venvs/camelot/bin/python -r baselines/requirements/camelot.txt
    .venvs/camelot/bin/python baselines/run_camelot.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import camelot
import pypdfium2 as pdfium

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import labeled_pages, page_pdf, write_result  # noqa: E402

warnings.filterwarnings("ignore")


def main() -> int:
    for flavor in ("lattice", "stream"):
        tool = f"camelot_{flavor}"
        for paper_id, _route, page in labeled_pages():
            pdf = page_pdf(paper_id, page)
            height = pdfium.PdfDocument(str(pdf))[0].get_height()
            t0 = time.time()
            try:
                found = camelot.read_pdf(str(pdf), pages="1", flavor=flavor, suppress_stdout=True)
            except Exception as exc:  # camelot raises on pages it can't parse at all
                print(f"{tool:16s} {paper_id:18s} p{page:<3d} failed: {exc}")
                found = []
            tables = []
            for t in found:
                x1, y1, x2, y2 = t._bbox  # PDF space, origin bottom-left
                tables.append({"rows": t.df.astype(str).values.tolist(), "caption": None,
                               "bbox": [round(x1, 2), round(height - y2, 2), round(x2, 2), round(height - y1, 2)]})
            write_result(tool, camelot.__version__, paper_id, page, time.time() - t0, tables,
                         settings={"flavor": flavor})
            print(f"{tool:16s} {paper_id:18s} p{page:<3d} {len(tables)} tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
