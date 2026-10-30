"""Cut every labeled page out of the input PDFs and render it, for the baseline runners.

Writes outputs/baselines/pages/<paper_id>_p<page>.pdf (one page, same coordinates as
the original) and a 200 DPI PNG of the same page. Run after `make data`.

Usage: python baselines/prepare_pages.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pikepdf
import pypdfium2 as pdfium

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PAGES, input_pdf, labeled_pages, page_pdf, page_png  # noqa: E402

DPI = 200


def main() -> int:
    PAGES.mkdir(parents=True, exist_ok=True)
    for paper_id, route, page in labeled_pages():
        src = input_pdf(paper_id, route)
        with pikepdf.open(src) as pdf:
            one = pikepdf.new()
            one.pages.append(pdf.pages[page - 1])
            one.save(page_pdf(paper_id, page))
        doc = pdfium.PdfDocument(str(src))
        doc[page - 1].render(scale=DPI / 72).to_pil().convert("RGB").save(page_png(paper_id, page))
        doc.close()
        print(f"ok  {paper_id} p{page}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
