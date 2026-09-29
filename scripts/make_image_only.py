"""Turn the papers marked `image_only` in papers.csv into scan-like PDFs.

Every page is rendered to a 300 DPI grayscale PNG and the PDF is rebuilt from
those images alone, so the text layer is gone and OCR is the only way in. The
original text layer stays available in data/raw/ as the reference for CER/WER.

Usage: python scripts/make_image_only.py [--dpi 300]
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import img2pdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tabletrace.paths import IMAGE_ONLY_PDFS, load_papers  # noqa: E402
from tabletrace.render import page_count, render_page  # noqa: E402


def make_image_only(src: Path, dest: Path, dpi: int) -> int:
    pages = []
    for page in range(1, page_count(src) + 1):
        img = render_page(src, page, dpi=dpi, grayscale=True)
        buf = io.BytesIO()
        img.save(buf, format="PNG", dpi=(dpi, dpi))
        pages.append(buf.getvalue())
    layout = img2pdf.get_fixed_dpi_layout_fun((dpi, dpi))
    dest.write_bytes(img2pdf.convert(pages, layout_fun=layout))
    return len(pages)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    IMAGE_ONLY_PDFS.mkdir(parents=True, exist_ok=True)
    for paper in load_papers():
        if paper.route != "image_only":
            continue
        n = make_image_only(paper.raw_pdf, paper.input_pdf, args.dpi)
        print(f"{paper.paper_id}: {n} pages -> {paper.input_pdf} ({paper.input_pdf.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
