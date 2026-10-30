"""Render the labeled pages for the GPU runs on Kaggle and write the dataset folder.

olmOCR is trained on pages rendered to 1288 px on the longest side, so it gets exactly
that. The other models get 1600 px, enough for the small type in the NASA scans.
Output: outputs/baselines/kaggle_pages/ (PNGs, pages.json, dataset-metadata.json).

Usage: python baselines/kaggle/prepare_upload.py
       kaggle datasets create -p outputs/baselines/kaggle_pages     (first time; private)
       kaggle datasets version -p outputs/baselines/kaggle_pages -m "pages"   (later)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pypdfium2 as pdfium

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, input_pdf, labeled_pages  # noqa: E402

OUT = ROOT / "outputs" / "baselines" / "kaggle_pages"
SIZES = {"img_olmocr": 1288, "img_page": 1600}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pages = []
    for paper_id, route, page in labeled_pages():
        doc = pdfium.PdfDocument(str(input_pdf(paper_id, route)))
        pg = doc[page - 1]
        w, h = pg.get_size()
        entry = {"paper_id": paper_id, "route": route, "page": page, "width_pt": w, "height_pt": h}
        for name, longest in SIZES.items():
            img = pg.render(scale=longest / max(w, h)).to_pil().convert("RGB")
            fname = f"{paper_id}_p{page}_{name[4:]}.png"
            img.save(OUT / fname)
            entry[name] = fname
        pages.append(entry)
        doc.close()
    (OUT / "pages.json").write_text(json.dumps(pages, indent=1))
    (OUT / "dataset-metadata.json").write_text(json.dumps(
        {"title": "tabletrace-pages", "id": "tanaymihani/tabletrace-pages", "licenses": [{"name": "other"}]}, indent=1))
    print(f"{len(pages)} pages in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
