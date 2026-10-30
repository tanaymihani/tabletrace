"""Build the Hugging Face dataset and model folders from the repo.

Dataset (outputs/hf/dataset): the card from release/dataset_card.md, the 514 labels, the
typed transcriptions they're generated from, the document list with licenses, and a 200 DPI
render of every labeled page. Model (outputs/hf/model): the card from release/model_card.md and
eng_tt.traineddata. Nothing is uploaded; the commands to do that are printed at the end.

Usage: python scripts/build_hf_release.py   (after make data)
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import pypdfium2 as pdfium

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tabletrace.paths import GROUND_TRUTH, LABELS, OUTPUTS, PAPERS_CSV, ROOT, load_papers  # noqa: E402

OUT = OUTPUTS / "hf"
DATASET_REPO = "tanaymihani/tabletrace-alloy-tables"
MODEL_REPO = "tanaymihani/tesseract-eng-tables"
PAPER_COLUMNS = ["paper_id", "doc_type", "first_author", "year", "venue", "identifier", "title", "license",
                 "route", "split", "labeled_tables", "url", "sha256"]


def main() -> int:
    ds, model = OUT / "dataset", OUT / "model"
    shutil.rmtree(OUT, ignore_errors=True)
    (ds / "data").mkdir(parents=True)
    (ds / "pages").mkdir()
    model.mkdir(parents=True)

    shutil.copy(ROOT / "release" / "dataset_card.md", ds / "README.md")
    shutil.copy(GROUND_TRUTH, ds / "data" / "ground_truth.csv")
    shutil.copy(LABELS / "transcriptions.py", ds / "transcriptions.py")
    with PAPERS_CSV.open(newline="", encoding="utf-8") as f, (ds / "data" / "papers.csv").open("w", newline="", encoding="utf-8") as g:
        w = csv.DictWriter(g, fieldnames=PAPER_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(csv.DictReader(f))
    for paper in load_papers():
        doc = pdfium.PdfDocument(str(paper.raw_pdf))
        for page in paper.labeled_pages:
            img = doc[page - 1].render(scale=200 / 72).to_pil().convert("RGB")
            img.save(ds / "pages" / f"{paper.paper_id}_p{page}.png", optimize=True)
        doc.close()

    shutil.copy(ROOT / "release" / "model_card.md", model / "README.md")
    shutil.copy(ROOT / "models" / "tessdata" / "eng_tt.traineddata", model / "eng_tt.traineddata")

    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"built {OUT.relative_to(ROOT)} ({size:.1f} MB)")
    print("upload (after `hf auth login`):")
    print(f"  hf upload {DATASET_REPO} {ds.relative_to(ROOT)} . --repo-type dataset")
    print(f"  hf upload {MODEL_REPO} {model.relative_to(ROOT)} . --repo-type model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
