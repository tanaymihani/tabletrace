"""Repository paths and the paper manifest."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPERS_CSV = ROOT / "papers.csv"
DATA = ROOT / "data"
RAW_PDFS = DATA / "raw"
IMAGE_ONLY_PDFS = DATA / "image_only"
LABELS = ROOT / "labels"
GROUND_TRUTH = LABELS / "ground_truth.csv"
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


@dataclass(frozen=True)
class Paper:
    paper_id: str
    doc_type: str
    route: str  # born_digital | image_only | scanned
    split: str  # calibration | test
    url: str
    sha256: str
    labeled_tables: tuple[tuple[int, int], ...]  # (table number, page)
    title: str
    identifier: str
    license: str

    @property
    def raw_pdf(self) -> Path:
        return RAW_PDFS / f"{self.paper_id}.pdf"

    @property
    def input_pdf(self) -> Path:
        """The PDF the pipeline actually sees: image-only papers lose their text layer."""
        if self.route == "image_only":
            return IMAGE_ONLY_PDFS / f"{self.paper_id}.pdf"
        return self.raw_pdf

    @property
    def labeled_pages(self) -> list[int]:
        return sorted({page for _, page in self.labeled_tables})


def _parse_tables(spec: str) -> tuple[tuple[int, int], ...]:
    out = []
    for item in spec.split(";"):
        table, page = item.split("@")
        out.append((int(table), int(page)))
    return tuple(out)


def load_papers() -> list[Paper]:
    with PAPERS_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        Paper(
            paper_id=r["paper_id"],
            doc_type=r["doc_type"],
            route=r["route"],
            split=r["split"],
            url=r["url"],
            sha256=r["sha256"],
            labeled_tables=_parse_tables(r["labeled_tables"]),
            title=r["title"],
            identifier=r["identifier"],
            license=r["license"],
        )
        for r in rows
    ]


def get_paper(paper_id: str) -> Paper:
    for p in load_papers():
        if p.paper_id == paper_id:
            return p
    raise KeyError(paper_id)
