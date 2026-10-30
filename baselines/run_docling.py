"""Docling with its default PDF pipeline (layout model, TableFormer, RapidOCR where there's no text).

Keeps what Docling returns per table: the cell grid with spans repeated, column-header flags,
the cell boxes and the caption it linked. Runs in its own environment:
    uv venv .venvs/docling --python 3.11
    uv pip install --python .venvs/docling/bin/python -r baselines/requirements/docling.txt
    .venvs/docling/bin/python baselines/run_docling.py
"""

from __future__ import annotations

import sys
import time
from importlib.metadata import version
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc import CoordOrigin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import labeled_pages, page_pdf, write_result  # noqa: E402


def top_left(bbox, height):
    b = bbox.to_top_left_origin(page_height=height) if bbox.coord_origin == CoordOrigin.BOTTOMLEFT else bbox
    return [round(b.l, 2), round(b.t, 2), round(b.r, 2), round(b.b, 2)]


def main() -> int:
    converter = DocumentConverter()
    for paper_id, _route, page in labeled_pages():
        t0 = time.time()
        doc = converter.convert(str(page_pdf(paper_id, page))).document
        seconds = time.time() - t0
        height = doc.pages[1].size.height
        tables = []
        for table in doc.tables:
            grid = table.data.grid
            rows = [[cell.text for cell in row] for row in grid]
            boxes = [[top_left(cell.bbox, height) if cell.bbox else None for cell in row] for row in grid]
            n_header = 0
            for row in grid:
                if row and all(cell.column_header for cell in row):
                    n_header += 1
                else:
                    break
            prov = table.prov[0] if table.prov else None
            tables.append({"rows": rows, "cell_boxes": boxes, "header_rows": n_header or None,
                           "caption": table.caption_text(doc) or None,
                           "bbox": top_left(prov.bbox, height) if prov else None})
        write_result("docling", version("docling"), paper_id, page, seconds, tables,
                     settings={"pipeline": "default PdfPipelineOptions"})
        print(f"docling  {paper_id:18s} p{page:<3d} {len(tables)} tables  {seconds:5.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
