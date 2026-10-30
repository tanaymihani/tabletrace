"""Marker 1.10 with its default settings (surya layout, OCR where needed, and table recognition).

Marker 2 moved OCR to a separate vision-model server (vLLM or llama.cpp), so this pins the last
1.x release, which runs its models in-process. Keeps each Table block's HTML and position, and
the caption when Marker groups one with the table. Runs in its own environment:
    uv venv .venvs/marker --python 3.11
    uv pip install --python .venvs/marker/bin/python "marker-pdf==1.10.2"
    .venvs/marker/bin/python baselines/run_marker.py
The reported run used the same code on a Kaggle CPU session (kaggle/marker_kaggle.py), because
Marker's models don't fit next to everything else in 8 GB of memory.
"""

from __future__ import annotations

import re
import sys
import time
from importlib.metadata import version
from pathlib import Path

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import labeled_pages, page_pdf, result_path, write_result  # noqa: E402


def plain(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def walk(block, group, tables, captions):
    kind = block.block_type
    if kind == "Table":
        tables.append((block, group))
    elif kind == "Caption":
        captions.append((block, group))
    for child in block.children or []:
        walk(child, block if kind == "TableGroup" else group, tables, captions)


def main() -> int:
    only = sys.argv[1:]
    converter = PdfConverter(artifact_dict=create_model_dict(), renderer="marker.renderers.json.JSONRenderer")
    for paper_id, _route, page in labeled_pages():
        if (only and f"{paper_id}_p{page}" not in only) or (not only and result_path("marker", paper_id, page).exists()):
            continue
        t0 = time.time()
        rendered = converter(str(page_pdf(paper_id, page)))
        seconds = time.time() - t0
        tables, captions = [], []
        for pg in rendered.children:
            walk(pg, None, tables, captions)
        out = []
        for block, group in tables:
            caption = " ".join(plain(c.html) for c, g in captions if group is not None and g is group) or None
            out.append({"html": block.html, "bbox": [round(v, 2) for v in block.bbox], "caption": caption})
        write_result("marker", version("marker-pdf"), paper_id, page, seconds, out,
                     settings={"renderer": "json", "use_llm": False})
        print(f"marker   {paper_id:18s} p{page:<3d} {len(out)} tables  {seconds:5.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
