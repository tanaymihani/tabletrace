"""Marker 1.10 on a Kaggle CPU session, same steps as baselines/run_marker.py.

Marker's models (3.3 GB) don't fit next to everything else on an 8 GB laptop, so this runs
on Kaggle's CPU machines (no GPU needed) over the single-page PDFs in the pages dataset and
writes /kaggle/working/raw/marker/<paper_id>_p<page>.json.
"""

import glob
import json
import os
import re
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "marker-pdf==1.10.2"], check=True)
os.environ.setdefault("TORCH_DEVICE", "cpu")

from importlib.metadata import version  # noqa: E402

from marker.converters.pdf import PdfConverter  # noqa: E402
from marker.models import create_model_dict  # noqa: E402

DATA = os.path.dirname(glob.glob("/kaggle/input/**/pages.json", recursive=True)[0])
OUT = "/kaggle/working/raw/marker"
PAGES = json.load(open(os.path.join(DATA, "pages.json")))


def plain(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def walk(block, group, tables, captions):
    kind = block.block_type
    if kind == "Table":
        tables.append((block, group))
    elif kind == "Caption":
        captions.append((block, group))
    for child in block.children or []:
        walk(child, block if kind == "TableGroup" else group, tables, captions)


def main():
    os.makedirs(OUT, exist_ok=True)
    converter = PdfConverter(artifact_dict=create_model_dict(), renderer="marker.renderers.json.JSONRenderer")
    for p in PAGES:
        pdf = os.path.join(DATA, f"{p['paper_id']}_p{p['page']}.pdf")
        t0 = time.time()
        rendered = converter(pdf)
        seconds = time.time() - t0
        tables, captions = [], []
        for pg in rendered.children:
            walk(pg, None, tables, captions)
        out = []
        for block, group in tables:
            caption = " ".join(plain(c.html) for c, g in captions if group is not None and g is group) or None
            out.append({"html": block.html, "bbox": [round(v, 2) for v in block.bbox], "caption": caption})
        doc = {"tool": "marker", "version": version("marker-pdf"), "paper_id": p["paper_id"], "page": p["page"],
               "seconds": round(seconds, 2), "settings": {"renderer": "json", "use_llm": False,
                                                          "device": "Kaggle CPU, 4 cores"},
               "tables": out, "text": None}
        with open(os.path.join(OUT, f"{p['paper_id']}_p{p['page']}.json"), "w") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        print(f"marker {p['paper_id']} p{p['page']}: {len(out)} tables, {seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
