"""Build a local review page: every labeled table as an image, next to its labels.

The point is a fast, independent second pass over ground_truth.csv: look at the
cell, read the label, mark anything wrong. Output goes to outputs/label_review/
(git-ignored, since it contains page images).

Usage: python scripts/label_review.py && open outputs/label_review/index.html
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tabletrace.captions import find_captions, link_caption  # noqa: E402
from tabletrace.evaluate import load_ground_truth  # noqa: E402
from tabletrace.paths import OUTPUTS, load_papers  # noqa: E402
from tabletrace.pipeline import page_ocr_words  # noqa: E402
from tabletrace.render import render_page  # noqa: E402
from tabletrace.router import route_page  # noqa: E402
from tabletrace.tables_image import detect_tables  # noqa: E402
from tabletrace.tables_text import text_words  # noqa: E402

OUT = OUTPUTS / "label_review"
DPI = 200


def main() -> int:
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    gt = load_ground_truth()
    sections = []
    for paper in load_papers():
        for table, page in paper.labeled_tables:
            labels = gt[(gt.paper_id == paper.paper_id) & (gt.page == page) & (gt.table_index == table)]
            img = render_page(paper.raw_pdf, page, dpi=DPI)
            dets = detect_tables(render_page(paper.raw_pdf, page, dpi=200), 200)
            decision = route_page(paper.raw_pdf, page)
            words = text_words(paper.raw_pdf, page) if decision.route == "text_layer" else page_ocr_words(paper.raw_pdf, page, 300)
            caps = find_captions(words)
            box = None
            for d in dets:
                cap = link_caption(caps, d.bbox)
                if cap and cap.number == table:
                    box = d.bbox
            if box is None:
                # detection missed this table: show the region under its caption instead
                cap = next((c for c in caps if c.number == table), None)
                if cap is not None:
                    box = (max(0.0, cap.bbox[0] - 120), cap.bbox[1], min(img.width / (DPI / 72.0), cap.bbox[2] + 120),
                           cap.bbox[3] + 150)
                elif dets:
                    box = max(dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])).bbox
            s = DPI / 72.0
            if box is not None:
                pad = 25
                crop = img.crop((max(0, int((box[0] - pad) * s)), max(0, int((box[1] - pad * 2) * s)),
                                 min(img.width, int((box[2] + pad) * s)), min(img.height, int((box[3] + pad) * s))))
            else:
                crop = img
            name = f"{paper.paper_id}_p{page}_t{table}.png"
            crop.save(OUT / "img" / name)
            rows = "".join(
                f"<tr><td><input type='checkbox' data-id='{html.escape(r.label_id)}'></td><td>{html.escape(r.label_id)}</td>"
                f"<td>{html.escape(str(r.material))}</td><td>{html.escape(r.property)}</td><td>{html.escape(str(r.raw_text))}</td>"
                f"<td>{r.value:g} {html.escape(r.unit)}</td><td>{'' if r.test_temperature_c != r.test_temperature_c else r.test_temperature_c}</td>"
                f"<td>{html.escape(str(r.condition or ''))}</td></tr>"
                for r in labels.fillna("").itertuples())
            sections.append(
                f"<section><h2>{html.escape(paper.paper_id)} · page {page} · Table {table} "
                f"<small>({len(labels)} labels)</small></h2><div class='pair'><img src='img/{name}' alt='table'>"
                f"<table><thead><tr><th>wrong?</th><th>id</th><th>material</th><th>property</th><th>cell</th>"
                f"<th>value</th><th>T (°C)</th><th>condition</th></tr></thead><tbody>{rows}</tbody></table></div></section>")
            print(f"{paper.paper_id} p{page} T{table}: {len(labels)} labels")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Label review</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {{ font: 14px system-ui, -apple-system, sans-serif; margin: 16px; background: #fcfcfb; color: #0b0b0b; }}
h1 {{ font-size: 20px; }} h2 {{ font-size: 16px; margin-top: 32px; }} small {{ color: #52514e; font-weight: normal; }}
.pair {{ display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; }}
.pair img {{ max-width: 100%; width: 720px; border: 1px solid #e1e0d9; }}
table {{ border-collapse: collapse; font-size: 13px; }} td, th {{ border-bottom: 1px solid #e1e0d9; padding: 3px 8px; text-align: left; }}
#out {{ position: sticky; top: 0; background: #fcfcfb; padding: 8px 0; }}
</style></head><body>
<h1>Label review</h1>
<p>Tick every label that does not match the image. The list of ticked ids below can be pasted into an issue.</p>
<div id="out">Flagged: <code id="flagged">none</code></div>
{''.join(sections)}
<script>
const box = document.getElementById('flagged');
function update() {{
  const ids = [...document.querySelectorAll('input[type=checkbox]:checked')].map(x => x.dataset.id);
  box.textContent = ids.length ? ids.join(', ') : 'none';
}}
document.addEventListener('change', update);
</script></body></html>"""
    (OUT / "index.html").write_text(page, encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({"tables": len(sections), "labels": int(len(gt))}))
    print(f"wrote {OUT / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
