"""Write tests/data/clipped_text.pdf: two pages of text that a viewer clips away but pdfminer returns.

Page 1 is the MDPI case. A Form XObject (a "figure") has a /BBox that doesn't start at the origin,
and the form also draws a line of text outside that box, the way a reused manuscript page carries
a whole table along with the figure. Page 2 is the pdfminer.six #414 case: text drawn outside an
explicit "re W n" clipping path. Everything is plain Helvetica, so the file is small and readable.

Usage: python scripts/make_clip_repro.py
"""

from __future__ import annotations

from pathlib import Path

import pikepdf

OUT = Path(__file__).resolve().parents[1] / "tests" / "data" / "clipped_text.pdf"

PAGE1 = b"""BT /F1 12 Tf 72 720 Td (Table 1. Hardness of the alloys) Tj ET
BT /F1 12 Tf 72 700 Td (HfNbTaTiZr 1155) Tj ET
q 1 0 0 1 22 350 cm /Fm1 Do Q
"""
# Form space: the box is [50 50 250 150]; on the page it lands at x 72-272, y 400-500.
FORM = b"""0.85 g 60 60 180 80 re f 0 g
BT /F1 10 Tf 70 100 Td (Figure label) Tj ET
BT /F1 12 Tf 70 300 Td (HfNbTaTiZr 1144) Tj ET
"""
PAGE2 = b"""BT /F1 12 Tf 72 720 Td (Visible line) Tj ET
q 72 600 200 30 re W n
BT /F1 12 Tf 80 610 Td (Inside the clip) Tj ET
BT /F1 12 Tf 80 560 Td (Outside the clip) Tj ET
Q
BT /F1 12 Tf 72 500 Td (After Q, visible again) Tj ET
"""


def main() -> int:
    pdf = pikepdf.new()
    font = pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name.Font, Subtype=pikepdf.Name.Type1,
                                                BaseFont=pikepdf.Name.Helvetica))
    fonts = pikepdf.Dictionary(F1=font)
    form = pikepdf.Stream(pdf, FORM)
    form.Type, form.Subtype = pikepdf.Name.XObject, pikepdf.Name.Form
    form.BBox = pikepdf.Array([50, 50, 250, 150])
    form.Resources = pikepdf.Dictionary(Font=fonts)
    for content, resources in ((PAGE1, pikepdf.Dictionary(Font=fonts, XObject=pikepdf.Dictionary(Fm1=form))),
                               (PAGE2, pikepdf.Dictionary(Font=fonts))):
        page = pikepdf.Dictionary(Type=pikepdf.Name.Page, MediaBox=pikepdf.Array([0, 0, 612, 792]),
                                  Resources=resources, Contents=pikepdf.Stream(pdf, content))
        pdf.pages.append(pikepdf.Page(page))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.save(OUT, compress_streams=False, deterministic_id=True)
    print(f"wrote {OUT.relative_to(OUT.parents[2])} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
