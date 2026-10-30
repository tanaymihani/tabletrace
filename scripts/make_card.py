"""Draw the provenance picture: a value on a scanned NASA page, boxed, with the record it became.

The record is read from reports/extractions.csv (the final run), not typed in, so the picture can't
drift from the data. Writes reports/figures/provenance.png (1600 x 900) and, with --og, a
1200 x 630 social preview next to it.

Usage: python scripts/make_card.py [--og]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tabletrace.paths import FIGURES, REPORTS, get_paper  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#2a78d6"
PAPER, PAGE, MATERIAL, RAW = "billinghurst1992", 14, "Ti-6Al-4 V ELI", "129.1"
CROP = (45.0, 395.0, 570.0, 527.0)  # PDF points, top-left origin: Table 4 of NASA TP-3288
FONTS = {"sans": ["/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf", "DejaVuSans.ttf"],
         "bold": ["/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial Bold.ttf", "DejaVuSans-Bold.ttf"],
         "mono": ["/System/Library/Fonts/Menlo.ttc", "/Library/Fonts/Courier New.ttf", "DejaVuSansMono.ttf"]}


def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for path in FONTS[kind]:
        try:
            return ImageFont.truetype(path, size, index=1 if kind == "bold" and path.endswith(".ttc") else 0)
        except OSError:
            continue
    return ImageFont.load_default(size)


def the_record() -> dict:
    with (REPORTS / "extractions.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["paper_id"] == PAPER and int(r["page"]) == PAGE and r["material"] == MATERIAL and r["raw_text"] == RAW:
                return r
    raise SystemExit("record not found in reports/extractions.csv")


def draw_card(width: int, height: int, scale: float) -> Image.Image:
    rec = the_record()
    bbox = json.loads(rec["bbox"])
    img = Image.new("RGB", (width, height), SURFACE)
    d = ImageDraw.Draw(img)
    s = lambda v: int(round(v * scale))  # noqa: E731

    d.text((s(64), s(52)), "TableTrace", font=font("bold", s(56)), fill=INK)
    d.text((s(66), s(122)), "Every number points back to the cell it came from", font=font("sans", s(28)), fill=INK_2)

    # the scanned table, rendered from the PDF
    dpi = 300
    k = dpi / 72
    page = pdfium.PdfDocument(str(get_paper(PAPER).raw_pdf))[PAGE - 1].render(scale=k).to_pil().convert("RGB")
    crop = page.crop(tuple(int(v * k) for v in CROP))
    target_w = s(860)
    ratio = target_w / crop.width
    crop = crop.resize((target_w, int(crop.height * ratio)), Image.LANCZOS)
    ox, oy = s(64), s(262)
    d.text((ox, oy - s(40)), "NASA TP-3288 (1992), a scanned report: page 14, Table 4",
           font=font("sans", s(22)), fill=MUTED)
    img.paste(crop, (ox, oy))
    d.rectangle((ox - 1, oy - 1, ox + crop.width, oy + crop.height), outline=GRID, width=max(1, s(2)))

    # the cell the value came from
    to_px = lambda x, y: (ox + (x - CROP[0]) * k * ratio, oy + (y - CROP[1]) * k * ratio)  # noqa: E731
    x0, y0 = to_px(bbox[0], bbox[1])
    x1, y1 = to_px(bbox[2], bbox[3])
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rounded_rectangle((x0, y0, x1, y1), radius=s(8), fill=(42, 120, 214, 40),
                                              outline=BLUE, width=max(2, s(4)))
    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))
    d = ImageDraw.Draw(img)

    # the record it became
    review = float(rec["confidence"]) < 0.85
    lines = [("{", None),
             ('  "paper_id"', f'"{rec["paper_id"]}",'), ('  "page"', f'{rec["page"]},'),
             ('  "table"', f'{rec["table_index"]},'), ('  "material"', f'"{rec["material"]}",'),
             ('  "property"', f'"{rec["property"]}",'), ('  "raw_text"', f'"{rec["raw_text"]}",'),
             ('  "raw_unit"', f'"{rec["raw_unit"]}",'), ('  "value"', f'{float(rec["value"]):.1f},'),
             ('  "unit"', f'"{rec["unit"]}",'), ('  "row"', f'{rec["row"]}, "col": {rec["col"]},'),
             ('  "bbox"', "[" + ", ".join(f"{v:.1f}" for v in bbox) + "],"),
             ('  "confidence"', f'{float(rec["confidence"]):.2f},'),
             ('  "review"', "true" if review else "false"), ("}", None)]
    mono = font("mono", s(21))
    px, py, lh = s(1012), s(214), s(36)
    panel = (px - s(28), py - s(24), width - s(56), py + lh * len(lines) + s(14))
    d.rounded_rectangle(panel, radius=s(14), fill="#ffffff", outline=GRID, width=max(1, s(2)))
    value_y = None
    for i, (key, val) in enumerate(lines):
        y = py + i * lh
        if val is None:
            d.text((px, y), key, font=mono, fill=INK_2)
            continue
        d.text((px, y), key + ":", font=mono, fill=INK_2)
        kx = px + d.textlength(key + ": ", font=mono)
        d.text((kx, y), val, font=mono, fill=BLUE if key.strip('" ') in ("value", "raw_text") else INK)
        if key.strip('" ') == "value":
            value_y = y + lh / 2

    # arrow from the cell to the record: down out of the table, then across
    ax = (x0 + x1) / 2
    bx, by = panel[0] - s(12), value_y
    w = max(2, s(4))
    d.line((ax, y1 + s(6), ax, by), fill=BLUE, width=w)
    d.line((ax - w / 2, by, bx, by), fill=BLUE, width=w)
    head = s(16)
    d.polygon([(bx, by), (bx - head, by - head * 0.55), (bx - head, by + head * 0.55)], fill=BLUE)

    note = (f"{rec['raw_text']} {rec['raw_unit']} read by OCR and converted to {float(rec['value']):.1f} {rec['unit']}. "
            f"Confidence {float(rec['confidence']):.2f} is {'below' if review else 'above'} the calibrated review "
            f"threshold, so it's {'sent to a person' if review else 'accepted'}.")
    d.text((s(64), height - s(76)), note, font=font("sans", s(24)), fill=INK_2)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--og", action="store_true", help="also write the 1200 x 630 social preview")
    args = ap.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    draw_card(1600, 900, 1.0).save(FIGURES / "provenance.png", optimize=True)
    print(f"wrote {FIGURES / 'provenance.png'}")
    if args.og:
        card = draw_card(1600, 900, 1.0)
        og = card.resize((1200, 675), Image.LANCZOS).crop((0, 22, 1200, 652))
        og.save(FIGURES / "provenance_og.png", optimize=True)
        print(f"wrote {FIGURES / 'provenance_og.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
