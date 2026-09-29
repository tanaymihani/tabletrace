"""Synthetic text lines for fine-tuning Tesseract on materials-science table cells.

The stock English LSTM model has no ± σ ε µ in its character set, and it has
never seen subscripted alloy formulas, so it misreads the most common table
content in this literature. This script renders the kinds of lines those
tables contain, with the ground truth written inline (subscripts as plain
digits), and writes image + .gt.txt pairs for training/finetune_tesseract.sh.

Nothing here is copied from the evaluation documents: compositions are random
draws over element symbols, values are random numbers, and headers come from a
generic vocabulary of property names and units.

Usage: python training/make_ocr_data.py --out data/ocr_training/lines --n 9000 --seed 7
"""

from __future__ import annotations

import argparse
import io
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SYS = Path("/System/Library/Fonts")
SUP = SYS / "Supplemental"
FONT_FILES = [
    # (path, ttc index, weight in the mix, typewriter-like)
    (SUP / "Times New Roman.ttf", 0, 3, False),
    (SUP / "Times New Roman Bold.ttf", 0, 1, False),
    (SYS / "Palatino.ttc", 0, 3, False),
    (SYS / "Palatino.ttc", 2, 1, False),
    (SUP / "Georgia.ttf", 0, 1, False),
    (SUP / "STIXTwoText.ttf", 0, 2, False),
    (SYS / "Supplemental" / "Baskerville.ttc", 0, 1, False),
    (SYS / "Helvetica.ttc", 0, 1, False),
    (SUP / "Arial.ttf", 0, 1, False),
    (SUP / "Courier New.ttf", 0, 2, True),
    (SYS / "Courier.ttc", 0, 2, True),
    (SUP / "AmericanTypewriter.ttc", 0, 1, True),
]

# Bold faces, used for header lines (journal tables usually set the header row in bold)
BOLD_FONTS = [
    (SUP / "Times New Roman Bold.ttf", 0),
    (SYS / "Palatino.ttc", 2),
    (SYS / "Helvetica.ttc", 1),
    (SUP / "Arial Bold.ttf", 0),
    (SUP / "Baskerville.ttc", 1),
    (SUP / "Charter.ttc", 3),
    (SUP / "Georgia Bold.ttf", 0),
]

ELEMENTS = ["Al", "Co", "Cr", "Fe", "Ni", "Ti", "V", "Nb", "Mo", "Ta", "W", "Zr", "Hf", "Cu", "Mn", "Si", "Mg",
            "Zn", "Li", "Sn", "B", "C", "Y", "Re", "Ru", "Pd", "Sc", "Ga", "Ca"]
HEADERS = [
    "σy (MPa)", "σ0.2 (MPa)", "σmax (MPa)", "σUTS (MPa)", "σ (MPa)", "εf (%)", "εmax (%)", "ε (%)", "εp (%)",
    "Yield strength (MPa)", "Ultimate tensile strength", "Tensile strength (MPa)", "Fracture strain (%)",
    "Elongation, percent", "Elongation (%)", "Reduction of area, percent", "Hardness (HV)", "Vickers hardness",
    "Hv (MPa)", "HV0.5", "HV30", "Hardness (HV0.1)", "Rp0.2 [MPa]", "Rm [MPa]", "A [%]", "E [GPa]",
    "E (GPa)", "UTS (ksi)", "YS (ksi)", "UTS in MPa", "YS in MPa", "εf in %", "MN/m²", "ksi", "ρ (g/cm³)",
    "Density, g/cm³", "Test temperature, °C", "Temperature (°C)", "Test Temp. (°F)", "Alloy", "Sample I.D.",
    "Nominal composition, wt %", "Percent EL", "Percent RA", "Compressive yield strength (MPa)",
    "Microhardness (GPa)", "Grain size (µm)", "Lattice parameter (Å)", "ΔHmix (kJ/mol)", "δ (%)",
]
MISSING = ["-", "--", "---", "N. A.", "NA", "n.a.", "–"]
# Domain words that should stay readable; mixed with random words from the base model's own wordlist
# so fine-tuning on digits and formulas does not make the model forget ordinary English.
DOMAIN_WORDS = """alloy alloys strength yield tensile ultimate hardness Vickers elongation fracture strain stress
compression compressive temperature room test tested specimen specimens sample samples composition nominal
density modulus reduction area percent average mean properties mechanical microstructure phase phases grain
cast annealed homogenized rolled forged heat treatment high entropy medium entropy refractory superalloy
measured reported values table data results summary series state condition direction longitudinal transverse
as-cast furnace cooled water quenched air vacuum argon hours minutes rate load indentation""".split()
WORDLIST: list[str] = []


@dataclass
class Run:
    text: str
    sub: bool = False  # drawn as a subscript (smaller, lowered)


def rnd_value(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.45:
        return str(rng.randint(1, 2400))
    if kind < 0.75:
        return f"{rng.uniform(0.1, 999):.{rng.choice([1, 1, 2])}f}"
    if kind < 0.85:
        v = rng.randint(1000, 9999)
        return f"{v:,}"
    return f"{rng.uniform(0, 60):.{rng.choice([1, 2])}f}"


def value_unc(rng: random.Random) -> list[Run]:
    v = rnd_value(rng)
    u = str(rng.randint(0, 150))
    if rng.random() < 0.15:
        u = f"0{rng.randint(1, 9)}"
    sep = rng.choice([" ± ", " ± ", "±", " ±", "± "])
    return [Run(f"{v}{sep}{u}")]


def number_cell(rng: random.Random) -> list[Run]:
    r = rng.random()
    if r < 0.12:
        return [Run(rng.choice(MISSING))]
    s = rnd_value(rng)
    if r < 0.25:
        s = rng.choice(["> ", ">", "< ", "~", "≈ "]) + s
    elif r < 0.35:
        s = s + rng.choice(["*", "†", "*", "a"])
    elif r < 0.42:
        s = "*" + s
    elif r < 0.5:
        s = "-" + s
    elif r < 0.56:
        s = f"{rng.randint(1, 4)}.{rng.randint(1, 9)}e{rng.choice(['', '-'])}{rng.randint(1, 5)}"
    return [Run(s)]


def alloy(rng: random.Random) -> list[Run]:
    k = rng.randint(3, 6)
    els = rng.sample(ELEMENTS, k)
    style = rng.random()
    runs: list[Run] = []
    if style < 0.25:  # equiatomic
        runs = [Run("".join(els))]
    elif style < 0.5:  # fractional subscript on one or two elements
        for e in els:
            runs.append(Run(e))
            if rng.random() < 0.35:
                runs.append(Run(rng.choice(["0.1", "0.25", "0.3", "0.5", "0.75", "1.5", "2", "x"]), sub=True))
    elif style < 0.8:  # atomic percent subscripts on every element
        spaced = rng.random() < 0.3
        for e in els:
            runs.append(Run(e))
            amt = str(rng.randint(1, 70))
            if rng.random() < 0.15:
                amt = f"0{rng.randint(1, 9)}"
            runs.append(Run(amt, sub=True))
            if spaced:
                runs.append(Run(" "))
    else:  # weight percent, hyphenated (Ti-6Al-4V style)
        base = els[0]
        parts = [base] + [f"{rng.choice(['1', '2', '3', '4', '5', '6', '8', '10', '18', '25', '35', '50', '1.5', '0.5'])}{e}"
                          for e in els[1:rng.randint(2, 4)]]
        runs = [Run("-".join(parts))]
    return runs


# Symbols as they are typeset in table headers: the qualifier is a real subscript (σ with "y" below
# the baseline), which a model trained on "σy" set inline never sees.
SYMBOLS = [("σ", "y"), ("σ", "0.2"), ("σ", "max"), ("σ", "UTS"), ("σ", "b"), ("σ", "s"), ("σ", "p"), ("σ", "f"),
           ("ε", "f"), ("ε", "max"), ("ε", "p"), ("ε", "u"), ("ε", "t"), ("R", "p0.2"), ("R", "m"), ("R", "p"),
           ("H", "v"), ("H", "V"), ("T", "m"), ("E", ""), ("A", ""), ("δ", ""), ("ρ", ""), ("ε", "")]
UNITS = {"σ": ["(MPa)", "[MPa]", "in MPa", "(GPa)", "(ksi)", ""], "ε": ["(%)", "[%]", "in %", ""],
         "R": ["[MPa]", "(MPa)", ""], "H": ["(MPa)", "(GPa)", "(HV)", ""], "T": ["(°C)", "(K)", ""],
         "E": ["(GPa)", "[GPa]"], "A": ["[%]", "(%)"], "δ": ["(%)", "[%]"], "ρ": ["(g/cm³)", "(g cm⁻³)"]}


HEADER_STYLE = "plain"  # "typeset" adds real subscripts and bold faces to header lines (see --header-style)


def header(rng: random.Random) -> list[Run]:
    if HEADER_STYLE == "plain":
        return [Run(rng.choice(HEADERS))]
    if rng.random() < 0.5:
        return [Run(rng.choice(HEADERS))]
    base, sub = rng.choice(SYMBOLS)
    unit = rng.choice(UNITS[base])
    runs = [Run(base)]
    if sub:
        runs.append(Run(sub, sub=True))
    if unit:
        runs.append(Run(" " + unit))
    if rng.random() < 0.15 and base == "σ":  # specific strength, "σmax/ρ"
        runs = [Run("σ"), Run(sub or "max", sub=True), Run("/ρ")]
    return runs


def _word(rng: random.Random) -> str:
    w = rng.choice(DOMAIN_WORDS) if (rng.random() < 0.35 or not WORDLIST) else rng.choice(WORDLIST)
    r = rng.random()
    return w.capitalize() if r < 0.15 else (w.upper() if r < 0.2 else w)


def prose(rng: random.Random) -> list[Run]:
    words = [_word(rng) for _ in range(rng.randint(2, 8))]
    text = " ".join(words)
    if rng.random() < 0.3:
        text += rng.choice([".", ",", ":", ";"])
    if rng.random() < 0.15:
        text = f"({text})"
    return [Run(text)]


def caption(rng: random.Random) -> list[Run]:
    num = rng.choice([str(rng.randint(1, 12)), rng.choice(["I", "II", "III", "IV", "V"])])
    lead = rng.choice(["Table", "TABLE", "Table", "Tab."])
    sep = rng.choice([". ", ". - ", " ", ": "])
    return [Run(f"{lead} {num}{sep}" + " ".join(_word(rng) for _ in range(rng.randint(2, 6))))]


def table_row(rng: random.Random) -> list[Run]:
    runs = alloy(rng) if rng.random() < 0.8 else [Run(f"{rng.randint(1, 3000)}")]
    for _ in range(rng.randint(2, 5)):
        runs.append(Run("\t"))  # a column gap, drawn wide and written as one space in the ground truth
        cell = value_unc(rng) if rng.random() < 0.3 else number_cell(rng)
        runs.extend(cell)
    return runs


GENERATORS = [(value_unc, 0.15), (number_cell, 0.10), (header, 0.12), (alloy, 0.13), (table_row, 0.15),
              (prose, 0.30), (caption, 0.05)]


def ground_truth(runs: list[Run]) -> str:
    return " ".join("".join(" " if r.text == "\t" else r.text for r in runs).split())


def has_glyphs(font: ImageFont.FreeTypeFont, text: str) -> bool:
    notdef = font.getmask("").getbbox()
    for ch in set(text):
        if ch.isspace():
            continue
        m = font.getmask(ch)
        if m.getbbox() is None or (m.getbbox() == notdef and ch not in "|-_"):
            return False
    return True


def render(runs: list[Run], font_path: Path, index: int, size: int, rng: random.Random, typewriter: bool) -> Image.Image | None:
    font = ImageFont.truetype(str(font_path), size, index=index)
    small = ImageFont.truetype(str(font_path), max(8, int(size * 0.68)), index=index)
    if not has_glyphs(font, "".join(r.text for r in runs)):
        return None
    pad_x, pad_y = rng.randint(6, 20), rng.randint(5, 14)
    ascent, descent = font.getmetrics()
    height = ascent + descent + 2 * pad_y + int(size * 0.2)
    widths = []
    for r in runs:
        if r.text == "\t":
            widths.append(rng.randint(int(size * 1.2), int(size * 3.5)))
        else:
            widths.append((small if r.sub else font).getlength(r.text))
    width = int(sum(widths) + 2 * pad_x)
    if HEADER_STYLE == "plain":
        bg = 255 if rng.random() > 0.15 else rng.randint(215, 245)  # some shaded header bands
    else:
        bg = 255 if rng.random() > 0.2 else rng.randint(200, 245)
    ink = rng.randint(0, 60)
    img = Image.new("L", (width, height), bg)
    d = ImageDraw.Draw(img)
    x = float(pad_x)
    base_y = pad_y
    gaps = []
    for r, w in zip(runs, widths):
        if r.text == "\t":
            gaps.append((x, x + w))
        elif r.sub:
            d.text((x, base_y + int(size * 0.32)), r.text, font=small, fill=ink)
        else:
            d.text((x, base_y), r.text, font=font, fill=ink)
        x += w
    # table rules through column gaps and along the edges
    if gaps and rng.random() < 0.45:
        for a, b in gaps:
            if rng.random() < 0.7:
                gx = int((a + b) / 2)
                d.line([(gx, 0), (gx, height)], fill=ink, width=rng.choice([1, 1, 2]))
    if rng.random() < 0.2:
        y = rng.choice([1, height - 2])
        d.line([(0, y), (width, y)], fill=ink, width=1)
    return degrade(img, rng, typewriter)


def degrade(img: Image.Image, rng: random.Random, typewriter: bool) -> Image.Image:
    if typewriter or rng.random() < 0.3:
        # a 200 dpi scan viewed at 300 dpi: lose detail, then interpolate back
        f = rng.uniform(0.5, 0.8)
        small = img.resize((max(1, int(img.width * f)), max(1, int(img.height * f))), Image.BILINEAR)
        img = small.resize(img.size, Image.BICUBIC)
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.0)))
    if typewriter and rng.random() < 0.5:
        img = img.filter(ImageFilter.MinFilter(3)) if rng.random() < 0.5 else img  # ink bleed
    arr = np.asarray(img, dtype=np.float32)
    if rng.random() < 0.5:
        arr = arr + np.random.default_rng(rng.randint(0, 10**9)).normal(0, rng.uniform(2, 12), arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.3:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=rng.randint(35, 85))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("L")
    if rng.random() < 0.3:
        img = img.rotate(rng.uniform(-0.6, 0.6), resample=Image.BICUBIC, fillcolor=255, expand=False)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/ocr_training/lines"))
    ap.add_argument("--n", type=int, default=9000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--header-style", choices=["plain", "typeset"], default="plain",
                    help="plain produced the shipped model; typeset (subscripted, bold headers) was tried and "
                         "lowered end-to-end F1 on the calibration papers")
    ap.add_argument("--wordlist", type=Path, default=Path("data/ocr_training/eng.wordlist"),
                    help="words from the base model (dawg2wordlist) for the prose lines")
    args = ap.parse_args()
    global HEADER_STYLE
    HEADER_STYLE = args.header_style
    if args.wordlist.exists():
        WORDLIST.extend(w for w in args.wordlist.read_text(encoding="utf-8").split() if w.isalpha() and 2 <= len(w) <= 12)
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    fonts = [(p, i, w, tw) for p, i, w, tw in FONT_FILES if p.exists()]
    font_weights = [w for _, _, w, _ in fonts]
    gens, gen_w = zip(*GENERATORS)
    written = attempts = 0
    while written < args.n and attempts < args.n * 3:
        attempts += 1
        gen = rng.choices(gens, weights=gen_w)[0]
        runs = gen(rng)
        path, index, _, typewriter = rng.choices(fonts, weights=font_weights)[0]
        if HEADER_STYLE == "typeset" and gen is header and rng.random() < 0.5:
            path, index = rng.choice([f for f in BOLD_FONTS if f[0].exists()])
            typewriter = False
        size = rng.randint(24, 46)
        img = render(runs, path, index, size, rng, typewriter)
        if img is None:
            continue
        stem = args.out / f"line_{written:05d}"
        text = ground_truth(runs)
        img.save(stem.with_suffix(".png"))
        stem.with_suffix(".gt.txt").write_text(text + "\n", encoding="utf-8")
        # line-level box file in Tesseract's WordStr format (what tesstrain's generate_line_box.py writes)
        stem.with_suffix(".box").write_text(
            f"WordStr 0 0 {img.width} {img.height} 0 #{unicodedata.normalize('NFC', text)}\n"
            f"\t {img.width} 0 {img.width + 1} {img.height} 0\n", encoding="utf-8")
        written += 1
    print(f"wrote {written} lines to {args.out} ({attempts} attempts, {len(fonts)} fonts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
