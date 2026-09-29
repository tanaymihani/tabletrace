"""Tesseract OCR: word text, boxes and confidence, returned in PDF-point coordinates."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytesseract
from PIL import Image, ImageOps

from .grid import BBox, Word

if shutil.which("tesseract") is None:  # pragma: no cover - environment check
    raise RuntimeError("Tesseract is not installed. On macOS: brew install tesseract")

# The fine-tuned model lives in models/tessdata. data/ocr_training/tessdata holds the float
# tessdata_best model it started from (eng_best), kept only as an experimental control.
# Anything else comes from Tesseract's own tessdata directory.
_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = _ROOT / "models" / "tessdata"
MODEL_DIRS = [MODELS_DIR, _ROOT / "data" / "ocr_training" / "tessdata"]


def model_fingerprint(lang: str) -> str:
    """Identify the exact model file behind a language name, so caches never outlive a retrain."""
    for d in MODEL_DIRS:
        f = d / f"{lang}.traineddata"
        if f.exists():
            st = f.stat()
            return f"{lang}:{st.st_size}:{int(st.st_mtime)}"
    return f"{lang}:system"


def _tess_config(psm: int, lang: str, extra: str = "") -> str:
    config = f"--psm {psm} --oem 1 {extra}".strip()
    for d in MODEL_DIRS:
        if (d / f"{lang}.traineddata").exists():
            return f"--tessdata-dir {d} " + config
    return config


def ocr_words(img: Image.Image, dpi: int, psm: int = 6, offset_pts: tuple[float, float] = (0.0, 0.0),
              min_conf: float = 0.0, lang: str = "eng") -> list[Word]:
    """Run Tesseract on an image and return words in PDF points.

    offset_pts is the position of the image's top-left corner on the page, so a
    crop of a table can be OCR'd and its words placed back on the page.
    """
    scale = 72.0 / dpi
    data = pytesseract.image_to_data(
        img, lang=lang, config=_tess_config(psm, lang, "-c preserve_interword_spaces=1"),
        output_type=pytesseract.Output.DICT,
    )
    ox, oy = offset_pts
    words = []
    for i, text in enumerate(data["text"]):
        text = (text or "").strip()
        conf = float(data["conf"][i])
        if not text or conf < 0:
            continue
        if conf / 100.0 < min_conf:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        words.append(Word(text, ox + x * scale, oy + y * scale, ox + (x + w) * scale, oy + (y + h) * scale, conf / 100.0,
                          h * scale))
    return words


def prepare(img: Image.Image, upscale: float = 1.0, binarize: bool = False) -> Image.Image:
    """Grayscale, optional upscale for small print, optional Otsu binarization for scans."""
    g = ImageOps.grayscale(img)
    if upscale != 1.0:
        g = g.resize((int(g.width * upscale), int(g.height * upscale)), Image.LANCZOS)
    if binarize:
        arr = np.asarray(g, dtype=np.uint8)
        hist = np.bincount(arr.ravel(), minlength=256).astype(float)
        total = arr.size
        sum_all = float(np.dot(np.arange(256), hist))
        w_b = sum_b = 0.0
        best_t, best_var = 128, -1.0
        for t in range(256):
            w_b += hist[t]
            if w_b == 0 or w_b == total:
                continue
            sum_b += t * hist[t]
            m_b = sum_b / w_b
            m_f = (sum_all - sum_b) / (total - w_b)
            var = w_b * (total - w_b) * (m_b - m_f) ** 2
            if var > best_var:
                best_var, best_t = var, t
        g = Image.fromarray(((arr > best_t) * 255).astype(np.uint8))
    return g


def ocr_region(page_img: Image.Image, dpi: int, bbox: BBox, psm: int = 6, pad_pts: float = 4.0,
               upscale: float = 1.0, binarize: bool = False, remove_rules: bool = False,
               lang: str = "eng") -> list[Word]:
    """OCR one region of a rendered page (a table) and return words in page coordinates."""
    s = dpi / 72.0
    x0, top, x1, bottom = bbox
    box = (max(0, int((x0 - pad_pts) * s)), max(0, int((top - pad_pts) * s)),
           min(page_img.width, int((x1 + pad_pts) * s)), min(page_img.height, int((bottom + pad_pts) * s)))
    crop = prepare(page_img.crop(box), upscale=upscale, binarize=binarize)
    if remove_rules:
        crop = erase_rules(crop)
    return ocr_words(crop, int(dpi * upscale), psm=psm, offset_pts=(box[0] / s, box[1] / s), lang=lang)


def erase_rules(img: Image.Image, frac: float = 0.5) -> Image.Image:
    """Paint over ruling lines: pixel rows or columns that are dark across most of the image.

    Table borders make Tesseract's block segmentation give up (psm 6 returns
    nothing on a fully ruled table); text lines never fill half the width.
    """
    arr = np.asarray(ImageOps.grayscale(img), dtype=np.uint8).copy()
    dark = arr < 128
    bg = int(np.median(arr))
    rows = dark.mean(axis=1) > frac
    cols = dark.mean(axis=0) > frac
    arr[rows, :] = bg
    arr[:, cols] = bg
    return Image.fromarray(arr)


NUMERIC_WHITELIST = "0123456789.,±+-~<>*()%/ "


def ocr_cell(page_img: Image.Image, dpi: int, bbox: BBox, upscale: float = 2.0, numeric: bool = False,
             multiline: bool = False, inset_pts: float = 0.6, lang: str = "eng") -> list[Word]:
    """OCR a single table cell, isolated from its neighbours and from the grid lines.

    Single-line cells use psm 7. `numeric` restricts the character set for body
    cells that should hold numbers, which removes O/0 and l/1 confusions.
    """
    s = dpi / 72.0
    x0, top, x1, bottom = bbox
    box = (max(0, int((x0 + inset_pts) * s)), max(0, int((top + inset_pts) * s)),
           min(page_img.width, int((x1 - inset_pts) * s)), min(page_img.height, int((bottom - inset_pts) * s)))
    if box[2] - box[0] < 4 or box[3] - box[1] < 4:
        return []
    crop = erase_rules(prepare(page_img.crop(box), upscale=upscale), frac=0.6)
    arr = np.asarray(crop)
    if (arr < 128).mean() < 0.002:  # blank cell
        return []
    # a white margin helps Tesseract find the text line
    pad = int(6 * upscale)
    canvas = Image.new("L", (crop.width + 2 * pad, crop.height + 2 * pad), 255)
    canvas.paste(crop, (pad, pad))
    psm = 6 if multiline else 7
    extra = f"-c tessedit_char_whitelist={NUMERIC_WHITELIST.replace(' ', '')}" if numeric else ""
    data = pytesseract.image_to_data(canvas, lang=lang, config=_tess_config(psm, lang, extra),
                                     output_type=pytesseract.Output.DICT)
    scale = 72.0 / (dpi * upscale)
    ox, oy = box[0] / s - pad * scale, box[1] / s - pad * scale
    words = []
    for i, text in enumerate(data["text"]):
        text = (text or "").strip()
        conf = float(data["conf"][i])
        if not text or conf < 0:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        words.append(Word(text, ox + x * scale, oy + y * scale, ox + (x + w) * scale, oy + (y + h) * scale, conf / 100.0,
                          h * scale))
    return words
