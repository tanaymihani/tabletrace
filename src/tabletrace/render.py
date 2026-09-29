"""Page rendering with pypdfium2 (no system dependency such as poppler)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

DEFAULT_DPI = 300


@lru_cache(maxsize=32)
def _document(path: str) -> pdfium.PdfDocument:
    return pdfium.PdfDocument(path)


def page_count(pdf_path: Path) -> int:
    return len(_document(str(pdf_path)))


def page_size_pts(pdf_path: Path, page: int) -> tuple[float, float]:
    """Width and height of a 1-based page in PDF points."""
    w, h = _document(str(pdf_path))[page - 1].get_size()
    return float(w), float(h)


def render_page(pdf_path: Path, page: int, dpi: int = DEFAULT_DPI, grayscale: bool = False) -> Image.Image:
    """Render a 1-based page to a PIL image at the given resolution."""
    pdf_page = _document(str(pdf_path))[page - 1]
    bitmap = pdf_page.render(scale=dpi / 72.0, grayscale=grayscale)
    img = bitmap.to_pil()
    return img.convert("L") if grayscale else img.convert("RGB")


def embedded_image_dpi(pdf_path: Path, page: int) -> float | None:
    """Resolution of the largest raster image on a page, or None if there is none.

    Scanned documents are one big image per page; rendering far above the scan's
    native resolution only interpolates pixels, so OCR uses this as a guide.
    """
    pdf_page = _document(str(pdf_path))[page - 1]
    width_pts, _ = pdf_page.get_size()
    best = None
    for obj in pdf_page.get_objects():
        if obj.type != pdfium.raw.FPDF_PAGEOBJ_IMAGE:
            continue
        px_w, _ = obj.get_px_size()
        left, _, right, _ = obj.get_pos()
        span_pts = max(right - left, 1e-6)
        dpi = px_w / (span_pts / 72.0)
        if span_pts > 0.5 * width_pts and (best is None or dpi > best):
            best = dpi
    return best
