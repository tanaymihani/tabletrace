"""Decide per page whether to trust the text layer or to OCR the rendered image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .tables_text import text_layer_info

MIN_CHARS = 50


@dataclass
class RouteDecision:
    page: int
    route: str  # text_layer | ocr
    reason: str
    n_chars: int
    n_ghost: int
    vendor_ocr: bool  # a scan that ships with someone else's OCR text layer


def route_page(pdf_path: Path, page: int) -> RouteDecision:
    info = text_layer_info(pdf_path, page)
    if info.n_chars < MIN_CHARS:
        return RouteDecision(page, "ocr", "no usable text layer", info.n_chars, info.n_ghost, False)
    if info.full_page_image:
        # A full-page raster with text on top is a scan with an OCR layer. The layer is kept
        # as a second opinion, but the primary reading comes from our own OCR.
        return RouteDecision(page, "ocr", "scanned page with a vendor OCR layer", info.n_chars, info.n_ghost, True)
    return RouteDecision(page, "text_layer", "born-digital text layer", info.n_chars, info.n_ghost, False)
