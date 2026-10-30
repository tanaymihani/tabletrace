"""Text that a viewer clips away: the repro PDF from scripts/make_clip_repro.py."""

from pathlib import Path

import numpy as np
import pdfplumber
import pypdfium2 as pdfium
import pytest

from tabletrace.tables_text import visible_page

PDF = Path(__file__).parent / "data" / "clipped_text.pdf"


def _ink(page_no: int, box_top_left) -> float:
    """Fraction of dark pixels in a box (PDF points, origin top-left) of the rendered page."""
    page = pdfium.PdfDocument(str(PDF))[page_no]
    img = np.asarray(page.render(scale=2).to_pil().convert("L"))
    x0, top, x1, bottom = (int(2 * v) for v in box_top_left)
    return float((img[top:bottom, x0:x1] < 128).mean())


@pytest.mark.parametrize("page_no, hidden, shown", [
    (0, (85, 135, 230, 150), (85, 85, 230, 100)),  # outside vs inside the form's /BBox
    (1, (75, 220, 200, 236), (75, 170, 200, 186)),  # outside vs inside "re W n"
])
def test_clipped_text_is_not_drawn(page_no, hidden, shown):
    assert _ink(page_no, hidden) == 0.0
    assert _ink(page_no, shown) > 0.01


def test_pdfminer_returns_the_clipped_text():
    # the premise: pdfminer (so pdfplumber) ignores both kinds of clip
    with pdfplumber.open(PDF) as pdf:
        assert "HfNbTaTiZr 1144" in pdf.pages[0].extract_text()
        assert "Outside the clip" in pdf.pages[1].extract_text()


def test_visible_page_drops_only_clipped_characters():
    with pdfplumber.open(PDF) as pdf:
        page1, n1 = visible_page(pdf.pages[0])
        page2, n2 = visible_page(pdf.pages[1])
        text1, text2 = page1.extract_text(), page2.extract_text()
    assert "1144" not in text1 and n1 == len("HfNbTaTiZr 1144")  # the space is a glyph too
    assert "HfNbTaTiZr 1155" in text1 and "Figure label" in text1  # text inside the box stays
    assert "Outside" not in text2 and n2 == len("Outside the clip")
    assert "Inside the clip" in text2 and "After Q, visible again" in text2  # Q restores the clip
