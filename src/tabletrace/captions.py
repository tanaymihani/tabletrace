"""Link detected tables to their printed captions ("Table 3.", "TABLE I. - ...")."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .grid import BBox, Word, join_words

# "Table 3." / "TABLE I. - TENSILE" / "Table 2 The Vicker's..." are captions; a sentence such as
# "Table 5 shows the effect of boron" is body text, recognisable by the lowercase word after the number.
CAPTION = re.compile(r"^\s*(?:table|tab\.)\s*([0-9]{1,2}|[ivxl]{1,6})\b(?!\s+(?-i:[a-z]))", re.I)
ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50}


def roman_to_int(s: str) -> int | None:
    s = s.lower()
    if not s or any(ch not in ROMAN for ch in s):
        return None
    total, prev = 0, 0
    for ch in reversed(s):
        v = ROMAN[ch]
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total


@dataclass
class Caption:
    number: int
    text: str
    bbox: BBox


def _lines(words: list[Word]) -> list[list[Word]]:
    lines: list[list[Word]] = []
    for w in sorted(words, key=lambda w: (w.cy, w.x0)):
        for line in lines:
            ref = line[0]
            ov = min(w.bottom, ref.bottom) - max(w.top, ref.top)
            if ov > 0.5 * min(w.bottom - w.top, ref.bottom - ref.top) and abs(w.x0 - line[-1].x1) < 60:
                line.append(w)
                break
        else:
            lines.append([w])
    for line in lines:
        line.sort(key=lambda w: w.x0)
    return sorted(lines, key=lambda ln: (ln[0].top, ln[0].x0))


def find_captions(words: list[Word]) -> list[Caption]:
    """Every caption on the page, with up to two continuation lines."""
    lines = _lines(words)
    caps = []
    for i, line in enumerate(lines):
        text = join_words(line)
        m = CAPTION.match(text)
        if not m:
            continue
        tok = m.group(1)
        # OCR often reads the roman numeral I as a lowercase l or the digit 1
        number = int(tok) if tok.isdigit() else roman_to_int(tok.replace("l", "i"))
        if number is None:
            continue
        body = list(line)
        for nxt in lines[i + 1:i + 3]:
            gap = nxt[0].top - max(w.bottom for w in body)
            same_column = nxt[0].x0 < max(w.x1 for w in body) and nxt[-1].x1 > min(w.x0 for w in body)
            if 0 <= gap < 1.2 * (line[0].bottom - line[0].top) and same_column and not CAPTION.match(join_words(nxt)):
                body.extend(nxt)
            else:
                break
        bbox = (min(w.x0 for w in body), min(w.top for w in body), max(w.x1 for w in body), max(w.bottom for w in body))
        caps.append(Caption(number, join_words(body), bbox))
    return caps


def link_caption(captions: list[Caption], table: BBox, max_above: float = 90.0, max_below: float = 60.0) -> Caption | None:
    """The caption closest above the table (or just below it, Nature style) that overlaps it horizontally."""
    x0, top, x1, bottom = table
    best, best_d = None, float("inf")
    for cap in captions:
        cx0, ctop, cx1, cbottom = cap.bbox
        if min(cx1, x1) - max(cx0, x0) <= 0:
            continue
        if cbottom <= top + 0.25 * (bottom - top) and top - cbottom < max_above:
            d = max(0.0, top - cbottom)  # above, or inside the top of the box
        elif ctop >= bottom - 5 and ctop - bottom < max_below:
            d = (ctop - bottom) + 20.0  # below: allowed, but an equally close caption above wins
        else:
            continue
        if d < best_d:
            best, best_d = cap, d
    return best
