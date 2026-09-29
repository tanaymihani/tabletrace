"""Parse numbers, uncertainties, ranges and units from table cells.

Everything here works on a single string so it can be tested in isolation.
The original cell text is always kept by the caller; parsing never mutates it.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

import pint

from .schema import CANONICAL_UNIT, PropertyKind

# --------------------------------------------------------------------------- text cleanup

_DEGREE = "◦˚º˝∘°"  # ring, ring above, ordinal, double acute (seen as a degree sign), ring operator, degree
_MINUS = "\u2212\u2012\u2013\u2014\u2010\u2011"  # minus sign, figure dash, en dash, em dash, hyphens
_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
_SCI_SUPER = re.compile(r"(?P<mant>\d+(?:\.\d+)?)\s*[×x*]\s*10\s*\^?\s*(?P<exp>[⁻\-−]?[⁰¹²³⁴-⁹\d]+)")


def clean_text(text: str) -> str:
    """Normalize the Unicode zoo found in PDFs and OCR output to plain forms."""
    if not text:
        return ""
    # scientific notation with superscript exponents must be handled before NFKC flattens them
    def _sci(m: re.Match) -> str:
        return f"{m.group('mant')}e{m.group('exp').translate(_SUPERSCRIPT).replace(chr(0x2212), '-')}"

    s = _SCI_SUPER.sub(_sci, text)
    for ch in _DEGREE:
        s = s.replace(ch, "°")
    s = unicodedata.normalize("NFKC", s)
    for ch in _MINUS:
        s = s.replace(ch, "-")
    s = s.replace("+/-", "±").replace("+-", "±")
    s = s.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    s = s.replace("° C", "°C").replace("° F", "°F")
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------- numbers

MISSING_TOKENS = {
    "", "-", "--", "---", "----", "na", "n.a", "n.a.", "n. a.", "n/a", "nd", "n.d.", "none", "\u2013", "\u2014",
}


@dataclass
class ParsedNumber:
    value: float
    uncertainty: float | None = None
    qualifier: str = "="  # "=", "<", ">", "~", "range"
    low: float | None = None
    high: float | None = None
    unit: str | None = None  # unit printed inside the cell, e.g. "Hv571"
    repaired: bool = False  # OCR character repair was needed (O->0, l->1)
    footnote: bool = False  # a footnote marker was stripped


def repair_ocr_cell(text: str) -> str:
    """Undo OCR artifacts that no table cell can legitimately contain.

    Ruling lines read as '|', '[' or ']' at the cell edges are dropped, and a '+'
    or '£' between two numbers becomes '±': property values never add two numbers.
    """
    s = clean_text(text)
    s = re.sub(r"^[\|\[\]!]+\s*|\s*[\|\[\]!]+$", "", s)
    s = re.sub(r"(?<=\d)\s*[+£]\s*(?=\d)", " ± ", s)
    return s.strip()


def is_missing(text: str) -> bool:
    s = clean_text(text).lower().replace(" ", "")
    return s in {t.replace(" ", "") for t in MISSING_TOKENS}


_NUM_TOKEN = r"[0-9OolI|][0-9OolI|,. ]*"
_FOOTNOTE_CHARS = "*†‡§#"
_CELL_UNITS = [
    (re.compile(r"^(?:hv|hv\d+(?:\.\d+)?)\s*", re.I), "HV"),  # prefix, e.g. "Hv571"
    (re.compile(r"\s*(?:hv\d*(?:\.\d+)?)$", re.I), "HV"),
    (re.compile(r"\s*mpa$", re.I), "MPa"),
    (re.compile(r"\s*gpa$", re.I), "GPa"),
    (re.compile(r"\s*ksi$", re.I), "ksi"),
    (re.compile(r"\s*%$"), "%"),
]


def _repair_ocr_digits(token: str) -> tuple[str, bool]:
    """Fix letter-for-digit OCR slips, but only inside tokens that already contain a digit."""
    if not re.search(r"\d", token) or not re.fullmatch(r"[0-9OolI|.,]+", token):
        return token, False
    fixed = token.translate(str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1"}))
    return fixed, fixed != token


def parse_decimal(token: str, decimal_comma: bool | None = None) -> float | None:
    """Parse one numeric token with either thousands or decimal-comma conventions.

    1,234.5 and 1 234,5 and 1.234,5 all mean 1234.5. A lone comma followed by
    exactly three digits is read as a thousands separator ("1,200" -> 1200)
    unless the table is known to use decimal commas.
    """
    t = token.strip()
    sign = -1.0 if t.startswith("-") else 1.0
    t = t.lstrip("+-").strip()
    if not t:
        return None
    # space as a thousands separator: "1 234" or "1 234,5"
    if re.fullmatch(r"\d{1,3}( \d{3})+([.,]\d+)?", t):
        t = t.replace(" ", "")
        if "," in t:
            t = t.replace(",", ".")
    if " " in t:
        return None
    if "," in t and "." in t:
        # whichever separator comes last is the decimal mark
        if t.rfind(",") > t.rfind("."):
            t = t.replace(".", "").replace(",", ".")
        else:
            t = t.replace(",", "")
    elif "," in t:
        parts = t.split(",")
        if decimal_comma is True:
            if len(parts) != 2:
                return None
            t = parts[0] + "." + parts[1]
        elif decimal_comma is None and len(parts) == 2 and len(parts[1]) != 3:
            t = parts[0] + "." + parts[1]  # "0,33" or "12,5": a decimal comma
        elif all(len(p) == 3 for p in parts[1:]) and 1 <= len(parts[0]) <= 3:
            t = "".join(parts)  # thousands separators
        else:
            return None
    try:
        return sign * float(t)
    except ValueError:
        return None


def parse_number(text: str, decimal_comma: bool | None = None) -> ParsedNumber | None:
    """Parse a table cell into a value, returning None if it is not a number.

    Handles 1,234.5 / 1 234,5 / 412 ± 8 / 5-50 / 1.2e3 / 1.2 x 10^3 / > 50 /
    ~23 / footnote marks (9.1*, 199.0†, *1190) and in-cell units (Hv1,049).
    """
    s = clean_text(text)
    if is_missing(s):
        return None
    footnote = False
    stripped = s.strip(_FOOTNOTE_CHARS + " ")
    if stripped != s:
        footnote = True
        s = stripped
    # trailing citation or footnote letter glued to a number: "929 [26]", "1512a"
    s2 = re.sub(r"\s*\[\d+(?:[,–-]\s*\d+)*\]$", "", s)
    if s2 != s:
        footnote, s = True, s2
    unit = None
    for pattern, name in _CELL_UNITS:
        if pattern.search(s):
            s_new = pattern.sub("", s).strip()
            if s_new and re.search(r"\d", s_new):
                unit, s = name, s_new
                break
    qualifier = "="
    m = re.match(r"^(>=|<=|≥|≤|>|<|~|≈|ca\.?|about|approx\.?)\s*", s, flags=re.I)
    if m:
        q = m.group(1).lower()
        qualifier = ">" if q in (">", ">=", "≥") else "<" if q in ("<", "<=", "≤") else "~"
        s = s[m.end():]
    repaired = False

    def num(tok: str) -> float | None:
        nonlocal repaired
        tok, fixed = _repair_ocr_digits(tok.strip())
        v = parse_decimal(tok, decimal_comma)
        if v is not None and fixed:
            repaired = True
        return v

    # value ± uncertainty
    if "±" in s:
        left, _, right = s.partition("±")
        v, u = num(left), num(right.strip(_FOOTNOTE_CHARS + " "))
        if v is None or u is None:
            return None
        return ParsedNumber(v, abs(u), qualifier, unit=unit, repaired=repaired, footnote=footnote)
    # scientific notation, already rewritten to 1.2e3 by clean_text
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)[eE]([+-]?\d+)", s)
    if m:
        return ParsedNumber(float(m.group(1)) * 10 ** int(m.group(2)), None, qualifier, unit=unit, footnote=footnote)
    # range a-b, a to b, a~b (a must be unsigned, and a < b)
    m = re.fullmatch(r"(\d[\d.,]*)\s*(?:-|to|~)\s*(\d[\d.,]*)", s)
    if m:
        a, b = num(m.group(1)), num(m.group(2))
        if a is not None and b is not None and a < b:
            return ParsedNumber((a + b) / 2, None, "range", low=a, high=b, unit=unit, repaired=repaired, footnote=footnote)
    if not re.fullmatch(r"-?" + _NUM_TOKEN, s):
        return None
    v = num(s)
    if v is None:
        return None
    return ParsedNumber(v, None, qualifier, unit=unit, repaired=repaired, footnote=footnote)


def split_pair(text: str) -> list[str] | None:
    """Split 'Calc/Exp' style cells such as '1097/4947' or '2528/-' into parts."""
    s = clean_text(text)
    if s.count("/") != 1:
        return None
    left, right = (p.strip() for p in s.split("/"))
    if not left or not right:
        return None
    if (parse_number(left) is None and not is_missing(left)) or (parse_number(right) is None and not is_missing(right)):
        return None
    return [left, right]


def detect_decimal_comma(cells: list[str]) -> bool | None:
    """Decide the decimal convention of a whole table from its numeric cells."""
    comma = point = 0
    for c in cells:
        s = clean_text(c)
        for tok in re.findall(r"\d+[.,]\d+", s):
            sep = "," if "," in tok else "."
            frac = tok.split(sep)[1]
            if sep == "," and len(frac) != 3:
                comma += 1
            elif sep == ".":
                point += 1
    if comma and not point:
        return True
    if point and not comma:
        return False
    return None


# --------------------------------------------------------------------------- units

@lru_cache(maxsize=1)
def units() -> pint.UnitRegistry:
    ureg = pint.UnitRegistry()
    # The Vickers hardness number is defined as load in kgf over indent area in mm^2.
    ureg.define("HV = kilogram_force / millimeter ** 2")
    return ureg


_UNIT_PATTERNS: list[tuple[re.Pattern, str]] = [
    # "MN/m?" and "MN/m*" are OCR readings of the superscript in MN/m²; MN/m alone is never a stress unit
    (re.compile(r"\bMN\s*/\s*(?:m|n|in)\s*(?:\^?\s*2\b|[?*\u00b0]|$|(?=\s))|\bMN\s*m-2\b", re.I), "MN/m**2"),
    (re.compile(r"\bN\s*/\s*mm\s*\^?\s*2\b", re.I), "N/mm**2"),
    (re.compile(r"\bkgf?\s*/\s*mm\s*\^?\s*2\b", re.I), "kgf/mm**2"),
    (re.compile(r"\bGPa\b", re.I), "GPa"),
    (re.compile(r"\bMPa\b", re.I), "MPa"),
    (re.compile(r"\bksi\b", re.I), "ksi"),
    (re.compile(r"\bpsi\b", re.I), "psi"),
    (re.compile(r"%|\bpercent\b|\bpct\b", re.I), "%"),
    (re.compile(r"\bH\s*V\s*\d*(?:[.,]\d+)?\b", re.I), "HV"),
]
_NON_TARGET_UNITS = re.compile(r"g\s*/\s*cm\s*\^?\s*3|\bK\b|°C|°F|\bnm\b|\bmm\b|Å|kJ|J/", re.I)


def find_unit(header: str) -> str | None:
    """Find the unit a header declares, preferring bracketed units over names.

    'Hv (MPa)' -> 'MPa' (hardness printed as a pressure), 'Rp0.2 [MPa]' -> 'MPa',
    'UTS in MPa' -> 'MPa', 'HV30' -> 'HV', 'Elongation, percent' -> '%'.
    """
    s = clean_text(header)
    for group in re.findall(r"[\(\[]([^\)\]]*)[\)\]]", s):
        for pattern, name in _UNIT_PATTERNS:
            if pattern.search(group):
                return name
    m = re.search(r"\bin\s+(\S+)", s, flags=re.I)
    if m:
        for pattern, name in _UNIT_PATTERNS:
            if pattern.search(m.group(1)):
                return name
    for pattern, name in _UNIT_PATTERNS:
        if pattern.search(s):
            return name
    return None


class UnitError(ValueError):
    pass


def to_canonical(value: float, unit: str | None, prop: PropertyKind) -> tuple[float, bool]:
    """Convert a value to the property's canonical unit.

    Returns (converted value, whether a non-trivial conversion happened). Raises
    UnitError for units that cannot describe the property, e.g. hardness in %.
    """
    target = CANONICAL_UNIT[prop]
    if unit is None:
        if prop in (PropertyKind.vickers_hardness, PropertyKind.elongation):
            return value, False  # unitless hardness is HV and unitless elongation is %, by convention
        raise UnitError(f"bad unit: no unit given for {prop.value}")
    if prop is PropertyKind.elongation:
        if unit != "%":
            raise UnitError(f"bad unit: elongation cannot be in {unit}")
        return value, False
    ureg = units()
    try:
        q = ureg.Quantity(value, unit)
        out = q.to(target).magnitude
    except (pint.DimensionalityError, pint.UndefinedUnitError) as exc:
        raise UnitError(f"bad unit: {unit} is not compatible with {prop.value} ({target})") from exc
    return float(out), abs(out - value) > 1e-9


def _to_celsius(v: float, scale: str) -> float:
    if scale == "F":
        return round((v - 32) / 1.8, 1)
    if scale == "K":
        return round(v - 273.15, 1)
    return v


def temperature_scale(text: str) -> str | None:
    """The scale a temperature header declares: 'Temperature (°C)' -> 'C'."""
    m = re.search(r"°\s*([CFK])\b|\(\s*([CFK])\s*\)", clean_text(text))
    if not m:
        return None
    return (m.group(1) or m.group(2)).upper()


def temperature_c(text: str, default_scale: str | None = None) -> float | None:
    """Pull a test temperature out of a header or row label, in Celsius.

    '800 °C' -> 800, '-423 °F' -> -252.8, '~23 °C' -> 23. A bare number such as
    '800' only counts when the caller knows the scale from a parent header.
    """
    s = clean_text(text)
    if re.search(r"\broom\s+temp|\bRT\b", s, flags=re.I):
        return None
    # "800 °C", or "800 C" with a space (so compositions like "Fe-0.5C" never read as temperatures)
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*°\s*([CFK])\b|(-?\d+(?:\.\d+)?)\s+([CFK])\b", s)
    if m:
        v = float(m.group(1) or m.group(3))
        return _to_celsius(v, (m.group(2) or m.group(4)).upper())
    if default_scale:
        m = re.fullmatch(r"[~≈]?\s*(-?\d+(?:\.\d+)?)", s)
        if m:
            return _to_celsius(float(m.group(1)), default_scale)
    return None
