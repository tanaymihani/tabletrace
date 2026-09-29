"""Map table headers to canonical properties with an alias table plus fuzzy matching.

Rules come first because headers in this domain are short and conventional:
EN ISO 6892 uses Rp0.2 / Rm / A, Chinese and Russian papers use sigma_s,
sigma_b and delta, US reports write YS, UTS and EL. Fuzzy matching only runs
on multi-word aliases, so a two-letter token like "RA" can never drift onto
"A" (elongation).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from .normalize import clean_text, find_unit
from .schema import PropertyKind

P = PropertyKind

# Whole-token aliases (matched exactly after normalization).
SHORT_ALIASES: dict[str, PropertyKind] = {
    # hardness
    "hv": P.vickers_hardness, "hardness": P.vickers_hardness, "microhardness": P.vickers_hardness,
    "hv0.1": P.vickers_hardness, "hv0.2": P.vickers_hardness, "hv0.5": P.vickers_hardness, "hv1": P.vickers_hardness,
    "hv5": P.vickers_hardness, "hv10": P.vickers_hardness, "hv30": P.vickers_hardness, "vickers": P.vickers_hardness,
    # yield
    "ys": P.yield_strength, "σy": P.yield_strength, "σys": P.yield_strength, "σ0.2": P.yield_strength,
    "σ0,2": P.yield_strength, "rp0.2": P.yield_strength, "rp0,2": P.yield_strength, "rp": P.yield_strength,
    "σs": P.yield_strength, "σyield": P.yield_strength, "cys": P.yield_strength, "tys": P.yield_strength,
    "sy": P.yield_strength, "rel": P.yield_strength, "reh": P.yield_strength, "yield": P.yield_strength,
    # ultimate
    "uts": P.ultimate_strength, "ucs": P.ultimate_strength, "rm": P.ultimate_strength, "σuts": P.ultimate_strength,
    "σb": P.ultimate_strength, "σmax": P.ultimate_strength, "σucs": P.ultimate_strength, "σp": P.ultimate_strength,
    "σf": P.ultimate_strength, "ftu": P.ultimate_strength,
    # elongation / strain to failure
    "a": P.elongation, "el": P.elongation, "elong": P.elongation, "elongation": P.elongation, "ε": P.elongation,
    "εf": P.elongation, "εp": P.elongation, "εmax": P.elongation, "εt": P.elongation, "εu": P.elongation,
    "ductility": P.elongation, "a%": P.elongation, "e%": P.elongation, "ef": P.elongation,
    # "δ" is elongation in GB/T notation but atomic size mismatch in nearly every HEA paper, so it is not an alias
}

# Multi-word aliases (substring match, then fuzzy).
PHRASE_ALIASES: dict[str, PropertyKind] = {
    "vickers hardness": P.vickers_hardness, "vickers microhardness": P.vickers_hardness,
    "hardness value": P.vickers_hardness, "micro hardness": P.vickers_hardness,
    "yield strength": P.yield_strength, "yield stress": P.yield_strength, "proof stress": P.yield_strength,
    "0.2% proof stress": P.yield_strength, "compressive yield strength": P.yield_strength,
    "tensile yield strength": P.yield_strength, "yield point": P.yield_strength, "offset yield": P.yield_strength,
    "ultimate tensile strength": P.ultimate_strength, "tensile strength": P.ultimate_strength,
    "ultimate strength": P.ultimate_strength, "ultimate compressive strength": P.ultimate_strength,
    "compressive strength": P.ultimate_strength, "maximum strength": P.ultimate_strength,
    "peak stress": P.ultimate_strength, "maximum stress": P.ultimate_strength,
    "elongation to failure": P.elongation, "elongation at break": P.elongation, "total elongation": P.elongation,
    "fracture strain": P.elongation, "strain to failure": P.elongation, "failure strain": P.elongation,
    "strain at fracture": P.elongation, "plastic strain": P.elongation, "compressive strain": P.elongation,
    "percent el": P.elongation, "elongation percent": P.elongation, "fracture elongation": P.elongation,
}

# Things that look like properties but are not ours; recognizing them blocks false mappings.
OTHER_PROPERTIES = [
    r"reduction\s+(of|in)\s+area", r"\bra\b", r"\bpercent\s+ra\b", r"modulus", r"\be\b", r"\bg\b", r"\bgm\b",
    r"\bgcal\b", r"density", r"\bρ\b", r"/\s*ρ", r"/\s*rho", r"specific", r"notch", r"shear\s+lip",
    r"lattice", r"\bvec\b", r"\btm\b", r"melting", r"grain\s+size", r"\bδσ\b", r"\bσc\b", r"\bσm\b",
    r"strain\s+rate", r"poisson", r"toughness", r"work\s+hardening", r"\bn\b", r"creep", r"rupture",
    r"crystallite", r"\bδh", r"\bδs", r"\bω\b", r"\bχ\b", r"\bδχ\b", r"\br\b", r"\bδ\b",
]
# A bracketed unit that no target property can have ("a (Å)" is a lattice parameter, not elongation).
FOREIGN_UNIT = re.compile(r"[\(\[]\s*(Å|nm|µm|μm|mm|g\s*/\s*cm|kg\s*/\s*m|K|°C|°F|J|kJ|eV|at\.?\s*%|wt\.?\s*%|GPa\s*m|MPa\s*m)", re.I)
DESCRIPTOR_PATTERNS = {
    "id": r"\b(sample|specimen|alloy)\s*(no\.?|number|i\.?\s*d\.?)|\bno\.?$|^no\.|\bi\.?\s*d\.?$|\bnumber\b",
    "material": r"\balloys?\b|\bmaterials?\b|\bcomposition|\bmetals?\b|\bsystems?\b|\bsamples?\b|\bspecimens?\b",
    "temperature": r"\btemp(erature)?\b|\btest\s+temp",
    "composition": r"\bmole?\b|\bat\.?\s*%|\bwt\.?\s*%|\belements?\b|\bfraction\b",
    "condition": r"\bmedia\b|\bmedium\b|\bstate\b|\bcondition\b|\bprocess|\btreatment\b|\bexposure\b|\bdirection\b",
    "reference": r"\bref\.?\b|\breference",
}
EXCLUDE_PATTERN = re.compile(
    r"\bcalc\.?\b|\bcalculated\b|\bpredict(ed|ion)?\b|\btheor(y|etical)\b|\bestimated\b|\bmodel(led)?\b|"
    r"\brom\b|\bfit(ted)?\b|\bsimulat(ed|ion)\b|\bcomputed\b|\bcal\b",
    re.I,
)
COMPRESSION_PATTERN = re.compile(r"compress|\bucs\b|\bcys\b|crush", re.I)
TENSION_PATTERN = re.compile(r"tensile|tension|\buts\b|\btys\b", re.I)


@dataclass
class HeaderMatch:
    kind: str  # property | other_property | descriptor | unknown
    property: PropertyKind | None = None
    score: float = 0.0  # 0..1, how clean the match was
    alias: str | None = None
    unit: str | None = None
    loading: str | None = None
    excluded: bool = False  # calculated / predicted values, not measurements
    descriptor: str | None = None  # id | material | temperature | composition | condition | reference
    notes: list[str] = field(default_factory=list)


def normalize_header(text: str) -> str:
    """Lowercase, drop units and citations, and repair line-break hyphenation."""
    s = clean_text(text)
    s = re.sub(r"(\w)-\s+(\w)", r"\1\2", s)  # "ten- sile" -> "tensile"
    s = re.sub(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]", " ", s)  # citations [25]
    s = re.sub(r"[\(\[][^\)\]]*[\)\]]", " ", s)  # bracketed units and remarks
    s = re.sub(r"\bin\s+(mpa|gpa|ksi|%|percent)\b", " ", s, flags=re.I)
    # OCR cannot read Greek letters with the stock English model: sigma comes back as o, O, 6 or c
    # and epsilon as e, E, € or ©. Recover them where a standard suffix follows.
    s = re.sub(r"(?<![A-Za-z])[oO06c]\s*,?\s*(y|ys|0[.,]2|max|uts|ucs)\b", r"σ\1", s)
    s = re.sub(r"(?<![A-Za-z])[eE€©]\s*,?\s*(max|f)\b", r"ε\1", s)
    s = s.lower()
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)  # decimal comma in "Rp0,2" survives the punctuation strip below
    # glue subscripts that arrived as separate tokens: "σ y" -> "σy", "rp 0.2" -> "rp0.2", "r m" -> "rm"
    s = re.sub(r"\b(σ|ε|rp|r|h|δ)\s+(y|ys|0[.,]2|p0[.,]2|m|max|f|p|b|s|v|uts|ucs|t|u)\b", r"\1\2", s)
    s = re.sub(r"(σ|ε)\s+(y|0[.,]2|max|f|p|b|s|uts|ucs|t|u)\b", r"\1\2", s)
    s = s.replace("hv ", "hv ").replace("h v", "hv")
    s = re.sub(r"[,;:]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokens(s: str) -> list[str]:
    return [t.strip(".") for t in s.split() if t.strip(".")]


def map_header(text: str) -> HeaderMatch:
    """Classify one header string and map it to a property if it names one."""
    raw = clean_text(text)
    unit = find_unit(raw)
    loading = "compression" if COMPRESSION_PATTERN.search(raw) else ("tension" if TENSION_PATTERN.search(raw) else None)
    excluded = bool(EXCLUDE_PATTERN.search(raw))
    h = normalize_header(raw)
    toks = _tokens(h)
    if not h:
        return HeaderMatch("unknown", unit=unit)
    if FOREIGN_UNIT.search(raw) and unit is None:
        kind = "descriptor" if re.search(DESCRIPTOR_PATTERNS["temperature"], h) else "other_property"
        return HeaderMatch(kind, unit=None, descriptor="temperature" if kind == "descriptor" else None,
                           notes=["foreign unit"])

    # 1. exact short aliases, as whole tokens
    for tok in toks:
        alias = tok
        # HV with a test load (HV0.1 ... HV100) is a header; "Hv571" is a value with its unit
        if re.fullmatch(r"hv(0\.\d+|[1-9]\d?|100)", tok):
            alias = "hv"
        if alias in SHORT_ALIASES:
            prop = SHORT_ALIASES[alias]
            # "E" is modulus and "A" elongation only when they stand alone or with a unit;
            # a token like "a" inside a sentence-length header is weak evidence
            score = 1.0 if len(toks) <= 3 else 0.9
            if any(re.search(p, h) for p in OTHER_PROPERTIES[:3]):
                break  # "reduction of area, percent" must not become elongation via "percent"
            return HeaderMatch("property", prop, score, alias, unit, loading, excluded)

    # 2. non-target properties
    for pattern in OTHER_PROPERTIES:
        if re.search(pattern, h):
            return HeaderMatch("other_property", unit=unit, notes=[pattern])

    # 3. multi-word aliases contained in the header
    for alias, prop in sorted(PHRASE_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if alias in h:
            return HeaderMatch("property", prop, 0.95, alias, unit, loading, excluded)

    # 4. descriptors (checked before fuzzy matching so "alloy" never fuzzes into a property)
    for name, pattern in DESCRIPTOR_PATTERNS.items():
        if re.search(pattern, h):
            return HeaderMatch("descriptor", unit=unit, descriptor=name)

    # 5. fuzzy match on phrase aliases for OCR-damaged or unusual wording. Plain ratio, not
    # token_set_ratio: the latter scores "percent" as a perfect match for "percent el".
    best, best_alias = 0.0, None
    for alias in PHRASE_ALIASES:
        s = fuzz.ratio(h, alias) if len(h) >= 6 else 0.0
        if s > best:
            best, best_alias = s, alias
    if best >= 85 and best_alias:
        return HeaderMatch("property", PHRASE_ALIASES[best_alias], round(0.9 * best / 100, 3), best_alias, unit, loading, excluded,
                           notes=["fuzzy"])
    return HeaderMatch("unknown", unit=unit, loading=loading)


def caption_property(caption: str) -> HeaderMatch | None:
    """Property named by a caption, for tables whose headers are temperatures or alloys.

    Only used when no column or row header maps to a property, e.g.
    'The elevated temperature compressive yield strength (MPa) of ...'.
    """
    s = clean_text(caption).lower()
    found: dict[PropertyKind, str] = {}
    for alias, prop in sorted(PHRASE_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if alias in s and prop not in found:
            found[prop] = alias
    if not found:
        m = re.search(r"\bhardness\b", s)
        if m:
            found[P.vickers_hardness] = "hardness"
    if len(found) != 1:
        return None  # a caption naming several properties does not tell us which cells are which
    prop, alias = next(iter(found.items()))
    loading = "compression" if COMPRESSION_PATTERN.search(s) else ("tension" if TENSION_PATTERN.search(s) else None)
    return HeaderMatch("property", prop, 0.8, alias, find_unit(caption), loading, bool(EXCLUDE_PATTERN.search(s)))


def caption_loading(caption: str) -> str | None:
    s = clean_text(caption)
    if COMPRESSION_PATTERN.search(s):
        return "compression"
    if TENSION_PATTERN.search(s):
        return "tension"
    return None
