"""Turn a table grid into validated property records.

The grid says nothing about meaning. This module decides which rows are
headers, which columns hold sample labels, whether properties run across the
columns or down the rows (transposed tables are common in this literature),
and then reads every value cell with its header path, unit and conditions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from pydantic import ValidationError

from .grid import TableGrid, split_merged_rows
from .mapping import HeaderMatch, caption_loading, caption_property, map_header
from .normalize import (
    ParsedNumber,
    UnitError,
    clean_text,
    detect_decimal_comma,
    find_unit,
    is_missing,
    parse_number,
    repair_ocr_cell,
    split_pair,
    temperature_c,
    temperature_scale,
    to_canonical,
)
from .schema import CANONICAL_UNIT, ExtractedValue, QuarantinedCell, quarantine_reason

SUMMARY_ROW = re.compile(r"^\s*(averages?|mean|avg\.?|total)\s*:?\s*$", re.I)
SI_UNITS = {"MPa", "MN/m**2", "N/mm**2", "GPa"}
CALC_WORDS = re.compile(r"calc|pred|theor|rom|model", re.I)


@dataclass
class Interpretation:
    values: list[ExtractedValue] = field(default_factory=list)
    quarantine: list[QuarantinedCell] = field(default_factory=list)
    orientation: str = ""
    header_rows: int = 0
    stub_cols: int = 0
    title: str = ""
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- cell classification

def is_valueish(text: str) -> bool:
    """A cell that holds a number, a missing marker, or a 'Calc/Exp' pair."""
    if not text:
        return False
    return is_missing(text) or parse_number(text) is not None or split_pair(text) is not None


def row_is_valueish(row: list[str], skip_first: bool = True) -> bool:
    cells = [t for t in (row[1:] if skip_first else row) if t]
    return bool(cells) and sum(is_valueish(t) for t in cells) / len(cells) >= 0.5


def drop_empty(texts: list[list[str]]) -> tuple[list[list[str]], list[int], list[int]]:
    rows = [i for i, r in enumerate(texts) if any(r)]
    cols = [j for j in range(len(texts[0]) if texts else 0) if any(texts[i][j] for i in rows)]
    return [[texts[i][j] for j in cols] for i in rows], rows, cols


def split_title_rows(texts: list[list[str]]) -> tuple[list[list[str]], str]:
    """Remove leading rows that hold one text across the whole table (a title, not a header)."""
    title = []
    while texts:
        vals = [t for t in texts[0] if t]
        distinct = set(vals)
        spans_all = len(distinct) == 1 and len(vals) >= max(2, int(0.7 * len(texts[0])))
        lone_long = len(vals) == 1 and len(vals[0]) > 30 and len(texts[0]) > 2
        if not (spans_all or lone_long) or row_is_valueish(texts[0], skip_first=False):
            break
        title.append(vals[0])
        texts = texts[1:]
    return texts, " ".join(title)


def find_header_rows(texts: list[list[str]], hint: int | None = None) -> int:
    """Rows above the first body row.

    A row stays in the header while it is text, or while it is numeric but has
    no row label and the next row is text again (grouping rows such as the
    '23 | 500 | 650' test temperatures above a second header row).
    """
    n = len(texts)
    k = 0
    while k < n:
        row = texts[k]
        if not row_is_valueish(row):
            k += 1
            continue
        if not row[0] and k + 1 < n and not row_is_valueish(texts[k + 1]):
            k += 1
            continue
        break
    if k == n:
        return hint or 1
    return k


def header_paths(texts: list[list[str]], header_rows: int) -> list[str]:
    """Join header rows top to bottom per column, carrying group labels across empty cells."""
    n_cols = len(texts[0]) if texts else 0
    if header_rows == 0:
        return [""] * n_cols
    grid = [list(texts[r]) for r in range(header_rows)]
    # A group label in an upper header row covers the empty cells to its right when the
    # rows below carry sub-headers there ("UTS in MPa" over "CC | DS").
    for r in range(header_rows - 1):
        for c in range(1, n_cols):
            left = grid[r][c - 1]
            if grid[r][c] or not left or not any(grid[rr][c] for rr in range(r + 1, header_rows)):
                continue
            m = map_header(left)
            group = (m.kind == "property" or temperature_scale(left) is not None or is_valueish(left)
                     or re.search(r"temperature|tested|longitudinal|transverse", left, re.I))
            if group:
                grid[r][c] = left
    paths = []
    for c in range(n_cols):
        parts: list[str] = []
        for r in range(header_rows):
            t = grid[r][c]
            if t and (not parts or parts[-1] != t):
                parts.append(t)
        paths.append(" | ".join(parts))
    return paths


# --------------------------------------------------------------------------- header analysis

@dataclass
class ColumnInfo:
    index: int
    path: str
    match: HeaderMatch | None  # property named by the header, if any
    temp_c: float | None = None
    condition: str = ""  # header parts that are neither the property, a unit nor a temperature


def analyse_path(path: str, index: int) -> ColumnInfo:
    parts = [p.strip() for p in path.split(" | ") if p.strip()]
    # The most specific header part decides: walking up from the bottom row, the first part that
    # names any property (ours or not) wins, so "Reduction of area" under a group label that
    # happens to mention elongation is still reduction of area.
    best, best_part = None, None
    for p in reversed(parts):
        m = map_header(p)
        if m.kind == "property":
            best, best_part = m, p
            break
        if m.kind == "other_property":
            break
    if best is None and len(parts) > 1:
        m = map_header(" ".join(parts))  # a name broken over two header rows ("Hardness" / "(HV)")
        if m.kind == "property":
            best, best_part = m, " ".join(parts)
    info = ColumnInfo(index, path, best)
    if best is not None and best.unit is None:
        # The unit may sit in a sub-header of its own ("MN/m2" under "Ultimate tensile strength").
        # Look below the property first, then above it, and never take a unit from a temperature
        # group label: OCR turns the superscript in "°C" into "%" often enough to matter.
        k = parts.index(best_part) if best_part in parts else len(parts)
        for p in parts[k + 1:] + parts[:k][::-1]:
            if re.search(r"\btemp", p, re.I):
                continue
            unit = find_unit(p)
            if unit:
                best.unit = unit
                break
    scale = next((temperature_scale(p) for p in parts if temperature_scale(p)), None)
    if scale is None and any(re.search(r"\btemp(erature)?\b[^|]{0,8}%", p, re.I) for p in parts):
        scale = "C"  # "Test temperature, %" is an OCR reading of "Test temperature, °C"
    others = []
    for p in parts:
        if p == best_part:
            continue
        t = temperature_c(p)
        if t is None and scale:
            t = _group_temperature(p, scale)
        if t is not None:
            info.temp_c = t
            rest = re.sub(r"\(?\s*[~≈]?-?\d+(\.\d+)?\s*°?\s*[CFK]\s*\)?", "", clean_text(p)).strip(" ,()")
            if rest and not re.fullmatch(r"(test\s+)?temp(erature)?\.?,?", rest, re.I):
                others.append(rest)
            continue
        if re.search(r"^(test\s+)?temperature", p, re.I) and temperature_scale(p):
            continue  # the "Test temperature, °C" group label itself
        if find_unit(p) and len(clean_text(p)) <= 8:
            continue  # a unit-only sub-header such as "MN/m2" or "ksi"
        others.append(p)
    info.condition = "; ".join(others)
    return info


def _group_temperature(part: str, scale: str) -> float | None:
    """A temperature group label under a declared scale: '800', '~23', or OCR'd '650 | oe'."""
    s = clean_text(part)
    if re.fullmatch(r"[~\u2248]?\s*-?\d+(\.\d+)?", s):
        return temperature_c(s, scale)
    m = re.match(r"^\W*(-?\d{2,4})\b", s)
    groups = re.findall(r"\d+", s)
    letters = len(re.findall(r"[A-Za-z]", s))
    if m or (len(groups) == 1 and 2 <= len(groups[0]) <= 4 and letters <= 3):
        return temperature_c(m.group(1) if m else groups[0], scale)
    return None


_OCR_DIGIT = str.maketrans({"O": "0", "o": "0", "S": "5", "l": "1", "I": "1"})


def repair_alloy_name(s: str) -> str:
    """Undo OCR letter-for-digit slips that alloy nomenclature makes detectable.

    Between a hyphen and an element symbol there can only be an amount:
    'Ti-SONb' is Ti-50Nb and 'Ti-50V-l.5Si' is Ti-50V-1.5Si.
    """
    def fix(m: re.Match) -> str:
        cand = m.group(1).translate(_OCR_DIGIT)
        return cand if re.fullmatch(r"\d+(\.\d+)?", cand) else m.group(1)

    return re.sub(r"(?<=-)([0-9OoSlI.]{1,5})(?=[A-Z][a-z]?)", fix, s)


def clean_material(text: str, ocr: bool = False) -> str:
    s = clean_text(text)
    if ocr:
        s = repair_alloy_name(s)
    s = re.sub(r"\s*\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]", "", s)  # citations like [25]
    s = re.sub(r"(?<=[A-Za-z0-9])\s*[\*†‡]+$", "", s)  # footnote marks
    if re.search(r"[A-Z][a-z]?-\d", s):  # 'Ti-50Nb-2Si a' -> drop a trailing footnote letter
        s = re.sub(r"(?<=[0-9A-Za-z])\s?\^?[a-d]$", "", s)
    s = re.sub(r"(\d)\.\s+(\d)", r"\1.\2", s)  # 'Ti-50V-1. 5Si' (typewriter spacing)
    return s.strip(" :;,")


def _is_id_or_label_header(m: HeaderMatch, path: str) -> bool:
    """Descriptor headers ('Alloy No.', 'Sample I.D.', 'Test Temp.') describe a label column.

    A header that itself contains an identifier ('Alloy No. 21') names a sample
    in a transposed table instead.
    """
    return m.kind == "descriptor" and not re.search(r"\d", re.sub(r"\(.*?\)", "", path))


# --------------------------------------------------------------------------- main entry

def interpret(grid: TableGrid, paper_id: str, route: str) -> Interpretation:
    out = Interpretation()
    grid = replace(grid, cells=split_merged_rows(grid.cells))
    fix = repair_ocr_cell if grid.source == "ocr" else clean_text
    raw = [[fix(c.text) for c in row] for row in grid.cells]
    if not raw or not raw[0]:
        return out
    texts, row_ids, col_ids = drop_empty(raw)
    n_title = len(texts)
    texts, title = split_title_rows(texts)
    row_ids = row_ids[n_title - len(texts):]
    out.title = title
    if len(texts) < 2 or len(texts[0]) < 2:
        out.notes.append("too small")
        return out

    header_rows = find_header_rows(texts, grid.header_rows)
    out.header_rows = header_rows
    paths = header_paths(texts, header_rows)
    n_cols = len(texts[0])
    body = texts[header_rows:]
    caption = clean_text(" ".join(x for x in (grid.caption, title) if x))
    cap_loading = caption_loading(caption)
    decimal_comma = detect_decimal_comma([t for r in body for t in r])

    # ---- stub columns: sample labels and conditions on the left
    n_stub = 0
    for c in range(n_cols - 1):
        cells = [r[c] for r in body if r[c]]
        m = map_header(paths[c]) if paths[c] else HeaderMatch("unknown")
        text_frac = sum(not is_valueish(t) for t in cells) / len(cells) if cells else 0.0
        is_label = text_frac >= 0.5 or _is_id_or_label_header(m, paths[c])
        if m.kind == "property":
            is_label = False  # a column headed by a property holds values, even when OCR garbled them
        if c == 0 and not paths[c] and cells and not is_label:
            # an unlabeled first column of numbers is a row label only if it holds temperatures
            is_label = all(temperature_c(t) is not None for t in cells)
        if not is_label:
            break
        n_stub += 1
    out.stub_cols = n_stub

    # ---- fold multi-line labels into the row they continue
    if n_stub:
        merged, origin = [], []
        for i, row in enumerate(body):
            if merged and not any(row[n_stub:]) and any(row[:n_stub]):
                for j in range(n_stub):
                    if row[j]:
                        merged[-1][j] = (merged[-1][j] + " " + row[j]).strip()
                continue
            merged.append(list(row))
            origin.append(i)
        body = merged
    else:
        origin = list(range(len(body)))

    # ---- roles of the stub columns
    stub_meta = [map_header(paths[c]) if paths[c] else HeaderMatch("unknown") for c in range(n_stub)]
    material_col = None
    temp_cols: list[tuple[int, str | None]] = []
    for c in range(n_stub):
        cells = [r[c] for r in body if r[c]]
        looks_temp = stub_meta[c].descriptor == "temperature" or (cells and all(temperature_c(t) is not None for t in cells))
        textual = bool(cells) and sum(not is_valueish(t) for t in cells) / len(cells) >= 0.5
        if looks_temp:
            temp_cols.append((c, temperature_scale(paths[c])))
        elif material_col is None and stub_meta[c].descriptor not in ("condition", "reference") \
                and (stub_meta[c].descriptor != "composition" or textual):
            # the first label column names the sample: an alloy name, or an ID when the table has one.
            # A "composition" header counts only over text ("Nominal composition, wt %" above
            # "Ti-25Nb"), not over numbers ("Minor element (mole %)" above "0.4").
            material_col = c

    # ---- orientation
    col_info = [analyse_path(paths[c], c) for c in range(n_cols)]
    col_props = sum(1 for c in range(n_stub, n_cols) if col_info[c].match is not None)
    row_matches = [map_header(r[0]) if r[0] else HeaderMatch("unknown") for r in body]
    row_props = sum(1 for m in row_matches if m.kind == "property")
    cap_match = caption_property(caption) if caption else None
    if row_props > col_props:
        out.orientation = "transposed"
    elif col_props:
        out.orientation = "normal"
    elif cap_match is not None:
        out.orientation = "caption"
    else:
        out.orientation = "none"
        out.notes.append("no property headers")
        return out

    ctx = _Context(grid, paper_id, route, out, decimal_comma, cap_loading, row_ids, col_ids, header_rows, origin)
    if out.orientation == "transposed":
        _read_transposed(ctx, body, row_matches, col_info, n_cols)
    else:
        _read_normal(ctx, body, col_info, n_stub, n_cols, material_col, temp_cols,
                     cap_match if out.orientation == "caption" else None)
    return out


@dataclass
class _Context:
    grid: TableGrid
    paper_id: str
    route: str
    out: Interpretation
    decimal_comma: bool | None
    cap_loading: str | None
    row_ids: list[int]
    col_ids: list[int]
    header_rows: int
    origin: list[int]

    def cell(self, r: int, c: int):
        src_r = self.row_ids[self.header_rows + self.origin[r]]
        src_c = self.col_ids[c]
        return src_r, src_c, self.grid.cells[src_r][src_c]


def _read_normal(ctx: _Context, body, col_info, n_stub, n_cols, material_col, temp_cols, cap_match) -> None:
    prop_cols = []
    for c in range(n_stub, n_cols):
        info = col_info[c]
        match = info.match
        if cap_match is not None:  # caption mode: every column shares the caption's property
            match = HeaderMatch("property", cap_match.property, cap_match.score, cap_match.alias,
                                cap_match.unit or find_unit(info.path), cap_match.loading, cap_match.excluded)
        if match is None or match.excluded:
            continue
        prop_cols.append((c, match, info))

    # dual-unit columns (MN/m2 next to ksi for the same test): keep the SI one, compare the other
    twins = _infer_dual_units(prop_cols, body, ctx.decimal_comma)
    paired = set(twins.values())
    keep = []
    for c, match, info in prop_cols:
        if c in paired:
            continue
        if c in twins:
            keep.append((c, match, info))
            continue
        twin = next((k for k in keep if k[1].property == match.property and k[2].temp_c == info.temp_c
                     and k[2].condition == info.condition and k[1].unit != match.unit and c - k[0] == 1), None)
        if twin is None:
            keep.append((c, match, info))
        elif twin[1].unit in SI_UNITS:
            twins[twin[0]] = c
        else:
            keep.remove(twin)
            keep.append((c, match, info))
            twins[c] = twin[0]

    last_material = ""
    for r, row in enumerate(body):
        if any(SUMMARY_ROW.match(t) for t in row[:max(n_stub, 1)] if t):
            continue
        replicate = False
        if material_col is not None:
            if row[material_col]:
                last_material = clean_material(row[material_col], ctx.grid.source == "ocr")
            else:
                replicate = True
        row_temp = None
        for tc, scale in temp_cols:
            if row[tc]:
                row_temp = temperature_c(row[tc])
                if row_temp is None and scale:
                    row_temp = temperature_c(row[tc], scale)
        stub_conditions = [row[c] for c in range(n_stub) if c != material_col and row[c]
                           and all(c != tc for tc, _ in temp_cols)]
        for c, match, info in keep:
            text = row[c]
            if not text:
                continue
            if material_col is not None:
                material, cond = last_material, [info.condition] if info.condition else []
            else:
                # no label column: the sample is named in the header ("CC"/"DS", or the alloy itself)
                material, cond = clean_material(info.condition or info.path, ctx.grid.source == "ocr"), []
            cond += stub_conditions
            if replicate:
                cond.append("replicate")
            temp = info.temp_c if info.temp_c is not None else row_temp
            agree = _unit_agreement(ctx, text, match, row[twins[c]], col_info[twins[c]].match) if c in twins else None
            _emit(ctx, material, match, text, r, c, info.path, temp, "; ".join(x for x in cond if x), agree)


KSI_PER_MPA = 6.894757
STRENGTHS = {"yield_strength", "ultimate_strength"}


def _infer_dual_units(prop_cols, body, decimal_comma) -> dict[int, int]:
    """Recover units OCR could not read from the values themselves; returns SI -> twin columns.

    US reports of the 1970s printed every stress twice, in MN/m2 and in ksi. Two
    neighbouring columns of the same strength whose values keep a ratio of
    6.895 row after row can only be that pair, whatever the unit cells say.
    """
    pairs: dict[int, int] = {}
    for (c1, m1, _), (c2, m2, _) in zip(prop_cols, prop_cols[1:]):
        if c2 - c1 != 1 or m1.property != m2.property or m1.property.value not in STRENGTHS:
            continue
        ratios = []
        for row in body:
            a, b = parse_number(row[c1], decimal_comma), parse_number(row[c2], decimal_comma)
            if a and b and a.value > 0 and b.value > 0:
                ratios.append(a.value / b.value)
        if len(ratios) < 3:
            continue
        ratios.sort()
        med = ratios[len(ratios) // 2]
        if abs(med / KSI_PER_MPA - 1) < 0.05:
            m1.unit = m1.unit if m1.unit in SI_UNITS else "MN/m**2"
            m2.unit = "ksi"
            pairs[c1] = c2
        elif abs(med * KSI_PER_MPA - 1) < 0.05:
            m1.unit = "ksi"
            m2.unit = m2.unit if m2.unit in SI_UNITS else "MN/m**2"
            pairs[c2] = c1
    return pairs


_FORMULA = re.compile(r"^(?:[A-Z][a-z]?(?:\d+(?:\.\d+)?)?)+$")  # element symbols with optional amounts


def _read_transposed(ctx: _Context, body, row_matches, col_info, n_cols) -> None:
    for r, row in enumerate(body):
        m = row_matches[r]
        if m.kind != "property" or m.excluded:
            continue
        for c in range(1, n_cols):
            text = row[c]
            if not text:
                continue
            info = col_info[c]
            header = row[0]
            parts = [x for x in info.path.split(" | ") if x] if info.path else []
            label = parts[-1] if parts else ""
            if len(parts) >= 2 and all(_FORMULA.match(re.sub(r"\s*\[[\d,\s–-]+\]|\s", "", x)) for x in parts[-2:]):
                label = parts[-2] + label  # a sample name broken over two header rows ("HfMoNb" over "TaTiZr [25]")
            pair_names = re.search(r"\(\s*([A-Za-z.]+)\s*/\s*([A-Za-z.]+)\s*\)", label)
            if pair_names:  # "MoNbTaV (Calc/Exp)"
                header = f"{row[0]} {pair_names.group(0)}"
                label = label.replace(pair_names.group(0), "")
            _emit(ctx, clean_material(label, ctx.grid.source == "ocr"), m, text, r, c, header, info.temp_c, "")


def _emit(ctx: _Context, material: str, match: HeaderMatch, text: str, r: int, c: int, header: str,
          temp: float | None, condition: str, unit_agree: float | None = None) -> None:
    out, grid = ctx.out, ctx.grid
    src_r, src_c, cell = ctx.cell(r, c)
    base = dict(paper_id=ctx.paper_id, page=grid.page, table_index=grid.table_index, material=material,
                property=match.property.value, raw_text=text, raw_unit=match.unit, header=header,
                source=grid.source, row=src_r, col=src_c)
    if not material.strip():
        out.quarantine.append(QuarantinedCell(**base, reason="missing material"))
        return
    cand = text
    pair = split_pair(text)
    if pair is not None:
        # 'Calc/Exp' cells: keep the component the header does not call calculated
        names = re.search(r"\(\s*([A-Za-z.]+)\s*/\s*([A-Za-z.]+)\s*\)", header)
        keep = 1
        if names and CALC_WORDS.search(names.group(2)) and not CALC_WORDS.search(names.group(1)):
            keep = 0
        cand = pair[keep]
    if is_missing(cand):
        return
    parsed: ParsedNumber | None = parse_number(cand, ctx.decimal_comma)
    if parsed is None:
        out.quarantine.append(QuarantinedCell(**base, reason="non-numeric value", detail=text))
        return
    unit = parsed.unit or match.unit
    prop = match.property
    try:
        value, converted = to_canonical(parsed.value, unit, prop)
        unc = to_canonical(parsed.uncertainty, unit, prop)[0] if parsed.uncertainty is not None else None
    except UnitError as exc:
        out.quarantine.append(QuarantinedCell(**{**base, "raw_unit": unit}, reason="bad unit", detail=str(exc)))
        return
    features = {
        "ocr_conf": cell.conf if grid.source == "ocr" else 1.0,
        "header_score": match.score,
        "repaired": float(parsed.repaired),
        "converted": float(converted),
        "footnote": float(parsed.footnote),
        "caption_property": float(out.orientation == "caption"),
        "has_uncertainty": float(parsed.uncertainty is not None),
    }
    if unit_agree is not None:
        features["unit_agreement"] = unit_agree
    try:
        ev = ExtractedValue(
            paper_id=ctx.paper_id, page=grid.page, table_index=grid.table_index, material=material, property=prop,
            value=value, uncertainty=unc, unit=CANONICAL_UNIT[prop], raw_text=text, source=grid.source,
            qualifier=parsed.qualifier, raw_unit=unit, header=header, test_temperature_c=temp,
            loading=match.loading or ctx.cap_loading, condition=condition or None, row=src_r, col=src_c,
            bbox=cell.bbox, route=ctx.route, features=features,
        )
    except ValidationError as exc:
        msg = "; ".join(e["msg"] for e in exc.errors())
        out.quarantine.append(QuarantinedCell(**{**base, "raw_unit": unit}, reason=quarantine_reason(msg), detail=msg))
        return
    out.values.append(ev)


def _unit_agreement(ctx: _Context, text: str, match: HeaderMatch, twin_text: str,
                    twin_match: HeaderMatch | None) -> float | None:
    """1.0 if a value and its twin in another unit (MN/m2 vs ksi) describe the same stress."""
    if twin_match is None or not twin_text:
        return None
    a, b = parse_number(text, ctx.decimal_comma), parse_number(twin_text, ctx.decimal_comma)
    if a is None or b is None:
        return 0.0
    try:
        va = to_canonical(a.value, match.unit, match.property)[0]
        vb = to_canonical(b.value, twin_match.unit, twin_match.property)[0]
    except UnitError:
        return None
    return 1.0 if abs(va - vb) <= max(0.02 * abs(vb), 5.0) else 0.0
