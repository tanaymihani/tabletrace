"""Other table extractors on the same labeled pages, scored the same way.

Each tool's runner (baselines/run_*.py, each in its own environment) saves what the tool returned
for every labeled page in baselines/raw/<tool>/. This module turns those tables into grids, sends
them through the same interpretation and normalization as TableTrace, and scores the records with
the same Hungarian matcher. The header rules are held fixed, so the comparison is about finding
the tables and reading their cells. Confidence intervals come from a bootstrap over documents.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

import numpy as np
import pandas as pd

from .captions import CAPTION, roman_to_int
from .evaluate import load_ground_truth, score
from .grid import Cell, TableGrid, Word
from .interpret import interpret
from .paths import REPORTS, ROOT, get_paper, load_papers

RAW = ROOT / "baselines" / "raw"
NO_BOX = (0.0, 0.0, 0.0, 0.0)
GROUPS = {"born_digital": "text layer", "image_only": "rendered pages", "scanned": "real scans"}


# --------------------------------------------------------------------------- cell text

_LATEX_SYMBOLS = {
    "pm": "±", "mp": "∓", "sigma": "σ", "varepsilon": "ε", "epsilon": "ε", "mu": "µ", "circ": "°",
    "degree": "°", "textdegree": "°", "textpm": "±", "textmu": "µ", "times": "×", "cdot": "·",
    "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "sim": "~", "approx": "≈", "alpha": "α", "beta": "β",
    "gamma": "γ", "delta": "δ", "Delta": "Δ", "theta": "θ", "lambda": "λ", "rho": "ρ", "tau": "τ",
    "eta": "η", "phi": "φ", "omega": "ω", "infty": "∞", "prime": "′", "ast": "*", "dagger": "†",
    "ddagger": "‡", "AA": "Å", "%": "%", "&": "&", "_": "_", "#": "#", "$": "$",
}
_LATEX_WRAPPERS = re.compile(r"\\(?:mathrm|text|textbf|textit|textrm|mathbf|mathit|boldsymbol|operatorname|emph|"
                             r"underline|mathsf|textsc|mbox|hbox)\s*\{([^{}]*)\}")


def delatex(s: str) -> str:
    """Turn the LaTeX that Nougat, olmOCR and some VLMs put in cells into plain text."""
    s = s.replace(r"\left", "").replace(r"\right", "")
    for _ in range(3):
        s = _LATEX_WRAPPERS.sub(r"\1", s)
    s = re.sub(r"\^\s*\{\s*\\circ\s*\}|\^\s*\\circ", "°", s)
    s = re.sub(r"\\([A-Za-z]+|[%&_#$])", lambda m: _LATEX_SYMBOLS.get(m.group(1), m.group(0)), s)
    s = re.sub(r"[_^]\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"[_^](\w)", r"\1", s)
    s = re.sub(r"\\[,;:! ]", " ", s)
    s = s.replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "").replace("$", "")
    s = re.sub(r"\\[A-Za-z]+", "", s)  # anything left, e.g. \hline inside a cell
    return s.replace("{", "").replace("}", "")


def clean_cell(s: str | None) -> str:
    if not s:
        return ""
    s = html.unescape(str(s))
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    s = re.sub(r"</?(?:sub|sup|b|i|em|strong|span|u|small|p|div)[^>]*>", "", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    if "\\" in s or "$" in s or re.search(r"[_^]\{", s):
        s = delatex(s)
    s = s.replace("**", "").replace("__", "")
    if len(s) > 2 and s.startswith("*") and s.endswith("*"):
        s = s[1:-1]  # markdown italics; a footnote mark sits on one side only ("*1190", "9.1*")
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------- table formats

@dataclass
class _HtmlCell:
    text: list[str] = field(default_factory=list)
    colspan: int = 1
    rowspan: int = 1
    header: bool = False


def _span(v) -> int:
    try:
        return max(1, min(int(str(v).strip().strip('"')), 50))
    except (TypeError, ValueError):
        return 1


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables: list[dict] = []
        self._stack: list[dict] = []
        self._cell: _HtmlCell | None = None
        self._in_caption = False

    def _close_cell(self):
        if self._cell is not None and self._stack:
            if not self._stack[-1]["rows"]:
                self._stack[-1]["rows"].append([])
            self._stack[-1]["rows"][-1].append(self._cell)
        self._cell = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            self._close_cell()
            self._stack.append({"rows": [], "caption": ""})
        elif not self._stack:
            return
        elif tag == "tr":
            self._close_cell()
            self._stack[-1]["rows"].append([])
        elif tag in ("td", "th"):
            self._close_cell()
            self._cell = _HtmlCell(colspan=_span(a.get("colspan")), rowspan=_span(a.get("rowspan")), header=tag == "th")
        elif tag == "br" and self._cell is not None:
            self._cell.text.append(" ")
        elif tag == "caption":
            self._in_caption = True

    def handle_endtag(self, tag):
        if not self._stack:
            return
        if tag in ("td", "th"):
            self._close_cell()
        elif tag == "caption":
            self._in_caption = False
        elif tag == "table":
            self._close_cell()
            self.tables.append(self._stack.pop())

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.text.append(data)
        elif self._in_caption and self._stack:
            self._stack[-1]["caption"] += data


def expand_spans(rows: list[list[tuple[str, int, int]]]) -> list[list[str]]:
    """(text, colspan, rowspan) rows to a full grid, repeating a spanning cell in every slot it covers."""
    grid: dict[tuple[int, int], str] = {}
    for r, row in enumerate(rows):
        c = 0
        for text, colspan, rowspan in row:
            while (r, c) in grid:
                c += 1
            for dr in range(rowspan):
                for dc in range(colspan):
                    grid.setdefault((r + dr, c + dc), text)
            c += colspan
    if not grid:
        return []
    n_rows = max(r for r, _ in grid) + 1
    n_cols = max(c for _, c in grid) + 1
    return [[grid.get((r, c), "") for c in range(n_cols)] for r in range(n_rows)]


def html_tables(text: str) -> list[dict]:
    """Every <table> in the text as {"rows", "caption", "header_rows", "start", "end"}."""
    p = _TableParser()
    p.feed(text)
    p.close()
    out = []
    starts = [m.start() for m in re.finditer(r"<table\b", text, re.I)]
    ends = [m.end() for m in re.finditer(r"</table\s*>", text, re.I)]
    for i, t in enumerate(p.tables):
        rows = [[(clean_cell("".join(c.text)), c.colspan, c.rowspan) for c in row] for row in t["rows"] if row]
        n_header = 0
        for row in t["rows"]:
            if row and all(c.header for c in row):
                n_header += 1
            else:
                break
        out.append({"rows": expand_spans(rows), "caption": clean_cell(t["caption"]) or None,
                    "header_rows": n_header or None,
                    "start": starts[i] if i < len(starts) else 0, "end": ends[i] if i < len(ends) else len(text)})
    return out


_MD_SEP = re.compile(r"^:?-{2,}:?$")


def markdown_tables(text: str) -> list[dict]:
    """Pipe tables in markdown. Rows are split on unescaped |, the --- row is dropped."""
    out, block, start = [], [], 0
    lines = text.splitlines(keepends=True)
    pos = 0

    def flush(end):
        if len(block) >= 2:
            rows = []
            for line in block:
                cells = [c.replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
                if all(_MD_SEP.match(c.strip()) or not c.strip() for c in cells):
                    continue
                rows.append([(clean_cell(c), 1, 1) for c in cells])
            if rows:
                out.append({"rows": expand_spans(rows), "caption": None, "header_rows": None, "start": start, "end": end})

    for line in lines:
        if line.strip().startswith("|") and line.strip().count("|") >= 2:
            if not block:
                start = pos
            block.append(line)
        else:
            if block:
                flush(pos)
            block = []
        pos += len(line)
    if block:
        flush(pos)
    return out


def _balanced(s: str, i: int) -> tuple[str, int]:
    """The {...} group starting at s[i] == '{', and the index after it."""
    depth, j = 0, i
    while j < len(s):
        if s[j] == "{" and (j == 0 or s[j - 1] != "\\"):
            depth += 1
        elif s[j] == "}" and s[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def _latex_cell(cell: str) -> tuple[str, int, int]:
    cell = cell.strip()
    colspan = rowspan = 1
    for name in ("multicolumn", "multirow"):
        m = re.match(r"\\" + name + r"\s*\{", cell)
        if m:
            n, j = _balanced(cell, m.end() - 1)
            j = cell.find("{", j)
            if j >= 0:
                _, j = _balanced(cell, j)  # column spec or width
                k = cell.find("{", j)
                if k >= 0:
                    cell, _ = _balanced(cell, k)
            if name == "multicolumn":
                colspan = _span(n)
            else:
                rowspan = _span(n)
            return _latex_cell(cell)[0], colspan, rowspan
    return clean_cell(delatex(cell)), colspan, rowspan


def _skip_ws(s: str, j: int) -> int:
    while j < len(s) and s[j] in " \t\n":
        j += 1
    return j


def latex_tables(text: str) -> list[dict]:
    """\\begin{tabular}...\\end{tabular} blocks (Nougat's table format)."""
    out = []
    for m in re.finditer(r"\\begin\{tabular\*?\}", text):
        j = _skip_ws(text, m.end())
        if j < len(text) and text[j] == "[":
            j = _skip_ws(text, text.find("]", j) + 1)
        for _ in range(2 if "*" in m.group(0) else 1):  # tabular* has a width before the column spec
            if j < len(text) and text[j] == "{":
                _, j = _balanced(text, j)
                j = _skip_ws(text, j)
        end = text.find("\\end{tabular", j)
        if end < 0:
            end = len(text)
        body = text[j:end]
        body = re.sub(r"\\(?:hline|toprule|midrule|bottomrule|cline\{[^}]*\}|cmidrule(?:\([^)]*\))?\{[^}]*\})", " ", body)
        rows = []
        for line in re.split(r"\\\\|\\tabularnewline", body):
            if not line.strip():
                continue
            rows.append([_latex_cell(c) for c in re.split(r"(?<!\\)&", line)])
        if rows:
            out.append({"rows": _latex_grid(rows), "caption": None, "header_rows": None,
                        "start": m.start(), "end": end})
    return out


def _latex_grid(rows: list[list[tuple[str, int, int]]]) -> list[list[str]]:
    """Unlike HTML, LaTeX keeps an empty placeholder cell under a \\multirow, so fill those in."""
    grid, multirows = [], []
    for r, row in enumerate(rows):
        flat: list[str] = []
        for text, colspan, rowspan in row:
            if rowspan > 1:
                multirows.append((r, len(flat), rowspan, text))
            flat.extend([text] * colspan)
        grid.append(flat)
    for r, c, n, text in multirows:
        for rr in range(r + 1, min(r + n, len(grid))):
            if c < len(grid[rr]) and not grid[rr][c]:
                grid[rr][c] = text
    width = max(len(row) for row in grid)
    return [row + [""] * (width - len(row)) for row in grid]


def _caption_line(line: str) -> str | None:
    s = clean_cell(re.sub(r"^[#>\s]+", "", line))
    return s if CAPTION.match(s) else None


def caption_near(text: str, start: int, end: int, before: int = 600, after: int = 400) -> str | None:
    """The closest caption line above a table in page text, or just below it (Nature style)."""
    above = [ln for ln in text[max(0, start - before):start].splitlines() if ln.strip()]
    for line in reversed(above[-6:]):
        cap = _caption_line(line)
        if cap:
            return cap
    below = [ln for ln in text[end:end + after].splitlines() if ln.strip()]
    for line in below[:3]:
        cap = _caption_line(line)
        if cap:
            return cap
    return None


def tables_from_text(text: str) -> list[dict]:
    """All tables in a page-level output (markdown, HTML or LaTeX), each with a nearby caption."""
    found = html_tables(text) + latex_tables(text)
    taken = [(t["start"], t["end"]) for t in found]
    for t in markdown_tables(text):
        if not any(a <= t["start"] < b for a, b in taken):
            found.append(t)
    found.sort(key=lambda t: t["start"])
    for t in found:
        t["caption"] = t["caption"] or caption_near(text, t["start"], t["end"])
    return found


def normalize_table(t: dict) -> dict:
    """One table from a raw file, in whatever format the tool gave it, as rows of strings."""
    if t.get("rows") is not None:
        rows = [[clean_cell(c) for c in row] for row in t["rows"]]
        return {**t, "rows": rows}
    for key, parse in (("html", html_tables), ("latex", latex_tables), ("markdown", markdown_tables)):
        if t.get(key):
            parsed = parse(t[key])
            if parsed:
                best = max(parsed, key=lambda x: sum(len(r) for r in x["rows"]))
                return {**t, "rows": best["rows"], "header_rows": t.get("header_rows") or best["header_rows"],
                        "caption": t.get("caption") or best["caption"]}
    return {**t, "rows": []}


# --------------------------------------------------------------------------- grids and records

def table_number(caption: str | None) -> int:
    if not caption:
        return -1
    m = CAPTION.match(caption)
    if not m:
        return -1
    tok = m.group(1)
    return int(tok) if tok.isdigit() else (roman_to_int(tok.replace("l", "i")) or -1)


def grid_from_rows(rows: list[list[str]], paper_id: str, page: int, source: str, caption: str | None,
                   bbox=None, cell_boxes=None, header_rows: int | None = None) -> TableGrid:
    """A TableGrid with one word per text line in each cell, so merged-row splitting still works."""
    width = max((len(r) for r in rows), default=0)
    cells = []
    for r, row in enumerate(rows):
        out_row = []
        for c in range(width):
            text = row[c] if c < len(row) else ""
            box = tuple(cell_boxes[r][c]) if cell_boxes and cell_boxes[r][c] else NO_BOX
            words = [Word(line.strip(), 0.0, 10.0 * i, 100.0, 10.0 * i + 8.0)
                     for i, line in enumerate(str(text).split("\n")) if line.strip()]
            out_row.append(Cell(r, c, box, words))
        cells.append(out_row)
    return TableGrid(paper_id, page, tuple(bbox) if bbox else NO_BOX, cells, source, "baseline",
                     caption or "", table_number(caption), 1.0, header_rows)


_PAGE_CAPTIONS: dict[tuple[str, int], list] = {}


def _page_captions(paper_id: str, page: int):
    """TableTrace's own caption finder on the page, for tools that give a table's position but no caption."""
    key = (paper_id, page)
    if key not in _PAGE_CAPTIONS:
        from .captions import find_captions
        from .router import route_page

        paper = get_paper(paper_id)
        pdf = paper.input_pdf
        if route_page(pdf, page).route == "text_layer":
            from .tables_text import text_words
            words = text_words(pdf, page)
        else:
            from .pipeline import page_ocr_words
            words = page_ocr_words(pdf, page, 300, "eng_tt")
        _PAGE_CAPTIONS[key] = find_captions(words)
    return _PAGE_CAPTIONS[key]


def records_from_text(text: str | None) -> list[dict]:
    """The JSON array in a model's answer, cut back to the last complete object if it ran out of tokens."""
    if not text:
        return []
    s = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    i = s.find("[")
    if i < 0:
        return []
    s = s[i:]
    for candidate in (s, s[: s.rfind("}") + 1] + "]"):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []
    return []


def values_from_records(doc: dict, tool: str, source: str):
    """End-to-end answers (material, property, value, unit, temperature) as validated records.

    Only the units and temperatures are converted here, with the same pint conversions
    TableTrace uses. Reading the table and naming the property is all the model's work.
    """
    from pydantic import ValidationError

    from .normalize import UnitError, find_unit, parse_number, temperature_c, to_canonical
    from .schema import CANONICAL_UNIT, ExtractedValue, PropertyKind, QuarantinedCell

    values, quarantine = [], []
    for rec in records_from_text(doc.get("text")):
        try:
            prop = PropertyKind(str(rec.get("property", "")).strip())
        except ValueError:
            continue
        material = clean_cell(str(rec.get("material") or ""))
        raw = clean_cell(str(rec.get("value", "")))
        unit_text = clean_cell(str(rec.get("unit") or ""))
        table = str(rec.get("table") or "")
        base = dict(paper_id=doc["paper_id"], page=doc["page"], table_index=int(table) if table.isdigit() else -1,
                    material=material, property=prop.value, raw_text=raw, raw_unit=unit_text or None, source=source)
        parsed = parse_number(raw)
        if parsed is None:
            quarantine.append(QuarantinedCell(**base, reason="non-numeric value"))
            continue
        unit = parsed.unit or (find_unit(unit_text) if unit_text else None)
        try:
            value, _ = to_canonical(parsed.value, unit, prop)
        except UnitError as exc:
            quarantine.append(QuarantinedCell(**base, reason="bad unit", detail=str(exc)))
            continue
        temp_text = rec.get("test_temperature")
        temp = temperature_c(str(temp_text)) if temp_text not in (None, "", "null") else None
        try:
            values.append(ExtractedValue(
                paper_id=doc["paper_id"], page=doc["page"], table_index=base["table_index"], material=material,
                property=prop, value=value, unit=CANONICAL_UNIT[prop], raw_text=raw, source=source,
                qualifier=parsed.qualifier, raw_unit=unit, test_temperature_c=temp, route=f"baseline/{tool}"))
        except ValidationError as exc:
            quarantine.append(QuarantinedCell(**base, reason="other", detail=str(exc.errors()[0]["msg"])))
    return values, quarantine


def load_raw(tool: str) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((RAW / tool).glob("*.json"))]


@dataclass
class ToolRun:
    tool: str
    version: str
    values: list = field(default_factory=list)
    quarantine: list = field(default_factory=list)
    n_tables: int = 0
    pages: int = 0
    seconds: float = 0.0
    provenance: str = "page"  # page | table | cell: the finest location the tool returns
    settings: dict = field(default_factory=dict)


def extract(tool: str) -> ToolRun:
    """Records from a tool's saved output, through TableTrace's interpretation."""
    from .captions import link_caption

    pages = load_raw(tool)
    run = ToolRun(tool, pages[0]["version"] if pages else "", settings=pages[0].get("settings", {}) if pages else {})
    has_table_box = has_cell_box = False
    for doc in pages:
        paper = get_paper(doc["paper_id"])
        source = "text_layer" if paper.route == "born_digital" else "ocr"
        if tool.endswith("_records"):
            vals, quar = values_from_records(doc, tool, source)
            run.values.extend(vals)
            run.quarantine.extend(quar)
            run.pages += 1
            run.seconds += float(doc.get("seconds") or 0.0)
            continue
        tables = [normalize_table(t) for t in doc.get("tables") or []]
        if doc.get("text"):
            tables += [normalize_table(t) for t in tables_from_text(doc["text"])]
        run.pages += 1
        run.seconds += float(doc.get("seconds") or 0.0)
        for t in tables:
            if not t["rows"]:
                continue
            caption = t.get("caption")
            if not caption and t.get("bbox"):
                cap = link_caption(_page_captions(doc["paper_id"], doc["page"]), tuple(t["bbox"]))
                caption = cap.text if cap else None
            has_table_box |= bool(t.get("bbox"))
            has_cell_box |= bool(t.get("cell_boxes"))
            grid = grid_from_rows(t["rows"], doc["paper_id"], doc["page"], source, caption, t.get("bbox"),
                                  t.get("cell_boxes"), t.get("header_rows"))
            interp = interpret(grid, doc["paper_id"], f"baseline/{tool}")
            for v in interp.values:
                if v.bbox == NO_BOX:
                    v.bbox = None
            run.values.extend(interp.values)
            run.quarantine.extend(interp.quarantine)
            run.n_tables += 1
    run.provenance = "cell" if has_cell_box else "table" if has_table_box else "page"
    return run


# --------------------------------------------------------------------------- scoring with CIs

def doc_counts(preds: list[dict], gts: list[dict]) -> dict[str, np.ndarray]:
    """Per document: [strict correct, lenient correct, predicted, gold]. Matching is per document,
    so resampling documents only needs these counts."""
    m = score(preds, gts)["match"]
    out = {p.paper_id: np.zeros(4) for p in load_papers()}
    for i, p in enumerate(preds):
        out[p["paper_id"]][0] += m.pred_status[i] == "correct"
        out[p["paper_id"]][1] += m.pred_status[i] in ("correct", "partial")
        out[p["paper_id"]][2] += 1
    for g in gts:
        out[g["paper_id"]][3] += 1
    return out


def f1_from(c: np.ndarray, lenient: bool = False) -> float:
    k = c[..., 1] if lenient else c[..., 0]
    denom = c[..., 2] + c[..., 3]
    return np.where(denom > 0, 2 * k / np.maximum(denom, 1e-12), 0.0)


def bootstrap(counts: dict[str, np.ndarray], docs: list[str], n_boot: int = 10000, seed: int = 0,
              other: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Resample documents with replacement. Returns [n_boot, 2] strict and lenient F1, or the
    difference from `other` on the same resamples when it's given."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(docs), size=(n_boot, len(docs)))
    a = np.stack([counts[d] for d in docs])[idx].sum(axis=1)
    res = np.stack([f1_from(a), f1_from(a, True)], axis=1)
    if other is not None:
        b = np.stack([other[d] for d in docs])[idx].sum(axis=1)
        res = res - np.stack([f1_from(b), f1_from(b, True)], axis=1)
    return res


def summarize(counts: dict[str, np.ndarray], docs: list[str], other=None, n_boot: int = 10000) -> dict:
    total = np.stack([counts[d] for d in docs]).sum(axis=0)
    point = np.array([f1_from(total), f1_from(total, True)])
    if other is not None:
        b = np.stack([other[d] for d in docs]).sum(axis=0)
        point = point - np.array([f1_from(b), f1_from(b, True)])
    boot = bootstrap(counts, docs, n_boot=n_boot, other=other)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return {"strict": float(point[0]), "strict_ci": [float(lo[0]), float(hi[0])],
            "lenient": float(point[1]), "lenient_ci": [float(lo[1]), float(hi[1])],
            "n_correct": int(total[0]), "n_pred": int(total[2]), "n_gold": int(total[3]), "docs": len(docs)}


def doc_sets() -> dict[str, list[str]]:
    papers = load_papers()
    sets = {"all documents": [p.paper_id for p in papers],
            "held-out documents": [p.paper_id for p in papers if p.split == "test"]}
    for route, name in GROUPS.items():
        sets[name] = [p.paper_id for p in papers if p.route == route]
    return sets


def tabletrace_predictions() -> list[dict]:
    """The final TableTrace run, from the committed reports/extractions.csv."""
    df = pd.read_csv(REPORTS / "extractions.csv", keep_default_na=False, na_values=[""])
    df["test_temperature_c"] = pd.to_numeric(df["test_temperature_c"], errors="coerce")
    df["value"] = df["value"].astype(float)
    recs = df.to_dict("records")
    for r in recs:
        if isinstance(r["test_temperature_c"], float) and np.isnan(r["test_temperature_c"]):
            r["test_temperature_c"] = None
    return recs


def available_tools() -> list[str]:
    return sorted(p.name for p in RAW.iterdir() if p.is_dir() and any(p.glob("*.json"))) if RAW.exists() else []


def evaluate_all(tools: list[str] | None = None, n_boot: int = 10000) -> dict:
    gts = load_ground_truth().to_dict("records")
    sets = doc_sets()
    reference = doc_counts(tabletrace_predictions(), gts)
    out = {"TableTrace": {"version": "final run", "provenance": "cell", "sets": {}, "diff": {}}}
    for name, docs in sets.items():
        out["TableTrace"]["sets"][name] = summarize(reference, docs, n_boot=n_boot)
    for tool in tools or available_tools():
        run = extract(tool)
        preds = [v.model_dump(mode="json") for v in run.values]
        counts = doc_counts(preds, gts)
        entry = {"version": run.version, "provenance": run.provenance, "tables": run.n_tables, "pages": run.pages,
                 "seconds_per_page": run.seconds / run.pages if run.pages else None, "settings": run.settings,
                 "quarantined": len(run.quarantine), "sets": {}, "diff": {}}
        for name, docs in sets.items():
            entry["sets"][name] = summarize(counts, docs, n_boot=n_boot)
            entry["diff"][name] = summarize(reference, docs, other=counts, n_boot=n_boot)
        out[tool] = entry
    return out


# --------------------------------------------------------------------------- report

# display name, what it was given, where it ran; rows appear in this order
TOOL_INFO = {
    "TableTrace": ("TableTrace (this repo)", "PDF page", "M2 CPU"),
    "pdfplumber": ("pdfplumber, ruling lines", "PDF text layer", "M2 CPU"),
    "pdfplumber_text": ("pdfplumber, word alignment", "PDF text layer", "M2 CPU"),
    "camelot_lattice": ("Camelot lattice", "PDF text layer", "M2 CPU"),
    "camelot_stream": ("Camelot stream", "PDF text layer", "M2 CPU"),
    "docling": ("Docling", "PDF page", "M2 CPU"),
    "marker": ("Marker", "PDF page", "M2 CPU"),
    "grobid": ("GROBID", "PDF text layer", "M2 CPU"),
    "nougat": ("Nougat base", "page image", "Kaggle T4"),
    "olmocr2": ("olmOCR-2 7B", "page image", "Kaggle T4 x2"),
    "qwen25vl_tables": ("Qwen2.5-VL 7B, tables", "page image", "Kaggle T4 x2"),
    "qwen25vl_records": ("Qwen2.5-VL 7B, records", "page image", "Kaggle T4 x2"),
    "qwen3vl_tables": ("Qwen3-VL 8B, tables", "page image", "Kaggle T4 x2"),
    "qwen3vl_records": ("Qwen3-VL 8B, records", "page image", "Kaggle T4 x2"),
}
SET_ORDER = ["held-out documents", "all documents", "text layer", "rendered pages", "real scans"]


def _ci(d: dict, kind: str = "strict") -> str:
    lo, hi = d[f"{kind}_ci"]
    return f"{d[kind]:.3f} [{lo:.2f}, {hi:.2f}]"


def _signed(d: dict, kind: str = "strict") -> str:
    lo, hi = d[f"{kind}_ci"]
    return f"{d[kind]:+.3f} [{lo:+.2f}, {hi:+.2f}]"


def third_route_lines(tr: dict) -> list[str]:
    label = TOOL_INFO.get(tr["tool"], (tr["tool"],))[0]
    a = tr["vlm_agreement"]
    rows = []
    for name, title in (("without", "without the VLM (TableTrace as published)"), ("penalize", "penalize"),
                        ("boost", "boost")):
        c = tr[name]
        t10, t05 = c["alphas"]["0.1"], c["alphas"]["0.05"]
        rows.append(f"| {title} | {c['auroc_by_split']['calibration']:.3f} | {c['auroc_by_split']['test']:.3f} | "
                    f"{t10['test']['accepted_error_rate']:.1%} | {t10['test']['accepted']} of {t10['test']['n']} | "
                    f"{t10['test']['errors_caught']} of {t10['test']['errors_total']} | "
                    f"{t10['all_splits']['share_holding']:.0%} | {t05['all_splits']['share_holding']:.0%} |")
    return [
        f"## A vision-language model as a third route ({label})", "",
        "TableTrace's confidence score already compares each value with a second reading of the same cell (OCR",
        "on text-layer pages, the PDF's own OCR layer on scans). Here the VLM's reading of the page is added",
        "as a third. Two ways of using it were fixed before looking at any result: *penalize* (a value the VLM",
        "reads differently, or doesn't find, costs what it costs for a second OCR reading) and *boost*",
        "(agreement closes half the gap to 1, nothing else changes). Neither adds a tuned weight.", "",
        "| Confidence | AUROC, calibration papers | AUROC, held-out papers | Error among accepted at alpha = 10% (held-out) "
        "| Accepted (held-out) | Errors caught | Splits where 10% holds | Splits where 5% holds |",
        "|---|---|---|---|---|---|---|---|", *rows, "",
        f"The VLM agreed with {a['agree']} of TableTrace's values, disagreed with {a['disagree']} and had no "
        f"counterpart for {a['missing']}. TableTrace was wrong on only {a['wrong_among_disagree']} of the "
        f"{a['disagree']} it disagreed with, so the VLM's disagreement mostly marks its own misreadings, and "
        "penalizing it hurts. Its agreement is more useful. Chosen on the calibration papers, the published "
        "score without the VLM stays the default (the boost ties it there); the boost does better on the "
        "held-out papers, which is a reason to test it on more documents, not to switch on this evidence.", ""]


def write_report(res: dict, intro: list[str] | None = None, third: dict | None = None) -> None:
    from .figures import interval_panels
    from .paths import FIGURES

    names = [t for t in TOOL_INFO if t in res] + [t for t in res if t not in TOOL_INFO]
    label = {t: TOOL_INFO.get(t, (t, "", ""))[0] for t in names}
    lines = list(intro or [])
    for kind, heading in (("strict", "Strict F1"), ("lenient", "Lenient F1")):
        lines += [f"### {heading} [95% bootstrap interval over documents]", "",
                  "| System | " + " | ".join(s.capitalize() for s in SET_ORDER) + " |",
                  "|---|" + "---|" * len(SET_ORDER)]
        for t in names:
            lines.append(f"| {label[t]} | " + " | ".join(_ci(res[t]["sets"][s], kind) for s in SET_ORDER) + " |")
        lines.append("")
    lines += ["### TableTrace minus each system, strict F1 (paired bootstrap, same resampled documents)", "",
              "| System | Held-out documents | All documents | Text layer | Rendered pages | Real scans |",
              "|---|---|---|---|---|---|"]
    for t in names[1:]:
        d = res[t]["diff"]
        lines.append(f"| {label[t]} | " + " | ".join(_signed(d[s]) for s in SET_ORDER) + " |")
    lines += ["", "### What each run returned", "",
              "| System | Version | Input | Ran on | Seconds per page | Tables | Values | Correct (strict) | Quarantined | Finest location |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for t in names:
        e, info = res[t], TOOL_INFO.get(t, (t, "", ""))
        a = e["sets"]["all documents"]
        spp = f"{e['seconds_per_page']:.1f}" if e.get("seconds_per_page") else "n/a"
        lines.append(f"| {label[t]} | {e['version']} | {info[1]} | {info[2]} | {spp} | {e.get('tables', 'n/a')} | "
                     f"{a['n_pred']} | {a['n_correct']} | {e.get('quarantined', 'n/a')} | {e['provenance']} |")
    if third is not None:
        lines += [""] + third_route_lines(third)
    (REPORTS / "baselines.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    payload = {"systems": res, "third_route": third}
    (REPORTS / "baselines.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    panels = {}
    for s, title in (("held-out documents", "Held-out documents"), ("text layer", "Text-layer pages"),
                     ("rendered pages", "Rendered pages"), ("real scans", "Real scans")):
        panels[title] = [(res[t]["sets"][s]["strict"], *res[t]["sets"][s]["strict_ci"]) for t in names]
    interval_panels([label[t] for t in names], panels, FIGURES / "baselines.png",
                    "Strict F1 on the same labeled pages, with 95% intervals over documents",
                    highlight=label["TableTrace"])


# --------------------------------------------------------------------------- a VLM as a third route

def third_route(tool: str = "olmocr2", alphas=(0.05, 0.10)) -> dict:
    """Re-score the final TableTrace run with a VLM's reading as a third route, then redo the
    calibration analysis. Needs outputs/runs/finetuned (make run)."""
    from .confidence import score_values
    from .report import auroc, calibration_analysis, evaluate_predictions, load_run
    from .schema import ExtractedValue

    run = load_run("finetuned")
    has_second = {r["paper_id"] for r in run["routes"] if r["route"] == "text_layer" or r.get("vendor_ocr")}
    secondary = [ExtractedValue(**d) for d in run["secondary"]]
    third = extract(tool).values
    meta = {p.paper_id: p.split for p in load_papers()}
    out = {"tool": tool}
    status = None
    for name, extra, mode in (("without", None, ""), ("penalize", third, "penalize"), ("boost", third, "boost")):
        values = [ExtractedValue(**d) for d in run["extractions"]]
        score_values(values, secondary, has_second, third=extra, third_mode=mode or "penalize")
        preds = [v.model_dump(mode="json") for v in values]
        if status is None:
            status = evaluate_predictions(preds, [], run["tables"])["pred_status"]
        cal = calibration_analysis(preds, status, alphas=alphas)
        cal.pop("curves", None)
        conf = np.array([p["confidence"] for p in preds])
        ok = np.array([s == "correct" for s in status])
        split = np.array([meta[p["paper_id"]] for p in preds])
        cal["auroc_by_split"] = {sp: auroc(conf[split == sp], ok[split == sp]) for sp in ("calibration", "test")}
        out[name] = cal
        if mode == "penalize":
            agree = [v.features.get("vlm_agreement") for v in values]
            out["vlm_agreement"] = {"agree": agree.count(1.0), "disagree": agree.count(0.0), "missing": agree.count(0.5),
                                    "wrong_among_disagree": sum(1 for a, s in zip(agree, status) if a == 0.0 and s != "correct"),
                                    "wrong_among_agree": sum(1 for a, s in zip(agree, status) if a == 1.0 and s != "correct")}
    return out
