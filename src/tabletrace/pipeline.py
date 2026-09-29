"""Run the extraction pipeline over the papers in papers.csv."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .captions import find_captions, link_caption
from .grid import TableGrid, Word, assign_words
from .interpret import interpret
from .ocr import model_fingerprint, ocr_cell, ocr_region, ocr_words, prepare
from .paths import OUTPUTS, Paper, load_papers
from .render import render_page
from .router import route_page
from .schema import ExtractedValue, QuarantinedCell
from .tables_image import (
    STRUCTURE_MODEL,
    apply_spanning,
    cells_from_structure,
    detect_tables,
    recognize_structure,
    ruled_header_spans,
)
from .tables_text import geometric_structure, plumber_structure, text_words

CACHE = OUTPUTS / "cache"


@dataclass
class Config:
    detect_dpi: int = 200
    ocr_dpi: int = 300
    text_structure: str = "tatr"  # tatr | plumber | geometric
    ocr_mode: str = "region"  # region (chosen on calibration tables: lower CER, 20x faster) | cell
    ocr_upscale: float = 2.0
    ocr_lang: str = "eng"
    structure_model: str = STRUCTURE_MODEL
    detection_threshold: float = 0.8
    second_route: bool = True
    use_raw_pdfs: bool = False  # read image-only papers from their original PDFs (a text-layer oracle)
    cell_fallback: bool = True  # re-read blank-but-inked cells and values whose ± was lost (calibration: +0.021 strict F1)
    ghost_filter: bool = True  # drop clipped text inside embedded figures (off only for the ablation)

    def key(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]


@dataclass
class TableResult:
    paper_id: str
    page: int
    table_index: int
    bbox: tuple
    route: str
    orientation: str
    caption: str
    n_values: int
    markdown: str


@dataclass
class PaperResult:
    paper_id: str
    values: list[ExtractedValue] = field(default_factory=list)
    secondary: list[ExtractedValue] = field(default_factory=list)
    quarantine: list[QuarantinedCell] = field(default_factory=list)
    tables: list[TableResult] = field(default_factory=list)
    routes: list[dict] = field(default_factory=list)
    seconds: float = 0.0


# --------------------------------------------------------------------------- cached OCR

def _cache_path(kind: str, *parts) -> Path:
    # OCR language names are replaced by the model file's fingerprint, so retraining invalidates the cache
    parts = tuple(model_fingerprint(x) if isinstance(x, str) and x.startswith("eng") else x for x in parts)
    h = hashlib.sha1(json.dumps(parts, default=str).encode()).hexdigest()[:16]
    return CACHE / kind / f"{h}.json"


def _cached_words(kind: str, parts: tuple, compute) -> list[Word]:
    path = _cache_path(kind, *parts)
    if path.exists():
        return [Word(**w) for w in json.loads(path.read_text())]
    words = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(w) for w in words]))
    return words


def page_ocr_words(pdf: Path, page: int, dpi: int, lang: str = "eng") -> list[Word]:
    """Full-page OCR (psm 3) used for captions and for the page-level CER evaluation."""
    def compute():
        img = prepare(render_page(pdf, page, dpi=dpi))
        return ocr_words(img, dpi, psm=3, lang=lang)
    return _cached_words("page_ocr", (str(pdf), page, dpi, lang), compute)


def ocr_table_grid(paper: Paper, pdf: Path, page: int, det_bbox, structure, img_ocr, cfg: Config) -> list[list]:
    """Cells for an OCR page.

    region (default): one sparse-text OCR pass over the table, words assigned to
    cells by their centers, and the header block re-read cell by cell, because
    short group labels ("500", "ksi") are lost in a whole-table pass.
    cell: every cell OCR'd on its own.
    """
    def one_cell(bbox, multiline):
        return _cached_words(
            "cell_ocr", (str(pdf), page, tuple(round(v, 2) for v in bbox), cfg.ocr_dpi, cfg.ocr_upscale, cfg.ocr_lang, multiline),
            lambda: ocr_cell(img_ocr, cfg.ocr_dpi, bbox, upscale=cfg.ocr_upscale, multiline=multiline, lang=cfg.ocr_lang))

    heights = sorted(b - a for a, b in structure.rows) or [10.0]
    median_h = heights[len(heights) // 2]
    if cfg.ocr_mode == "region":
        words = _cached_words("region_ocr", (str(pdf), page, det_bbox, cfg.ocr_dpi, cfg.ocr_lang),
                              lambda: ocr_region(img_ocr, cfg.ocr_dpi, det_bbox, psm=11, lang=cfg.ocr_lang))
        cells = assign_words(words, structure.rows, structure.cols, det_bbox)
        header_bottom = structure.header_bbox[3] if structure.header_bbox is not None else None
        for row in cells:
            for cell in row:
                multiline = (cell.bbox[3] - cell.bbox[1]) > 1.6 * median_h
                in_header = header_bottom is not None and (cell.bbox[1] + cell.bbox[3]) / 2 <= header_bottom + 2
                if in_header:
                    if cell.words:
                        cell.words = one_cell(cell.bbox, multiline)
                elif cfg.cell_fallback and needs_reread(cell, img_ocr, cfg.ocr_dpi):
                    again = one_cell(cell.bbox, multiline)
                    if reading_score(again) > reading_score(cell.words):
                        cell.words = again
    else:
        cells = assign_words([], structure.rows, structure.cols, det_bbox)
        for row in cells:
            for cell in row:
                cell.words = one_cell(cell.bbox, (cell.bbox[3] - cell.bbox[1]) > 1.6 * median_h)
    # a spanning header is OCR'd as one unit so its words are never split between sub-cells
    apply_spanning(cells, structure, lambda span: one_cell(span, True))
    return cells


# "422 75", "696 T 33", "470 1 8": a value and its uncertainty with the ± lost or misread
_LOST_PLUSMINUS = re.compile(r"^\s*\d[\d.,]*\s+(?:\S\s+)?\d[\d.,]*\s*$")


def needs_reread(cell, img, dpi: int) -> bool:
    """Body cells the whole-table pass probably misread: blank but inked cells
    (sparse-text mode drops isolated short numbers) and values whose ± was lost."""
    text = cell.text
    if not text.strip():
        return cell_has_ink(img, dpi, cell.bbox)
    return bool(_LOST_PLUSMINUS.match(text))


def reading_score(words) -> float:
    """Prefer readings that parse as one number (optionally with ±), then higher OCR confidence."""
    from .grid import join_words
    from .normalize import parse_number, repair_ocr_cell

    if not words:
        return 0.0
    text = repair_ocr_cell(join_words(words))
    conf = sum(w.conf for w in words) / len(words)
    return conf + (1.0 if parse_number(text) is not None else 0.0) + (0.2 if "\u00b1" in text else 0.0)


def cell_has_ink(img, dpi: int, bbox, inset: float = 1.0, frac: float = 0.004) -> bool:
    import numpy as np

    s = dpi / 72.0
    x0, top, x1, bottom = bbox
    box = (int((x0 + inset) * s), int((top + inset) * s), int((x1 - inset) * s), int((bottom - inset) * s))
    if box[2] - box[0] < 3 or box[3] - box[1] < 3:
        return False
    arr = np.asarray(img.crop(box).convert("L"))
    # ignore ruling lines along the cell edges: count dark pixels away from full dark rows/columns
    dark = arr < 128
    rows = dark.mean(axis=1) > 0.6
    cols = dark.mean(axis=0) > 0.6
    dark[rows, :] = False
    dark[:, cols] = False
    return dark.mean() > frac


def text_table_cells(pdf: Path, page: int, det_bbox, structure, cfg: Config):
    words = text_words(pdf, page, det_bbox)
    if cfg.text_structure == "plumber":
        st = plumber_structure(pdf, page, det_bbox)
        if st is not None:
            return assign_words(words, st[0], st[1], det_bbox), None
    if cfg.text_structure == "geometric":
        rows, cols = geometric_structure(words, det_bbox)
        return assign_words(words, rows, cols, det_bbox), None
    return cells_from_structure(words, structure, det_bbox)


# --------------------------------------------------------------------------- per page

def process_page(paper: Paper, page: int, cfg: Config, result: PaperResult) -> None:
    pdf = paper.raw_pdf if cfg.use_raw_pdfs else paper.input_pdf
    decision = route_page(pdf, page)
    result.routes.append({"page": page, **asdict(decision)})
    img = render_page(pdf, page, dpi=cfg.detect_dpi)
    img_ocr = render_page(pdf, page, dpi=cfg.ocr_dpi)
    detections = detect_tables(img, cfg.detect_dpi, cfg.detection_threshold)
    if decision.route == "text_layer":
        caption_words = text_words(pdf, page)
    else:
        caption_words = page_ocr_words(pdf, page, cfg.ocr_dpi, cfg.ocr_lang)
    captions = find_captions(caption_words)
    for det in detections:
        cap = link_caption(captions, det.bbox)
        structure = recognize_structure(img, cfg.detect_dpi, det.bbox, model=cfg.structure_model)
        if decision.route == "ocr":
            # ruled headers: take spans from the ruling lines rather than from TATR's spanning-cell guesses
            ruled = ruled_header_spans(img, cfg.detect_dpi, structure, det.bbox)
            if ruled:
                structure.spanning = ruled
        table_index = cap.number if cap else -1
        caption = cap.text if cap else ""

        def grid_for(source: str, cells, header_rows=None, method="", det=det, caption=caption,
                     table_index=table_index) -> TableGrid:
            return TableGrid(paper.paper_id, page, det.bbox, cells, source, method, caption, table_index,
                             det.score, header_rows)

        if decision.route == "text_layer":
            cells, header_rows = text_table_cells(pdf, page, det.bbox, structure, cfg)
            primary = grid_for("text_layer", cells, header_rows, cfg.text_structure)
            route_name = f"text_layer/{cfg.text_structure}"
        else:
            cells = ocr_table_grid(paper, pdf, page, det.bbox, structure, img_ocr, cfg)
            primary = grid_for("ocr", cells, None, f"tatr+{cfg.ocr_mode}")
            route_name = f"ocr/{cfg.ocr_mode}"
        interp = interpret(primary, paper.paper_id, route_name)
        result.values.extend(interp.values)
        result.quarantine.extend(interp.quarantine)
        result.tables.append(TableResult(paper.paper_id, page, table_index, tuple(round(v, 1) for v in det.bbox),
                                         route_name, interp.orientation, caption[:160], len(interp.values),
                                         primary.to_markdown()))
        if not cfg.second_route:
            continue
        # second opinion for the confidence score
        if decision.route == "text_layer":
            cells2 = ocr_table_grid(paper, pdf, page, det.bbox, structure, img_ocr, cfg)
            second = grid_for("ocr", cells2, None, "second:ocr")
            route2 = "second/ocr"
        elif decision.vendor_ocr:
            cells2, header_rows2 = text_table_cells(pdf, page, det.bbox, structure, cfg)
            second = grid_for("text_layer", cells2, header_rows2, "second:vendor_layer")
            route2 = "second/vendor_layer"
        else:
            continue
        result.secondary.extend(interpret(second, paper.paper_id, route2).values)


def process_paper(paper: Paper, cfg: Config) -> PaperResult:
    t0 = time.time()
    result = PaperResult(paper.paper_id)
    for page in paper.labeled_pages:
        process_page(paper, page, cfg, result)
    result.seconds = time.time() - t0
    return result


def run(cfg: Config | None = None, paper_ids: list[str] | None = None, verbose: bool = True) -> list[PaperResult]:
    """Extract every labeled page, then score each value's confidence."""
    from . import tables_text
    from .confidence import score_values

    cfg = cfg or Config()
    tables_text.GHOST_FILTER = cfg.ghost_filter
    results = []
    for paper in load_papers():
        if paper_ids and paper.paper_id not in paper_ids:
            continue
        res = process_paper(paper, cfg)
        # the second route exists for born-digital pages (our OCR) and scans with a vendor OCR layer
        ran_second = cfg.second_route and any(r["route"] == "text_layer" or r["vendor_ocr"] for r in res.routes)
        has_second = {paper.paper_id} if ran_second else set()
        score_values(res.values, res.secondary, has_second)
        if verbose:
            print(f"{paper.paper_id:18s} {len(res.values):4d} values  {len(res.quarantine):3d} quarantined  "
                  f"{len(res.tables)} tables  {res.seconds:5.1f}s")
        results.append(res)
    return results


def save(results: list[PaperResult], out_dir: Path = OUTPUTS) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "extractions.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            for v in r.values:
                f.write(v.model_dump_json() + "\n")
    with (out_dir / "secondary.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            for v in r.secondary:
                f.write(v.model_dump_json() + "\n")
    with (out_dir / "quarantine.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            for q in r.quarantine:
                f.write(q.model_dump_json() + "\n")
    with (out_dir / "tables.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            for t in r.tables:
                f.write(json.dumps(asdict(t)) + "\n")
    with (out_dir / "routes.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            for d in r.routes:
                f.write(json.dumps({"paper_id": r.paper_id, **d}) + "\n")
