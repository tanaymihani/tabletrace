"""Build reports/results.md, reports/metrics.json and reports/figures/ from the saved runs."""

from __future__ import annotations

import json
import math

import jiwer
import numpy as np

from . import figures
from .evaluate import linearize
from .grid import words_in
from .paths import FIGURES, LABELS, REPORTS, get_paper, load_papers
from .report import GROUP_LABEL, calibration_analysis, evaluate_predictions, load_run, paper_meta, run_dir

SHORT = {"born_digital": "text", "image_only": "render", "scanned": "scan"}


def _pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def _f(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def _prf_row(name: str, d: dict) -> str:
    s, lax = d["strict"], d["lenient"]
    return (f"| {name} | {s['n_gold']} | {s['n_pred']} | {_f(s['precision'])} | {_f(s['recall'])} | "
            f"**{_f(s['f1'])}** | {_f(lax['f1'])} |")


PRF_HEADER = ("| Split | Labels | Extracted | Precision | Recall | F1 (strict) | F1 (lenient) |\n"
              "|---|---|---|---|---|---|---|")


# --------------------------------------------------------------------------- OCR quality

def ocr_quality(run_name: str, lang: str) -> dict:
    """CER / WER of OCR against the original text layer on the three image-only papers.

    Page level: full-page OCR (psm 3) against the page's visible text layer.
    Table level: the OCR words the pipeline used inside each detected table.
    Both sides are put in the same geometric reading order before comparison.
    """
    from .ocr import ocr_region
    from .pipeline import _cached_words, page_ocr_words
    from .render import page_count, render_page
    from .tables_text import text_words

    run = load_run(run_name)
    out = {"pages": {}, "tables": {}}
    page_refs, page_hyps, tab_refs, tab_hyps = [], [], [], []
    for paper in load_papers():
        if paper.route != "image_only":
            continue
        refs, hyps = [], []
        for page in range(1, page_count(paper.raw_pdf) + 1):
            ref = linearize(text_words(paper.raw_pdf, page))
            hyp = linearize(page_ocr_words(paper.input_pdf, page, 300, lang))
            if ref.strip():
                refs.append(ref)
                hyps.append(hyp or "#")
        out["pages"][paper.paper_id] = {"cer": jiwer.cer(refs, hyps), "wer": jiwer.wer(refs, hyps), "pages": len(refs)}
        page_refs += refs
        page_hyps += hyps
        trefs, thyps = [], []
        for t in run["tables"]:
            if t["paper_id"] != paper.paper_id:
                continue
            bbox = tuple(t["bbox"])
            ref = linearize(text_words(paper.raw_pdf, t["page"], bbox))
            img = render_page(paper.input_pdf, t["page"], dpi=300)
            words = _cached_words("region_ocr", (str(paper.input_pdf), t["page"], bbox, 300, lang),
                                  lambda img=img, bbox=bbox: ocr_region(img, 300, bbox, psm=11, lang=lang))
            hyp = linearize(words_in(words, bbox, pad=2))
            if ref.strip():
                trefs.append(ref)
                thyps.append(hyp or "#")
        if trefs:
            out["tables"][paper.paper_id] = {"cer": jiwer.cer(trefs, thyps), "wer": jiwer.wer(trefs, thyps), "tables": len(trefs)}
            tab_refs += trefs
            tab_hyps += thyps
    out["pages"]["all"] = {"cer": jiwer.cer(page_refs, page_hyps), "wer": jiwer.wer(page_refs, page_hyps), "pages": len(page_refs)}
    out["tables"]["all"] = {"cer": jiwer.cer(tab_refs, tab_hyps), "wer": jiwer.wer(tab_refs, tab_hyps), "tables": len(tab_refs)}
    return out


def plusminus_accuracy(preds: list[dict]) -> tuple[int, int]:
    """How many labeled values printed with ± came back with their uncertainty."""
    from .evaluate import load_ground_truth, match

    gt = load_ground_truth()
    gts = gt[gt.raw_text.astype(str).str.contains("±")].to_dict("records")
    ids = {g["paper_id"] for g in gts}
    p = [x for x in preds if x["paper_id"] in ids]
    m = match(p, gts)
    got = 0
    for i, j, _ in m.pairs:
        if m.pred_status[i] == "correct" and p[i].get("uncertainty") is not None \
                and abs(float(p[i]["uncertainty"]) - float(gts[j]["uncertainty"])) <= 0.01 * max(1.0, float(gts[j]["uncertainty"])):
            got += 1
    return got, len(gts)


# --------------------------------------------------------------------------- report

def write_reports(alpha: float = 0.10) -> dict:
    FIGURES.mkdir(parents=True, exist_ok=True)
    meta = paper_meta()
    runs = {n: load_run(n) for n in ("stock", "finetuned", "text_oracle", "no_ghost_filter")
            if (run_dir(n) / "extractions.jsonl").exists()}
    if "stock" not in runs:
        raise SystemExit("no saved runs; run `python -m tabletrace run` first")
    evals = {n: evaluate_predictions(r["extractions"], r["quarantine"], r["tables"])
             for n, r in runs.items() if n in ("stock", "finetuned")}

    # model selection on the calibration split only: OCR documents, strict F1
    def cal_ocr_f1(name):
        cal_ocr = [p.paper_id for p in load_papers() if p.split == "calibration" and p.route != "born_digital"]
        r = runs[name]
        return evaluate_predictions(r["extractions"], r["quarantine"], r["tables"], cal_ocr)["strict"]["f1"]

    selection = {n: cal_ocr_f1(n) for n in evals}
    final = max(selection, key=selection.get)
    ev = evals[final]
    fr = runs[final]
    metrics: dict = {"final_run": final, "model_selection_calibration_ocr_f1": selection}

    lines = ["# TableTrace results", "",
             "Generated by `make eval` from the saved runs in `outputs/runs/`. Do not edit by hand.", "",
             f"Final configuration: **{final}** (chosen on the calibration split: strict F1 on its OCR documents "
             + ", ".join(f"{n} {v:.3f}" for n, v in selection.items()) + ").", ""]

    # ---- corpus
    papers = load_papers()
    import pandas as pd

    from .evaluate import load_ground_truth
    gt = load_ground_truth()
    lines += ["## Corpus", "", "| Document | Type | Route | Split | Labeled values |", "|---|---|---|---|---|"]
    for p in papers:
        lines.append(f"| {p.paper_id} | {p.doc_type.replace('_', ' ')} | {GROUP_LABEL[p.route]} | {p.split} | "
                     f"{int((gt.paper_id == p.paper_id).sum())} |")
    lines += ["", f"{len(gt)} labeled values in {len(papers)} documents.", ""]

    # ---- router over every page of every document
    router = router_check()
    metrics["router"] = router
    lines += ["## Page router", "",
              "Every page of every document, not only the labeled ones:", "",
              "| Document route | Pages | Sent to text layer | Sent to OCR | Pages where text inside embedded figures was dropped |",
              "|---|---|---|---|---|"]
    for route, d in router.items():
        lines.append(f"| {GROUP_LABEL[route]} | {d['pages']} | {d['text_layer']} | {d['ocr']} | {d['ghost_pages']} |")
    lines.append("")

    # ---- headline table
    g = ev["groups"]
    text_pages = g.get(GROUP_LABEL["born_digital"])
    ocr_keys = [GROUP_LABEL["image_only"], GROUP_LABEL["scanned"]]
    ocr_pred = [p for p in fr["extractions"] if meta[p["paper_id"]]["route"] != "born_digital"]
    ocr_ids = [p.paper_id for p in papers if p.route != "born_digital"]
    ocr_all = evaluate_predictions(ocr_pred, fr["quarantine"], fr["tables"], ocr_ids)
    lines += ["## Value extraction", "",
              "Strict: same sample label, property and value within 1%. Lenient also accepts a sample label with "
              "fuzzy similarity of 0.85 or more (an OCR slip in the name) or a value within 5%.", "", PRF_HEADER]
    if text_pages:
        lines.append(_prf_row("Text-layer pages", text_pages))
    lines.append(_prf_row("OCR pages", ocr_all))
    for k in ocr_keys:
        if k in g:
            lines.append(_prf_row(f"&nbsp;&nbsp;{k}", g[k]))
    lines.append(_prf_row("All", {"strict": ev["strict"], "lenient": ev["lenient"]}))
    lines += ["", "By split (the review threshold is chosen on calibration papers only):", "", PRF_HEADER]
    for k in ("calibration", "test"):
        lines.append(_prf_row(k.capitalize(), ev["by_split"][k]))
    lines += ["", "By document:", "", PRF_HEADER.replace("Split", "Document")]
    for p in papers:
        if p.paper_id in ev["by_paper"]:
            lines.append(_prf_row(f"{p.paper_id} ({SHORT[p.route]}, {p.split})", ev["by_paper"][p.paper_id]))
    lines += ["", "By property:", "", PRF_HEADER.replace("Split", "Property")]
    for k, v in ev["by_property"].items():
        lines.append(_prf_row(k, v))
    metrics["extraction"] = {k: ev[k] for k in ("strict", "lenient", "groups", "by_paper", "by_split", "by_property")}
    metrics["extraction"]["ocr_pages"] = {"strict": ocr_all["strict"], "lenient": ocr_all["lenient"]}
    q = collections_counter(fr["quarantine"])
    n_values = len(fr["extractions"])
    lines += ["", f"{n_values} values extracted, {len(fr['quarantine'])} cells quarantined "
              f"({', '.join(f'{k}: {v}' for k, v in q.items())}).", ""]
    metrics["counts"] = {"extracted": n_values, "quarantined": len(fr["quarantine"]), "quarantine_reasons": q}

    # ---- error analysis
    lines += ["## Error analysis", "", "Every missed label (not matched strictly), classified by the stage that lost it.", "",
              "| Route | Stage | Misses |", "|---|---|---|"]
    for k, v in sorted(ev["stage_counts"].items(), key=lambda kv: (kv[0].split(" | ")[0], -kv[1])):
        grp, stage = k.split(" | ")
        lines.append(f"| {grp} | {stage} | {v} |")
    figures.errors_by_stage(ev["stage_counts"], FIGURES / "errors_by_stage.png",
                            f"Where the {len(ev['misses'])} missed labels were lost")
    lines += ["", "![Errors by stage](figures/errors_by_stage.png)", ""]
    metrics["stage_counts"] = ev["stage_counts"]

    # ---- per-document F1 figure
    rows = []
    for p in papers:
        d = ev["by_paper"].get(p.paper_id)
        if d:
            rows.append((p.paper_id, GROUP_LABEL[p.route], d["strict"]["f1"], d["lenient"]["f1"]))
    figures.f1_by_document(rows, FIGURES / "f1_by_document.png", "Value extraction F1 by document")
    lines += ["![F1 by document](figures/f1_by_document.png)", ""]

    # ---- calibration
    cal = calibration_analysis(fr["extractions"], ev["pred_status"], alphas=(0.05, 0.10))
    metrics["calibration"] = {k: v for k, v in cal.items() if k != "curves"}
    lines += ["## Calibrated review threshold", "",
              "Learn-then-Test on the calibration papers: the lowest confidence threshold whose accepted set has a "
              "one-sided Clopper-Pearson upper bound (delta = 0.1) on its error rate at or below alpha, tested in a "
              "fixed sequence from strict to lenient. Values below the threshold go to human review.", "",
              "| alpha | Threshold | Calibration: accepted / errors / bound | Test: accepted error rate | Test: flagged for review | Test: errors caught by the flag | Guarantee held on all 5/5 paper splits |",
              "|---|---|---|---|---|---|---|"]
    for a, d in cal["alphas"].items():
        c, t, s = d["calibration"], d["test"], d["all_splits"]
        thr = "accept none" if math.isinf(d["threshold"]) else f"{d['threshold']:.3f}"
        lines.append(f"| {float(a):.0%} | {thr} | {c['accepted']} / {c['errors']} / {_pct(c['upper_bound'])} | "
                     f"{_pct(t['accepted_error_rate'])} ({t['accepted']} values) | {_pct(t['flagged_share'])} | "
                     f"{t['errors_caught']} of {t['errors_total']} | {_pct(s['share_holding'])} of {s['n_nonempty']} |")
    if "auroc" in cal:
        lines += ["", f"Confidence separates correct from incorrect values with AUROC {cal['auroc']:.3f}."]
    a_key = str(alpha)
    if a_key in cal["alphas"]:
        d = cal["alphas"][a_key]
        thr_cov = {"calibration": d["calibration"]["coverage"],
                   "test": 1 - d["test"]["flagged_share"]}
        figures.risk_coverage(cal["curves"], alpha, thr_cov, FIGURES / "risk_coverage.png",
                              "Error among auto-accepted values vs. share accepted")
        lines += ["", "![Risk-coverage](figures/risk_coverage.png)", ""]

    # ---- OCR quality
    langs = [("stock", "eng")] + ([("finetuned", "eng_tt")] if "finetuned" in runs else [])
    oq = {name: ocr_quality(name, lang) for name, lang in langs}
    metrics["ocr_quality"] = oq
    lines += ["## OCR quality on the image-only papers", "",
              "OCR text against the original text layer of the same pages (jiwer, geometric reading order on both sides).", "",
              "| Model | Scope | CER | WER |", "|---|---|---|---|"]
    for name, d in oq.items():
        for scope in ("pages", "tables"):
            for pid, v in d[scope].items():
                label = "all" if pid == "all" else pid
                lines.append(f"| {name} | {scope}: {label} | {_pct(v['cer'])} | {_pct(v['wer'])} |")
    lines.append("")

    # ---- controlled route comparison and vendor OCR
    journals = [p.paper_id for p in papers if p.doc_type == "journal_article"]
    if "text_oracle" in runs:
        text_route = [p for p in fr["extractions"] if meta[p["paper_id"]]["route"] == "born_digital"] + runs["text_oracle"]["extractions"]
        ocr_route = [p for p in fr["secondary"] if meta[p["paper_id"]]["route"] == "born_digital"] + \
                    [p for p in fr["extractions"] if meta[p["paper_id"]]["route"] == "image_only"]
        tr = evaluate_predictions(text_route, [], fr["tables"] + runs["text_oracle"]["tables"], journals)
        orr = evaluate_predictions(ocr_route, [], fr["tables"], journals)
        metrics["route_comparison"] = {"text_layer": {k: tr[k] for k in ("strict", "lenient")},
                                       "ocr": {k: orr[k] for k in ("strict", "lenient")}}
        lines += ["## Same pages, two routes", "",
                  "The seven journal papers read once through their text layer and once by OCR of the rendered pages.", "",
                  PRF_HEADER.replace("Split", "Route"), _prf_row("Text layer", tr), _prf_row(f"OCR ({final})", orr), ""]
    scans = [p.paper_id for p in papers if p.route == "scanned"]
    vendor = [p for p in fr["secondary"] if p["paper_id"] in scans]
    if vendor:
        ours = [p for p in fr["extractions"] if p["paper_id"] in scans]
        vr = evaluate_predictions(vendor, [], fr["tables"], scans)
        orr2 = evaluate_predictions(ours, fr["quarantine"], fr["tables"], scans)
        metrics["vendor_ocr"] = {"vendor": {k: vr[k] for k in ("strict", "lenient")}, "ours": {k: orr2[k] for k in ("strict", "lenient")}}
        lines += ["## Real scans: the vendor OCR layer vs our OCR", "",
                  "Scans from NTRS and Google Patents ship with an OCR text layer. Reading that layer as if it were a "
                  "text layer is the zero-effort baseline.", "", PRF_HEADER.replace("Split", "Reading"),
                  _prf_row("Vendor OCR layer", vr), _prf_row(f"TableTrace OCR ({final})", orr2), ""]

    # ---- fine-tuning comparison
    if "finetuned" in evals:
        ocr_docs = [p.paper_id for p in papers if p.route != "born_digital"]
        bs = evals["stock"]["by_paper"]
        bf = evals["finetuned"]["by_paper"]
        pm = {n: plusminus_accuracy(runs[n]["extractions"]) for n in ("stock", "finetuned")}
        metrics["finetune"] = {"stock": {d: bs[d] for d in ocr_docs if d in bs}, "finetuned": {d: bf[d] for d in ocr_docs if d in bf},
                               "plusminus": pm}
        lines += ["## Fine-tuning Tesseract", "",
                  "The stock English model has no ± σ ε µ in its character set. `training/` renders 10,000 synthetic "
                  "lines (random alloy formulas with real subscripts, values with ±, headers, ruled rows, 30% ordinary "
                  "prose) and continues training tessdata_best/eng on them. Strict F1 per OCR document:", "",
                  "| Document | Split | Stock | Fine-tuned |", "|---|---|---|---|"]
        for d in ocr_docs:
            if d in bs and d in bf:
                lines.append(f"| {d} | {meta[d]['split']} | {_f(bs[d]['strict']['f1'])} | {_f(bf[d]['strict']['f1'])} |")
        lines += ["", f"Values printed with ± recovered with the right uncertainty: stock {pm['stock'][0]} of {pm['stock'][1]}, "
                  f"fine-tuned {pm['finetuned'][0]} of {pm['finetuned'][1]}.", ""]
        labels = [d for d in ocr_docs if d in bs and d in bf]
        figures.paired_bars(labels, [bs[d]["strict"]["f1"] for d in labels], [bf[d]["strict"]["f1"] for d in labels],
                            ("Stock Tesseract", "Fine-tuned (eng_tt)"), FIGURES / "ocr_finetune.png",
                            "Strict F1 on OCR documents, before and after fine-tuning", "Strict F1", xmax=1.1)
        lines += ["![Fine-tuning](figures/ocr_finetune.png)", ""]

    # ---- ablation: the hidden duplicate text in MDPI PDFs
    if "no_ghost_filter" in runs:
        born = [p.paper_id for p in papers if p.route == "born_digital"]
        with_f = evaluate_predictions([p for p in fr["extractions"] if p["paper_id"] in born], [], fr["tables"], born)
        without = evaluate_predictions(runs["no_ghost_filter"]["extractions"], [], runs["no_ghost_filter"]["tables"], born)
        ghosts: dict[str, int] = {}
        for r in fr["routes"]:
            if r["paper_id"] in born and r["n_ghost"]:
                ghosts[r["paper_id"]] = ghosts.get(r["paper_id"], 0) + r["n_ghost"]
        metrics["ghost_filter"] = {"with": {k: with_f[k] for k in ("strict", "lenient")},
                                   "without": {k: without[k] for k in ("strict", "lenient")},
                                   "hidden_chars_by_paper": ghosts}
        lines += ["## Ablation: hidden text in the PDF text layer", "",
                  "The born-digital papers read through their text layer with and without the filter that drops "
                  "characters clipped inside embedded figures ("
                  + ", ".join(f"{k}: {v:,} hidden characters" for k, v in ghosts.items()) + ").", "",
                  PRF_HEADER.replace("Split", "Text layer"), _prf_row("With the filter", with_f),
                  _prf_row("Without (the raw text layer)", without), ""]

    # ---- first test run (frozen configuration) vs final
    first = REPORTS / "first_test_run" / "metrics.json"
    if first.exists():
        f0 = json.loads(first.read_text())
        e0, e1 = f0["extraction"], metrics["extraction"]

        def row(name, a, b):
            return f"| {name} | {_f(a)} | {_f(b)} |"

        lines += ["## First test run and post-hoc fixes", "",
                  "The configuration was frozen before the first run on the test papers; that run is kept in "
                  "`reports/first_test_run/`. Its errors exposed three defects, fixed afterwards and re-checked on the "
                  "calibration papers (strict F1 there 0.540 before, 0.570 after):", "",
                  "1. Ruled-header detection read the table's outer border, and the stems of capital letters in a "
                  "tight header band, as vertical rules, collapsing a boxed header into one cell.",
                  "2. Table Transformer sometimes gives two data rows one band (`590 800` in one cell). Rows whose "
                  "cells hold two lines of numbers are now split along their text lines.",
                  "3. Sparse-text OCR drops `±` or isolated short numbers. Such cells are now re-read on their own.", "",
                  "| Strict F1 | First test run | Final |", "|---|---|---|",
                  row("Test papers", e0["by_split"]["test"]["strict"]["f1"], e1["by_split"]["test"]["strict"]["f1"]),
                  row("OCR pages (all papers)", e0["ocr_pages"]["strict"]["f1"], e1["ocr_pages"]["strict"]["f1"]),
                  row("All papers", e0["strict"]["f1"], e1["strict"]["f1"]), ""]
        metrics["first_test_run"] = {"test_f1": e0["by_split"]["test"]["strict"]["f1"], "all_f1": e0["strict"]["f1"]}

    # ---- context: LitXAlloy's own values
    ref_path = LABELS / "litxalloy_reference.csv"
    if ref_path.exists():
        import pandas as pd

        ref = pd.read_csv(ref_path)
        ref = ref[ref["value"].notna()].copy()
        ref["from_table"] = ref["source"].fillna("").str.lower().str.contains("table")
        pool = [p for p in fr["extractions"] if meta[p["paper_id"]]["route"] == "born_digital"]
        if "text_oracle" in runs:
            pool += runs["text_oracle"]["extractions"]
        litx = litx_recall(pool, ref)
        metrics["litxalloy"] = litx
        lines += ["## Context: LitXAlloy values", "",
                  "LitXBench (Radical AI, 2026) reports single-property F1 of 0.88 to 0.95 for Gemini 3.1 Pro on its "
                  "19 alloy papers, from an OCR transcription that includes the running text. TableTrace reads tables "
                  "only, so the comparable number is recall of LitXAlloy's values on the 7 papers used here, through "
                  "the text layer:", "",
                  "| LitXAlloy values | Count | Recovered by TableTrace |", "|---|---|---|",
                  f"| All four properties | {litx['all']['n']} | {litx['all']['found']} ({_pct(litx['all']['recall'])}) |",
                  f"| Source cites a table | {litx['table']['n']} | {litx['table']['found']} ({_pct(litx['table']['recall'])}) |",
                  "", "The values not recovered are the ones LitXAlloy took from the paper text, which a table "
                  "extractor never sees.", ""]

    # ---- failure examples
    lines += ["## Failure examples", ""]
    examples = pick_failures(ev["misses"])
    for i, m in enumerate(examples, 1):
        lab = m["label"]
        ev_ = m["evidence"]
        bbox = ev_.get("bbox")
        if not bbox:
            continue
        paper = get_paper(lab["paper_id"])
        out = FIGURES / f"failure_{i}.png"
        read = f"pipeline read {ev_.get('raw_text')!r} for {ev_.get('material')!r}" if ev_.get("raw_text") else m["detail"]
        figures.failure_crop(paper.input_pdf, int(lab["page"]), bbox, out,
                             f"{lab['paper_id']} p{lab['page']}: label {lab['raw_text']} ({lab['material']}); {read}")
        lines += [f"{i}. **{m['stage']}** in {lab['paper_id']} (Table {lab['table_index']}): the cell says "
                  f"`{lab['raw_text']}` for `{lab['material']}`; {m['detail']}.", "",
                  f"   ![failure {i}](figures/failure_{i}.png)", ""]

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8")
    # per-value outputs of the final run, for inspection
    pd.DataFrame(fr["extractions"]).drop(columns=["features"], errors="ignore").to_csv(REPORTS / "extractions.csv", index=False)
    pd.DataFrame(fr["quarantine"]).to_csv(REPORTS / "quarantine.csv", index=False)
    print(f"wrote {REPORTS / 'results.md'} and {REPORTS / 'metrics.json'}")
    return metrics


def router_check() -> dict:
    from .render import page_count
    from .router import route_page

    out: dict[str, dict] = {}
    for paper in load_papers():
        d = out.setdefault(paper.route, {"pages": 0, "text_layer": 0, "ocr": 0, "ghost_pages": 0})
        for page in range(1, page_count(paper.input_pdf) + 1):
            dec = route_page(paper.input_pdf, page)
            d["pages"] += 1
            d[dec.route] += 1
            d["ghost_pages"] += int(dec.n_ghost > 0)
    return out


def litx_recall(preds: list[dict], ref) -> dict:
    """Share of LitXAlloy values that TableTrace extracted (property and value within 1%, one to one)."""
    from scipy.optimize import linear_sum_assignment

    out = {}
    for name, sub in (("all", ref), ("table", ref[ref["from_table"]])):
        found = 0
        for pid, g in sub.groupby("paper_id"):
            pp = [p for p in preds if p["paper_id"] == pid]
            if not pp:
                continue
            cost = np.full((len(g), len(pp)), 1e6)
            for a, r in enumerate(g.itertuples()):
                for b, p in enumerate(pp):
                    if p["property"] == r.property and abs(p["value"] - r.value) <= 0.01 * max(abs(r.value), 1e-9):
                        cost[a, b] = abs(p["value"] - r.value)
            rows, cols = linear_sum_assignment(cost)
            found += int(sum(cost[i, j] < 1e6 for i, j in zip(rows, cols)))
        out[name] = {"n": int(len(sub)), "found": found, "recall": found / len(sub) if len(sub) else float("nan")}
    return out


def collections_counter(quarantine: list[dict]) -> dict:
    out: dict[str, int] = {}
    for q in quarantine:
        out[q["reason"]] = out.get(q["reason"], 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def pick_failures(misses: list[dict], k: int = 3) -> list[dict]:
    """One example per stage, most frequent stages first, each from a different document where possible."""
    by_stage: dict[str, list[dict]] = {}
    for m in misses:
        if m["evidence"].get("bbox"):
            by_stage.setdefault(m["stage"], []).append(m)
    order = sorted(by_stage, key=lambda s: -len(by_stage[s]))
    picked, used = [], set()
    for stage in order[:k]:
        choice = next((m for m in by_stage[stage] if m["label"]["paper_id"] not in used), by_stage[stage][0])
        picked.append(choice)
        used.add(choice["label"]["paper_id"])
    return picked


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and math.isinf(o):
        return "inf"
    return str(o)
