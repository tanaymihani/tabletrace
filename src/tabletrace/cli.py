"""Command line: tabletrace data | run | eval | all."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from .paths import ROOT, load_papers

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Every configuration the report compares. "stock" and "finetuned" differ only in the OCR model;
# "text_oracle" reads the three image-only papers from their original PDFs, which gives the
# same pages through the text layer for a controlled text-layer vs OCR comparison.
EXPERIMENTS = {
    "stock": dict(ocr_lang="eng"),
    "finetuned": dict(ocr_lang="eng_tt"),
    "text_oracle": dict(use_raw_pdfs=True, second_route=False),
    "no_ghost_filter": dict(ghost_filter=False, second_route=False),
}


def cmd_data(_args) -> int:
    for script in ("download_papers.py", "make_image_only.py"):
        rc = subprocess.call([sys.executable, str(ROOT / "scripts" / script)])
        if rc:
            return rc
    return 0


def cmd_run(args) -> int:
    from .pipeline import Config, run
    from .report import save_run

    names = args.experiments or list(EXPERIMENTS)
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        print(f"unknown experiment(s): {', '.join(unknown)}; choose from {', '.join(EXPERIMENTS)}")
        return 2
    for name in names:
        cfg = replace(Config(), **EXPERIMENTS[name])
        if cfg.ocr_lang != "eng" and not (ROOT / "models" / "tessdata" / f"{cfg.ocr_lang}.traineddata").exists():
            print(f"skip {name}: models/tessdata/{cfg.ocr_lang}.traineddata not found (run `make model` to download it)")
            continue
        papers = [p.paper_id for p in load_papers()
                  if (name != "text_oracle" or p.route == "image_only") and (name != "no_ghost_filter" or p.route == "born_digital")]
        print(f"== run {name}: {cfg}")
        results = run(cfg, papers)
        out = save_run(name, results, cfg)
        print(f"   saved to {out.relative_to(ROOT)}")
    return 0


def cmd_eval(args) -> int:
    from .results import write_reports

    write_reports(alpha=args.alpha)
    return 0


def cmd_extract(args) -> int:
    """Extract one page and print every record with where it came from and whether it needs review."""
    import json
    from types import SimpleNamespace

    from .confidence import score_values
    from .paths import load_papers
    from .pipeline import Config, PaperResult, process_page

    known = {p.paper_id: p for p in load_papers()}
    if args.pdf in known:
        paper = known[args.pdf]
    else:
        path = Path(args.pdf).resolve()
        paper = SimpleNamespace(paper_id=path.stem, input_pdf=path, raw_pdf=path)
    cfg = Config(ocr_lang="eng_tt" if (ROOT / "models" / "tessdata" / "eng_tt.traineddata").exists() else "eng")
    threshold = args.threshold
    if threshold is None:
        metrics = ROOT / "reports" / "metrics.json"
        threshold = json.loads(metrics.read_text())["calibration"]["alphas"]["0.1"]["threshold"] if metrics.exists() else 0.85
    res = PaperResult(paper.paper_id)
    process_page(paper, args.page, cfg, res)
    route = res.routes[-1]
    has_second = {paper.paper_id} if route["route"] == "text_layer" or route["vendor_ocr"] else set()
    score_values(res.values, res.secondary, has_second)
    if args.json:
        for v in res.values:
            print(json.dumps({**v.model_dump(mode="json", exclude={"features"}), "review": v.confidence < threshold}))
        return 0
    print(f"page {args.page}: {route['reason']} -> {route['route'].replace('_', ' ')} route")
    for t in res.tables:
        print(f"table {t.table_index if t.table_index > 0 else '?'} at {tuple(round(x) for x in t.bbox)}: "
              f"{t.n_values} values ({t.orientation})")
    for v in res.values:
        temp = f" @ {v.test_temperature_c:g} C" if v.test_temperature_c is not None else ""
        flag = "REVIEW" if v.confidence < threshold else "accept"
        print(f"  {v.material[:22]:22s} {v.property.value:17s} {v.value:8.1f} {v.unit:3s}{temp:11s} "
              f"cell {v.row},{v.col}  printed {v.raw_text!r:12s} conf {v.confidence:.2f}  {flag}")
    flagged = sum(v.confidence < threshold for v in res.values)
    print(f"{len(res.values)} values, {flagged} sent to review (threshold {threshold:.2f}), {len(res.quarantine)} quarantined")
    return 0


def cmd_baselines(args) -> int:
    from .baselines import available_tools, evaluate_all, third_route, write_report

    tools = args.tools or available_tools()
    res = evaluate_all(tools, n_boot=args.n_boot)
    third = None
    if "olmocr2" in tools and (ROOT / "outputs" / "runs" / "finetuned" / "extractions.jsonl").exists():
        third = third_route("olmocr2")
    write_report(res, intro=BASELINES_INTRO, third=third)
    print(f"scored {len(tools)} tools: reports/baselines.md, reports/baselines.json, reports/figures/baselines.png")
    return 0


BASELINES_INTRO = [
    "# Other tools on the same labels",
    "",
    "Every system gets the 16 labeled pages that TableTrace reads, and its tables go through",
    "TableTrace's interpretation, normalization and Hungarian matching, so these numbers measure",
    "finding tables and reading their cells. See baselines/README.md for how each tool was run.",
    "Intervals come from resampling the 10 documents with replacement (10,000 times), so a",
    "document counts once however many labels it has. The \"records\" rows skip TableTrace's",
    "interpretation: the model returns material, property, value and unit itself, and only the",
    "unit conversion is shared.",
    "",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tabletrace", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data", help="download the corpus and build the image-only PDFs")
    r = sub.add_parser("run", help="run the extraction experiments")
    r.add_argument("experiments", nargs="*", help=f"any of {', '.join(EXPERIMENTS)} (default: all)")
    e = sub.add_parser("eval", help="score the saved runs and write reports/")
    e.add_argument("--alpha", type=float, default=0.10)
    x = sub.add_parser("extract", help="extract one page and print its records with provenance")
    x.add_argument("pdf", help="a PDF path, or a paper_id from papers.csv")
    x.add_argument("--page", type=int, required=True, help="1-based page number")
    x.add_argument("--json", action="store_true", help="one JSON record per line")
    x.add_argument("--threshold", type=float, default=None, help="review threshold (default: the calibrated one)")
    b = sub.add_parser("baselines", help="score the saved outputs of other tools (baselines/raw/) against the labels")
    b.add_argument("tools", nargs="*", help="tool folders in baselines/raw (default: all)")
    b.add_argument("--n-boot", type=int, default=10000)
    a = sub.add_parser("all", help="data, run, eval")
    a.add_argument("--alpha", type=float, default=0.10)
    args = ap.parse_args(argv)
    if args.cmd == "data":
        return cmd_data(args)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "eval":
        return cmd_eval(args)
    if args.cmd == "extract":
        return cmd_extract(args)
    if args.cmd == "baselines":
        return cmd_baselines(args)
    if args.cmd == "all":
        args.experiments = None
        return cmd_data(args) or cmd_run(args) or cmd_eval(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
