"""Command line: tabletrace data | run | eval | all."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import replace

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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tabletrace", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data", help="download the corpus and build the image-only PDFs")
    r = sub.add_parser("run", help="run the extraction experiments")
    r.add_argument("experiments", nargs="*", help=f"any of {', '.join(EXPERIMENTS)} (default: all)")
    e = sub.add_parser("eval", help="score the saved runs and write reports/")
    e.add_argument("--alpha", type=float, default=0.10)
    a = sub.add_parser("all", help="data, run, eval")
    a.add_argument("--alpha", type=float, default=0.10)
    args = ap.parse_args(argv)
    if args.cmd == "data":
        return cmd_data(args)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "eval":
        return cmd_eval(args)
    if args.cmd == "all":
        args.experiments = None
        return cmd_data(args) or cmd_run(args) or cmd_eval(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
