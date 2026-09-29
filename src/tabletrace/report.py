"""Score saved runs and write reports/results.md, reports/metrics.json and the figures."""

from __future__ import annotations

import collections
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .calibrate import apply_threshold, choose_threshold, split_robustness
from .evaluate import classify_miss, load_ground_truth, score
from .paths import OUTPUTS, load_papers

RUNS = OUTPUTS / "runs"
GROUP_LABEL = {"born_digital": "Text layer", "image_only": "OCR: rendered pages", "scanned": "OCR: real scans"}


def run_dir(name: str) -> Path:
    return RUNS / name


def save_run(name: str, results, cfg) -> Path:
    from .pipeline import save

    out = run_dir(name)
    save(results, out)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    return out


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_run(name: str) -> dict:
    d = run_dir(name)
    return {k: load_jsonl(d / f"{k}.jsonl") for k in ("extractions", "secondary", "quarantine", "tables", "routes")}


def paper_meta() -> dict[str, dict]:
    return {p.paper_id: {"route": p.route, "split": p.split, "doc_type": p.doc_type} for p in load_papers()}


def evaluate_predictions(preds: list[dict], quarantine: list[dict], tables: list[dict], paper_ids=None) -> dict:
    """P/R/F1 overall, by route group, by split and by paper, plus the error taxonomy."""
    meta = paper_meta()
    gt = load_ground_truth()
    if paper_ids is not None:
        gt = gt[gt.paper_id.isin(paper_ids)]
        preds = [p for p in preds if p["paper_id"] in paper_ids]
    gts = gt.to_dict("records")

    def group(x):
        return GROUP_LABEL[meta[x["paper_id"]]["route"]]

    res = score(preds, gts, group_of=group)
    by_paper = score(preds, gts, group_of=lambda x: x["paper_id"])["groups"]
    by_split = score(preds, gts, group_of=lambda x: meta[x["paper_id"]]["split"])["groups"]
    by_property = score(preds, gts, group_of=lambda x: x["property"])["groups"]
    m = res["match"]
    misses = []
    for j, g in enumerate(gts):
        if m.gt_status[j] != "found":
            a = classify_miss(g, preds, quarantine, tables)
            misses.append({"label": g, "stage": a.stage, "detail": a.detail, "group": group(g),
                           "evidence": {k: a.evidence.get(k) for k in ("material", "raw_text", "value", "bbox", "page",
                                                                          "property", "source")} if a.evidence else {}})
    stages = collections.Counter((x["group"], x["stage"]) for x in misses)
    return {
        "strict": res["strict"], "lenient": res["lenient"], "groups": res["groups"], "by_paper": by_paper,
        "by_split": by_split, "by_property": by_property, "misses": misses,
        "stage_counts": {f"{g} | {s}": n for (g, s), n in sorted(stages.items())},
        "pred_status": m.pred_status,
    }


def calibration_analysis(preds: list[dict], pred_status: list[str], alphas=(0.05, 0.10), delta: float = 0.1) -> dict:
    """Choose the review threshold on calibration papers and check it on test papers."""
    meta = paper_meta()
    conf = np.array([p["confidence"] for p in preds], dtype=float)
    correct = np.array([s == "correct" for s in pred_status], dtype=bool)
    split = np.array([meta[p["paper_id"]]["split"] for p in preds])
    papers = np.array([p["paper_id"] for p in preds])
    cal, test = split == "calibration", split == "test"
    out = {"n_calibration": int(cal.sum()), "n_test": int(test.sum()), "delta": delta, "alphas": {}}
    for alpha in alphas:
        c = choose_threshold(conf[cal], correct[cal], alpha, delta)
        rep = apply_threshold(conf[test], correct[test], c.threshold)
        robust = split_robustness(conf, correct, papers, alpha, delta)
        holds = [r["holds"] for r in robust if r["test_accepted"] > 0]
        out["alphas"][str(alpha)] = {
            "threshold": c.threshold,
            "calibration": {"accepted": c.n_accepted, "errors": c.n_errors_accepted, "upper_bound": c.upper_bound,
                            "coverage": c.coverage},
            "test": {"accepted": rep.n_accepted, "n": rep.n, "accepted_error_rate": rep.accepted_error_rate,
                     "flagged_share": rep.flagged_share, "errors_total": rep.errors_total,
                     "errors_caught": rep.errors_caught, "catch_rate": rep.catch_rate},
            "all_splits": {"n_splits": len(robust), "n_nonempty": len(holds),
                           "share_holding": float(np.mean(holds)) if holds else float("nan"),
                           "median_test_flagged_share": float(np.median([r["test_flagged_share"] for r in robust]))},
        }
    # risk-coverage curves for the figure
    curves = {}
    for name, mask in (("calibration", cal), ("test", test)):
        order = np.argsort(-conf[mask], kind="stable")
        err = (~correct[mask])[order]
        n = np.arange(1, len(err) + 1)
        curves[name] = {"coverage": (n / len(err)).tolist(), "error": (np.cumsum(err) / n).tolist(),
                        "confidence": conf[mask][order].tolist()}
    out["curves"] = curves
    # how well confidence separates right from wrong (AUROC)
    pos, neg = conf[correct], conf[~correct]
    if len(pos) and len(neg):
        ranks = np.argsort(np.argsort(np.concatenate([pos, neg])))
        auc = (ranks[: len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))
        out["auroc"] = float(auc)
    return out
