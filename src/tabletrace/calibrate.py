"""A review threshold with a finite-sample error guarantee (Learn-then-Test).

Values with confidence >= threshold are accepted automatically; the rest go
to a human. The threshold is chosen on the calibration papers so that, with
probability at least 1 - delta over the draw of those papers, the error rate
among accepted values is at most alpha. The error rate of each candidate
threshold is bounded with a one-sided Clopper-Pearson interval, and candidate
thresholds are tested in a fixed sequence from strict to lenient, stopping at
the first one whose bound exceeds alpha. That ordering is what makes the
multiple testing valid without a Bonferroni correction.

Caveat worth stating in the write-up: the guarantee assumes calibration and
test values are exchangeable. Values from one table are correlated, so with
ten papers the effective sample size is much smaller than the value count.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta


def clopper_pearson_upper(errors: int, n: int, delta: float) -> float:
    """One-sided (1 - delta) upper confidence bound on a binomial error rate."""
    if n == 0:
        return 1.0
    if errors >= n:
        return 1.0
    return float(beta.ppf(1 - delta, errors + 1, n - errors))


def min_n_to_certify(alpha: float, delta: float) -> int:
    """Smallest accepted set that could pass with zero errors: (1 - alpha)^n <= delta."""
    return math.ceil(math.log(delta) / math.log(1 - alpha))


@dataclass
class Calibration:
    threshold: float  # accept values with confidence >= threshold (inf means accept nothing)
    alpha: float
    delta: float
    n_accepted: int
    n_errors_accepted: int
    upper_bound: float
    coverage: float  # share of calibration values accepted
    curve: list[tuple[float, int, int, float]]  # (threshold, n accepted, errors, CP upper bound)


def choose_threshold(conf: np.ndarray, correct: np.ndarray, alpha: float = 0.05, delta: float = 0.1) -> Calibration:
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    candidates = np.unique(conf)[::-1]  # strict to lenient
    n_min = min_n_to_certify(alpha, delta)
    curve = []
    chosen = None
    for lam in candidates:
        acc = conf >= lam
        n, k = int(acc.sum()), int((~correct[acc]).sum())
        ub = clopper_pearson_upper(k, n, delta)
        curve.append((float(lam), n, k, ub))
        if n < n_min:
            continue  # too few values for any bound to pass; the sequence starts once n is large enough
        if ub <= alpha:
            chosen = (float(lam), n, k, ub)
        else:
            break  # fixed-sequence testing stops at the first failure
    if chosen is None:
        return Calibration(float("inf"), alpha, delta, 0, 0, 1.0, 0.0, curve)
    lam, n, k, ub = chosen
    return Calibration(lam, alpha, delta, n, k, ub, n / len(conf), curve)


@dataclass
class ReviewReport:
    threshold: float
    n: int
    n_accepted: int
    accepted_error_rate: float
    flagged_share: float
    errors_total: int
    errors_caught: int  # errors that the flag sent to review

    @property
    def catch_rate(self) -> float:
        return self.errors_caught / self.errors_total if self.errors_total else float("nan")


def apply_threshold(conf: np.ndarray, correct: np.ndarray, threshold: float) -> ReviewReport:
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    acc = conf >= threshold
    n_acc = int(acc.sum())
    err_acc = int((~correct[acc]).sum())
    errors = int((~correct).sum())
    return ReviewReport(
        threshold=threshold,
        n=len(conf),
        n_accepted=n_acc,
        accepted_error_rate=err_acc / n_acc if n_acc else 0.0,
        flagged_share=1 - n_acc / len(conf) if len(conf) else 0.0,
        errors_total=errors,
        errors_caught=errors - err_acc,
    )


def split_robustness(conf: np.ndarray, correct: np.ndarray, groups: np.ndarray, alpha: float, delta: float,
                     k: int | None = None) -> list[dict]:
    """Recalibrate on every way of choosing k papers and test on the rest.

    With 10 papers and k = 5 that is all 252 splits, which shows how often the
    guarantee holds on held-out papers instead of relying on one lucky split.
    """
    papers = sorted(set(groups))
    k = k or len(papers) // 2
    out = []
    for cal in itertools.combinations(papers, k):
        cal_mask = np.isin(groups, cal)
        c = choose_threshold(conf[cal_mask], correct[cal_mask], alpha, delta)
        rep = apply_threshold(conf[~cal_mask], correct[~cal_mask], c.threshold)
        out.append({"calibration": cal, "threshold": c.threshold, "test_error": rep.accepted_error_rate,
                    "test_accepted": rep.n_accepted, "test_flagged_share": rep.flagged_share,
                    "holds": rep.accepted_error_rate <= alpha})
    return out
