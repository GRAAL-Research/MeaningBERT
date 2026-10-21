"""Significance tests shared by the v4 reports (``discovery_fr.py``, ``xnli_multi.py``).

Every comparison the reports make is paired: two heads on the same items, or one head on
the same items in two languages. The tests are therefore the paired ones: DeLong for two
AUCs, McNemar for two sets of binary decisions, Welch across seeds, Holm over a table.
"""

from __future__ import annotations

import numpy as np


def delong(
    positive_a: np.ndarray, negative_a: np.ndarray, positive_b: np.ndarray, negative_b: np.ndarray
) -> dict[str, float]:
    """Paired DeLong test of ``AUC(a) - AUC(b)`` on the same items (DeLong et al., 1988).

    The z-test for two correlated AUCs: each AUC is a Mann-Whitney statistic, its variance
    and the covariance between the two come from the structural components, computed with
    midranks (Sun and Xu, 2014). Ties count one half, as in :func:`auc`.

    Returns:
        Both AUCs, their difference, its standard error, ``z`` and the two-sided ``p``.
    """
    from scipy.stats import norm, rankdata

    m, n = len(positive_a), len(negative_a)
    aucs, v10, v01 = [], [], []
    for positive, negative in ((positive_a, negative_a), (positive_b, negative_b)):
        ranks = rankdata(np.concatenate([positive, negative]))
        within_pos, within_neg = rankdata(positive), rankdata(negative)
        aucs.append(float((ranks[:m].sum() - m * (m + 1) / 2) / (m * n)))
        v10.append((ranks[:m] - within_pos) / n)
        v01.append(1.0 - (ranks[m:] - within_neg) / m)
    s10, s01 = np.cov(np.vstack(v10)), np.cov(np.vstack(v01))
    variance = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    diff = aucs[0] - aucs[1]
    se = float(np.sqrt(max(variance, 0.0)))
    if se == 0.0:
        z, p = (0.0, 1.0) if diff == 0 else (float(np.sign(diff) * np.inf), 0.0)
    else:
        z = diff / se
        p = float(2 * norm.sf(abs(z)))
    return {"auc_a": aucs[0], "auc_b": aucs[1], "diff": diff, "se": se, "z": float(z), "p": p}


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, float]:
    """Exact McNemar test on two paired sets of binary outcomes over the same items."""
    from scipy.stats import binomtest

    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    p = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_a": only_a, "only_b": only_b, "p": float(p)}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjustment, monotone, capped at one."""
    order = sorted(pvalues, key=pvalues.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def welch(sample_a: list[float], sample_b: list[float]) -> dict[str, float]:
    """Welch t-test between two sets of per-seed scores."""
    from scipy.stats import ttest_ind

    if len(sample_a) < 2 or len(sample_b) < 2:
        return {"t": float("nan"), "p": float("nan")}
    result = ttest_ind(sample_a, sample_b, equal_var=False)
    return {"t": float(result.statistic), "p": float(result.pvalue)}


def williams(r_a: float, r_b: float, r_ab: float, n: int) -> dict[str, float]:
    """Williams' t-test of ``corr(a, y) - corr(b, y)`` when *a* and *b* score the same items.

    The two correlations share the human judgement *y* and are themselves correlated through
    ``r_ab``; Fisher's independent-samples z would ignore that and lose most of its power
    (Williams, 1959; recommended by Steiger, 1980, and by Graham and Baldwin, 2014, for
    comparing metrics).

    Returns:
        ``t`` with ``n - 3`` degrees of freedom and its two-sided ``p``.
    """
    from scipy.stats import t as student

    det = 1 - r_a**2 - r_b**2 - r_ab**2 + 2 * r_a * r_b * r_ab
    mean = (r_a + r_b) / 2
    denominator = 2 * (n - 1) / (n - 3) * det + mean**2 * (1 - r_ab) ** 3
    if n <= 3 or denominator <= 0:
        return {"t": float("nan"), "p": float("nan")}
    value = (r_a - r_b) * np.sqrt((n - 1) * (1 + r_ab)) / np.sqrt(denominator)
    return {"t": float(value), "p": float(2 * student.sf(abs(value), n - 3))}
