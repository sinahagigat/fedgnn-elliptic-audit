# -*- coding: utf-8 -*-
"""
stats.py — statistical testing layer (single source of truth for variant comparisons)
==============================================================================
Goal: no statistical formula (Holm-Bonferroni, TOST, Wilcoxon, Cohen's d) may be
rewritten inline in analyze.py or anywhere else.

⚠️ Statistical honesty: with only 3 seeds per variant, the statistical power of
parametric tests is very limited. This module provides both the paired
t-test and the non-parametric Wilcoxon test, and the output of both explicitly
carries a small-n warning. With few seeds these tests should be presented as
an indication of the direction of an effect, not as definitive proof of
superiority; a strong significance claim with n=3 would not be defensible, and
stating this limitation openly belongs in the Limitations section.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger("fedgnn.stats")

MIN_RECOMMENDED_N = 5  # below this number of seeds, a statistical-power warning is issued


# ==============================================================================
# Effect size — Cohen's d for paired samples
# ==============================================================================
def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Standard effect size for paired samples: d = mean(diff) / std(diff).
    It does not depend on n and can be reported and interpreted qualitatively
    even with n=3 (small/medium/large, following Cohen 1988), unlike the p-value,
    which becomes unstable for small n.
    """
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a_arr.shape == b_arr.shape, "❌ cohens_d_paired: the two samples must have equal length."
    diff = a_arr - b_arr
    std_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    if std_diff == 0.0:
        return 0.0 if float(np.mean(diff)) == 0.0 else float("inf")
    return float(np.mean(diff) / std_diff)


def interpret_cohens_d(d: float) -> str:
    """Qualitative interpretation following Cohen's (1988) convention; a reporting guide only."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"


# ==============================================================================
# Paired comparison — parametric t-test and non-parametric Wilcoxon, together
# ==============================================================================
@dataclass
class PairwiseComparisonResult:
    metric_name: str
    variant_a: str
    variant_b: str
    n_seeds: int
    mean_a: float
    mean_b: float
    mean_diff: float
    t_stat: float
    t_pvalue: float
    wilcoxon_stat: Optional[float]
    wilcoxon_pvalue: Optional[float]
    cohens_d: float
    effect_size_label: str
    low_power_warning: bool
    note: str = ""


def paired_comparison(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    variant_a: str,
    variant_b: str,
    metric_name: str,
) -> PairwiseComparisonResult:
    """
    Full paired comparison between two variants on the same set of seeds. Wilcoxon
    is computed only when n>=6 (below this threshold the exact Wilcoxon p-value
    table is unreliable); otherwise None is returned and the caller should rely
    only on Cohen's d and the mean.
    """
    a_arr, b_arr = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    assert a_arr.shape == b_arr.shape, "❌ paired_comparison: the two score vectors differ in length."
    n = len(a_arr)
    assert n >= 2, "❌ paired_comparison: at least 2 seeds are required for any paired test."

    t_stat, t_pvalue = scipy_stats.ttest_rel(a_arr, b_arr)

    wilcoxon_stat, wilcoxon_pvalue = None, None
    if n >= 6:
        try:
            wilcoxon_stat, wilcoxon_pvalue = scipy_stats.wilcoxon(a_arr, b_arr)
        except ValueError as exc:
            logger.warning(f"⚠️ Wilcoxon could not be computed ({variant_a} vs {variant_b}): {exc}")

    d = cohens_d_paired(a_arr, b_arr)
    low_power = n < MIN_RECOMMENDED_N
    note = (
        f"⚠️ n={n} seeds — limited statistical power; the p-value should only be "
        f"read as an indication of effect direction, not as proof of superiority."
        if low_power else ""
    )

    return PairwiseComparisonResult(
        metric_name=metric_name,
        variant_a=variant_a,
        variant_b=variant_b,
        n_seeds=n,
        mean_a=float(np.mean(a_arr)),
        mean_b=float(np.mean(b_arr)),
        mean_diff=float(np.mean(a_arr) - np.mean(b_arr)),
        t_stat=float(t_stat),
        t_pvalue=float(t_pvalue),
        wilcoxon_stat=float(wilcoxon_stat) if wilcoxon_stat is not None else None,
        wilcoxon_pvalue=float(wilcoxon_pvalue) if wilcoxon_pvalue is not None else None,
        cohens_d=d,
        effect_size_label=interpret_cohens_d(d),
        low_power_warning=low_power,
        note=note,
    )


# ==============================================================================
# Multiple-comparison correction — Holm-Bonferroni (step-down, less conservative than plain Bonferroni)
# ==============================================================================
@dataclass
class HolmCorrectionResult:
    original_pvalues: List[float]
    sorted_indices: List[int]
    adjusted_pvalues: List[float]
    reject_null: List[bool]
    alpha: float


def holm_bonferroni_correction(
    p_values: Sequence[float], alpha: float = 0.05
) -> HolmCorrectionResult:
    """
    Holm-Bonferroni correction: p-values sorted in ascending order are each compared
    with alpha/(m-i) (i the zero-based index in the sorted order, m the total number
    of comparisons). It has more power than plain Bonferroni (alpha/m for all) and
    still controls the family-wise error rate (FWER); a standard and defensible
    choice.

    Stopping rule: as soon as the first non-significant p-value is met in ascending
    order, all later p-values are not rejected either (even if their raw value is
    small); this is exactly the correct behaviour of the Holm procedure.
    """
    m = len(p_values)
    assert m >= 1, "❌ holm_bonferroni_correction: at least one p-value is required."

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    sorted_indices = [idx for idx, _ in indexed]
    sorted_pvals = [p for _, p in indexed]

    adjusted = [0.0] * m
    reject = [False] * m
    stop = False
    for rank, p in enumerate(sorted_pvals):
        threshold = alpha / (m - rank)
        adjusted[rank] = min(p * (m - rank), 1.0)  # reportable adjusted statistic
        if not stop and p <= threshold:
            reject[rank] = True
        else:
            stop = True  # from this point on, nothing is rejected (correct Holm behaviour)

    # restore the original input order
    final_adjusted = [0.0] * m
    final_reject = [False] * m
    for rank, orig_idx in enumerate(sorted_indices):
        final_adjusted[orig_idx] = adjusted[rank]
        final_reject[orig_idx] = reject[rank]

    return HolmCorrectionResult(
        original_pvalues=list(p_values),
        sorted_indices=sorted_indices,
        adjusted_pvalues=final_adjusted,
        reject_null=final_reject,
        alpha=alpha,
    )


# ==============================================================================
# TOST — two one-sided equivalence tests; to show that "the difference is negligible"
# ==============================================================================
@dataclass
class TOSTResult:
    mean_diff: float
    lower_bound: float
    upper_bound: float
    p_lower: float
    p_upper: float
    p_tost: float
    equivalent: bool
    alpha: float
    n: int
    note: str = ""


def tost_equivalence_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    equivalence_margin: float,
    alpha: float = 0.05,
) -> TOSTResult:
    """
    Two One-Sided Tests (TOST). Use in this project: showing that a lighter variant
    (e.g. FedGNN_SoftSelective without virtual nodes) has no practically meaningful
    difference from the full version (equivalence), not merely that "no significant
    difference was found" (which is not evidence of equivalence; a common confusion).

    Logic: H0_lower: mean_diff <= -margin   (rejected => the difference is not below
           the lower bound)
          H0_upper: mean_diff >= +margin   (rejected => the difference is not above
           the upper bound)
    If both H0 are rejected => equivalence within [-margin, +margin] is accepted;
    p_tost = max(p_lower, p_upper), following the standard TOST rule.
    """
    a_arr, b_arr = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    diff = a_arr - b_arr
    n = len(diff)
    assert n >= 2, "❌ tost_equivalence_test: at least 2 seeds are required."
    assert equivalence_margin > 0, "❌ tost_equivalence_test: the margin must be positive."

    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 1e-12
    se = std_diff / math.sqrt(n) if std_diff > 0 else 1e-12
    df = n - 1

    t_lower = (mean_diff - (-equivalence_margin)) / se
    p_lower = 1.0 - scipy_stats.t.cdf(t_lower, df)

    t_upper = (mean_diff - equivalence_margin) / se
    p_upper = scipy_stats.t.cdf(t_upper, df)

    p_tost = max(p_lower, p_upper)
    equivalent = p_tost < alpha

    note = (
        f"⚠️ n={n} seeds — a TOST result with a small sample must be read with "
        f"care; margin={equivalence_margin} was chosen by hand on the basis of "
        f"practical (not statistical) importance."
    )

    return TOSTResult(
        mean_diff=mean_diff, lower_bound=-equivalence_margin, upper_bound=equivalence_margin,
        p_lower=float(p_lower), p_upper=float(p_upper), p_tost=float(p_tost),
        equivalent=equivalent, alpha=alpha, n=n, note=note,
    )


# ==============================================================================
# High-level orchestrator — all paired comparisons of one metric with one Holm correction
# ==============================================================================
def compare_all_variants_vs_baseline(
    variant_scores: Dict[str, List[float]],
    baseline_name: str,
    metric_name: str,
    alpha: float = 0.05,
) -> Dict[str, object]:
    """
    Compare all variants against a given baseline (e.g. FedAvg_GCN), applying
    the Holm-Bonferroni correction over the whole family of comparisons (not each
    comparison on its own), as expected for a statistical-significance section
    of a paper.
    """
    assert baseline_name in variant_scores, f"❌ baseline '{baseline_name}' is not in variant_scores."
    baseline_scores = variant_scores[baseline_name]

    comparisons: List[PairwiseComparisonResult] = []
    other_variants = [v for v in variant_scores if v != baseline_name]
    for variant_name in other_variants:
        result = paired_comparison(
            variant_scores[variant_name], baseline_scores, variant_name, baseline_name, metric_name
        )
        comparisons.append(result)

    holm = holm_bonferroni_correction(
        [c.t_pvalue for c in comparisons], alpha=alpha
    )

    return {
        "baseline": baseline_name,
        "metric": metric_name,
        "comparisons": comparisons,
        "holm_correction": holm,
        "significant_after_holm": [
            other_variants[i] for i, rej in enumerate(holm.reject_null) if rej
        ],
    }
