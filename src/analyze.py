# -*- coding: utf-8 -*-
"""Summary tables and paired tests from an artifacts.json file (completed runs only).
Used for quick checks; the article's numbers come from analysis/build_numbers.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.stats import compare_all_variants_vs_baseline, tost_equivalence_test

logger = logging.getLogger("fedgnn.analyze")

METRICS_OF_INTEREST = ["ap", "auc", "f1", "precision", "recall"]


# Loading and filtering artifacts
def load_completed_runs(artifacts_path: Path) -> Dict[str, dict]:
    assert artifacts_path.exists(), f"artifacts file not found: {artifacts_path}"
    with open(artifacts_path, "r", encoding="utf-8") as f:
        artifacts = json.load(f)
    runs = artifacts.get("runs", {})
    completed = {k: v for k, v in runs.items() if v.get("status") == "completed"}
    failed = {k: v for k, v in runs.items() if v.get("status") == "failed"}
    if failed:
        logger.warning(f"{len(failed)} runs with status='failed' were excluded from the analysis: {list(failed.keys())}")
    logger.info(f"{len(completed)} completed runs loaded for analysis.")
    return completed


def group_runs_by_variant(completed_runs: Dict[str, dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for run in completed_runs.values():
        grouped.setdefault(run["variant_name"], []).append(run)
    return grouped


# Summary table — mean ± std over seeds for every metric
def build_summary_table(grouped_runs: Dict[str, List[dict]]) -> pd.DataFrame:
    """
    Build a DataFrame with one row per variant, containing the mean/std of every
    metric over all seeds completed for that variant, plus the actual seed count
    (so that a variant with fewer than N completed seeds is visible in the table
    and is not mistakenly presented as a complete result).
    """
    rows = []
    for variant_name, runs in grouped_runs.items():
        row: Dict[str, object] = {"variant": variant_name, "n_seeds": len(runs)}
        row["n_converged"] = sum(1 for r in runs if r.get("converged", False))

        for metric in METRICS_OF_INTEREST:
            values = [r["test_metrics"][metric] for r in runs if not np.isnan(r["test_metrics"][metric])]
            if values:
                row[f"{metric}_mean"] = round(float(np.mean(values)), 4)
                row[f"{metric}_std"] = round(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, 4)
            else:
                row[f"{metric}_mean"], row[f"{metric}_std"] = float("nan"), float("nan")

        row["avg_runtime_sec"] = round(float(np.mean([r["runtime_sec"] for r in runs])), 2)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("ap_mean", ascending=False).reset_index(drop=True)
    return df


# Convergence report — full transparency on runs that did not reach converged=True
def convergence_report(grouped_runs: Dict[str, List[dict]]) -> pd.DataFrame:
    rows = []
    for variant_name, runs in grouped_runs.items():
        for r in runs:
            rows.append({
                "variant": variant_name,
                "seed": r["seed"],
                "converged": r.get("converged", False),
                "best_round_idx": r.get("best_round_idx", None),
                "total_rounds_run": r.get("total_rounds_run", None),
                "best_val_ap": round(r.get("best_val_ap", float("nan")), 4),
            })
    df = pd.DataFrame(rows)
    n_not_converged = int((~df["converged"]).sum())
    if n_not_converged > 0:
        logger.warning(
            f"{n_not_converged}/{len(df)} runs recorded with converged=False; "
            f"report this alongside the results."
        )
    return df


# Statistical comparison of all variants against the baseline
def run_statistical_comparisons(
    grouped_runs: Dict[str, List[dict]],
    baseline_name: str,
    metric: str = "ap",
    alpha: float = 0.05,
) -> Dict[str, object]:
    assert baseline_name in grouped_runs, f"baseline '{baseline_name}' is not in the results."

    variant_scores: Dict[str, List[float]] = {}
    for variant_name, runs in grouped_runs.items():
        # seed order must be identical across variants for the paired test to stay meaningful
        runs_sorted = sorted(runs, key=lambda r: r["seed"])
        variant_scores[variant_name] = [r["test_metrics"][metric] for r in runs_sorted]

    seed_counts = {v: len(s) for v, s in variant_scores.items()}
    if len(set(seed_counts.values())) > 1:
        logger.warning(
            f"seed counts differ between variants {seed_counts}; the paired "
            f"comparison is meaningful only on the common subset of seeds."
        )
        min_n = min(seed_counts.values())
        variant_scores = {v: s[:min_n] for v, s in variant_scores.items()}

    n_min = min((len(v) for v in variant_scores.values()), default=0)
    if n_min < 2:
        logger.warning(
            f"paired tests skipped: minimum seed count = {n_min} (required: 2). "
            f"This is normal for a smoke test; a full run with ≥3 seeds executes them."
        )
        return {"skipped": True, "reason": f"n_seeds={n_min} < 2", "comparisons": {}}
    return compare_all_variants_vs_baseline(variant_scores, baseline_name, metric, alpha)


def run_ablation_equivalence_check(
    grouped_runs: Dict[str, List[dict]],
    variant_a: str,
    variant_b: str,
    equivalence_margin: float,
    metric: str = "ap",
) -> dict:
    """
    Specific use: check whether contribution 2 (virtual nodes) really produces a
    meaningful performance difference, through TOST between
    FedGNN_SoftSelective_GCN (without virtual nodes) and FedGNN_SoftSelective_Virtual_GCN.
    """
    assert variant_a in grouped_runs and variant_b in grouped_runs, "both variants must be present in the results."
    scores_a = sorted(grouped_runs[variant_a], key=lambda r: r["seed"])
    scores_b = sorted(grouped_runs[variant_b], key=lambda r: r["seed"])
    n = min(len(scores_a), len(scores_b))

    a_vals = [r["test_metrics"][metric] for r in scores_a[:n]]
    b_vals = [r["test_metrics"][metric] for r in scores_b[:n]]
    result = tost_equivalence_test(a_vals, b_vals, equivalence_margin)
    return {
        "variant_a": variant_a, "variant_b": variant_b, "metric": metric,
        "tost_result": result,
    }


# LaTeX output — for direct inclusion in the paper
def dataframe_to_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Standard booktabs LaTeX table, compatible with most journal templates."""
    cols_to_show = ["variant", "ap_mean", "ap_std", "auc_mean", "auc_std", "f1_mean", "f1_std", "n_seeds"]
    subset = df[[c for c in cols_to_show if c in df.columns]].copy()
    latex_body = subset.to_latex(index=False, float_format="%.4f", escape=True)
    return (
        f"% table generated automatically by analyze.py; review manually before use in the paper\n"
        f"\\begin{{table}}[htbp]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"{latex_body}\n\\end{{table}}\n"
    )


# Final orchestrator — one function, full analysis output
def generate_full_report(
    artifacts_path: Path,
    output_dir: Path,
    baseline_name: str = "FedAvg_GCN",
    primary_metric: str = "ap",
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    completed = load_completed_runs(artifacts_path)
    assert len(completed) > 0, "no completed run is available for analysis."
    grouped = group_runs_by_variant(completed)

    summary_df = build_summary_table(grouped)
    summary_df.to_csv(output_dir / "summary_table.csv", index=False)
    logger.info(f"summary table saved: {output_dir / 'summary_table.csv'}")

    conv_df = convergence_report(grouped)
    conv_df.to_csv(output_dir / "convergence_report.csv", index=False)

    stats_result = None
    if baseline_name in grouped:
        stats_result = run_statistical_comparisons(grouped, baseline_name, primary_metric)
        with open(output_dir / "statistical_comparisons.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "baseline": stats_result["baseline"],
                    "metric": stats_result["metric"],
                    "significant_after_holm": stats_result["significant_after_holm"],
                    "comparisons": [vars(c) for c in stats_result["comparisons"]],
                },
                f, ensure_ascii=False, indent=2, default=str,
            )
    else:
        logger.warning(f"baseline '{baseline_name}' not in the results; statistical comparison skipped.")

    latex_table = dataframe_to_latex_table(
        summary_df, caption="Performance comparison across all variants (mean $\\pm$ std over seeds)",
        label="tab:main_results",
    )
    with open(output_dir / "main_results_table.tex", "w", encoding="utf-8") as f:
        f.write(latex_table)

    logger.info("analysis complete; all outputs in " + str(output_dir))
    return {
        "summary_table": summary_df,
        "convergence_report": conv_df,
        "statistical_comparison": stats_result,
        "output_dir": str(output_dir),
    }
