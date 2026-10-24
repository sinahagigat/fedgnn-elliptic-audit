# -*- coding: utf-8 -*-
"""
main.py — entry point of the full pipeline (the only file run directly)
==============================================================================
Execution flow:
  1) parse CLI arguments (config path, resume on/off, output path)
  2) load and validate the Config (fail fast if the config is invalid)
  3) run the full experiment matrix from train.py (resume by default)
  4) produce the final report from analyze.py (tables + statistical tests + LaTeX)
  5) print a human-readable summary to stdout

Usage in Colab:
    !python main.py --config configs/main_config.yaml --output-dir /content/results
"""

from __future__ import annotations

import os
# v7: cuBLAS needs this before CUDA initialises for deterministic GEMMs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import logging
import sys
from pathlib import Path

from src.config import load_config
from src.train import run_all_experiments
from src.analyze import generate_full_report
from src.utils.common import ensure_dir


# ==============================================================================
# Unified logging set-up: the same format in every module
# ==============================================================================
def setup_logging(log_path: Path, verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger("fedgnn")
    root_logger.setLevel(level)
    root_logger.handlers.clear()  # avoid duplicate handlers on re-runs

  # --- console handler (live log in Colab) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # --- file handler (kept for later inspection, even if Colab disconnects) ---
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    root_logger.info(f"📝 logging initialised | level={logging.getLevelName(level)} | file={log_path}")


# ==============================================================================
# CLI argument parsing
# ==============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FedGNN-Fraud v5.0.0 — run the full experiment matrix + final analysis"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="path to the YAML configuration file (e.g. configs/main_config.yaml)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="/content/results",
        help="folder for artifacts.json, logs and the final report",
    )
    parser.add_argument(
        "--artifacts-name", type=str, default="artifacts.json",
        help="name of the artifacts file (to keep several versions side by side)",
    )
    parser.add_argument(
        "--baseline", type=str, default="FedAvg_GCN",
        help="name of the variant used as the baseline in the statistical comparisons",
    )
    parser.add_argument(
        "--skip-analysis", action="store_true",
        help="run only the experiment matrix and skip the final analysis (useful for quick resume)",
    )
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="v7: run only these seeds, e.g. '42-46' or '42,43,50'",
    )
    parser.add_argument(
        "--variants", type=str, default=None,
        help="v7: run only these variants, comma-separated names",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="v7: folder with the three Elliptic CSV files (overrides data.raw_dir; not part of the config hash)",
    )
    parser.add_argument(
        "--threads", type=int, default=None,
        help="v7: number of CPU threads for torch in this process",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="enable DEBUG-level logging (full per-round/per-client detail)",
    )
    return parser


# ==============================================================================
# Main function
# ==============================================================================
# --- install the config-name compatibility layer (must run before any Config is built) ---
from src.compat import install_config_compat, compat_report  # noqa: E402
import src.config as _fedgnn_config_module  # noqa: E402
install_config_compat(_fedgnn_config_module)


def _parse_seeds(text: str):
    """'42-46' -> [42..46]; '42,45,50' -> [42,45,50]; mixtures allowed."""
    out = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main() -> int:
    args = build_arg_parser().parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    artifacts_path = output_dir / args.artifacts_name

    setup_logging(log_path=output_dir / "run.log", verbose=args.verbose)
    logger = logging.getLogger("fedgnn.main")

    config_path = Path(args.config)
    assert config_path.exists(), f"❌ config file not found: {config_path}"

    logger.info("=" * 70)
    logger.info("🚀 starting FedGNN-Fraud v5.0.0")
    logger.info(f"📄 config: {config_path}")
    logger.info(f"💾 output: {output_dir}")
    logger.info(f"📦 artifacts: {artifacts_path}")
    logger.info("=" * 70)

    try:
        cfg = load_config(config_path)
        if args.data_dir:
            # raw_dir and cache_dir are volatile paths: the config hash is unchanged
            cfg.data.raw_dir = str(Path(args.data_dir))
            cfg.data.cache_dir = str(Path(args.data_dir) / "cache")
        if args.threads:
            import torch
            torch.set_num_threads(int(args.threads))
            logger.info(f"torch threads = {torch.get_num_threads()}")
    except Exception as exc:
        logger.error(f"❌ error while loading/validating the config — run stopped (fail fast): {exc}")
        return 1

    logger.info(
        f"📊 experiment matrix: {len(cfg.experiment.variants)} variants × "
        f"{len(cfg.experiment.seeds)} seed = "
        f"{len(cfg.experiment.variants) * len(cfg.experiment.seeds)} runs in total."
    )

    # --- stage 1: run the full experiment matrix (resume by default) ---
    only_seeds = _parse_seeds(args.seeds) if args.seeds else None
    only_variants = [v.strip() for v in args.variants.split(",")] if args.variants else None
    if only_seeds or only_variants:
        logger.info(f"shard: seeds={only_seeds} variants={only_variants}")
    try:
        run_all_experiments(cfg, artifacts_path, only_seeds=only_seeds, only_variants=only_variants)
    except Exception as exc:
        logger.error(f"❌ unexpected error while running the experiment matrix: {exc}", exc_info=True)
        return 1

    if args.skip_analysis:
        logger.info("⏭️ final analysis skipped (--skip-analysis). Run finished successfully.")
        return 0

    # --- stage 2: final analysis + tables/statistical tests/LaTeX ---
    try:
        report = generate_full_report(
            artifacts_path=artifacts_path,
            output_dir=output_dir / "analysis",
            baseline_name=args.baseline,
        )
    except Exception as exc:
        logger.error(
            f"❌ error in the final analysis stage — artifacts.json is complete "
            f"and analyze.py can be run separately later: {exc}", exc_info=True
        )
        return 1

    # --- final summary (robust to key-name differences in analyze.py) ---
    logger.info("=" * 70)
    logger.info("🏁 full run finished successfully.")
    if isinstance(report, dict):
        logger.info(f"🔑 report keys: {list(report.keys())}")
    df = report.get("summary_table") if isinstance(report, dict) else None
    if df is None and isinstance(report, dict):
        for k, v in report.items():
            if hasattr(v, "iterrows"):
                df = v
                break
    if df is not None and hasattr(df, "head"):
        logger.info("📋 summary table (first 3 rows):")
        for _, row in df.head(3).iterrows():
            logger.info("   • " + " | ".join(f"{c}={row[c]}" for c in df.columns))
    else:
        logger.warning("⚠️ summary table not found — check the output of analyze.py.")
    stat = None
    if isinstance(report, dict):
        stat = report.get("statistical_comparison") or report.get("comparisons")
    if isinstance(stat, dict):
        sig = stat.get("significant_after_holm")
        logger.info(f"📈 significant after Holm (vs {args.baseline}): {sig or 'none'}")
    logger.info(f"📂 all outputs in: {output_dir}")
    compat_report()
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
