# -*- coding: utf-8 -*-
"""Logging to file and console; one log file per variant and seed, named with
the configuration hash.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_LOGGERS: dict = {}


def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    variant: Optional[str] = None,
    seed: Optional[int] = None,
    config_hash: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create or retrieve a logger with a unique name. If a logger with this name
    already exists, it is returned without adding duplicate handlers (avoids
    doubled logs in notebooks whose cells are run several times).
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        parts = [p for p in [variant, f"seed{seed}" if seed is not None else None,
                              config_hash] if p]
        fname = "_".join(parts) if parts else "run"
        fh = logging.FileHandler(Path(log_dir) / f"{fname}.log", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    _LOGGERS[name] = logger
    return logger


def get_run_logger(variant: str, seed: int, config_hash: str, results_dir: str = "./results"):
    """Shortcut for creating the logger of one federated run."""
    name = f"run.{variant}.{seed}.{config_hash}"
    return setup_logger(
        name=name,
        log_dir=f"{results_dir}/logs",
        variant=variant,
        seed=seed,
        config_hash=config_hash,
    )
