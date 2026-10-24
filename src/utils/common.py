# -*- coding: utf-8 -*-
"""
common.py — shared helpers (single source of truth for seeding, atomic I/O and time)
==============================================================================
Every function needed by more than one module (seeding, atomic JSON read/write,
commit hash, unique run keys) is defined here.
train.py, main.py and analyze.py all use these helpers.
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger("fedgnn.utils.common")


# ==============================================================================
# Full reproducibility: single source of truth for seeding every RNG
# ==============================================================================
def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """
    Seed every RNG (random, numpy, torch CPU and CUDA).

    v7: with deterministic=True, torch.use_deterministic_algorithms(True) is
    also enabled. Without it, CUDA scatter/index_add kernels used by PyG
    message passing are non-deterministic, so two runs with the same seed can
    differ (the v6a runs measured up to 0.0057 AP between FedAvg and FedProx,
    which are mathematically identical with one local step).
    CUBLAS_WORKSPACE_CONFIG must be set before CUDA initialises; main.py does
    that at import time. warn_only=True means an op without a deterministic
    kernel emits a warning instead of crashing; the determinism check in the
    notebook verifies the outcome empirically.
    """
    assert isinstance(seed, int) and seed >= 0, f"invalid seed: {seed}"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    logger.debug(f"seed={seed} set (deterministic={deterministic})")


def code_fingerprint(root: Optional[Path] = None) -> str:
    """
    SHA-256 over the contents of every .py file under src/ plus main.py,
    in sorted path order. Stored in every run record, so the exact code
    that produced a result can be identified later even without git.
    """
    import hashlib
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    files = sorted(list((root / "src").rglob("*.py")) + [root / "main.py"])
    h = hashlib.sha256()
    for f in files:
        if f.exists() and "__pycache__" not in f.parts:
            h.update(str(f.relative_to(root)).replace("\\", "/").encode("utf-8"))
            h.update(f.read_bytes())
    return h.hexdigest()[:12]


def environment_info() -> Dict[str, Any]:
    """Library versions and hardware, stored with every run record."""
    import platform
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }
    try:
        import torch_geometric
        info["torch_geometric"] = torch_geometric.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0)
    else:
        info["cpu"] = platform.processor() or platform.machine()
    return info


# ==============================================================================
# Time and run identifiers
# ==============================================================================
def utc_timestamp() -> str:
    """ISO-8601 timestamp in UTC (no time-zone ambiguity)."""
    return datetime.now(timezone.utc).isoformat()


def make_run_key(variant_name: str, seed: int) -> str:
    """Unique key of each (variant, seed), used to index artifacts.json."""
    assert variant_name, "❌ make_run_key: variant_name is empty."
    return f"{variant_name}__seed{seed}"


# ==============================================================================
# Version information: commit hash for exact code reproducibility
# ==============================================================================
def get_git_commit_hash() -> Optional[str]:
    """
    Current git commit hash (if the project is inside a git repository). On
    Colab there is usually no git repository; in that case the function
    degrades gracefully and returns None, which is stored in the artifact as
    "git_commit": null instead of an error that would stop the run.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        logger.debug("ℹ️ git commit hash not available (probably Colab without a git repo).")
        return None


# ==============================================================================
# Atomic JSON I/O: protects files against a sudden Colab disconnect
# ==============================================================================
def load_json_if_exists(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    """
    If the file exists, return its content (basis of resume by default);
    otherwise return the default (without writing a new file yet).
    """
    path = Path(path)
    if not path.exists():
        logger.info(f"ℹ️ file {path} does not exist — starting from the default state.")
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        logger.error(
            f"❌ file {path} is corrupt/incomplete ({exc}) — instead of crashing the whole project, "
            f"a backup {path}.corrupted is made and execution continues from the default state."
        )
        path.rename(path.with_suffix(path.suffix + ".corrupted"))
        return default


def atomic_save_json(path: Path, data: Dict[str, Any]) -> None:
    """
    Atomic JSON write: the data is first written to a temporary file in the same
    directory and then moved over the target with os.replace (atomic on POSIX/NTFS).
    Even if Colab disconnects in the middle of writing, artifacts.json is never
    left half-written: either the previous complete version remains or the new
    complete version replaces it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp_path, path)  # atomic, not copy+delete
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def ensure_dir(path: Path) -> Path:
    """Create a directory (with parents) if it does not exist (idempotent)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
