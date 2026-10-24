# -*- coding: utf-8 -*-
"""
io_utils.py — atomic writes + checkpoint/resume
================================================
Goal: a crash in run 17 of 18 must not send you back to run 1 (resume by default),
and a half-written JSON file must never corrupt earlier artifacts (atomic writes).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch


# ------------------------------------------------------------------------------
# Atomic writes
# ------------------------------------------------------------------------------
def atomic_write_json(data: Dict[str, Any], path: str) -> None:
    """
    Write JSON atomically: write to a temporary file, then rename it
    (rename is atomic on POSIX), so that after a crash or power loss the
    final file is never half-written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, p)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_json(path: str) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------------------
# Checkpoint / Resume
# ------------------------------------------------------------------------------
def run_signature(variant: str, seed: int, config_hash: str) -> str:
    """Unique identifier of one run."""
    return f"{variant}__seed{seed}__{config_hash}"


def artifact_path(results_dir: str, variant: str, seed: int, config_hash: str) -> Path:
    sig = run_signature(variant, seed, config_hash)
    return Path(results_dir) / f"artifacts_{sig}.json"


def is_run_completed(results_dir: str, variant: str, seed: int, config_hash: str) -> bool:
    """
    Check whether a complete, valid artifact exists for this (variant, seed, hash).
    The file existing is not enough; the key 'completed': True must also be set
    (so that a half-written artifact from a crash is not treated as complete).
    """
    p = artifact_path(results_dir, variant, seed, config_hash)
    if not p.exists():
        return False
    data = load_json(str(p))
    return bool(data and data.get("completed") is True)


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    """
    Atomically save a model/server checkpoint (weight state_dict + round metadata).
    Tensors are written with torch.save to a temporary file followed by os.replace.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = p.with_suffix(p.suffix + ".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, p)


def load_checkpoint(path: str, map_location: str = "cpu") -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    return torch.load(p, map_location=map_location)


def checkpoint_path(results_dir: str, variant: str, seed: int, config_hash: str) -> Path:
    sig = run_signature(variant, seed, config_hash)
    d = Path(results_dir) / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"ckpt_{sig}.pt"
