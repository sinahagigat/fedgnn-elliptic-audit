# -*- coding: utf-8 -*-
"""
seed.py — full reproducibility control (determinism)
================================================
Seeds random, numpy, torch (CPU/GPU) and PyTorch Geometric in one place.
Without it, two runs with the same configuration hash could produce different
results, which would undermine any reproducibility claim.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int, deterministic_algorithms: bool = True) -> None:
    """
    Seed every source of randomness.

    Args:
        seed: seed value.
        deterministic_algorithms: if True, non-deterministic CUDA algorithms
            (such as some scatter/gather kernels in PyG) are disabled. This can
            slow execution by 5-15% but makes runs reproducible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_torch_generator(seed: int) -> torch.Generator:
    """An independent torch.Generator for a DataLoader/Sampler with a given seed."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def worker_init_fn(worker_id: int, base_seed: Optional[int] = None) -> None:
    """For DataLoader(num_workers>0): give each worker its own deterministic seed."""
    seed = (base_seed if base_seed is not None else torch.initial_seed()) + worker_id
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)
