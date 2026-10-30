
# -*- coding: utf-8 -*-
"""Boundary-node computation: uses the compiled extension fedgnn_rust if it is
installed, otherwise the numpy implementation in fallback.py (fast enough for
Elliptic; all reported runs used it).
"""

from __future__ import annotations

import logging

logger = logging.getLogger("fedgnn.rust_ext")

_BACKEND = "fallback"

try:
    import fedgnn_rust as _rust_backend  # type: ignore
    _BACKEND = "rust"
    logger.info("Rust extension (fedgnn_rust) loaded — using the fast backend.")
except ImportError:
    from src.rust_ext.fallback import compute_boundary_fast as _rust_backend_compute_boundary_fallback
    logger.info(
        "Rust extension not found; using the vectorized numpy fallback "
        "(fully sufficient for the current dataset size)."
    )


def compute_boundary_fast(edge_index_np, assignment_np, num_clients: int):
    """
    Uniform interface: the Rust backend and the fallback share this signature.

    Returns:
        boundary_degree: dict[global_node_idx -> int]
        neighbor_client_map: dict[global_node_idx -> set[int]]
    """
    if _BACKEND == "rust":
        return _rust_backend.compute_boundary(edge_index_np, assignment_np, num_clients)
    return _rust_backend_compute_boundary_fallback(edge_index_np, assignment_np, num_clients)


def get_active_backend() -> str:
    return _BACKEND
