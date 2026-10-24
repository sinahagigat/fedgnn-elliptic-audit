
# -*- coding: utf-8 -*-
"""
rust_ext — optional Rust accelerator for computing boundary nodes
==============================================================================
Why this module? The Python loop in federated_dataset._compute_boundary_info
could become a bottleneck on graphs larger than Elliptic (future scaling).
This package provides an extension point:

  1) try to import a compiled Rust (PyO3) extension named
     `fedgnn_rust`, if it has been built and installed (it is not needed in
     the current environment).
  2) otherwise, degrade gracefully to the vectorized numpy implementation
     in fallback.py, which is fast enough for the size of Elliptic
     (about 46 thousand nodes).

As a result the project never crashes because a Rust toolchain is missing; it only
emits an informational log message.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("fedgnn.rust_ext")

_BACKEND = "fallback"

try:
    import fedgnn_rust as _rust_backend  # type: ignore
    _BACKEND = "rust"
    logger.info("🦀 Rust extension (fedgnn_rust) loaded — using the fast backend.")
except ImportError:
    from src.rust_ext.fallback import compute_boundary_fast as _rust_backend_compute_boundary_fallback
    logger.info(
        "ℹ️ Rust extension not found; using the vectorized numpy fallback "
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
