# -*- coding: utf-8 -*-
"""Timing of the numpy boundary computation against a plain Python loop.

    python -m src.rust_ext.benchmark
"""

from __future__ import annotations

import time

import numpy as np

from src.rust_ext.fallback import compute_boundary_fast


def _naive_python_loop(edge_index_np, assignment_np):
    src, dst = edge_index_np[0], edge_index_np[1]
    boundary_degree = {}
    neighbor_map = {}
    for s, d in zip(src.tolist(), dst.tolist()):
        if assignment_np[s] != assignment_np[d]:
            boundary_degree[s] = boundary_degree.get(s, 0) + 1
            neighbor_map.setdefault(s, set()).add(int(assignment_np[d]))
    return boundary_degree, neighbor_map


def run_benchmark(num_nodes: int = 46564, num_edges: int = 73248, num_clients: int = 10):
    rng = np.random.RandomState(42)
    src = rng.randint(0, num_nodes, size=num_edges)
    dst = rng.randint(0, num_nodes, size=num_edges)
    edge_index_np = np.stack([src, dst])
    assignment_np = rng.randint(0, num_clients, size=num_nodes)

    t0 = time.perf_counter()
    _naive_python_loop(edge_index_np, assignment_np)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    compute_boundary_fast(edge_index_np, assignment_np, num_clients)
    t_vectorized = time.perf_counter() - t0

    print(f"plain Python loop : {t_naive * 1000:.2f} ms")
    print(f"vectorized numpy  : {t_vectorized * 1000:.2f} ms")
    print(f"speed-up          : {t_naive / max(t_vectorized, 1e-9):.1f}x")


if __name__ == "__main__":
    run_benchmark()
