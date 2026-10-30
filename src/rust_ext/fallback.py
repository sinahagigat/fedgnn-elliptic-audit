# -*- coding: utf-8 -*-
"""Vectorized numpy computation of boundary nodes, O(E); a few milliseconds
for the arcs of the labelled Elliptic subgraph.
"""

from __future__ import annotations

from typing import Dict, Set, Tuple

import numpy as np


def compute_boundary_fast(
    edge_index_np: np.ndarray,
    assignment_np: np.ndarray,
    num_clients: int,
) -> Tuple[Dict[int, int], Dict[int, Set[int]]]:
    """
    Args:
        edge_index_np: [2, E] unsorted arc array (after make_undirected)
        assignment_np: [N] client assignment of each node
        num_clients: total number of clients

    Returns:
        boundary_degree: {global_node_idx: number of external arcs}
        neighbor_client_map: {global_node_idx: set(ids of external neighbour clients)}
    """
    src, dst = edge_index_np[0], edge_index_np[1]
    client_src = assignment_np[src]
    client_dst = assignment_np[dst]

    is_boundary_edge = client_src != client_dst
    b_src = src[is_boundary_edge]
    b_dst = dst[is_boundary_edge]
    b_client_dst = client_dst[is_boundary_edge]  # client on the other side of src

    boundary_degree: Dict[int, int] = {}
    neighbor_client_map: Dict[int, Set[int]] = {}

    # fast boundary-degree count with np.bincount over boundary src nodes
    if len(b_src) > 0:
        counts = np.bincount(b_src, minlength=assignment_np.shape[0])
        nonzero_nodes = np.nonzero(counts)[0]
        for node in nonzero_nodes:
            boundary_degree[int(node)] = int(counts[node])

        # build the neighbour-client map (this part still has a light loop, but
        # it only runs over the subset of boundary arcs, not the whole graph)
        order = np.argsort(b_src, kind="stable")
        b_src_sorted = b_src[order]
        b_client_dst_sorted = b_client_dst[order]

        unique_nodes, start_idx = np.unique(b_src_sorted, return_index=True)
        start_idx = list(start_idx) + [len(b_src_sorted)]

        for i, node in enumerate(unique_nodes):
            s, e = start_idx[i], start_idx[i + 1]
            neighbor_client_map[int(node)] = set(b_client_dst_sorted[s:e].tolist())

    return boundary_degree, neighbor_client_map
