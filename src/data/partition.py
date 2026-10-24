# -*- coding: utf-8 -*-
"""
partition.py — federated partitioning of the graph across clients
========================================================
🔴 This module directly addresses the most serious weakness of the earlier version:

    "A random node partition with K=10 reduces edge retention to ≈15.6% and the
     mean within-client degree to ≈0.49; in this regime graph message passing
     is almost ineffective and the GNN effectively becomes an MLP."

Solution: the default partition is **community-based (Louvain)**, which keeps as
many edges as possible inside each client instead of cutting the graph at random.
The "random_node" and "dirichlet_label" partitions are kept only for the
sensitivity analysis in the paper, precisely to be able to show why Louvain
was chosen.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from src.config import PartitionConfig

logger = logging.getLogger("fedgnn.data.partition")


# ==============================================================================
# Shared helpers
# ==============================================================================
def _to_networkx_undirected(data: Data) -> nx.Graph:
    """Build a light nx.Graph (structure only, no feature copy) for community detection."""
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    edge_index = data.edge_index.cpu().numpy()
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    G.add_edges_from(edges)
    return G


def _community_metric(nodes: List[int], data: Data, objective: str) -> float:
    """Size measure of a community for bin packing, according to balance_objective."""
    if objective == "nodes":
        return float(len(nodes))
    if objective == "positives":
        y = data.y.cpu().numpy()
        return float(y[nodes].sum())
    if objective == "edges":
        # fast estimate: sum of node degrees (instead of an exact count of internal edges, for speed)
        deg = torch.zeros(data.num_nodes, dtype=torch.long)
        deg.scatter_add_(0, data.edge_index[0], torch.ones(data.edge_index.shape[1], dtype=torch.long))
        return float(deg[nodes].sum().item())
    raise ValueError(f"unknown balance_objective: {objective}")


def _greedy_bin_pack(
    communities: List[List[int]],
    num_clients: int,
    data: Data,
    objective: str,
) -> Dict[int, List[int]]:
    """
    Greedy bin packing: communities are sorted from large to small and each is
    added to the bin (client) with the currently lowest load. This is the
    standard LPT (Longest Processing Time first) strategy and bounds the final
    deviation from the mean.
    """
    community_sizes = [
        (_community_metric(c, data, objective), c) for c in communities
    ]
    community_sizes.sort(key=lambda t: t[0], reverse=True)

    bins: Dict[int, List[int]] = {k: [] for k in range(num_clients)}
    bin_loads = np.zeros(num_clients, dtype=np.float64)

    for size, community in community_sizes:
        target = int(np.argmin(bin_loads))
        bins[target].extend(community)
        bin_loads[target] += size

    return bins


def _assignment_from_bins(bins: Dict[int, List[int]], num_nodes: int) -> np.ndarray:
    assignment = np.full(num_nodes, -1, dtype=np.int64)
    for client_id, nodes in bins.items():
        assignment[nodes] = client_id
    if (assignment == -1).any():
        raise RuntimeError("❌ some nodes were not assigned to any client (internal partition bug).")
    return assignment


# ==============================================================================
# 1) Louvain scheme (default)
# ==============================================================================
def _partition_louvain(data: Data, cfg: PartitionConfig, seed: int) -> np.ndarray:
    G = _to_networkx_undirected(data)
    communities = nx.algorithms.community.louvain_communities(
        G, resolution=cfg.louvain_resolution, seed=seed
    )
    communities = [list(c) for c in communities]
    logger.info(f"Louvain: {len(communities)} communities found (resolution={cfg.louvain_resolution}).")

    bins = _greedy_bin_pack(communities, cfg.num_clients, data, cfg.balance_objective)
    return _assignment_from_bins(bins, data.num_nodes)


# ==============================================================================
# 2) METIS scheme (optional, with automatic fallback)
# ==============================================================================
def _partition_metis(data: Data, cfg: PartitionConfig, seed: int) -> np.ndarray:
    try:
        import metis  # type: ignore
    except ImportError:
        logger.warning(
            "⚠️ the 'metis' library is not installed; graceful degradation -> falling back to Louvain."
        )
        return _partition_louvain(data, cfg, seed)

    G = _to_networkx_undirected(data)
    G.graph["node_weight_attr"] = None
    try:
        _, parts = metis.part_graph(G, nparts=cfg.num_clients, ufactor=cfg.metis_ufactor, seed=seed)
    except Exception as e:
        logger.warning(f"⚠️ METIS failed ({e}); falling back to Louvain.")
        return _partition_louvain(data, cfg, seed)

    return np.array(parts, dtype=np.int64)


# ==============================================================================
# 3) Random node scheme (old baseline, for sensitivity analysis only)
# ==============================================================================
def _partition_random_node(data: Data, cfg: PartitionConfig, seed: int) -> np.ndarray:
    """
    ⚠️ This is the scheme that destroyed the graph structure in the earlier version.
    It is kept deliberately so that the paper can show with numbers why it
    should not be used (ablation table over partition.scheme).
    """
    rng = np.random.RandomState(seed)
    assignment = rng.randint(0, cfg.num_clients, size=data.num_nodes)
    return assignment.astype(np.int64)


# ==============================================================================
# 4) Dirichlet scheme on labels (strong label-skew non-IID)
# ==============================================================================
def _partition_dirichlet_label(data: Data, cfg: PartitionConfig, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    y = data.y.cpu().numpy()
    assignment = np.full(data.num_nodes, -1, dtype=np.int64)

    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        proportions = rng.dirichlet(alpha=[cfg.dirichlet_alpha] * cfg.num_clients)
        # cut indices from the cumulative proportions
        cut_points = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        splits = np.split(idx, cut_points)
        for client_id, split_idx in enumerate(splits):
            assignment[split_idx] = client_id

    return assignment


# ==============================================================================
# 5) Temporal shard scheme (split by time window; another sensitivity setting)
# ==============================================================================
def _partition_temporal_shard(data: Data, cfg: PartitionConfig, seed: int) -> np.ndarray:
    time_step = data.time_step.cpu().numpy()
    unique_steps = np.sort(np.unique(time_step))
    shards = np.array_split(unique_steps, cfg.num_clients)
    step_to_client = {}
    for client_id, steps in enumerate(shards):
        for s in steps:
            step_to_client[int(s)] = client_id
    assignment = np.array([step_to_client[int(t)] for t in time_step], dtype=np.int64)
    return assignment


# ==============================================================================
# Run the selected scheme
# ==============================================================================
_SCHEME_DISPATch = {
    "louvain": _partition_louvain,
    "metis": _partition_metis,
    "random_node": _partition_random_node,
    "dirichlet_label": _partition_dirichlet_label,
    "temporal_shard": _partition_temporal_shard,
}


def _run_scheme(scheme: str, data: Data, cfg: PartitionConfig, seed: int) -> np.ndarray:
    if scheme not in _SCHEME_DISPATch:
        raise ValueError(f"unknown partition.scheme: {scheme}")
    return _SCHEME_DISPATch[scheme](data, cfg, seed)


# ==============================================================================
# Repair of invalid assignments (reassignment)
# ==============================================================================
def _validate_and_fix(
    assignment: np.ndarray, data: Data, cfg: PartitionConfig, seed: int
) -> np.ndarray:
    """
    Check that every client has at least min_nodes_per_client nodes and
    min_positives_per_client_train positive nodes in its train part.
    On violation, a node is moved from the largest client to the smallest one.
    """
    rng = np.random.RandomState(seed + 999)
    y = data.y.cpu().numpy()
    train_mask = data.train_mask.cpu().numpy()

    for attempt in range(cfg.max_repartition_attempts):
        counts = np.bincount(assignment, minlength=cfg.num_clients)
        pos_counts = np.array([
            int(((assignment == k) & train_mask & (y == 1)).sum())
            for k in range(cfg.num_clients)
        ])

        violations = np.where(
            (counts < cfg.min_nodes_per_client) |
            (pos_counts < cfg.min_positives_per_client_train)
        )[0]

        if len(violations) == 0:
            logger.info(f"✓ client assignment is valid (attempt {attempt}).")
            return assignment

        if not cfg.reassign_if_infeasible:
            logger.warning(
                f"⚠️ {len(violations)} clients do not meet the minimum conditions and "
                f"reassign_if_infeasible=False; continuing without repair."
            )
            return assignment

        donor = int(np.argmax(counts))
        needy = int(violations[0])
        donor_nodes = np.where(assignment == donor)[0]

        # if the shortfall is of type 'positive', positive nodes are moved first
        if pos_counts[needy] < cfg.min_positives_per_client_train:
            donor_pos_nodes = donor_nodes[
                (y[donor_nodes] == 1) & train_mask[donor_nodes]
            ]
            candidates = donor_pos_nodes if len(donor_pos_nodes) > 0 else donor_nodes
        else:
            candidates = donor_nodes

        n_move = min(
            max(cfg.min_nodes_per_client - counts[needy], 1),
            max(len(candidates) - cfg.min_nodes_per_client, 1),
        )
        n_move = max(1, min(n_move, len(candidates)))
        move_idx = rng.choice(candidates, size=n_move, replace=False)
        assignment[move_idx] = needy

    logger.warning(
        f"⚠️ after {cfg.max_repartition_attempts} attempts not all minimum constraints "
        f"were satisfied. Record the final distribution in the report (transparency)."
    )
    return assignment


# ==============================================================================
# Partition diagnostics (for the paper and for aggregation.py)
# ==============================================================================
def compute_partition_diagnostics(
    assignment: np.ndarray, data: Data, cfg: PartitionConfig
) -> Dict:
    """
    Key statistics for:
      (a) the partition-quality table in the paper (edge retention, avg degree),
      (b) structure-quality weighting in AggregationConfig (contribution 1).
    """
    edge_index = data.edge_index.cpu().numpy()
    src, dst = edge_index[0], edge_index[1]
    same_client = assignment[src] == assignment[dst]

    total_edges = len(src)
    intra_edges = int(same_client.sum())
    edge_retention_rate = intra_edges / max(total_edges, 1)

    per_client = {}
    y = data.y.cpu().numpy()
    train_mask = data.train_mask.cpu().numpy()

    for k in range(cfg.num_clients):
        node_mask = assignment == k
        n_nodes_k = int(node_mask.sum())
        edge_mask_k = same_client & (assignment[src] == k)
        n_edges_k = int(edge_mask_k.sum())
        avg_degree_k = (2 * n_edges_k) / max(n_nodes_k, 1)  # each edge has 2 endpoints

        n_pos_train_k = int((node_mask & train_mask & (y == 1)).sum())
        n_train_k = int((node_mask & train_mask).sum())
        local_pos_rate_k = n_pos_train_k / max(n_train_k, 1)

        per_client[k] = {
            "num_nodes": n_nodes_k,
            "num_intra_edges": n_edges_k,
            "avg_degree": avg_degree_k,
            "num_train": n_train_k,
            "num_positives_train": n_pos_train_k,
            "local_positive_rate": local_pos_rate_k,
        }

    sizes = np.array([v["num_nodes"] for v in per_client.values()], dtype=np.float64)
    imbalance_ratio = float(sizes.std() / max(sizes.mean(), 1e-9))

    diagnostics = {
        "scheme": cfg.scheme,
        "num_clients": cfg.num_clients,
        "total_edges_global": total_edges,
        "intra_client_edges": intra_edges,
        "edge_retention_rate": edge_retention_rate,
        "size_imbalance_ratio": imbalance_ratio,
        "per_client": per_client,
    }

    logger.info(
        f"📊 partition '{cfg.scheme}': edge_retention={edge_retention_rate:.3f} | "
        f"imbalance={imbalance_ratio:.3f} | "
        f"avg_degree_overall={np.mean([v['avg_degree'] for v in per_client.values()]):.3f}"
    )
    return diagnostics


def partition_graph(
    data: Data, cfg: PartitionConfig, seed: int
) -> Tuple[np.ndarray, Dict]:
    """
    Main entry point of federated partitioning.

    Following the lessons of the earlier version, this function:
      1) runs the selected scheme (default: louvain),
      2) validates the assignment against the minimum constraints (min_nodes/min_positives)
         and repairs it if needed,
      3) computes the full diagnostics (edge retention, avg degree, imbalance) and
         returns them with the assignment, so that they can be used directly both
         in the run log and in the paper tables.

    Args:
        data: global Elliptic graph (output of elliptic.load_elliptic_graph).
        cfg: partition settings (PartitionConfig).
        seed: seed of this run. If cfg.repartition_per_seed=True, the
            partition itself is one of the sources of between-seed variance
            (a more honest approach than keeping the partition fixed).

    Returns:
        (assignment, diagnostics):
            assignment: array [N] with values {0..num_clients-1}
            diagnostics: dict of full partition-quality statistics
    """
    # v7: the fixed seed is configurable (v6a hard-coded 0).
    effective_seed = seed if cfg.repartition_per_seed else int(getattr(cfg, "fixed_partition_seed", 0))

    logger.info(
        f"🔧 partitioning started | scheme={cfg.scheme} | "
        f"num_clients={cfg.num_clients} | seed={effective_seed} "
        f"(repartition_per_seed={cfg.repartition_per_seed})"
    )

    assignment = _run_scheme(cfg.scheme, data, cfg, effective_seed)
    assignment = _validate_and_fix(assignment, data, cfg, effective_seed)
    diagnostics = compute_partition_diagnostics(assignment, data, cfg)

    # record the effective seed for full transparency in the final artifact
    diagnostics["effective_partition_seed"] = effective_seed
    diagnostics["repartition_per_seed"] = cfg.repartition_per_seed

    return assignment, diagnostics


def run_partition_sensitivity(
    data: Data, cfg: PartitionConfig, seed: int
) -> Dict[str, Dict]:
    """
    Run several partition schemes (cfg.sensitivity_schemes) on the same graph
    and the same seed, to produce the comparison table of sensitivity to the
    partition type in the paper. The absence of exactly this analysis in the
    earlier version is why the random-partition problem went unnoticed.

    Returns:
        dict {scheme_name: diagnostics} for all requested schemes.
    """
    results: Dict[str, Dict] = {}
    original_scheme = cfg.scheme

    for scheme in cfg.sensitivity_schemes:
        cfg.scheme = scheme
        logger.info(f"📐 partition sensitivity analysis: running scheme '{scheme}'...")
        assignment, diagnostics = partition_graph(data, cfg, seed)
        results[scheme] = diagnostics

    cfg.scheme = original_scheme  # restore the original setting; no side effect
    return results


# ==============================================================================
# Compatibility layer with the src/train.py contract
# ==============================================================================
def partition_graph_federated(data, cfg, seed: int = 42, **kw):
    """Name expected by train.py; delegates directly to partition_graph."""
    return partition_graph(data, cfg, seed, **kw)
