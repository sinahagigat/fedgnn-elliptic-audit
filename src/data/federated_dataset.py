# -*- coding: utf-8 -*-
"""
federated_dataset.py — client subgraphs + masks + boundary statistics
=====================================================================
This module takes the global graph + the partition assignment and builds an
independent torch_geometric.data.Data (local subgraph) for every client, which is
passed directly to src/fed/client.py.

Key points:
  1) each client subgraph contains only within-client edges (induced
     subgraph). Boundary edges (between two clients) are counted in this module
     and extracted as boundary-node statistics, so that the virtual-node layer
     can align clients without revealing the identity of the data.
  2) local<->global indices are kept for every client so that they can be
     traced in the final analysis (e.g. per-client tables).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from src.config import PartitionConfig, VirtualNodeConfig

logger = logging.getLogger("fedgnn.data.federated_dataset")


# ==============================================================================
# Data structure for the boundary information of each client
# ==============================================================================
@dataclass
class BoundaryInfo:
    """
    Boundary-node information of a client, i.e. nodes that have at least one
    edge to another client. This information is the basis of the virtual
    nodes (contribution 2).
    """
    local_boundary_idx: np.ndarray               # local indices of boundary nodes
    global_boundary_idx: np.ndarray               # matching global indices
    neighbor_client_ids: Dict[int, List[int]] = field(default_factory=dict)
    # {local index of a boundary node -> list of external neighbour client_ids}
    boundary_degree: np.ndarray = None            # number of external edges of each boundary node


# ==============================================================================
# Subgraph of one client
# ==============================================================================
def _build_client_subgraph(
    data: Data,
    assignment: np.ndarray,
    client_id: int,
) -> Tuple[Data, np.ndarray]:
    """
    Extract the induced subgraph of client client_id.

    Returns:
        (client_data, global_node_idx):
            client_data: independent Data with local x, y, edge_index, masks
            global_node_idx: [n_local] mapping local index -> global index
    """
    global_node_idx = np.where(assignment == client_id)[0]
    node_idx_t = torch.tensor(global_node_idx, dtype=torch.long)

    edge_index_local, _ = subgraph(
        node_idx_t,
        data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
    )

    client_data = Data(
        x=data.x[node_idx_t].clone(),
        y=data.y[node_idx_t].clone(),
        edge_index=edge_index_local,
    )
    client_data.train_mask = data.train_mask[node_idx_t].clone()
    client_data.val_mask = data.val_mask[node_idx_t].clone()
    client_data.test_mask = data.test_mask[node_idx_t].clone()
    client_data.time_step = data.time_step[node_idx_t].clone()

    return client_data, global_node_idx


# ==============================================================================
# Boundary information (without revealing raw data)
# ==============================================================================
def _compute_boundary_info(
    data: Data,
    assignment: np.ndarray,
    client_id: int,
    global_node_idx: np.ndarray,
    global_to_local: Dict[int, int],
) -> BoundaryInfo:
    """
    Identify the boundary nodes of client client_id: nodes that have at least one
    global edge to a node in another client.

    Important for privacy: only counts/statistics are computed in this function;
    no raw feature of an external neighbour is read or stored. What is exchanged
    later (by the virtual-node layer) is only aggregate statistics (mean/variance),
    not raw data.
    """
    edge_index = data.edge_index.cpu().numpy()
    src, dst = edge_index[0], edge_index[1]

    client_node_set = set(global_node_idx.tolist())

    boundary_degree_map: Dict[int, int] = {}
    neighbor_client_map: Dict[int, set] = {}

    for s, d in zip(src, dst):
        s_in = s in client_node_set
        d_in = d in client_node_set
        if s_in and not d_in:
            boundary_degree_map[s] = boundary_degree_map.get(s, 0) + 1
            neighbor_client_map.setdefault(s, set()).add(int(assignment[d]))
        elif d_in and not s_in:
            boundary_degree_map[d] = boundary_degree_map.get(d, 0) + 1
            neighbor_client_map.setdefault(d, set()).add(int(assignment[s]))

    global_boundary_idx = np.array(sorted(boundary_degree_map.keys()), dtype=np.int64)
    local_boundary_idx = np.array(
        [global_to_local[g] for g in global_boundary_idx], dtype=np.int64
    )
    boundary_degree = np.array(
        [boundary_degree_map[g] for g in global_boundary_idx], dtype=np.int64
    )
    neighbor_client_ids = {
        global_to_local[g]: sorted(neighbor_client_map[g])
        for g in global_boundary_idx
    }

    return BoundaryInfo(
        local_boundary_idx=local_boundary_idx,
        global_boundary_idx=global_boundary_idx,
        neighbor_client_ids=neighbor_client_ids,
        boundary_degree=boundary_degree,
    )


# ==============================================================================
# Full federated dataset (public entry point)
# ==============================================================================
@dataclass
class FederatedClientData:
    """Complete data bundle of one client; direct input of src/fed/client.py."""
    client_id: int
    data: Data
    global_node_idx: np.ndarray
    boundary_info: BoundaryInfo
    num_nodes: int
    num_edges: int
    num_train: int
    num_positives_train: int


def build_federated_dataset(
    data: Data,
    assignment: np.ndarray,
    partition_cfg: PartitionConfig,
) -> List[FederatedClientData]:
    """
    Build the complete list of federated data for all clients.

    This function is the only meeting point between the output of partition.py and the
    input of fed/client.py; any change to the data representation must be made here only.
    """
    clients: List[FederatedClientData] = []

    for client_id in range(partition_cfg.num_clients):
        client_data, global_node_idx = _build_client_subgraph(data, assignment, client_id)

        if client_data.num_nodes == 0:
            logger.warning(f"⚠️ client {client_id} has no nodes; removed from the registry.")
            continue

        global_to_local = {g: i for i, g in enumerate(global_node_idx)}
        boundary_info = _compute_boundary_info(
            data, assignment, client_id, global_node_idx, global_to_local
        )

        fcd = FederatedClientData(
            client_id=client_id,
            data=client_data,
            global_node_idx=global_node_idx,
            boundary_info=boundary_info,
            num_nodes=int(client_data.num_nodes),
            num_edges=int(client_data.edge_index.shape[1]),
            num_train=int(client_data.train_mask.sum().item()),
            num_positives_train=int(
                client_data.y[client_data.train_mask].sum().item()
            ),
        )
        clients.append(fcd)

        logger.info(
            f"👤 client {client_id}: n_nodes={fcd.num_nodes} | n_edges={fcd.num_edges} | "
            f"n_train={fcd.num_train} | n_pos_train={fcd.num_positives_train} | "
            f"n_boundary={len(boundary_info.local_boundary_idx)}"
        )

    total_boundary = sum(len(c.boundary_info.local_boundary_idx) for c in clients)
    logger.info(
        f"✅ federated dataset built: {len(clients)} active clients | "
        f"total boundary nodes (counted repeatedly across clients)={total_boundary}"
    )
    return clients


# ==============================================================================
# Aggregate boundary statistics for the virtual nodes
# ==============================================================================
def compute_boundary_exchange_payload(
    client: FederatedClientData,
    virtual_cfg: VirtualNodeConfig,
) -> Dict[str, torch.Tensor]:
    """
    Build a client's exchange payload for virtual-node alignment,
    following the exchange_payload policy in VirtualNodeConfig.

    ⚠️ Privacy: the output of this function is only aggregate statistics (mean/variance
    of boundary-node features); no raw feature vector or node identifier is sent.
    Optional noise / secure-aggregation protection (src/fed/privacy.py) is applied to
    this same payload.

    Args:
        client: client data bundle (output of build_federated_dataset).
        virtual_cfg: virtual-node settings.

    Returns:
        dict containing at least 'mean' and, if required, 'var', together with
        'boundary_degree' for the attach_policy weighting.
    """
    bidx = client.boundary_info.local_boundary_idx
    if len(bidx) == 0:
        feat_dim = client.data.x.shape[1]
        zeros = torch.zeros(feat_dim)
        payload = {"mean": zeros, "boundary_degree": torch.tensor(0.0)}
        if virtual_cfg.exchange_payload == "mean_var":
            payload["var"] = zeros.clone()
        return payload

    x_boundary = client.data.x[bidx]  # [n_boundary, F]
    mean = x_boundary.mean(dim=0)

    payload: Dict[str, torch.Tensor] = {
        "mean": mean,
        "boundary_degree": torch.tensor(
            float(client.boundary_info.boundary_degree.sum())
        ),
    }

    if virtual_cfg.exchange_payload == "mean_var":
        var = x_boundary.var(dim=0, unbiased=False)
        payload["var"] = var
    elif virtual_cfg.exchange_payload == "sketch":
        # fixed random projection (Johnson–Lindenstrauss style) for dimensionality reduction
        # before exchange: lower communication cost without losing the overall statistics
        feat_dim = x_boundary.shape[1]
        rng = torch.Generator().manual_seed(client.client_id)
        proj = torch.randn(feat_dim, virtual_cfg.sketch_dim, generator=rng)
        proj = proj / (proj.norm(dim=0, keepdim=True) + 1e-8)
        payload["sketch"] = x_boundary.mean(dim=0) @ proj

    return payload


# ==============================================================================
# Compatibility layer with the src/train.py contract
# ==============================================================================
import copy as _copy
import torch as _torch


def build_federated_client_data(*args, **kw):
    """Name expected by train.py; delegates to build_federated_dataset."""
    return build_federated_dataset(*args, **kw)


def build_centralized_splits(full_data, seed: int = 42, cfg=None):
    """
    Centralized baseline: no client split is performed and the full graph is returned
    with the temporal masks built in elliptic.py.

    Important for the validity of the paper: the test mask here is exactly the mask that
    build_global_test_split gives to the federated variants. Centralized and
    FedGNN are therefore measured on one and the same test set, and the comparison is fair.
    """
    d = _copy.copy(full_data)
    for m in ("train_mask", "val_mask", "test_mask"):
        assert hasattr(d, m), f"[federated_dataset] mask '{m}' is missing."
    assert int(d.test_mask.sum()) > 0, "[federated_dataset] the test set is empty."
    return d


def build_global_test_split(full_data, cfg=None, seed: int = 42):
    """
    Global test set for evaluating the federated global model.

    The full graph (with all edges, including cross-client edges) is returned and
    only the test mask is active. Reason: at evaluation time the server has access to
    the full topology; the federated constraint concerns the *training* phase, not
    measurement. If evaluation were done on the clients' cut subgraphs, all federated
    methods would be penalized artificially and the comparison with Centralized would be invalid.
    """
    d = _copy.copy(full_data)
    assert hasattr(d, "test_mask"), "[federated_dataset] test_mask is missing."
    n_test = int(d.test_mask.sum())
    n_pos = int(d.y[d.test_mask].sum())
    assert n_test > 0, "[federated_dataset] the global test set is empty."
    assert n_pos > 0, (
        "[federated_dataset] the global test set has no positive example; "
        "AP/F1 would be undefined."
    )
    d.global_test_size = n_test
    d.global_test_positives = n_pos
    return d
