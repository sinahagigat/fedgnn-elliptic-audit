# -*- coding: utf-8 -*-
"""Server aggregation strategies.

fedavg and fedprox: data-size weighted average (the proximal term is applied in
client.py); fedproto: as fedavg, plus class-prototype averaging; fedgnn_hard:
average of the top-k clients by selection score; fedgnn_soft: softmax(score/T)
times data share. Virtual-node payloads are aggregated separately.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

from src.fed.client import ClientUpdate


def _align_devices(*tensors):
    """Align the input tensors on one device and a common floating dtype.

    The virtual-node payload is created on the model device (cuda), while quality
    weights come from a Python list and therefore live on the cpu. This function
    only aligns device/dtype and performs no statistical computation.
    """
    ts = [x for x in tensors if isinstance(x, torch.Tensor)]
    if not ts:
        return tensors if len(tensors) != 1 else tensors[0]
    dev = next((x.device for x in ts if x.is_cuda), ts[0].device)
    fdt = next((x.dtype for x in ts if x.is_floating_point()), None)
    out = []
    for x in tensors:
        if isinstance(x, torch.Tensor):
            if fdt is not None and x.is_floating_point():
                x = x.to(device=dev, dtype=fdt)
            else:
                x = x.to(device=dev)
        out.append(x)
    return tuple(out) if len(out) != 1 else out[0]


logger = logging.getLogger("fedgnn.fed.aggregation")


# Model-weight aggregation: FedAvg / FedProx (weighted by num_train)
def fedavg_aggregate(
    updates: List[ClientUpdate], selected_ids: Optional[List[int]] = None
) -> Dict[str, torch.Tensor]:
    """
    Weighted mean of the clients' state_dicts by number of training samples.
    If selected_ids is given, only that subset takes part in the aggregation
    (shared basis of hard_selective_aggregate).
    """
    chosen = [u for u in updates if selected_ids is None or u.client_id in selected_ids]
    if not chosen:
        raise ValueError("no client selected for aggregation.")

    total_n = sum(u.num_train for u in chosen)
    weights = [u.num_train / total_n for u in chosen]

    agg_state: Dict[str, torch.Tensor] = {}
    keys = chosen[0].state_dict.keys()
    for key in keys:
        stacked = torch.stack(
            [u.state_dict[key].float() * w for u, w in zip(chosen, weights)], dim=0
        )
        agg_state[key] = stacked.sum(dim=0).to(chosen[0].state_dict[key].dtype)

    return agg_state


# Global prototype aggregation: FedProto / Hard / Soft
def fedproto_aggregate_prototypes(
    updates: List[ClientUpdate], selected_ids: Optional[List[int]] = None
) -> Dict[int, torch.Tensor]:
    """
    Weighted mean of each class prototype across clients, weighted by the number
    of samples of that class at each client (not the client's total size),
    so that the prototype of class 1 is influenced only by clients that have
    seen class-1 samples.
    """
    chosen = [u for u in updates if selected_ids is None or u.client_id in selected_ids]
    class_sums: Dict[int, torch.Tensor] = {}
    class_weight_totals: Dict[int, float] = {}

    for u in chosen:
        for c, proto in u.local_prototypes.items():
            n_c = u.class_counts.get(c, 0)
            if n_c == 0:
                continue
            if c not in class_sums:
                class_sums[c] = proto.clone().float() * n_c
                class_weight_totals[c] = float(n_c)
            else:
                class_sums[c] += proto.float() * n_c
                class_weight_totals[c] += float(n_c)

    global_prototypes = {
        c: (class_sums[c] / class_weight_totals[c])
        for c in class_sums
        if class_weight_totals[c] > 0
    }
    return global_prototypes


# Hard selective aggregation
def hard_selective_aggregate(
    updates: List[ClientUpdate], top_k_ratio: float
) -> Dict[str, torch.Tensor]:
    """
    Hard selection: only the top-k clients by selection_score take part in the
    aggregation (weak or noisy clients are excluded entirely). This is the
    hard version; the soft version is soft_selective_aggregate.

    Args:
        top_k_ratio: share of clients kept (e.g. 0.5 keeps the upper half).
    """
    n_keep = max(1, int(round(len(updates) * top_k_ratio)))
    sorted_updates = sorted(updates, key=lambda u: u.selection_score, reverse=True)
    selected_ids = [u.client_id for u in sorted_updates[:n_keep]]

    logger.info(
        f"Hard-Selective: {n_keep}/{len(updates)} clients selected "
        f"(scores: {[round(u.selection_score, 3) for u in sorted_updates]})"
    )
    return fedavg_aggregate(updates, selected_ids=selected_ids)


# Soft selective aggregation
def soft_selective_aggregate(
    updates: List[ClientUpdate], temperature: float
) -> Dict[str, torch.Tensor]:
    """
    Soft weighting: instead of excluding weak clients, their aggregation weight is
    reduced by softmax(selection_score / temperature) but never set to zero,
    so every client keeps a share (full exclusion would harm federated
    fairness) while the influence of low-quality clients is limited.


    final weight of each client = softmax(score/T) ⊙ (num_train / sum(num_train))
    renormalized to sum to 1, so that both quality (score) and data volume
    (num_train) enter the decision.
    """
    scores = np.array([u.selection_score for u in updates], dtype=np.float64)
    scores_scaled = scores / max(temperature, 1e-8)
    scores_scaled -= scores_scaled.max()  # numerical stability of the softmax
    soft_weights = np.exp(scores_scaled)
    soft_weights /= soft_weights.sum()

    data_weights = np.array([u.num_train for u in updates], dtype=np.float64)
    data_weights /= data_weights.sum()

    combined = soft_weights * data_weights
    combined /= combined.sum()

    logger.info(
        f"Soft-Selective: final client weights = "
        f"{[round(w, 4) for w in combined.tolist()]}"
    )

    agg_state: Dict[str, torch.Tensor] = {}
    keys = updates[0].state_dict.keys()
    for key in keys:
        stacked = torch.stack(
            [u.state_dict[key].float() * w for u, w in zip(updates, combined)], dim=0
        )
        agg_state[key] = stacked.sum(dim=0).to(updates[0].state_dict[key].dtype)

    return agg_state


# Virtual-node payload aggregation (independent of the main algorithm)
def aggregate_virtual_payloads(
    updates: List[ClientUpdate],
) -> Optional[torch.Tensor]:
    """
    Simple average of all clients' raw boundary payloads (weighted by each
    client's boundary_degree), returning one global reference vector that is
    broadcast to all clients in the next round (client.set_virtual_reference).

    Note: only the client-level 'mean' statistic is aggregated (no raw node
    features are exchanged).
    """
    payloads = [u.virtual_payload for u in updates if u.virtual_payload is not None]
    if not payloads:
        return None

    weights = torch.tensor(
        [max(p["boundary_degree"].item(), 1e-6) for p in payloads]
    )
    weights = weights / weights.sum()

    means = torch.stack([p["mean"] for p in payloads], dim=0)  # [n_clients, F]
    means, weights = _align_devices(means, weights)
    global_mean = (means * weights.unsqueeze(1)).sum(dim=0)

    return global_mean
