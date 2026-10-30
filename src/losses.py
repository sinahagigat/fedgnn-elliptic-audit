# -*- coding: utf-8 -*-
"""Loss functions shared by several modules: positive-class weight, weighted
cross-entropy, focal loss, prototype contrastive loss and the virtual-node
consistency loss.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("fedgnn.losses")


# 1) positive-class weight (client.py._compute_pos_weight_tensor computes the same value)
def compute_pos_weight(
    y: torch.Tensor,
    mask: torch.Tensor,
    max_pos_weight: float,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Weight of class 1 (illicit) relative to class 0, against the strong imbalance
    of Elliptic (positive rate ~9.76%). Formula: min(n_neg / n_pos, max_pos_weight);
    the cap max_pos_weight prevents the loss from exploding on clients with very
    few positive examples (e.g. 1 or 2).
    """
    dev = device or y.device
    y_masked = y[mask]
    n_pos = int((y_masked == 1).sum().item())
    n_neg = int((y_masked == 0).sum().item())

    if n_pos == 0:
        pos_weight = 1.0  # no positive example; weighting is meaningless
        logger.debug("compute_pos_weight: no positive example in the mask; weight=1.0")
    else:
        pos_weight = min(n_neg / max(n_pos, 1), max_pos_weight)

    return torch.tensor([1.0, pos_weight], dtype=torch.float32, device=dev)


# 2) standard weighted cross-entropy wrapper
class WeightedCrossEntropyLoss(nn.Module):
    """
    Stable wrapper around F.cross_entropy with a precomputed class weight.
    The weight must be set in the constructor or refreshed with update_weight
    (class composition differs across clients and rounds).
    """

    def __init__(self, class_weight: torch.Tensor):
        super().__init__()
        self.register_buffer("class_weight", class_weight)

    def update_weight(self, new_weight: torch.Tensor) -> None:
        self.class_weight = new_weight.to(self.class_weight.device)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.class_weight)


# 3) Focal loss: optional alternative (for sensitivity analysis)
class FocalLoss(nn.Module):
    """
    Focal loss (Lin et al., 2017) for cases where weighted CE alone is not
    sufficient; a larger gamma focuses more on hard / minority examples. It is
    not used by any reported variant (all use weighted CE) and is kept only
    for optional ablations.

    """

    def __init__(self, gamma: float = 2.0, class_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("class_weight", class_weight if class_weight is not None else torch.tensor([1.0, 1.0]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.class_weight, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


# 4) Prototype contrastive loss: module version
class PrototypeContrastiveLoss(nn.Module):
    """
    Module version of the contrastive formula between a node embedding and the
    global prototype of its class (client.FederatedClient._prototype_loss is
    logically equivalent). Formula:

        L_proto = CrossEntropy( sim(emb, prototypes) / T , label_idx )

    where sim is the normalized cosine similarity and T the prototype temperature
    (fed_cfg.prototype_temperature). A lower temperature gives a sharper sim
    distribution and a stricter penalty for being far from the correct prototype.
    """

    def __init__(self, temperature: float):
        super().__init__()
        assert temperature > 0, "the prototype temperature must be positive."
        self.temperature = temperature

    def forward(
        self,
        node_emb: torch.Tensor,
        labels: torch.Tensor,
        global_prototypes: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        if not global_prototypes or node_emb.shape[0] == 0:
            return torch.tensor(0.0, device=node_emb.device)

        class_ids = sorted(global_prototypes.keys())
        if len(class_ids) < 2:
            return torch.tensor(0.0, device=node_emb.device)

        proto_matrix = torch.stack([global_prototypes[c] for c in class_ids]).to(node_emb.device)

        valid_mask = torch.tensor(
            [int(l.item()) in class_ids for l in labels], device=node_emb.device
        )
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=node_emb.device)

        emb_valid = node_emb[valid_mask]
        labels_valid = labels[valid_mask]
        label_idx = torch.tensor(
            [class_ids.index(int(l.item())) for l in labels_valid], device=node_emb.device
        )

        emb_norm = F.normalize(emb_valid, dim=-1)
        proto_norm = F.normalize(proto_matrix, dim=-1)
        sims = (emb_norm @ proto_norm.t()) / self.temperature

        return F.cross_entropy(sims, label_idx)


# 5) Virtual-node consistency loss: general version
def virtual_node_consistency_loss(
    local_virtual_emb: torch.Tensor,
    global_virtual_ref: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    MSE penalty between the virtual-node vectors of the current client and the
    global reference (weighted mean of all clients from the previous round). By
    minimizing it, each client aligns its virtual nodes with the shared
    cross-client space without exposing the raw features of its boundary nodes.

    If no global reference exists yet (first round), the loss is zero; this is
    expected, not an error.
    """
    if global_virtual_ref is None:
        return torch.tensor(0.0, device=local_virtual_emb.device)

    ref = global_virtual_ref.to(local_virtual_emb.device)
    # the local virtual-node dimension must match the global reference
    assert local_virtual_emb.shape[-1] == ref.shape[-1], (
        f"virtual-node dimension mismatch: local={local_virtual_emb.shape[-1]}, "
        f"global reference={ref.shape[-1]}"
    )
    return F.mse_loss(local_virtual_emb.mean(dim=0), ref)
