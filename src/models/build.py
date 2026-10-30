# -*- coding: utf-8 -*-
"""Model factory. Every design uses the same FedGNNModel (encoder, optional
virtual-node layer, linear head); designs differ in training and aggregation
and, for Soft-VN-GAT, in the encoder.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from src.config import ModelConfig, VirtualNodeConfig
from src.models.encoders import build_encoder
from src.models.virtual_nodes import VirtualNodeModule

logger = logging.getLogger("fedgnn.models.build")


class FedGNNModel(nn.Module):
    """
    Complete model of one client (or of the centralized version): GCN/GAT/MLP
    encoder, optionally followed by the virtual-node layer.

    The final logits are computed from node_emb *after* the
    virtual-node layer (when enabled), so the virtual nodes do affect the
    classification of boundary nodes whenever boundary indices are passed.
    """

    def __init__(
        self,
        in_dim: int,
        model_cfg: ModelConfig,
        virtual_cfg: Optional[VirtualNodeConfig] = None,
        use_virtual_nodes: bool = False,
        num_classes: int = 2,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.use_virtual_nodes = use_virtual_nodes and virtual_cfg is not None

        self.encoder = build_encoder(in_dim, model_cfg, num_classes=num_classes)

        # separate classification head (after the optional virtual-node layer);
        # encoder.classifier is not used in forward(); this second head keeps
        # the weights of the two paths independent.
        self.final_classifier = nn.Linear(model_cfg.hidden_dim, num_classes)

        if self.use_virtual_nodes:
            self.virtual_module = VirtualNodeModule(model_cfg.hidden_dim, virtual_cfg)
            # projection of the raw-feature payload into the hidden space (the
            # payload lives in the raw feature space F, not in the hidden space)
            self.payload_proj = nn.Linear(in_dim, model_cfg.hidden_dim)
            # The output of this layer only feeds .data and a detached anchor, so
            # it never receives gradients. It is frozen explicitly so that it is not
            # counted as trainable; it acts as a fixed random projection.
            for _p in self.payload_proj.parameters():
                _p.requires_grad_(False)
        else:
            self.virtual_module = None
            self.payload_proj = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        boundary_local_idx: Optional[torch.Tensor] = None,
    ):
        node_emb, _ = self.encoder(x, edge_index)

        if self.use_virtual_nodes and boundary_local_idx is not None:
            node_emb = self.virtual_module(node_emb, boundary_local_idx)

        logits = self.final_classifier(node_emb)
        return node_emb, logits

    def set_virtual_reference(self, raw_payload: torch.Tensor):
        """Project the raw boundary payload (from federated_dataset.py) into the
        hidden space and pass it to VirtualNodeModule as this round's reference."""
        if not self.use_virtual_nodes:
            return
        projected = self.payload_proj(raw_payload.unsqueeze(0))  # [1, H]
        self.virtual_module.set_external_reference(projected)

    def get_virtual_node_loss(self) -> torch.Tensor:
        if not self.use_virtual_nodes:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.virtual_module.compute_virtual_node_loss()


def build_model(
    in_dim: int,
    model_cfg: ModelConfig,
    virtual_cfg: Optional[VirtualNodeConfig] = None,
    use_virtual_nodes: bool = False,
    num_classes: int = 2,
    device: str = "cuda",
) -> FedGNNModel:
    """Single entry point for building a model; every variant goes through it."""
    model = FedGNNModel(
        in_dim=in_dim,
        model_cfg=model_cfg,
        virtual_cfg=virtual_cfg,
        use_virtual_nodes=use_virtual_nodes,
        num_classes=num_classes,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"model built | arch={model_cfg.arch} | virtual_nodes={use_virtual_nodes} | "
        f"parameters={n_params:,} | device={device}"
    )
    return model
