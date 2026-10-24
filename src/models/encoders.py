# -*- coding: utf-8 -*-
"""
encoders.py — GCN, GAT and MLP encoders
==============================================================================
All encoders share one interface, forward(x, edge_index) -> (node_emb, logits),
so that build.py can attach the virtual-node layer and the classification
head without special cases. node_emb is the hidden vector of the last layer
(before classification); it is used both for class prototypes and for the
virtual-node attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

from src.config import ModelConfig


class GCNEncoder(nn.Module):
    """
    Multi-layer GCN encoder with dropout, LayerNorm and an optional residual
    connection between layers (disabled in all reported experiments).
    
    
    """

    def __init__(self, in_dim: int, cfg: ModelConfig, num_classes: int = 2):
        super().__init__()
        self.cfg = cfg
        hidden = cfg.hidden_dim
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        dims = [in_dim] + [hidden] * cfg.num_layers
        for i in range(cfg.num_layers):
            self.convs.append(GCNConv(dims[i], dims[i + 1], add_self_loops=True))
            self.norms.append(nn.LayerNorm(dims[i + 1]))

        self.dropout = nn.Dropout(cfg.dropout)
        self.use_residual = getattr(cfg, "residual", False) and in_dim == hidden
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)
            if self.use_residual and i > 0 and h_new.shape == h.shape:
                h_new = h_new + h
            h = h_new

        logits = self.classifier(h)
        return h, logits


class GATEncoder(nn.Module):
    """
    Multi-layer GAT encoder with concat=True in intermediate layers (keeps the
    multi-head capacity) and concat=False (head averaging) in the last layer,
    so that the output dimension equals hidden_dim, as expected by build.py
    and the virtual-node layer.
    """

    def __init__(self, in_dim: int, cfg: ModelConfig, num_classes: int = 2):
        super().__init__()
        self.cfg = cfg
        hidden = cfg.hidden_dim
        heads = cfg.gat_heads
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dims = [in_dim] + [hidden] * (cfg.num_layers - 1)
        for i in range(cfg.num_layers):
            is_last = i == cfg.num_layers - 1
            out_per_head = hidden if is_last else hidden // heads
            concat = not is_last
            self.convs.append(
                GATConv(
                    in_dims[i],
                    out_per_head,
                    heads=heads,
                    concat=concat,
                    dropout=cfg.dropout,
                    add_self_loops=True,
                )
            )
            out_dim = out_per_head * heads if concat else out_per_head
            self.norms.append(nn.LayerNorm(out_dim))

        self.dropout = nn.Dropout(cfg.dropout)
        self.use_residual = getattr(cfg, "residual", False) and in_dim == hidden
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index)
            h_new = norm(h_new)
            h_new = F.elu(h_new)
            h_new = self.dropout(h_new)
            if self.use_residual and i > 0 and h_new.shape == h.shape:
                h_new = h_new + h
            h = h_new

        logits = self.classifier(h)
        return h, logits


class MLPEncoder(nn.Module):
    """
    v7: graph-free control. Same depth, width, LayerNorm, ReLU and dropout as
    GCNEncoder, but each layer is a Linear map, so edge_index is ignored.
    A GCN that does not beat this encoder is not using the graph.
    """

    def __init__(self, in_dim: int, cfg: ModelConfig, num_classes: int = 2):
        super().__init__()
        self.cfg = cfg
        hidden = cfg.hidden_dim
        dims = [in_dim] + [hidden] * cfg.num_layers
        self.lins = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(cfg.num_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(dims[i + 1]) for i in range(cfg.num_layers))
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        h = x
        for lin, norm in zip(self.lins, self.norms):
            h = self.dropout(F.relu(norm(lin(h))))
        return h, self.classifier(h)


_ENCODER_REGISTRY = {"gcn": GCNEncoder, "gat": GATEncoder, "mlp": MLPEncoder}


def build_encoder(in_dim: int, cfg: ModelConfig, num_classes: int = 2) -> nn.Module:
    """Encoder factory selected by cfg.arch (\"gcn\", \"gat\" or \"mlp\")."""
    arch = cfg.arch.lower()
    if arch not in _ENCODER_REGISTRY:
        raise ValueError(f"❌ unknown architecture: {arch}. Valid options: {list(_ENCODER_REGISTRY)}")
    return _ENCODER_REGISTRY[arch](in_dim, cfg, num_classes)
