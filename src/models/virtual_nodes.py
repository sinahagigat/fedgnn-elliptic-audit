# -*- coding: utf-8 -*-
"""Virtual-node layer for boundary nodes.

Each client keeps K learnable virtual nodes (default 8). Their states are blended
towards a server reference computed from the clients' aggregated boundary
payloads. Boundary nodes attend to the virtual nodes through single-head
attention, and an MSE term keeps the virtual-node states close to the reference.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import VirtualNodeConfig


class VirtualNodeModule(nn.Module):
    """
    Virtual-node module, applied as a separate layer after the main encoder
    (GCN/GAT) to the hidden vectors of boundary nodes.
    """

    # class-level default of the reference anchor (before the first server consensus)
    _last_reference_payload = None

    def __init__(self, hidden_dim: int, cfg: VirtualNodeConfig):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = hidden_dim
        self.num_virtual = cfg.num_virtual

        # base embedding of the virtual nodes, blended with the external payload each round
        self.virtual_embed = nn.Parameter(torch.zeros(self.num_virtual, hidden_dim))
        nn.init.xavier_uniform_(self.virtual_embed)

        # light attention layer connecting boundary nodes and virtual nodes
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

        self._last_reference_payload: Optional[torch.Tensor] = None  # for virtual_node_loss

    def set_external_reference(self, incoming_payload_matrix):
        """
        Receive the server's consensus payload and store it as this round's anchor.

        Implementation notes:
          (1) the consensus variable is defined in both branches.
          (2) the anchor is the pure server consensus, before momentum blending.
          (3) shape, device and dtype of the input are normalized and checked.
        """
        if incoming_payload_matrix is None:
            self._last_reference_payload = None
            return

        with torch.no_grad():
            payload = incoming_payload_matrix
            if not torch.is_tensor(payload):
                payload = torch.as_tensor(payload)
            payload = payload.to(device=self.virtual_embed.device,
                                 dtype=self.virtual_embed.dtype)
            if payload.dim() == 1:
                payload = payload.unsqueeze(0)

            H = self.virtual_embed.shape[-1]
            assert payload.shape[-1] == H, (
                f"[virtual_nodes] payload dimension ({payload.shape[-1]}) does not match the "
                f"virtual-node hidden space ({H}); a projection is required in client.py."
            )

            if payload.shape[0] == self.num_virtual:
                consensus = payload
            else:
                consensus = payload.mean(dim=0, keepdim=True) \
                                   .expand(self.num_virtual, -1)
            consensus = consensus.contiguous().clone()

            m = float(self.cfg.momentum)
            self.virtual_embed.data = (
                m * self.virtual_embed.data + (1.0 - m) * consensus
            )
            self._last_reference_payload = consensus.detach().clone()

    def forward(
        self,
        node_emb: torch.Tensor,
        boundary_local_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update the boundary-node vectors with information from the virtual nodes
        through one single-head scaled dot-product attention block, and return
        the full updated node_emb (all other nodes are left unchanged).

        Args:
            node_emb: [N_local, H] hidden vectors of all client nodes
            boundary_local_idx: [n_boundary] local indices of boundary nodes

        Returns:
            node_emb_updated: [N_local, H]
        """
        if boundary_local_idx.numel() == 0:
            return node_emb  # client without boundary nodes: unchanged

        h_boundary = node_emb[boundary_local_idx]              # [B, H]
        q = self.query_proj(h_boundary)                        # [B, H]
        k = self.key_proj(self.virtual_embed)                  # [V, H]
        v = self.value_proj(self.virtual_embed)                # [V, H]

        attn_scores = (q @ k.t()) / self.scale                 # [B, V]
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = attn_weights @ v                              # [B, H]
        context = self.out_proj(context)

        h_boundary_updated = h_boundary + self.cfg.residual_weight * context

        node_emb_updated = node_emb.clone()
        node_emb_updated[boundary_local_idx] = h_boundary_updated
        return node_emb_updated

    def compute_virtual_node_loss(self) -> torch.Tensor:
        """
        MSE penalty between the current virtual nodes and the latest reference from
        the server; anchors the virtual nodes so that they do not drift from their
        aggregated representation during the local steps.
        """
        if self._last_reference_payload is None:
            return torch.tensor(0.0, device=self.virtual_embed.device)
        return F.mse_loss(self.virtual_embed, self._last_reference_payload)

    def get_outgoing_payload(self) -> torch.Tensor:
        """Summary vector of the virtual nodes to send to the server in the next round."""
        return self.virtual_embed.detach().clone()
