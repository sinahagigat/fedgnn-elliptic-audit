# -*- coding: utf-8 -*-
"""Federated client: local training on the client subgraph, class prototypes,
prototype loss, virtual-node payload and the selection score used by the
selective aggregation designs. Clients send model weights and class- or
boundary-level statistics only.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import ModelConfig, VirtualNodeConfig, FedConfig
from src.data.federated_dataset import FederatedClientData, compute_boundary_exchange_payload
from src.models.build import FedGNNModel, build_model

logger = logging.getLogger("fedgnn.fed.client")


# Output bundle of a client after one round of local training
@dataclass
class ClientUpdate:
    """
    Complete output of a client after local_train; direct input of aggregation.py.
    """
    client_id: int
    state_dict: Dict[str, torch.Tensor]
    num_train: int
    num_val: int
    num_positives_train: int
    class_counts: Dict[int, int]                       # for weighting the global prototypes
    local_prototypes: Dict[int, torch.Tensor]           # local prototype of each class
    virtual_payload: Optional[Dict[str, torch.Tensor]]  # raw boundary payload (constant across rounds)
    selection_score: float                               # participation score
    metrics: Dict[str, float]                             # train_loss, val_ap, val_auc, ...


# Federated client
class FederatedClient:
    """
    An independent client with its model, local data, local prototypes and
    (when enabled) the virtual-node layer. Internal state (self.local_prototypes)
    persists across rounds so that the prototype momentum is meaningful.
    """

    def __init__(
        self,
        client_id: int,
        fcd: FederatedClientData,
        in_dim: int,
        model_cfg: ModelConfig,
        virtual_cfg: Optional[VirtualNodeConfig],
        fed_cfg: FedConfig,
        use_virtual_nodes: bool,
        device: str = "cuda",
    ):
        self.client_id = client_id
        self.fcd = fcd
        self.model_cfg = model_cfg
        self.virtual_cfg = virtual_cfg
        self.fed_cfg = fed_cfg
        self.use_virtual_nodes = use_virtual_nodes
        self.device = device

        self.use_prototype = fed_cfg.algorithm in (
            "fedproto", "fedgnn_hard", "fedgnn_soft"
        )

        self.model: FedGNNModel = build_model(
            in_dim=in_dim,
            model_cfg=model_cfg,
            virtual_cfg=virtual_cfg,
            use_virtual_nodes=use_virtual_nodes,
            num_classes=2,
            device=device,
        )

        self.data = fcd.data.to(device)
        self.boundary_idx = torch.tensor(
            fcd.boundary_info.local_boundary_idx, dtype=torch.long, device=device
        )

        # local prototypes: state kept across rounds (momentum on themselves)
        self.local_prototypes: Dict[int, torch.Tensor] = {}

        # raw boundary payload: raw node features do not change across rounds,
        # so it is computed once in the constructor
        if self.use_virtual_nodes:
            self.raw_virtual_payload = compute_boundary_exchange_payload(fcd, virtual_cfg)
            self.raw_virtual_payload = {
                k: v.to(device) for k, v in self.raw_virtual_payload.items()
            }
        else:
            self.raw_virtual_payload = None

        self._pos_weight_tensor = self._compute_pos_weight_tensor()

        # optional split of the validation nodes (see
        # FedConfig.val_split_weighting). The split uses its own generator with a
        # fixed seed, so it does not touch the global random state and is the
        # same for every training seed.
        self.val_select_mask = self.data.val_mask
        self.val_weight_mask = self.data.val_mask
        if bool(getattr(fed_cfg, "val_split_weighting", False)):
            self.val_weight_mask, self.val_select_mask = self._split_validation_mask()

    VAL_SPLIT_SEED = 20261025

    def _split_validation_mask(self):
        """Stratified random halves of this client's validation nodes:
        (mask used for the selection score, mask used for checkpoint selection)."""
        vm = self.data.val_mask
        idx = torch.nonzero(vm, as_tuple=False).flatten().cpu().numpy()
        yv = self.data.y[vm].cpu().numpy()
        rng = np.random.default_rng(self.VAL_SPLIT_SEED + int(self.client_id))
        take = np.zeros(len(idx), dtype=bool)
        for c in (0, 1):
            ii = np.flatnonzero(yv == c)
            rng.shuffle(ii)
            take[ii[: (len(ii) + 1) // 2]] = True
        weight_mask = torch.zeros_like(vm)
        weight_mask[torch.as_tensor(idx[take], dtype=torch.long, device=vm.device)] = True
        select_mask = vm & ~weight_mask
        return weight_mask, select_mask

    def _compute_pos_weight_tensor(self) -> torch.Tensor:
        y_train = self.data.y[self.data.train_mask]
        n_pos = int((y_train == 1).sum().item())
        n_neg = int((y_train == 0).sum().item())
        if n_pos == 0:
            pos_weight = 1.0
        else:
            pos_weight = min(n_neg / max(n_pos, 1), self.fed_cfg.max_pos_weight)
        return torch.tensor([1.0, pos_weight], device=self.device)

    def _update_local_prototypes(
        self, node_emb: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
    ):
        """Momentum update of each class's local prototype, using training nodes only."""
        classes_present = torch.unique(y[mask]).tolist()
        momentum = self.fed_cfg.prototype_momentum
        for c in classes_present:
            class_mask = mask & (y == c)
            if class_mask.sum() == 0:
                continue
            mean_emb = node_emb[class_mask].mean(dim=0).detach()
            if c in self.local_prototypes:
                self.local_prototypes[c] = (
                    momentum * self.local_prototypes[c] + (1 - momentum) * mean_emb
                )
            else:
                self.local_prototypes[c] = mean_emb.clone()

    def _prototype_loss(
        self,
        node_emb: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        global_prototypes: Optional[Dict[int, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Soft contrastive penalty between each node embedding and the global
        prototype of its class, relative to the other prototypes (softmax over
        cosine similarities). It encourages alignment of the embedding space
        across heterogeneous clients.
        """
        if not self.use_prototype or not global_prototypes or mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        class_ids = sorted(global_prototypes.keys())
        if len(class_ids) < 2:
            return torch.tensor(0.0, device=self.device)  # at least two classes are needed

        proto_matrix = torch.stack([global_prototypes[c] for c in class_ids]).to(self.device)
        emb = node_emb[mask]
        labels = y[mask]

        # drop nodes whose class has no global prototype
        valid = torch.tensor([int(l.item()) in class_ids for l in labels], device=self.device)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        emb = emb[valid]
        labels = labels[valid]
        label_idx = torch.tensor(
            [class_ids.index(int(l.item())) for l in labels], device=self.device
        )

        emb_norm = F.normalize(emb, dim=-1)
        proto_norm = F.normalize(proto_matrix, dim=-1)
        sims = (emb_norm @ proto_norm.t()) / self.fed_cfg.prototype_temperature  # [n, C]

        return F.cross_entropy(sims, label_idx)

    def _quick_eval(self, mask: torch.Tensor) -> Dict[str, float]:
        """
        Light local evaluation (AP/AUC). The final reported metrics come from
        src/evaluate.py; this function is only used for model selection and the
        selection score during federated training.
        """
        self.model.eval()
        with torch.no_grad():
            _, logits = self.model(self.data.x, self.data.edge_index, self.boundary_idx)
            probs = F.softmax(logits, dim=-1)[:, 1]
            y_true = self.data.y[mask].cpu().numpy()
            y_score = probs[mask].cpu().numpy()

        if len(np.unique(y_true)) < 2:
            ap, auc = 0.0, 0.5
        else:
            ap = float(average_precision_score(y_true, y_score))
            auc = float(roc_auc_score(y_true, y_score))
        self.model.train()
        return {"ap": ap, "auc": auc}

    def _compute_selection_score(self, val_metrics: Dict[str, float]) -> float:
        """
        Client participation score for hard/soft selective aggregation.

        Formula: val_ap weighted by log(1+n_pos_train). A client that both
        generalizes well on its own validation data and has enough positive
        examples (so its training is more reliable than minority-class noise)
        receives a higher score.
        """
        n_pos = int(self.data.y[self.data.train_mask].sum().item())
        reliability = float(np.log1p(n_pos))
        return val_metrics["ap"] * reliability

    def evaluate_aggregate(self, state_dict, virtual_ref=None) -> Dict[str, float]:
        """Validation AP/AUC of the aggregated model on this client's checkpoint-selection
        nodes (FedConfig.select_on_aggregate). Works on a copy of the local model, so local
        training and the random state are not affected."""
        model = copy.deepcopy(self.model)
        model.load_state_dict(state_dict)
        if self.use_virtual_nodes and virtual_ref is not None:
            model.set_virtual_reference(virtual_ref)
        model.eval()
        with torch.no_grad():
            _, logits = model(self.data.x, self.data.edge_index, self.boundary_idx)
            probs = F.softmax(logits, dim=-1)[:, 1]
        y_true = self.data.y[self.val_select_mask].cpu().numpy()
        y_score = probs[self.val_select_mask].cpu().numpy()
        if len(np.unique(y_true)) < 2:
            return {"ap": 0.0, "auc": 0.5}
        return {"ap": float(average_precision_score(y_true, y_score)),
                "auc": float(roc_auc_score(y_true, y_score))}

    def local_train(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        global_prototypes: Optional[Dict[int, torch.Tensor]],
        external_virtual_ref: Optional[torch.Tensor],
        round_idx: int,
    ) -> ClientUpdate:
        """
        One complete round of local training: load the global weights, take
        local_epochs steps, compute prototypes/virtual payload/score, return ClientUpdate.
        """
        self.model.load_state_dict(copy.deepcopy(global_state_dict))

        if self.use_virtual_nodes and external_virtual_ref is not None:
            self.model.set_virtual_reference(external_virtual_ref)

        # FedProx term: global weights as reference for the local-global proximity penalty
        global_params_ref = None
        if self.fed_cfg.algorithm == "fedprox":
            global_params_ref = {
                k: v.clone().detach().to(self.device) for k, v in global_state_dict.items()
            }

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.fed_cfg.lr,
            weight_decay=self.fed_cfg.weight_decay,
        )

        train_mask = self.data.train_mask
        last_loss = 0.0

        for _epoch in range(self.fed_cfg.local_epochs):
            optimizer.zero_grad()
            node_emb, logits = self.model(self.data.x, self.data.edge_index, self.boundary_idx)

            ce_loss = F.cross_entropy(
                logits[train_mask], self.data.y[train_mask], weight=self._pos_weight_tensor
            )

            proto_loss = self._prototype_loss(
                node_emb, self.data.y, train_mask, global_prototypes
            )

            virtual_loss = (
                self.model.get_virtual_node_loss() if self.use_virtual_nodes
                else torch.tensor(0.0, device=self.device)
            )

            prox_loss = torch.tensor(0.0, device=self.device)
            if global_params_ref is not None:
                for name, param in self.model.named_parameters():
                    if name in global_params_ref:
                        prox_loss = prox_loss + torch.sum(
                            (param - global_params_ref[name]) ** 2
                        )
                prox_loss = (self.fed_cfg.fedprox_mu / 2.0) * prox_loss

            total_loss = (
                ce_loss
                + self.fed_cfg.prototype_loss_weight * proto_loss
                + self.fed_cfg.virtual_loss_weight * virtual_loss
                + prox_loss
            )

            total_loss.backward()
            clip_grad_norm_(self.model.parameters(), self.fed_cfg.grad_clip_norm)
            optimizer.step()

            self._update_local_prototypes(node_emb.detach(), self.data.y, train_mask)
            last_loss = float(total_loss.item())

        val_metrics = self._quick_eval(self.val_select_mask)
        if self.val_weight_mask is self.val_select_mask:
            selection_score = self._compute_selection_score(val_metrics)
        else:
            selection_score = self._compute_selection_score(self._quick_eval(self.val_weight_mask))

        class_counts = {
            c: int((self.data.y[train_mask] == c).sum().item())
            for c in torch.unique(self.data.y[train_mask]).tolist()
        }

        # log L_align (the tag separates the first round from rounds with a reference)
        vloss_val = 0.0
        _vn_tag = "off"
        if self.use_virtual_nodes:
            _vn = next((mod for mod in self.model.modules()
                        if hasattr(mod, "compute_virtual_node_loss")), None)
            if _vn is None:
                _vn_tag = "no_module"
            elif getattr(_vn, "_last_reference_payload", None) is None:
                _vn_tag = "no_ref_yet"
            else:
                with torch.no_grad():
                    vloss_val = float(_vn.compute_virtual_node_loss().item())
                _vn_tag = "ok"
            logger.info(
                f"[VNLOG] round={round_idx} client={self.client_id} "
                f"L_align={vloss_val:.6e} tag={_vn_tag}"
            )

        # replace non-finite values in the outgoing payload (tensor, dict or list)
        def _sanitize_payload(o, _path="payload"):
            if torch.is_tensor(o):
                o = o.detach()
                if not torch.isfinite(o).all():
                    logger.warning(
                        f"client {self.client_id}: {_path} non-finite, set to zero")
                    o = torch.zeros_like(o)
                return o
            if isinstance(o, dict):
                return {k: _sanitize_payload(v, f"{_path}.{k}") for k, v in o.items()}
            if isinstance(o, list):
                return [_sanitize_payload(v, f"{_path}[{i}]") for i, v in enumerate(o)]
            if isinstance(o, tuple):
                return tuple(_sanitize_payload(v, f"{_path}[{i}]") for i, v in enumerate(o))
            return o

        virtual_payload_out = None
        _rvp = getattr(self, "raw_virtual_payload", None)
        if self.use_virtual_nodes and _rvp is not None:
            virtual_payload_out = _sanitize_payload(_rvp)
            # payload protection (virtual.protection); every reported configuration uses "none"
            protection = str(getattr(self.virtual_cfg, "protection", "none") or "none").lower()
            self._payload_protection_applied = "none"
            if protection not in ("none", "off", ""):
                logger.warning(f"payload protection '{protection}' is not applied to the "
                               f"virtual-node payload; the raw payload is sent")
            if int(round_idx) == 1 and int(self.client_id) == 0:
                logger.info(f"[VNLOG] protection_cfg={protection} "
                            f"applied={self._payload_protection_applied}")
            if int(round_idx) == 1 and int(self.client_id) == 0:
                _o = virtual_payload_out
                if isinstance(_o, dict):
                    _st = {k: (tuple(v.shape) if torch.is_tensor(v) else type(v).__name__)
                            for k, v in _o.items()}
                elif torch.is_tensor(_o):
                    _st = tuple(_o.shape)
                else:
                    _st = type(_o).__name__
                logger.info(f"[VNLOG] payload_struct={_st}")

        logger.debug(
            f"client {self.client_id} | round {round_idx} | "
            f"train_loss={last_loss:.4f} | val_ap={val_metrics['ap']:.4f} | "
            f"score={selection_score:.4f}"
        )

        return ClientUpdate(
            client_id=self.client_id,
            state_dict=copy.deepcopy(self.model.state_dict()),
            num_train=int(train_mask.sum().item()),
            num_val=int(self.data.val_mask.sum().item()),
            num_positives_train=int(self.data.y[train_mask].sum().item()),
            class_counts=class_counts,
            local_prototypes={k: v.clone() for k, v in self.local_prototypes.items()},
            virtual_payload=virtual_payload_out,
            selection_score=selection_score,
            metrics={"train_loss": last_loss, "virtual_loss": vloss_val, "val_ap": val_metrics["ap"], "val_auc": val_metrics["auc"]},
        )
