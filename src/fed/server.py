# -*- coding: utf-8 -*-
"""
server.py — server-side orchestrator for one variant and seed
==============================================================================
Main federated loop: broadcast global weights + prototypes + virtual reference,
collect a ClientUpdate from every client, aggregate according to fed_cfg.algorithm,
select the best checkpoint by validation AP, and detect convergence.

The output of this module is passed to src/train.py, which evaluates the selected
model on the test period.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from src.config import FedConfig
from src.fed.client import FederatedClient, ClientUpdate
from src.fed.aggregation import (
    fedavg_aggregate,
    fedproto_aggregate_prototypes,
    hard_selective_aggregate,
    soft_selective_aggregate,
    aggregate_virtual_payloads,
)

logger = logging.getLogger("fedgnn.fed.server")


# ==============================================================================
# History record of one round, for reporting and convergence detection
# ==============================================================================
@dataclass
class RoundRecord:
    round_idx: int
    avg_train_loss: float
    avg_val_ap: float
    avg_val_auc: float
    per_client_val_ap: Dict[int, float]
    selection_scores: Optional[Dict[int, float]] = None


# ==============================================================================
# Federated server
# ==============================================================================
class FederatedServer:
    """
    Orchestrator of one complete experiment (one variant on one seed): manages the
    global state (model weights, prototypes, virtual-node reference), runs num_rounds
    rounds, selects the best checkpoint by validation AP, and sets the converged
    flag when there is no meaningful improvement within the patience window.
    """

    def __init__(
        self,
        clients: List[FederatedClient],
        initial_state_dict: Dict[str, torch.Tensor],
        fed_cfg: FedConfig,
    ):
        self.clients = clients
        self.fed_cfg = fed_cfg

        # global state: single source of truth during federated training
        self.global_state_dict: Dict[str, torch.Tensor] = copy.deepcopy(initial_state_dict)
        self.global_prototypes: Optional[Dict[int, torch.Tensor]] = None
        self.global_virtual_ref: Optional[torch.Tensor] = None

        # best checkpoint seen so far (by mean client validation AP)
        self.best_state_dict: Dict[str, torch.Tensor] = copy.deepcopy(initial_state_dict)
        self.best_val_ap: float = -1.0
        self.best_round_idx: int = -1

        self.history: List[RoundRecord] = []
        self._rounds_without_improve: int = 0
        self.converged: bool = False

    # -------------------------------------------------------------- #
    def _select_aggregation_fn(self, updates: List[ClientUpdate]) -> Dict[str, torch.Tensor]:
        """Select the weight-aggregation strategy from fed_cfg.algorithm."""
        algo = self.fed_cfg.algorithm

        if algo == "fedgnn_hard":
            return hard_selective_aggregate(updates, top_k_ratio=self.fed_cfg.top_k_ratio)
        elif algo == "fedgnn_soft":
            return soft_selective_aggregate(updates, temperature=self.fed_cfg.aggregation_temperature)
        elif algo in ("fedavg", "fedprox", "fedproto"):
            return fedavg_aggregate(updates)
        else:
            raise ValueError(f"❌ unknown aggregation algorithm: {algo}")

    # -------------------------------------------------------------- #
    def run_round(self, round_idx: int) -> RoundRecord:
        """
        Run one complete federated round:
          1) broadcast the current global state to all clients + local_train
          2) aggregate the model weights with the selected algorithm
          3) aggregate the global class prototypes (prototype-based variants)
          4) aggregate/update the virtual-node reference (virtual-node variants)
          5) record history and update the best checkpoint
        """
        updates: List[ClientUpdate] = []
        for client in self.clients:
            update = client.local_train(
                global_state_dict=self.global_state_dict,
                global_prototypes=self.global_prototypes,
                external_virtual_ref=self.global_virtual_ref,
                round_idx=round_idx,
            )
            updates.append(update)

        # --- weight aggregation (fail fast if no client returned an update) ---
        assert len(updates) > 0, "❌ no ClientUpdate received from the clients."
        self.global_state_dict = self._select_aggregation_fn(updates)

        # --- global prototype aggregation ---
        if self.fed_cfg.algorithm in ("fedproto", "fedgnn_hard", "fedgnn_soft"):
            self.global_prototypes = fedproto_aggregate_prototypes(updates)

        # --- virtual-node reference aggregation (independent of the main algorithm) ---
        if self.fed_cfg.use_virtual_nodes:
            new_ref = aggregate_virtual_payloads(updates)
            if new_ref is not None:
                self.global_virtual_ref = new_ref

        # --- aggregate the metrics reported for this round ---
        avg_train_loss = float(np.mean([u.metrics["train_loss"] for u in updates]))
        avg_val_ap = float(np.mean([u.metrics["val_ap"] for u in updates]))
        avg_val_auc = float(np.mean([u.metrics["val_auc"] for u in updates]))
        per_client_val_ap = {u.client_id: u.metrics["val_ap"] for u in updates}
        selection_scores = (
            {u.client_id: u.selection_score for u in updates}
            if self.fed_cfg.algorithm in ("fedgnn_hard", "fedgnn_soft")
            else None
        )

        record = RoundRecord(
            round_idx=round_idx,
            avg_train_loss=avg_train_loss,
            avg_val_ap=avg_val_ap,
            avg_val_auc=avg_val_auc,
            per_client_val_ap=per_client_val_ap,
            selection_scores=selection_scores,
        )
        self.history.append(record)

        # --- best-checkpoint update (single criterion: validation AP) ---
        improved = avg_val_ap > self.best_val_ap + self.fed_cfg.min_delta
        if improved:
            self.best_val_ap = avg_val_ap
            self.best_state_dict = copy.deepcopy(self.global_state_dict)
            self.best_round_idx = round_idx
            self._rounds_without_improve = 0
        else:
            self._rounds_without_improve += 1

        # --- convergence detection (converged flag) ---
        if self._rounds_without_improve >= self.fed_cfg.patience:
            self.converged = True

        logger.info(
            f"🌐 server | round {round_idx}/{self.fed_cfg.num_rounds} | "
            f"train_loss={avg_train_loss:.4f} | val_ap={avg_val_ap:.4f} | "
            f"best_val_ap={self.best_val_ap:.4f} @ round {self.best_round_idx} | "
            f"patience={self._rounds_without_improve}/{self.fed_cfg.patience} | "
            f"converged={self.converged}"
        )

        return record

    # -------------------------------------------------------------- #
    def run(self) -> Dict[str, object]:
        """
        Run up to num_rounds rounds with early stopping by patience. If the
        convergence criterion is met earlier, the loop stops and this is reported
        explicitly in the output.
        """
        for round_idx in range(1, self.fed_cfg.num_rounds + 1):
            self.run_round(round_idx)
            if self.converged and self.fed_cfg.early_stop_enabled:
                logger.info(
                    f"⏹️ early stop at round {round_idx}: no val_ap improvement "
                    f"for {self.fed_cfg.patience} consecutive rounds."
                )
                break

        return {
            "best_state_dict": self.best_state_dict,
            "best_val_ap": self.best_val_ap,
            "best_round_idx": self.best_round_idx,
            "converged": self.converged,
            "total_rounds_run": len(self.history),
            "global_prototypes": self.global_prototypes,
            "history": self.history,
        }

    # -------------------------------------------------------------- #
    def get_best_model_state(self) -> Dict[str, torch.Tensor]:
        """Return the best weights found during federated training."""
        return self.best_state_dict
