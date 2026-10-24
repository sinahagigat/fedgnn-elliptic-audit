# -*- coding: utf-8 -*-
"""
privacy.py — optional differential-privacy layer (not used in the reported experiments)
==============================================================================
⚠️ Scope: the mechanism implemented here is a standard Gaussian-mechanism
approximation with naive (basic) composition of the epsilon budget, not an
exact Rényi-DP or moments accountant. Any epsilon value it reports is an
approximate monitor of relative noise, not a formal (ε,δ)-DP guarantee.
The module is entirely optional: with privacy disabled no noise is added
and the system behaves exactly as if the module did not exist.



Functions:
  1) clip_tensor / clip_state_dict     — L2-norm clipping (prerequisite for noise)
  2) add_gaussian_noise                — calibrated Gaussian noise
  3) privatize_prototype               — privacy on a local prototype
  4) privatize_virtual_payload         — privacy on the boundary payload
  5) privatize_model_update            — privacy on the model-weight delta (light DP-FedAvg)
  6) PrivacyAccountant                 — tracking of the consumed epsilon budget
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from src.config import PrivacyConfig

logger = logging.getLogger("fedgnn.fed.privacy")


# ==============================================================================
# Clipping: mandatory prerequisite of any calibrated noise mechanism
# ==============================================================================
def clip_tensor(tensor: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Clip the L2 norm of an arbitrary tensor to max_norm."""
    norm = torch.norm(tensor.detach(), p=2)
    if norm <= max_norm or norm == 0:
        return tensor
    return tensor * (max_norm / (norm + 1e-8))


def clip_state_dict(
    state_dict: Dict[str, torch.Tensor], max_norm: float
) -> Dict[str, torch.Tensor]:
    """
    Clip the L2 norm of the *whole* flattened parameter vector (not layer by
    layer), the standard DP-SGD/DP-FedAvg way to bound update sensitivity.
    """
    flat = torch.cat([v.flatten().float() for v in state_dict.values()])
    total_norm = torch.norm(flat, p=2)
    if total_norm <= max_norm or total_norm == 0:
        return state_dict

    scale = max_norm / (total_norm + 1e-8)
    return {k: v * scale for k, v in state_dict.items()}


# ==============================================================================
# Gaussian noise
# ==============================================================================
def add_gaussian_noise(
    tensor: torch.Tensor,
    noise_std: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Add i.i.d. Gaussian noise with standard deviation noise_std to a tensor."""
    if noise_std <= 0:
        return tensor
    noise = torch.randn(tensor.shape, generator=generator, device=tensor.device) * noise_std
    return tensor + noise


# ==============================================================================
# Privacy on the local prototype (before sending it to the server)
# ==============================================================================
def privatize_prototype(
    prototype: torch.Tensor,
    privacy_cfg: PrivacyConfig,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Clip + noise on one class prototype before it leaves the client. A
    prototype is already an aggregate statistic (mean class embedding), not
    raw data, so its sensitivity is naturally limited; clipping is only
    needed to calibrate the noise.
    """
    if not privacy_cfg.enabled:
        return prototype
    clipped = clip_tensor(prototype, privacy_cfg.clip_norm)
    noise_std = privacy_cfg.clip_norm * privacy_cfg.noise_multiplier
    return add_gaussian_noise(clipped, noise_std, generator)


# ==============================================================================
# Privacy on the virtual-node payload
# ==============================================================================
def privatize_virtual_payload(
    payload: Dict[str, torch.Tensor],
    privacy_cfg: PrivacyConfig,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """
    Apply privacy to the 'mean' vector of the boundary payload (the
    'boundary_degree' key stays noise-free: it is a single summed integer needed
    for weighting in aggregate_virtual_payloads and leaks no meaningful information).
    """
    if not privacy_cfg.enabled:
        return payload
    noised = dict(payload)
    noised["mean"] = privatize_prototype(payload["mean"], privacy_cfg, generator)
    return noised


# ==============================================================================
# Privacy on the model-weight delta (light DP-FedAvg); optional, costly in accuracy
# ==============================================================================
def privatize_model_update(
    local_state: Dict[str, torch.Tensor],
    global_state: Dict[str, torch.Tensor],
    privacy_cfg: PrivacyConfig,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute the delta (local weights - global weights), clip its total norm,
    add Gaussian noise, and rebuild the local weights as global + noisy delta,
    i.e. DP-FedAvg (McMahan et al., 2018) at the client level.

    ⚠️ Not enabled in any reported experiment: its accuracy cost on the small,
    imbalanced Elliptic data is considerable; it is kept only for a possible
    privacy sensitivity analysis.

    """
    if not privacy_cfg.enabled:
        return local_state

    delta = {k: local_state[k].float() - global_state[k].float() for k in local_state}
    clipped_delta = clip_state_dict(delta, privacy_cfg.clip_norm)

    noise_std = privacy_cfg.clip_norm * privacy_cfg.noise_multiplier
    noised_delta = {
        k: add_gaussian_noise(v, noise_std, generator) for k, v in clipped_delta.items()
    }

    return {
        k: (global_state[k].float() + noised_delta[k]).to(local_state[k].dtype)
        for k in local_state
    }


# ==============================================================================
# epsilon budget tracking (naive composition): relative monitoring only, no formal guarantee
# ==============================================================================
@dataclass
class PrivacyAccountant:
    """
    Approximate tracking of the epsilon consumed over successive rounds under
    basic composition: ε_total ≈ Σ ε_round. This estimate is conservative
    (worse than reality) and deliberately not more elaborate (e.g. an RDP
    accountant), so that no claim about its tightness is made.
    """
    delta: float
    epsilon_per_round: float = field(default=0.0, init=False)
    total_epsilon_consumed: float = field(default=0.0, init=False)
    rounds_accounted: int = field(default=0, init=False)

    def compute_epsilon_for_round(self, noise_multiplier: float) -> float:
        """
        Approximate Gaussian-mechanism formula: ε ≈ sqrt(2 * ln(1.25/δ)) / σ
        (σ = noise_multiplier). This standard relation (Dwork & Roth, 2014,
        Theorem 3.22) applies to a single application of the Gaussian mechanism.
        """
        if noise_multiplier <= 0:
            return float("inf")  # no noise means no DP guarantee
        eps = math.sqrt(2.0 * math.log(1.25 / self.delta)) / noise_multiplier
        return eps

    def record_round(self, noise_multiplier: float) -> float:
        eps_round = self.compute_epsilon_for_round(noise_multiplier)
        self.epsilon_per_round = eps_round
        self.total_epsilon_consumed += eps_round
        self.rounds_accounted += 1
        logger.debug(
            f"🔒 PrivacyAccountant | round {self.rounds_accounted} | "
            f"eps_round≈{eps_round:.4f} | eps_total≈{self.total_epsilon_consumed:.4f}"
        )
        return self.total_epsilon_consumed

    def summary(self) -> Dict[str, float]:
        return {
            "delta": self.delta,
            "epsilon_per_round_approx": self.epsilon_per_round,
            "total_epsilon_consumed_approx_naive_composition": self.total_epsilon_consumed,
            "rounds_accounted": self.rounds_accounted,
        }
