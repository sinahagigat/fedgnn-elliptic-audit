# -*- coding: utf-8 -*-
"""Evaluation metrics (AP, ROC-AUC, F1, precision, recall, confusion matrix)
on a node mask, global-test evaluation of a trained model and validation-tuned
thresholds.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
)
from src.compat import safe_forward

logger = logging.getLogger("fedgnn.evaluate")

# client.py computes validation AP and AUC with the same sklearn calls as this
# module, so the values are identical. Threshold-dependent metrics (F1,
# precision, recall, MCC, confusion matrix) come only from evaluate_split.


# Evaluation on an arbitrary mask of a single Data object
@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    device: str,
    boundary_idx: Optional[torch.Tensor] = None,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Full computation of the standard binary classification metrics on the mask subset.

    Notes:
      - if the mask contains no positive example, AP/AUC are meaningless; in
        that case NaN is returned instead of a silent skip, and handling it is
        left to the caller.
      - the hard classification threshold (for F1/Precision/Recall) is argmax over
        the two-dimensional logits, not an arbitrary threshold on p(class 1).
    """
    model.eval()
    node_emb, logits = safe_forward(model, x, edge_index, boundary_idx)

    y_masked = y[mask].cpu().numpy()
    logits_masked = logits[mask]

    probs = F.softmax(logits_masked, dim=-1)[:, 1].cpu().numpy()

    if threshold is None:
        # historical behaviour: argmax over two logits == threshold 0.5 on p(y=1)
        # (they differ only on exact ties, an event of measure zero)
        preds = logits_masked.argmax(dim=-1).cpu().numpy()
        thr_used = 0.5
    else:
        thr_used = float(threshold)
        preds = (probs >= thr_used).astype(np.int64)

    n_pos = int((y_masked == 1).sum())
    n_total = len(y_masked)

    if n_pos == 0 or n_pos == n_total:
        logger.warning(
            f"evaluate_split: mask with {n_total} samples has only one class "
            f"(n_pos={n_pos}); AP/AUC set to NaN."
        )
        ap, auc = float("nan"), float("nan")
    else:
        ap = float(average_precision_score(y_masked, probs))
        auc = float(roc_auc_score(y_masked, probs))

    f1 = float(f1_score(y_masked, preds, zero_division=0))
    precision = float(precision_score(y_masked, preds, zero_division=0))
    recall = float(recall_score(y_masked, preds, zero_division=0))
    try:
        mcc = float(matthews_corrcoef(y_masked, preds))
    except Exception:
        mcc = 0.0

    cm = confusion_matrix(y_masked, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "ap": ap,
        "auc": auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "mcc": mcc,
        "threshold": float(thr_used),
        "n_samples": n_total,
        "n_positive": n_pos,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# v7: metrics from a score vector (used by per-client inference, where scores
# from several client subgraphs are concatenated before computing metrics)
@torch.no_grad()
def predict_scores(model, x, edge_index, boundary_idx=None) -> np.ndarray:
    """p(y=1) for every node of one graph, model in eval mode."""
    model.eval()
    _, logits = safe_forward(model, x, edge_index, boundary_idx)
    return F.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()


def metrics_from_scores(
    y_true: np.ndarray, probs: np.ndarray, threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Same metric set as evaluate_split, computed from scores.
    threshold=None reproduces evaluate_split's default rule exactly:
    argmax over two logits picks class 1 iff p(y=1) > 0.5 (ties go to class 0).
    """
    y_true = np.asarray(y_true).astype(np.int64).ravel()
    probs = np.asarray(probs, dtype=np.float64).ravel()
    if threshold is None:
        preds = (probs > 0.5).astype(np.int64)
        thr_used = 0.5
    else:
        thr_used = float(threshold)
        preds = (probs >= thr_used).astype(np.int64)
    n_pos = int((y_true == 1).sum())
    n_total = int(y_true.size)
    if n_pos == 0 or n_pos == n_total:
        ap, auc = float("nan"), float("nan")
    else:
        ap = float(average_precision_score(y_true, probs))
        auc = float(roc_auc_score(y_true, probs))
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    try:
        mcc = float(matthews_corrcoef(y_true, preds))
    except Exception:
        mcc = 0.0
    return {
        "ap": ap,
        "auc": auc,
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "mcc": mcc,
        "threshold": float(thr_used),
        "n_samples": n_total,
        "n_positive": n_pos,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def tuned_metrics_from_scores(
    y_val: np.ndarray, p_val: np.ndarray, y_test: np.ndarray, p_test: np.ndarray
) -> Dict[str, float]:
    """Threshold chosen on validation (max F1), then applied once to test."""
    sel = find_best_threshold(y_val, p_val, metric="f1")
    out = metrics_from_scores(y_test, p_test, threshold=sel["threshold"])
    out["threshold_source"] = "val"
    out["val_tuned_f1"] = float(sel["score"])
    return out


# Thin wrapper for the final global test evaluation (used in train.py)
@torch.no_grad()
def evaluate_variant_on_global_test(
    model: torch.nn.Module,
    data,
    mask: torch.Tensor,
    device: str,
) -> Dict[str, float]:
    """
    Thin wrapper over evaluate_split for evaluating the optimized global model on
    the global test set (no client split); the only path used in train.py for
    the final report.
    """
    boundary_idx = getattr(data, "boundary_idx", None)
    return evaluate_split(
        model=model,
        x=data.x,
        edge_index=data.edge_index,
        y=data.y,
        mask=mask,
        device=device,
        boundary_idx=boundary_idx,
    )


# Threshold selection on the validation period (never on test)
def find_best_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    metric: str = "f1",
) -> Dict[str, float]:
    """
    F1-maximizing threshold (positive class) on one split, by exact search.

    Why precision_recall_curve rather than a uniform grid? F1 as a function of the
    threshold is a step function that changes only at observed score values, so
    the global maximum is necessarily at one of those points. A uniform grid is
    slower and can miss the maximum (this is an exhaustive search, not an approximation).

    Tie rule: on equal F1 the largest threshold is chosen (higher precision,
    fewer FP); a deterministic, reproducible rule, not an arbitrary choice.
    """
    if metric != "f1":
        raise NotImplementedError(f"[evaluate] unsupported metric: {metric}")

    y_true = np.asarray(y_true).astype(np.int64).ravel()
    probs = np.asarray(probs, dtype=np.float64).ravel()
    if y_true.shape != probs.shape:
        raise ValueError(f"[evaluate] shape mismatch: {y_true.shape} vs {probs.shape}")

    n_pos = int((y_true == 1).sum())
    if n_pos == 0 or n_pos == y_true.size:
        logger.warning(
            "find_best_threshold: tuning split has a single class; "
            "falling back to the default threshold 0.5 (no calibration)."
        )
        return {"threshold": 0.5, "score": float("nan"), "n_candidates": 0}

    prec, rec, thr = precision_recall_curve(y_true, probs)
    prec, rec = prec[:-1], rec[:-1]          # the last point has no matching threshold
    denom = prec + rec
    f1 = np.where(denom > 0.0, 2.0 * prec * rec / np.maximum(denom, 1e-12), 0.0)

    best = f1.max()
    cand = np.flatnonzero(f1 >= best - 1e-12)
    i = int(cand[np.argmax(thr[cand])])
    return {
        "threshold": float(thr[i]),
        "score": float(f1[i]),
        "n_candidates": int(thr.size),
    }


@torch.no_grad()
def evaluate_variant_on_global_test_tuned(
    model: torch.nn.Module,
    data,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    device: str,
    metric: str = "f1",
) -> Dict[str, float]:
    """
    Final test evaluation with the threshold calibrated on val.

    Strict order of operations (prevents information leakage):
      1) extract p(y=1) on val_mask only
      2) find τ* on that same val split
      3) apply τ* to test; test never takes part in choosing τ*

    The output keeps both the τ* metrics and the 0.5-threshold metrics, so that
    the report is auditable and tuning on test can be ruled out.
    AP/AUC are threshold-independent and stay the same in both cases.
    """
    boundary_idx = getattr(data, "boundary_idx", None)
    model.eval()
    _, logits = safe_forward(model, data.x, data.edge_index, boundary_idx)

    probs_val = F.softmax(logits[val_mask], dim=-1)[:, 1].cpu().numpy()
    y_val = data.y[val_mask].cpu().numpy()
    sel = find_best_threshold(y_val, probs_val, metric=metric)
    tau = float(sel["threshold"])

    common = dict(
        model=model, x=data.x, edge_index=data.edge_index, y=data.y,
        mask=test_mask, device=device, boundary_idx=boundary_idx,
    )
    m_def = evaluate_split(threshold=None, **common)
    out = dict(evaluate_split(threshold=tau, **common))

    out["threshold_source"] = "val"
    out["threshold_metric"] = metric
    out["val_tuned_score"] = float(sel["score"])
    out["n_threshold_candidates"] = int(sel["n_candidates"])
    for k in ("f1", "precision", "recall", "mcc"):
        out[f"{k}_at_default"] = float(m_def[k])
    out["confusion_matrix_at_default"] = m_def["confusion_matrix"]

    logger.info(
        f"threshold from val: τ*={tau:.4f} (val_F1={sel['score']:.4f}) | "
        f"test_F1: {m_def['f1']:.4f} @0.5 → {out['f1']:.4f} @τ* | "
        f"P={out['precision']:.4f} R={out['recall']:.4f} MCC={out['mcc']:.4f} | "
        f"AP={out['ap']:.4f} (unchanged)"
    )
    return out
