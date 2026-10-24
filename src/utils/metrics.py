# -*- coding: utf-8 -*-
"""
metrics.py — evaluation metrics for fraud detection (imbalanced classes)
==============================================================
AP, ROC-AUC, F1, recall, precision, MCC, P@k and optimal-threshold selection.
Model selection uses val_ap by default (config.py) because at a positive rate of
about 9.76% accuracy and ROC-AUC alone are misleading.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    recall_score,
    precision_score,
    matthews_corrcoef,
    confusion_matrix,
)


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def find_best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Find the threshold that maximizes F1 on the validation set.
    The threshold is then applied unchanged to the test set (no threshold leakage).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(precisions),
        where=(precisions + recalls) != 0,
    )
    if len(thresholds) == 0:
        return 0.5, float(f1s[-1]) if len(f1s) else 0.0
    best_idx = int(np.argmax(f1s[:-1])) if len(f1s) > 1 else 0
    best_idx = min(best_idx, len(thresholds) - 1)
    return float(thresholds[best_idx]), float(f1s[best_idx])


def precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    """Precision@k: share of truly fraudulent transactions among the k highest-scored."""
    if k <= 0 or len(y_prob) == 0:
        return 0.0
    k = min(k, len(y_prob))
    top_idx = np.argsort(-y_prob)[:k]
    return float(np.mean(y_true[top_idx]))


def compute_all_metrics(
    y_true,
    y_prob,
    threshold: Optional[float] = None,
    k_for_precision_at_k: int = 100,
) -> Dict[str, float]:
    """
    Compute the full metric set.

    Args:
        y_true: true labels (0/1), Tensor or ndarray.
        y_prob: predicted probability of the positive class, Tensor or ndarray.
        threshold: if None, 0.5 is used (do not use this for validation; for the
            test set use the threshold fitted on validation).
        k_for_precision_at_k: value of k for Precision@k.

    Returns:
        dict with AP, AUC_ROC, F1, Recall, Precision, MCC, P@k, threshold.
    """
    y_true = _to_numpy(y_true).astype(int).ravel()
    y_prob = _to_numpy(y_prob).astype(float).ravel()

    out: Dict[str, float] = {}

    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)

    if n_pos == 0 or n_neg == 0:
        # both classes must be present, otherwise AP/AUC are undefined
        out.update({
            "AP": float("nan"), "AUC_ROC": float("nan"),
            "F1": 0.0, "Recall": 0.0, "Precision": 0.0, "MCC": 0.0,
            f"P@{k_for_precision_at_k}": 0.0, "threshold": threshold or 0.5,
            "n_pos": n_pos, "n_neg": n_neg,
        })
        return out

    out["AP"] = float(average_precision_score(y_true, y_prob))
    out["AUC_ROC"] = float(roc_auc_score(y_true, y_prob))

    thr = threshold if threshold is not None else 0.5
    y_pred = (y_prob >= thr).astype(int)

    out["F1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["Recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["Precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["MCC"] = float(matthews_corrcoef(y_true, y_pred)) if len(set(y_pred)) > 1 else 0.0
    out[f"P@{k_for_precision_at_k}"] = precision_at_k(y_true, y_prob, k_for_precision_at_k)
    out["threshold"] = float(thr)
    out["n_pos"] = n_pos
    out["n_neg"] = n_neg

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["TP"], out["FP"], out["FN"], out["TN"] = int(tp), int(fp), int(fn), int(tn)

    return out


def compute_pos_weight(y_train, max_pos_weight: float = 50.0) -> float:
    """
    Positive-class weight for BCEWithLogitsLoss:
        pos_weight = n_neg / n_pos  (capped at max_pos_weight)
    """
    y_train = _to_numpy(y_train).astype(int).ravel()
    n_pos = max(int(y_train.sum()), 1)
    n_neg = int(len(y_train) - n_pos)
    w = n_neg / n_pos
    return float(min(w, max_pos_weight))
