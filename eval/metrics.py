from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BinaryMetrics:
    auc: float
    f1: float
    precision: float
    recall: float
    fpr_at_tpr_99: float
    fpr_at_tpr_95: float
    threshold_at_tpr_99: float


def binary_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> BinaryMetrics:
    from sklearn.metrics import (  # type: ignore[import-not-found]
        precision_recall_fscore_support,
        roc_auc_score,
        roc_curve,
    )

    preds = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    auc = float(roc_auc_score(labels, scores))

    fpr_curve, tpr_curve, thresholds = roc_curve(labels, scores)
    fpr_at_tpr_99 = _fpr_at_tpr(fpr_curve, tpr_curve, target_tpr=0.99)
    fpr_at_tpr_95 = _fpr_at_tpr(fpr_curve, tpr_curve, target_tpr=0.95)
    threshold_at_tpr_99 = _threshold_at_tpr(tpr_curve, thresholds, target_tpr=0.99)

    return BinaryMetrics(
        auc=auc,
        f1=float(f1),
        precision=float(precision),
        recall=float(recall),
        fpr_at_tpr_99=fpr_at_tpr_99,
        fpr_at_tpr_95=fpr_at_tpr_95,
        threshold_at_tpr_99=threshold_at_tpr_99,
    )


def _fpr_at_tpr(fpr: np.ndarray, tpr: np.ndarray, target_tpr: float) -> float:
    idx = np.searchsorted(tpr, target_tpr, side="left")
    if idx >= len(fpr):
        return 1.0
    return float(fpr[idx])


def _threshold_at_tpr(tpr: np.ndarray, thresholds: np.ndarray, target_tpr: float) -> float:
    idx = np.searchsorted(tpr, target_tpr, side="left")
    if idx >= len(thresholds):
        return 0.0
    return float(thresholds[idx])
