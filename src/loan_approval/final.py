"""Scoring helpers for the single, final evaluation on the held-out test split."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from loan_approval.data import SEED
from loan_approval.threshold import metrics_at

BOOTSTRAP_METRICS = ("roc_auc", "precision", "recall", "f1", "accuracy")


def score_at_threshold(y_true, proba, threshold: float) -> dict:
    """Threshold metrics (plus the confusion counts) and the threshold-free ROC-AUC."""
    scores = metrics_at(y_true, proba, threshold)
    scores["roc_auc"] = float(roc_auc_score(y_true, proba))
    return scores


def bootstrap_ci(
    y_true,
    proba,
    threshold: float,
    n_boot: int = 2000,
    seed: int = SEED,
    level: float = 0.95,
) -> dict:
    """Percentile bootstrap confidence intervals: resample the test rows with replacement,
    re-score each resample, and take the middle `level` share of the results."""
    y = np.asarray(y_true)
    p = np.asarray(proba)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {m: [] for m in BOOTSTRAP_METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():  # ROC-AUC needs both classes in the resample
            continue
        s = score_at_threshold(y[idx], p[idx], threshold)
        for m in draws:
            draws[m].append(s[m])
    alpha = (1 - level) / 2
    return {
        m: {"low": float(np.quantile(v, alpha)), "high": float(np.quantile(v, 1 - alpha))}
        for m, v in draws.items()
    }
