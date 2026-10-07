"""Choosing the decision threshold from out-of-fold predictions on TRAINING data only.

Label meaning: loan_status is the historical decision (1 = approved), not later repayment.
So "false approval" (model says approve, history says reject) is only a stand-in for a bad loan.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from loan_approval.data import SEED

DEFAULT_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def oof_probabilities(model, X, y, n_splits: int = 5, seed: int = SEED) -> np.ndarray:
    """Probability of 'approved' for every training row, each predicted by a model that
    never saw that row."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]


def metrics_at(y_true, proba, threshold: float) -> dict:
    y = np.asarray(y_true).astype(bool)
    pred = np.asarray(proba) >= threshold
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())  # false approval
    fn = int((~pred & y).sum())  # false rejection
    tn = int((~pred & ~y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / len(y),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def threshold_table(y_true, proba, grid=DEFAULT_GRID) -> pd.DataFrame:
    return pd.DataFrame([metrics_at(y_true, proba, t) for t in grid])


def best_threshold(y_true, proba, fp_cost: float, fn_cost: float, grid=DEFAULT_GRID) -> dict:
    """Threshold that minimises fp_cost * false_approvals + fn_cost * false_rejections.

    On ties the lowest threshold is returned.
    """
    table = threshold_table(y_true, proba, grid)
    table["cost"] = fp_cost * table["fp"] + fn_cost * table["fn"]
    row = table.loc[table["cost"].idxmin()]
    return {k: (int(v) if k in ("tp", "fp", "fn", "tn") else float(v)) for k, v in row.items()}
