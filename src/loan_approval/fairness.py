"""A simple group-by-group audit of the final predictions.

This is a check, not a certificate of fairness: it only looks at the groups listed and at
a few standard gaps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AGE_BANDS = [0, 24, 29, 39, 200]
AGE_LABELS = ["24 or under", "25-29", "30-39", "40 and over"]


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=AGE_BANDS, labels=AGE_LABELS).astype(str)


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def group_report(y_true, proba, groups, threshold: float) -> dict:
    y = np.asarray(y_true).astype(bool)
    pred = np.asarray(proba) >= threshold
    g = np.asarray(groups)
    report: dict = {}
    for name in sorted(set(g.tolist())):
        mask = g == name
        yy, pp = y[mask], pred[mask]
        tp = int((pp & yy).sum())
        fp = int((pp & ~yy).sum())
        fn = int((~pp & yy).sum())
        tn = int((~pp & ~yy).sum())
        report[str(name)] = {
            "n": int(mask.sum()),
            "true_approval_rate": float(yy.mean()),
            "model_approval_rate": float(pp.mean()),
            "recall": _rate(tp, tp + fn),
            "false_approval_rate": _rate(fp, fp + tn),
        }
    return report


def max_gap(report: dict, key: str) -> float:
    values = [r[key] for r in report.values()]
    return float(max(values) - min(values))
