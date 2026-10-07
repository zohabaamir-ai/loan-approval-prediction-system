"""Cross-validation helpers. Only ever used on the TRAINING part of the data."""

from __future__ import annotations

from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from loan_approval.data import SEED

SCORING = {
    "roc_auc": "roc_auc",
    "f1": make_scorer(f1_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "accuracy": "accuracy",
}
# ROC-AUC is the metric used to choose between models (it does not depend on a threshold).
PRIMARY_METRIC = "roc_auc"


def cross_validate_model(estimator, X, y, n_splits: int = 5, seed: int = SEED) -> dict:
    """Stratified k-fold cross-validation. Returns mean, std and every fold's score."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    res = cross_validate(estimator, X, y, cv=cv, scoring=SCORING, error_score="raise")
    out: dict = {}
    for metric in SCORING:
        folds = res[f"test_{metric}"]
        out[metric] = {
            "mean": float(folds.mean()),
            "std": float(folds.std(ddof=1)),
            "folds": [round(float(v), 5) for v in folds],
        }
    out["fit_seconds_mean"] = float(res["fit_time"].mean())
    return out


def format_table(results: dict) -> str:
    """Plain-text table of mean +/- std per metric."""
    metrics = list(SCORING)
    lines = ["model".ljust(24) + "".join(m.rjust(18) for m in metrics)]
    for name, res in results.items():
        cells = "".join(f"{res[m]['mean']:.3f} +/- {res[m]['std']:.3f}".rjust(18) for m in metrics)
        lines.append(name.ljust(24) + cells)
    return "\n".join(lines)
