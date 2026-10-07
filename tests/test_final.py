import numpy as np
import pandas as pd

from loan_approval.fairness import age_band, group_report, max_gap
from loan_approval.final import bootstrap_ci, score_at_threshold


def _toy(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = np.clip(0.55 * y + rng.normal(0.2, 0.25, n), 0, 1)
    return y, p


def test_score_at_threshold_has_all_fields():
    y, p = _toy()
    s = score_at_threshold(y, p, 0.5)
    for key in ("roc_auc", "precision", "recall", "f1", "accuracy", "tp", "fp", "fn", "tn"):
        assert key in s
    assert s["tp"] + s["fp"] + s["fn"] + s["tn"] == len(y)


def test_bootstrap_interval_contains_the_point_estimate_and_is_reproducible():
    y, p = _toy()
    point = score_at_threshold(y, p, 0.5)
    ci = bootstrap_ci(y, p, 0.5, n_boot=200, seed=1)
    for metric, bounds in ci.items():
        assert bounds["low"] < bounds["high"]
        assert bounds["low"] <= point[metric] <= bounds["high"]
    assert ci == bootstrap_ci(y, p, 0.5, n_boot=200, seed=1)


def test_group_report_hand_checked():
    y = [1, 1, 0, 0, 1, 0]
    p = [0.9, 0.2, 0.8, 0.1, 0.9, 0.1]
    groups = ["a", "a", "a", "b", "b", "b"]
    r = group_report(y, p, groups, 0.5)
    assert r["a"]["n"] == 3 and r["b"]["n"] == 3
    assert abs(r["a"]["model_approval_rate"] - 2 / 3) < 1e-9  # rows 0 and 2 approved
    assert r["a"]["recall"] == 0.5  # one of two truly approved found
    assert r["b"]["false_approval_rate"] == 0.0
    assert abs(max_gap(r, "model_approval_rate") - (2 / 3 - 1 / 3)) < 1e-9


def test_age_band_edges():
    bands = age_band(pd.Series([21, 24, 25, 29, 30, 39, 40, 80]))
    assert bands.tolist() == [
        "24 or under",
        "24 or under",
        "25-29",
        "25-29",
        "30-39",
        "30-39",
        "40 and over",
        "40 and over",
    ]
