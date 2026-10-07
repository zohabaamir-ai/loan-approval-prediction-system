"""Stronger candidate models and the pre-agreed rule for adopting one.

All candidates are built WITHOUT the sensitive feature (person_gender), because that is how
the final model will be deployed. The Random Forest without gender is the fallback: it is what
we keep if no challenger clearly wins.
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from loan_approval.data import SEED
from loan_approval.evaluate import PRIMARY_METRIC
from loan_approval.pipeline import build_pipeline, build_preprocessor

# The fallback, shown so we can see what dropping gender costs.
FALLBACK_NAME = "random_forest_no_gender"
# Challengers have to pass the adoption rule below.
CHALLENGER_NAMES = ("hist_gradient_boosting_no_gender",)
BASELINE_NAME = "random_forest"  # the honest baseline (all original features)


def build_candidate(name: str, seed: int = SEED) -> Pipeline:
    if name == "random_forest_no_gender":
        return build_pipeline("random_forest", include_sensitive=False, seed=seed)
    if name == "hist_gradient_boosting_no_gender":
        return Pipeline(
            [
                ("preprocess", build_preprocessor(include_sensitive=False)),
                ("model", HistGradientBoostingClassifier(random_state=seed)),
            ]
        )
    raise ValueError(f"Unknown candidate '{name}'.")


def beats_baseline(baseline: dict, candidate: dict, metric: str = PRIMARY_METRIC) -> dict:
    """The adoption rule, fixed before any candidate was run.

    A challenger is adopted only if its mean cross-validated score is higher than the baseline's
    mean PLUS the baseline's fold-to-fold standard deviation (that is, the cross-validation noise).
    """
    required = baseline[metric]["mean"] + baseline[metric]["std"]
    got = candidate[metric]["mean"]
    return {
        "metric": metric,
        "required_above": required,
        "candidate_mean": got,
        "wins": got > required,
    }
