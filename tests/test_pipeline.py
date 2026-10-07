import numpy as np
import pandas as pd
import pytest

from loan_approval.data import TARGET
from loan_approval.evaluate import SCORING, cross_validate_model
from loan_approval.pipeline import (
    MODEL_NAMES,
    PreviousDefaultRule,
    build_pipeline,
    build_reference,
)
from loan_approval.synthetic import make_synthetic


@pytest.fixture(scope="module")
def data():
    df = make_synthetic(400, seed=3)
    return df.drop(columns=[TARGET]), df[TARGET]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_each_pipeline_fits_and_predicts(name, data):
    X, y = data
    model = build_pipeline(name).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (len(X),)
    assert set(np.unique(pred)) <= {0, 1}


def test_unseen_category_does_not_crash(data):
    X, y = data
    model = build_pipeline("logistic_regression").fit(X, y)
    new = X.iloc[:3].copy()
    new["loan_intent"] = "SOMETHING_NEW"
    assert model.predict(new).shape == (3,)


def test_sensitive_feature_is_ignored_when_dropped(data):
    X, y = data
    model = build_pipeline("logistic_regression", include_sensitive=False).fit(X, y)
    flipped = X.copy()
    flipped["person_gender"] = flipped["person_gender"].map({"male": "female", "female": "male"})
    assert np.allclose(model.predict_proba(X), model.predict_proba(flipped))


def test_previous_default_rule_logic():
    X = pd.DataFrame({"previous_loan_defaults_on_file": ["Yes", "No", "No"]})
    rule = PreviousDefaultRule().fit(X, [0, 1, 1])
    assert rule.predict(X).tolist() == [0, 1, 1]


def test_cross_validation_returns_all_metrics_and_folds(data):
    X, y = data
    res = cross_validate_model(build_reference("previous_default_rule"), X, y, n_splits=3)
    for metric in SCORING:
        assert len(res[metric]["folds"]) == 3
        assert 0.0 <= res[metric]["mean"] <= 1.0
