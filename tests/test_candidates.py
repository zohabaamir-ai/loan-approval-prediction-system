import pytest

from loan_approval.candidates import (
    CHALLENGER_NAMES,
    FALLBACK_NAME,
    beats_baseline,
    build_candidate,
)
from loan_approval.data import TARGET
from loan_approval.synthetic import make_synthetic


@pytest.mark.parametrize("name", (FALLBACK_NAME, *CHALLENGER_NAMES))
def test_candidates_fit_and_predict_without_gender(name):
    df = make_synthetic(300, seed=5)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    model = build_candidate(name).fit(X, y)
    assert model.predict(X).shape == (len(X),)
    flipped = X.copy()
    flipped["person_gender"] = flipped["person_gender"].map({"male": "female", "female": "male"})
    assert (model.predict(X) == model.predict(flipped)).all()


def test_adoption_rule_needs_more_than_the_noise_margin():
    baseline = {"roc_auc": {"mean": 0.973, "std": 0.002}}
    just_above = {"roc_auc": {"mean": 0.9749}}
    clearly_above = {"roc_auc": {"mean": 0.9760}}
    assert beats_baseline(baseline, just_above)["wins"] is False
    assert beats_baseline(baseline, clearly_above)["wins"] is True


def test_unknown_candidate_is_rejected():
    with pytest.raises(ValueError):
        build_candidate("nope")
