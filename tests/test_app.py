import pytest

from loan_approval.app import build_input_row, describe_prediction, fit_final_model
from loan_approval.data import ALL_COLUMNS, TARGET
from loan_approval.synthetic import make_synthetic

GOOD = dict(
    age=30,
    income=60000,
    emp_exp=5,
    loan_amnt=12000,
    loan_intent="PERSONAL",
    loan_int_rate=11.0,
    cred_hist_length=5,
    credit_score=650,
    home_ownership="RENT",
    education="Bachelor",
    previous_default="No",
)


def test_input_row_has_model_columns_and_computed_ratio():
    row = build_input_row(**GOOD)
    assert len(row) == 1
    assert row.loc[0, "loan_percent_income"] == 0.2  # 12000 / 60000
    assert "person_gender" not in row.columns  # gender is never asked for or used
    assert set(row.columns) <= set(ALL_COLUMNS) - {TARGET}


@pytest.mark.parametrize(
    "change",
    [
        {"age": 17},
        {"income": 0},
        {"loan_amnt": -5},
        {"credit_score": 900},
        {"loan_int_rate": 55},
        {"emp_exp": 50},
        {"income": None},
    ],
)
def test_bad_input_is_rejected_with_a_message(change):
    with pytest.raises(ValueError):
        build_input_row(**{**GOOD, **change})


def test_description_depends_on_threshold():
    assert "Likely approved" in describe_prediction(0.9, 0.78)
    assert "Likely rejected" in describe_prediction(0.6, 0.78)
    assert "91.0%" in describe_prediction(0.91, 0.78)


def test_final_model_gives_a_probability_for_a_typed_in_applicant():
    model = fit_final_model(make_synthetic(500, seed=2), name="hist_gradient_boosting_no_gender")
    probability = model.predict_proba(build_input_row(**GOOD))[0, 1]
    assert 0.0 <= probability <= 1.0
