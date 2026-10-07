"""Small Gradio demo: type in an applicant, get an approval probability and a decision.

Run with:  python -m loan_approval.app      (needs data/loan_data.csv, see DATA.md)

The model is rebuilt from the data every time the app starts (a few seconds), so no pickle
tied to a library version is shipped. It is the exact final model from the evaluation: trained
on the training split only, with the decision threshold stored in reports/threshold.json.
This is a demo on synthetic data, not a real lending tool.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import pandas as pd

from loan_approval.candidates import build_candidate
from loan_approval.data import REPO_ROOT, clean_data, load_data, split_data
from loan_approval.selection import choose_final_name

THRESHOLD_PATH = REPO_ROOT / "reports" / "threshold.json"

CHOICES = {
    "person_education": ["High School", "Associate", "Bachelor", "Master", "Doctorate"],
    "person_home_ownership": ["RENT", "MORTGAGE", "OWN", "OTHER"],
    "loan_intent": [
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "PERSONAL",
        "DEBTCONSOLIDATION",
        "HOMEIMPROVEMENT",
    ],
    "previous_loan_defaults_on_file": ["No", "Yes"],
}

DISCLAIMER = (
    "Demo only. The model was trained on a synthetic dataset and predicts the *historical "
    "decision* in that data, not whether someone would repay. It must not be used for real "
    "lending decisions. Gender is not used by the model."
)


def load_threshold(path=THRESHOLD_PATH) -> float:
    return float(json.loads(path.read_text(encoding="utf-8"))["chosen"]["threshold"])


def fit_final_model(df: pd.DataFrame, name: str | None = None):
    """Fit the final model on the training split of `df` (never on the held-out test rows)."""
    X_train, _unused_test, y_train, _unused_test_y = split_data(clean_data(df))
    return build_candidate(name or choose_final_name()).fit(X_train, y_train)


@lru_cache(maxsize=1)
def get_model():
    return fit_final_model(load_data())


def build_input_row(
    age,
    income,
    emp_exp,
    loan_amnt,
    loan_intent,
    loan_int_rate,
    cred_hist_length,
    credit_score,
    home_ownership,
    education,
    previous_default,
) -> pd.DataFrame:
    """Validate the typed-in values and turn them into the one-row table the model expects.

    loan_percent_income is computed here instead of being asked for.
    """
    values = {
        "age": age,
        "income": income,
        "years employed": emp_exp,
        "loan amount": loan_amnt,
        "interest rate": loan_int_rate,
        "credit history length": cred_hist_length,
        "credit score": credit_score,
    }
    for label, value in values.items():
        if value is None:
            raise ValueError(f"Please enter a value for {label}.")
    if not 18 <= age <= 100:
        raise ValueError("Age must be between 18 and 100.")
    if income <= 0 or loan_amnt <= 0:
        raise ValueError("Income and loan amount must be greater than zero.")
    if not 0 <= emp_exp <= age:
        raise ValueError("Years employed must be between 0 and the applicant's age.")
    if not 0 <= loan_int_rate <= 40:
        raise ValueError("Interest rate must be between 0 and 40 (percent).")
    if not 0 <= cred_hist_length <= age:
        raise ValueError("Credit history length must be between 0 and the applicant's age.")
    if not 300 <= credit_score <= 850:
        raise ValueError("Credit score must be between 300 and 850.")

    return pd.DataFrame(
        [
            {
                "person_age": float(age),
                "person_income": float(income),
                "person_emp_exp": int(emp_exp),
                "loan_amnt": float(loan_amnt),
                "loan_int_rate": float(loan_int_rate),
                "loan_percent_income": round(loan_amnt / income, 2),
                "cb_person_cred_hist_length": float(cred_hist_length),
                "credit_score": int(credit_score),
                "person_education": education,
                "person_home_ownership": home_ownership,
                "loan_intent": loan_intent,
                "previous_loan_defaults_on_file": previous_default,
            }
        ]
    )


def describe_prediction(probability: float, threshold: float) -> str:
    verdict = "Likely approved" if probability >= threshold else "Likely rejected"
    return (
        f"## {verdict}\n\n"
        f"Estimated approval probability: **{probability:.1%}**\n\n"
        f"The decision threshold is {threshold:.2f}: the model says 'approved' only when the "
        "probability reaches it. It was set high on purpose to avoid approving applicants the "
        "historical data would have rejected.\n\n"
        f"*{DISCLAIMER}*"
    )


def build_demo(model, threshold: float):
    # Gradio is imported here so the rest of the package (and the tests) work without it.
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    import gradio as gr

    def run(*inputs) -> str:
        try:
            row = build_input_row(*inputs)
        except ValueError as error:
            raise gr.Error(str(error), print_exception=False) from error
        probability = float(model.predict_proba(row)[0, 1])
        return describe_prediction(probability, threshold)

    with gr.Blocks(title="Loan approval demo") as demo:
        gr.Markdown("# Loan approval prediction demo\n" + DISCLAIMER)
        with gr.Row():
            with gr.Column():
                age = gr.Number(label="Age", value=30, precision=0)
                income = gr.Number(label="Yearly income", value=60000)
                emp_exp = gr.Number(label="Years employed", value=5, precision=0)
                education = gr.Dropdown(
                    CHOICES["person_education"], value="Bachelor", label="Education"
                )
                home = gr.Dropdown(
                    CHOICES["person_home_ownership"], value="RENT", label="Home ownership"
                )
            with gr.Column():
                loan_amnt = gr.Number(label="Loan amount", value=10000)
                intent = gr.Dropdown(CHOICES["loan_intent"], value="PERSONAL", label="Loan purpose")
                rate = gr.Number(label="Interest rate (%)", value=11.0)
                history = gr.Number(label="Credit history length (years)", value=5, precision=0)
                score = gr.Number(label="Credit score (300-850)", value=650, precision=0)
                previous = gr.Dropdown(
                    CHOICES["previous_loan_defaults_on_file"],
                    value="No",
                    label="Previous loan default on file?",
                )
        button = gr.Button("Predict", variant="primary")
        result = gr.Markdown()
        button.click(
            run,
            inputs=[
                age,
                income,
                emp_exp,
                loan_amnt,
                intent,
                rate,
                history,
                score,
                home,
                education,
                previous,
            ],
            outputs=result,
        )
    return demo


def main() -> None:
    print("Training the model from data/loan_data.csv (a few seconds)...")
    model = get_model()  # fails fast, with a helpful message, if the dataset is missing
    demo = build_demo(model, load_threshold())
    demo.launch(server_name="127.0.0.1")


if __name__ == "__main__":
    main()
