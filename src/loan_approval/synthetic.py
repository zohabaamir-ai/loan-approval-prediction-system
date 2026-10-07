"""Generate a small FAKE dataset with the same columns as the real one.

It is used by the tests and CI so they run without the real data. The values are random
and carry no real signal about lending: never report results from it.

Regenerate the committed sample with:  python -m loan_approval.synthetic
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from loan_approval.data import ALL_COLUMNS, SAMPLE_DATA_PATH


def make_synthetic(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 60, n)
    income = rng.lognormal(11, 0.5, n).round(0)
    loan_amnt = rng.integers(1000, 30000, n).astype(float)
    int_rate = rng.uniform(6, 20, n).round(2)
    prev_default = rng.choice(["No", "Yes"], n, p=[0.5, 0.5])
    score = rng.integers(400, 850, n)
    # Made-up rule, only so that both classes exist and a model has something to learn.
    logit = 0.01 * (score - 600) - 2.0 * (prev_default == "Yes") - 0.8 + rng.normal(0, 0.5, n)
    status = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame(
        {
            "person_age": age.astype(float),
            "person_gender": rng.choice(["female", "male"], n),
            "person_education": rng.choice(
                ["High School", "Associate", "Bachelor", "Master", "Doctorate"], n
            ),
            "person_income": income,
            "person_emp_exp": np.clip(age - 22, 0, None) // 2,
            "person_home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE", "OTHER"], n),
            "loan_amnt": loan_amnt,
            "loan_intent": rng.choice(
                [
                    "EDUCATION",
                    "MEDICAL",
                    "VENTURE",
                    "PERSONAL",
                    "DEBTCONSOLIDATION",
                    "HOMEIMPROVEMENT",
                ],
                n,
            ),
            "loan_int_rate": int_rate,
            "loan_percent_income": (loan_amnt / income).round(2),
            "cb_person_cred_hist_length": rng.integers(2, 20, n).astype(float),
            "credit_score": score,
            "previous_loan_defaults_on_file": prev_default,
            "loan_status": status,
        }
    )
    return df[ALL_COLUMNS]


if __name__ == "__main__":
    SAMPLE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    make_synthetic().to_csv(SAMPLE_DATA_PATH, index=False)
    print(f"Wrote {SAMPLE_DATA_PATH}")
