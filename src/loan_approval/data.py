"""Load, clean and split the loan dataset.

The raw CSV is NOT stored in this repository (see DATA.md). Place it at
data/loan_data.csv before running anything that needs the real data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "loan_data.csv"
SAMPLE_DATA_PATH = REPO_ROOT / "data" / "sample_synthetic.csv"

TARGET = "loan_status"
SEED = 42
TEST_SIZE = 0.2
MAX_PLAUSIBLE_AGE = 100

NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
]
CATEGORICAL_FEATURES = [
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent",
    "previous_loan_defaults_on_file",
]
# Sensitive in a lending context. Kept in the baseline (to match the original project),
# dropped from the final model.
SENSITIVE_FEATURES = ["person_gender"]

ALL_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]


def feature_columns(include_sensitive: bool = True) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) feature names, optionally without sensitive ones."""
    if include_sensitive:
        return list(NUMERIC_FEATURES), list(CATEGORICAL_FEATURES)
    numeric = [c for c in NUMERIC_FEATURES if c not in SENSITIVE_FEATURES]
    categorical = [c for c in CATEGORICAL_FEATURES if c not in SENSITIVE_FEATURES]
    return numeric, categorical


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Read the CSV and check that every expected column is present."""
    path = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. The raw data is not included in this repository. "
            "See DATA.md for where to download it, then save it as data/loan_data.csv."
        )
    df = pd.read_csv(path)
    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return df[ALL_COLUMNS].copy()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows and rows with an impossible age.

    Only row removal happens here. Anything that learns from the data (imputing,
    scaling, encoding) lives inside the scikit-learn Pipeline so it is fitted on
    training data only.
    """
    out = df.drop_duplicates()
    out = out[out["person_age"] <= MAX_PLAUSIBLE_AGE]
    return out.reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split. The test part is only used once, at the very end."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
