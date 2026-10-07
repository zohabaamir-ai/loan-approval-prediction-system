import pandas as pd
import pytest

from loan_approval.data import (
    ALL_COLUMNS,
    SAMPLE_DATA_PATH,
    TARGET,
    clean_data,
    feature_columns,
    load_data,
    split_data,
)
from loan_approval.synthetic import make_synthetic


def test_sample_file_loads_with_expected_schema():
    df = load_data(SAMPLE_DATA_PATH)
    assert list(df.columns) == ALL_COLUMNS
    assert df[TARGET].nunique() == 2


def test_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="DATA.md"):
        load_data(tmp_path / "nope.csv")


def test_missing_columns_are_reported(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing expected columns"):
        load_data(bad)


def test_clean_drops_duplicates_and_impossible_ages():
    df = make_synthetic(50)
    dirty = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # one exact duplicate
    dirty.loc[1, "person_age"] = 144
    cleaned = clean_data(dirty)
    assert len(cleaned) == len(df) - 1  # duplicate removed, age-144 row removed
    assert cleaned["person_age"].max() <= 100


def test_split_is_stratified_disjoint_and_reproducible():
    df = make_synthetic(500)
    X_tr, X_te, y_tr, y_te = split_data(df, test_size=0.2, seed=1)
    assert len(X_te) == 100 and len(X_tr) == 400
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert abs(y_tr.mean() - y_te.mean()) < 0.02  # same class balance in both parts
    X_tr2, *_ = split_data(df, test_size=0.2, seed=1)
    assert X_tr.index.equals(X_tr2.index)


def test_sensitive_feature_can_be_dropped():
    num, cat = feature_columns(include_sensitive=False)
    assert "person_gender" not in cat
    assert "person_gender" in feature_columns()[1]
