"""Model pipelines.

Everything that learns from data (imputing, scaling, one-hot encoding, the classifier itself)
lives INSIDE one scikit-learn Pipeline. Cross-validation therefore re-fits all of it on each
training fold only, so nothing leaks from the validation or test rows.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from loan_approval.data import SEED, feature_columns

MODEL_NAMES = ("random_forest", "logistic_regression", "decision_tree", "svm")


def build_preprocessor(include_sensitive: bool = True) -> ColumnTransformer:
    """Impute, scale numbers and one-hot encode categories.

    Columns that are not listed (for example a dropped sensitive feature) are discarded.
    """
    numeric, categorical = feature_columns(include_sensitive)
    numeric_steps = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_steps, numeric), ("cat", categorical_steps, categorical)]
    )


def make_classifier(name: str, seed: int = SEED):
    """The four classifiers from the original project, with default settings."""
    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "decision_tree": DecisionTreeClassifier(random_state=seed),
        "svm": SVC(random_state=seed),
    }
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Choose from {sorted(models)}.")
    return models[name]


def build_pipeline(name: str, include_sensitive: bool = True, seed: int = SEED) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(include_sensitive)),
            ("model", make_classifier(name, seed)),
        ]
    )


class PreviousDefaultRule(ClassifierMixin, BaseEstimator):
    """Reference rule, nothing is learned: reject anyone with a previous default, approve the rest.

    It shows how much of this task a one-line rule already solves.
    """

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        approve = (X["previous_loan_defaults_on_file"] == "No").to_numpy().astype(float)
        return np.column_stack([1 - approve, approve])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_reference(name: str):
    """Reference points that every real model has to beat."""
    if name == "predict_rejected":
        return DummyClassifier(strategy="most_frequent")
    if name == "previous_default_rule":
        return PreviousDefaultRule()
    raise ValueError(f"Unknown reference '{name}'.")


REFERENCE_NAMES = ("predict_rejected", "previous_default_rule")
