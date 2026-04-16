from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn.config import DEFAULT_SEED
from churn.features.custom_transformers import TotalChargesCleaner
from churn.features.preprocessing import build_preprocessor


def build_estimator(estimator_name: str = "logistic_regression"):
    if estimator_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=DEFAULT_SEED,
            n_jobs=1,
            class_weight="balanced",
        )
    if estimator_name == "logistic_regression":
        return LogisticRegression(
            max_iter=5000,
            n_jobs=1,
            random_state=DEFAULT_SEED,
            class_weight="balanced",
        )
    raise ValueError(f"Unsupported estimator_name '{estimator_name}'.")


def build_churn_pipeline(features, estimator_name: str = "logistic_regression") -> Pipeline:
    preprocessor = build_preprocessor(features)
    estimator = build_estimator(estimator_name=estimator_name)

    return Pipeline(
        steps=[
            ("clean_total_charges", TotalChargesCleaner()),
            ("preprocessor", preprocessor),
            ("estimator", estimator),
        ]
    )

