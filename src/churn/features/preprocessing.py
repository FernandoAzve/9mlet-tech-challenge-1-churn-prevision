from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def infer_column_groups(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = features.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(
    features: pd.DataFrame,
) -> ColumnTransformer:
    numeric_cols, categorical_cols = infer_column_groups(features)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

