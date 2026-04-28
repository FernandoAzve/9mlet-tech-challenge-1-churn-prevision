from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from churn.config import TARGET_COLUMN

REMOVED_COLUMNS_FROM_RAW = (
    "CustomerID",
    "Count",
    "Country",
    "State",
    "Lat Long",
    "Churn Label",
    "Churn Score",
    "Churn Reason",
)


@dataclass(frozen=True)
class DatasetContract:
    target_column: str = TARGET_COLUMN
    id_like_columns: tuple[str, ...] = ("CustomerID", "Count")
    post_event_columns: tuple[str, ...] = ("Churn Label", "Churn Score", "Churn Reason")


def validate_ready_dataset(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> None:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")
    if df[target_column].isna().any():
        raise ValueError("Target column contains null values.")
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise ValueError("Target column must be numeric.")


def split_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    validate_ready_dataset(df, target_column=target_column)
    features = df.drop(columns=[target_column]).copy()
    target = df[target_column].astype(int).copy()
    return features, target

