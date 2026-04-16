from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TotalChargesCleaner(BaseEstimator, TransformerMixin):
    """Converts `Total Charges` to numeric, filling invalid values with zero."""

    def __init__(self, column_name: str = "Total Charges"):
        self.column_name = column_name

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TotalChargesCleaner":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        if self.column_name in X_copy.columns:
            X_copy[self.column_name] = pd.to_numeric(
                X_copy[self.column_name],
                errors="coerce",
            ).fillna(0.0)
        return X_copy

