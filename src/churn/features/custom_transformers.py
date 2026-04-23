from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TotalChargesCleaner(BaseEstimator, TransformerMixin):
    """
    Garante que a coluna ``Total Charges`` seja numérica (como no EDA / notebooks).

    Se vier texto ou valor inválido, vira NaN e depois 0 — evita quebrar o restante
    do pipeline quando alguém envia CSV “cru” em vez do arquivo já tratado.
    """

    def __init__(self, column_name: str = "Total Charges"):
        self.column_name = column_name

    def fit(self, X: pd.DataFrame, y: Any = None) -> TotalChargesCleaner:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        if self.column_name in X_copy.columns:
            X_copy[self.column_name] = pd.to_numeric(
                X_copy[self.column_name],
                errors="coerce",
            ).fillna(0.0)
        return X_copy


class FeatureColumnAligner(BaseEstimator, TransformerMixin):
    """
    Fixa a **ordem e o conjunto de colunas** vistos no treino (fit).

    No notebook 04 as features já vêm todas numéricas do CSV do EDA; na API o cliente
    manda só alguns campos — preenchemos o restante com 0 para bater com o one-hot
    esperado, igual à ideia de ``reindex(..., fill_value=0)`` usada na inferência.
    """

    def __init__(self) -> None:
        self.columns_: list[str] | None = None
        self.feature_names_in_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: Any = None) -> FeatureColumnAligner:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureColumnAligner espera pandas.DataFrame na entrada do fit.")
        self.columns_ = X.columns.tolist()
        self.feature_names_in_ = np.array(self.columns_, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.columns_ is None:
            raise RuntimeError("O transformador precisa ser ajustado com fit antes.")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        aligned = X.reindex(columns=self.columns_, fill_value=0)
        return aligned.to_numpy(dtype=np.float64)

