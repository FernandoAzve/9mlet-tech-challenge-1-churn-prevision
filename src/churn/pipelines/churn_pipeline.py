"""
Compatibilidade com notebooks que importam ``build_churn_pipeline``.

O fluxo oficial de produção é **somente** MLP + pré-processador sklearn
(ver ``build_mlp_preprocessing_pipeline``). Os parâmetros ``estimator_name``
e o antigo classificador sklearn **não** existem mais — são ignorados ou
levantam erro se alguém esperar ``predict_proba`` de um pipeline completo sklearn.
"""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from churn.features.preprocessing import build_mlp_preprocessing_pipeline


def build_churn_pipeline(
    features: pd.DataFrame,
    estimator_name: str = "logistic_regression",
) -> Pipeline:
    """
    Retorna o **pré-processador sklearn** usado antes da MLP.

    ``estimator_name`` é mantido só para não quebrar chamadas antigas; é ignorado.
    """
    _ = features, estimator_name
    return build_mlp_preprocessing_pipeline()
