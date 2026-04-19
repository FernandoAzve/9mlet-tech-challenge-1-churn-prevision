# -*- coding: utf-8 -*-
"""
Pré-processamento oficial para o pipeline **MLP + notebooks**.

O fluxo legado (``ColumnTransformer`` + classificador sklearn) foi removido.
Aqui montamos um único ``sklearn.pipeline.Pipeline`` reprodutível:

1. ``TotalChargesCleaner`` — coluna ``Total Charges`` sempre numérica.
2. ``FeatureColumnAligner`` — mesma ordem de colunas do treino; faltantes viram 0.
3. ``StandardScaler`` — média/desvio **aprendidos só no conjunto de treino** (notebook 04).

O dataset esperado é o CSV pronto do EDA (``Telco_customer_churn_ready.csv``), já com one-hot.
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from churn.features.custom_transformers import FeatureColumnAligner, TotalChargesCleaner


def build_mlp_preprocessing_pipeline() -> Pipeline:
    """
    Retorna o pipeline sklearn que deve ser ``fit`` no treino e salvo em disco
    junto com a MLP (ver ``ChurnMLPBundle``).
    """
    return Pipeline(
        steps=[
            ("clean_total_charges", TotalChargesCleaner()),
            ("align_features", FeatureColumnAligner()),
            ("scale", StandardScaler()),
        ]
    )
