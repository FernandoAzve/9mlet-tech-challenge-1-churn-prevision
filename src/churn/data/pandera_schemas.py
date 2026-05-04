"""
Contratos Pandera para o CSV de treino (com alvo) e para inferência (sem alvo obrigatório).
"""

from __future__ import annotations

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

from churn.config import TARGET_COLUMN


def _non_empty(df: pd.DataFrame) -> bool:
    return len(df) > 0


def _target_no_null(df: pd.DataFrame) -> bool:
    return bool(df[TARGET_COLUMN].notna().all())


ChurnTrainingDatasetSchema = DataFrameSchema(
    {
        TARGET_COLUMN: Column(int, checks=[Check.isin([0, 1])], nullable=False),
        # Presentes no CSV completo; required=False aceita recortes mínimos em testes.
        "Tenure Months": Column(float, checks=[Check.ge(0)], nullable=False, required=False),
        "Monthly Charges": Column(float, nullable=False, required=False),
        "Total Charges": Column(float, checks=[Check.ge(0)], nullable=False, required=False),
        "CLTV": Column(float, checks=[Check.ge(0)], nullable=False, required=False),
    },
    strict=False,
    coerce=True,
    checks=[
        pa.Check(_non_empty, error="O dataset de treino está vazio."),
        pa.Check(_target_no_null, error=f"A coluna '{TARGET_COLUMN}' contém valores nulos."),
    ],
)


ChurnInferenceFeaturesSchema = DataFrameSchema(
    {
        "Tenure Months": Column(float, checks=[Check.ge(0)], nullable=False, required=False),
        "Monthly Charges": Column(float, nullable=False, required=False),
        "Total Charges": Column(float, checks=[Check.ge(0)], nullable=False, required=False),
        "CLTV": Column(float, checks=[Check.ge(0)], nullable=False, required=False),
    },
    strict=False,
    coerce=True,
    checks=[
        pa.Check(_non_empty, error="O arquivo de entrada está vazio."),
    ],
)


def validate_churn_training_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Valida o CSV bruto de treino antes do split e do pré-processamento."""
    return ChurnTrainingDatasetSchema.validate(df, lazy=True)


def validate_churn_inference_features(df: pd.DataFrame) -> pd.DataFrame:
    """Valida features de inferência (colunas extras do one-hot são permitidas)."""
    return ChurnInferenceFeaturesSchema.validate(df, lazy=True)
