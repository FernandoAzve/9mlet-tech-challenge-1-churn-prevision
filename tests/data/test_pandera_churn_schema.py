from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from churn.config import TARGET_COLUMN
from churn.data.pandera_schemas import (
    ChurnInferenceFeaturesSchema,
    ChurnTrainingDatasetSchema,
    validate_churn_inference_features,
    validate_churn_training_dataset,
)


def test_training_schema_accepts_minimal_contract() -> None:
    df = pd.DataFrame(
        {
            "Tenure Months": [12],
            "Monthly Charges": [50.0],
            "Total Charges": [100.0],
            TARGET_COLUMN: [0],
        }
    )
    ChurnTrainingDatasetSchema.validate(df)


def test_training_schema_rejects_invalid_target() -> None:
    df = pd.DataFrame(
        {
            "Tenure Months": [12],
            "Monthly Charges": [50.0],
            "Total Charges": [100.0],
            TARGET_COLUMN: [2],
        }
    )

    with pytest.raises(SchemaErrors):
        validate_churn_training_dataset(df)


def test_inference_schema_rejects_empty_dataframe() -> None:
    df = pd.DataFrame()

    with pytest.raises(SchemaErrors):
        validate_churn_inference_features(df)


def test_inference_schema_accepts_partial_columns() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "Tenure Months": rng.integers(1, 24, size=3),
            "Monthly Charges": rng.uniform(20.0, 100.0, size=3),
            "Total Charges": rng.uniform(0.0, 500.0, size=3),
            "Extra_dummy_X": [1, 0, 1],
        }
    )
    out = validate_churn_inference_features(df)
    assert len(out) == 3


def test_inference_schema_optional_core_when_absent() -> None:
    df = pd.DataFrame({"Zip Code": [90001], "Latitude": [34.0], "Longitude": [-118.0]})
    ChurnInferenceFeaturesSchema.validate(df)

