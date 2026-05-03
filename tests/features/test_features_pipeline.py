from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn.features.custom_transformers import FeatureColumnAligner, TotalChargesCleaner
from churn.features.preprocessing import build_mlp_preprocessing_pipeline
from churn.pipelines.churn_pipeline import build_churn_pipeline


def test_total_charges_cleaner_converts_values() -> None:
    df = pd.DataFrame({"Total Charges": ["10.5", "bad", None], "Other": [1, 2, 3]})

    cleaner = TotalChargesCleaner()
    out = cleaner.fit(df).transform(df)

    assert out["Total Charges"].tolist() == [10.5, 0.0, 0.0]


def test_total_charges_cleaner_ignores_missing_column() -> None:
    df = pd.DataFrame({"Other": [1, 2]})

    cleaner = TotalChargesCleaner()
    out = cleaner.transform(df)

    pd.testing.assert_frame_equal(out, df)


def test_feature_column_aligner_validations() -> None:
    aligner = FeatureColumnAligner()

    with pytest.raises(TypeError):
        aligner.fit(np.array([[1.0, 2.0]]))

    with pytest.raises(RuntimeError):
        aligner.transform(pd.DataFrame({"a": [1]}))


def test_feature_column_aligner_aligns_and_fills() -> None:
    aligner = FeatureColumnAligner().fit(pd.DataFrame({"a": [1.0], "b": [2.0]}))

    out = aligner.transform(pd.DataFrame({"b": [5.0]}))
    assert out.shape == (1, 2)
    assert out[0, 0] == 0.0
    assert out[0, 1] == 5.0

    out_from_list = aligner.transform([{"a": 3.0, "b": 4.0}])
    assert out_from_list.shape == (1, 2)


def test_build_mlp_preprocessing_pipeline() -> None:
    pipeline = build_mlp_preprocessing_pipeline()
    step_names = [name for name, _ in pipeline.steps]
    assert step_names == ["clean_total_charges", "align_features", "scale"]

    df = pd.DataFrame(
        {
            "Tenure Months": [1, 2],
            "Monthly Charges": [20.0, 30.0],
            "Total Charges": [100.0, 200.0],
        }
    )

    pipeline.fit(df)
    out = pipeline.transform(df)
    assert out.shape == (2, 3)


def test_build_churn_pipeline_returns_mlp_preprocessor() -> None:
    df = pd.DataFrame(
        {
            "Tenure Months": [1],
            "Monthly Charges": [20.0],
            "Total Charges": [100.0],
        }
    )

    pipeline = build_churn_pipeline(df)
    step_names = [name for name, _ in pipeline.steps]
    assert step_names == ["clean_total_charges", "align_features", "scale"]
