from __future__ import annotations

import numpy as np
import pandas as pd

from churn.models.predict import predict_with_pipeline


class DummyPipeline:
    feature_names_in_ = np.array(["a", "b", "c"])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        assert list(features.columns) == ["a", "b", "c"]
        probability = pd.to_numeric(features["a"], errors="coerce").fillna(0.0) / 10.0
        probability = np.clip(probability, 0.0, 1.0)
        return np.column_stack([1.0 - probability, probability])


def test_predict_with_pipeline_aligns_missing_columns_and_applies_threshold():
    pipeline = DummyPipeline()
    features = pd.DataFrame([{"a": 8}])

    prediction_df = predict_with_pipeline(
        pipeline=pipeline,
        features=features,
        threshold=0.7,
    )

    assert prediction_df.shape == (1, 2)
    assert int(prediction_df.loc[0, "prediction"]) == 1
    assert float(prediction_df.loc[0, "probability_churn"]) == 0.8


def test_predict_with_pipeline_rejects_invalid_threshold():
    pipeline = DummyPipeline()

    try:
        predict_with_pipeline(
            pipeline=pipeline,
            features=pd.DataFrame([{"a": 1}]),
            threshold=2.0,
        )
    except ValueError as exc:
        assert "threshold" in str(exc)
    else:
        raise AssertionError("ValueError was expected for invalid threshold")
