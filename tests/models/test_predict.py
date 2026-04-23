from __future__ import annotations

import numpy as np
import pandas as pd

from churn.models.predict import predict_with_churn_bundle


class DummyBundle:
    """Falso bundle: só precisa expor ``predict_proba_positive`` como o ``ChurnMLPBundle`` real."""

    feature_names_in_ = np.array(["a", "b", "c"])

    def predict_proba_positive(self, features: pd.DataFrame) -> np.ndarray:
        aligned = features.reindex(columns=["a", "b", "c"], fill_value=0)
        probability = pd.to_numeric(aligned["a"], errors="coerce").fillna(0.0) / 10.0
        p = np.clip(probability.to_numpy(dtype=np.float64), 0.0, 1.0)
        return np.asarray(p).ravel()


def test_predict_with_churn_bundle_aligns_missing_columns_and_applies_threshold():
    bundle = DummyBundle()
    features = pd.DataFrame([{"a": 8}])

    prediction_df = predict_with_churn_bundle(
        bundle=bundle,  # type: ignore[arg-type]
        features=features,
        threshold=0.7,
    )

    assert prediction_df.shape == (1, 2)
    assert int(prediction_df.loc[0, "prediction"]) == 1
    assert float(prediction_df.loc[0, "probability_churn"]) == 0.8


def test_predict_with_churn_bundle_rejects_invalid_threshold():
    bundle = DummyBundle()

    try:
        predict_with_churn_bundle(
            bundle=bundle,  # type: ignore[arg-type]
            features=pd.DataFrame([{"a": 1}]),
            threshold=2.0,
        )
    except ValueError as exc:
        assert "threshold" in str(exc)
    else:
        raise AssertionError("ValueError was expected for invalid threshold")
