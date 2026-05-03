from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from churn.features.preprocessing import build_mlp_preprocessing_pipeline
from churn.models.mlp_bundle import ChurnMLPBundle
from churn.models.mlp_torch import MLPChurn


class DummyPreprocessor:
    def __init__(self) -> None:
        self.named_steps: dict[str, object] = {}

    def transform(self, features: pd.DataFrame):
        return [[0.0] * len(features.columns)] * len(features)


def _make_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Tenure Months": [1, 2, 3],
            "Monthly Charges": [20.0, 30.0, 40.0],
            "Total Charges": [100.0, 200.0, 300.0],
        }
    )


def _make_preprocessor(features: pd.DataFrame):
    preprocessor = build_mlp_preprocessing_pipeline()
    preprocessor.fit(features)
    return preprocessor


def _zero_model(input_dim: int) -> MLPChurn:
    model = MLPChurn(input_dim=input_dim, hidden_dim_1=64, hidden_dim_2=32)
    for param in model.parameters():
        torch.nn.init.constant_(param, 0.0)
    return model


def test_bundle_predict_proba_positive_constant() -> None:
    features = _make_features()
    preprocessor = _make_preprocessor(features)
    model = _zero_model(input_dim=features.shape[1])

    bundle = ChurnMLPBundle(
        preprocessor=preprocessor,
        model=model,
        feature_columns=list(features.columns),
        threshold_prediction=0.5,
        threshold_otimo_f1_validacao=0.6,
        device=torch.device("cpu"),
    )

    proba = bundle.predict_proba_positive(features.iloc[:2])

    assert proba.shape == (2,)
    assert np.allclose(proba, 0.5, atol=1e-6)


def test_bundle_predict_proba_accepts_non_numpy_transform() -> None:
    features = _make_features().iloc[:1]
    model = _zero_model(input_dim=features.shape[1])

    bundle = ChurnMLPBundle(
        preprocessor=DummyPreprocessor(),
        model=model,
        feature_columns=list(features.columns),
        threshold_prediction=0.5,
        threshold_otimo_f1_validacao=0.6,
        device=torch.device("cpu"),
    )

    proba = bundle.predict_proba_positive(features)

    assert np.allclose(proba, 0.5, atol=1e-6)
    assert np.array_equal(bundle.feature_names_in_, np.array(list(features.columns), dtype=object))


def test_bundle_save_and_load_roundtrip(tmp_path: Path) -> None:
    features = _make_features()
    preprocessor = _make_preprocessor(features)
    model = _zero_model(input_dim=features.shape[1])

    bundle = ChurnMLPBundle(
        preprocessor=preprocessor,
        model=model,
        feature_columns=list(features.columns),
        threshold_prediction=0.5,
        threshold_otimo_f1_validacao=0.6,
        device=torch.device("cpu"),
    )

    bundle_dir = bundle.save(tmp_path / "bundle", extra_metadata={"source": "unit_test"})
    loaded = ChurnMLPBundle.load(bundle_dir)

    assert loaded.feature_columns == bundle.feature_columns

    proba = loaded.predict_proba_positive(features.iloc[:1])
    assert np.allclose(proba, 0.5, atol=1e-6)


def test_bundle_load_raises_when_metadata_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ChurnMLPBundle.load(tmp_path)


def test_bundle_load_raises_when_preprocessor_missing(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "feature_columns": ["a", "b"],
        "input_dim": 2,
        "hidden_dim_1": 64,
        "hidden_dim_2": 32,
        "threshold_prediction": 0.5,
        "threshold_otimo_f1_validacao": 0.6,
    }
    (bundle_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        ChurnMLPBundle.load(bundle_dir)
