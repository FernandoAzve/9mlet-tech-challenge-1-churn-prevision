from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from fastapi.testclient import TestClient

from churn.api.main import create_app
from churn.features.preprocessing import build_mlp_preprocessing_pipeline
from churn.models.mlp_bundle import ChurnMLPBundle
from churn.models.mlp_torch import MLPChurn


def _create_bundle(bundle_dir: Path) -> Path:
    features = pd.DataFrame(
        {
            "Tenure Months": [1, 2],
            "Monthly Charges": [20.0, 30.0],
            "Total Charges": [100.0, 200.0],
        }
    )
    preprocessor = build_mlp_preprocessing_pipeline()
    preprocessor.fit(features)

    model = MLPChurn(input_dim=features.shape[1], hidden_dim_1=64, hidden_dim_2=32)
    for param in model.parameters():
        torch.nn.init.constant_(param, 0.0)

    bundle = ChurnMLPBundle(
        preprocessor=preprocessor,
        model=model,
        feature_columns=list(features.columns),
        threshold_prediction=0.5,
        threshold_otimo_f1_validacao=0.6,
        device=torch.device("cpu"),
    )
    bundle.save(bundle_dir)
    return bundle_dir


def test_lifespan_sets_model_none_when_bundle_missing(monkeypatch, tmp_path: Path) -> None:
    bundle_dir = tmp_path / "missing_bundle"
    monkeypatch.setenv("CHURN_MODEL_BUNDLE_DIR", str(bundle_dir))

    app = create_app()

    with TestClient(app) as client:
        assert client.app.state.model is None
        assert client.app.state.model_path == str(bundle_dir.resolve())


def test_lifespan_loads_bundle_when_present(monkeypatch, tmp_path: Path) -> None:
    bundle_dir = _create_bundle(tmp_path / "bundle")
    monkeypatch.setenv("CHURN_MODEL_BUNDLE_DIR", str(bundle_dir))

    app = create_app()

    with TestClient(app) as client:
        assert client.app.state.model is not None
        assert client.app.state.model_path == str(bundle_dir.resolve())
