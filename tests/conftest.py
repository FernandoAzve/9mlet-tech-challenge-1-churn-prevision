from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


class FakePipeline:
    feature_names_in_ = np.array(["Tenure Months", "Monthly Charges", "Total Charges"])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        tenure = pd.to_numeric(features["Tenure Months"], errors="coerce").fillna(0.0)
        monthly = pd.to_numeric(features["Monthly Charges"], errors="coerce").fillna(0.0)
        total = pd.to_numeric(features["Total Charges"], errors="coerce").fillna(0.0)

        score = (0.03 * monthly) + (0.0005 * total) - (0.01 * tenure)
        probability = 1.0 / (1.0 + np.exp(-score / 10.0))
        probability = np.clip(probability, 0.0, 1.0)

        return np.column_stack([1.0 - probability, probability])


@pytest.fixture
def client_with_model() -> TestClient:
    from churn.api.main import create_app

    app = create_app()
    with TestClient(app) as client:
        client.app.state.model = FakePipeline()
        client.app.state.model_path = "fake://in-memory-model"
        yield client


@pytest.fixture
def client_without_model() -> TestClient:
    from churn.api.main import create_app

    app = create_app()
    with TestClient(app) as client:
        client.app.state.model = None
        client.app.state.model_path = "missing://model"
        yield client
