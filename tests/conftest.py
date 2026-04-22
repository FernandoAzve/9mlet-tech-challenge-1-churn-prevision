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


class FakeChurnBundle:
    """
    Simula ``ChurnMLPBundle`` na memória para testes de API (sem carregar disco/PyTorch).

    A rota usa ``predict_with_churn_bundle``, que chama ``predict_proba_positive`` — não ``predict_proba``.
    """

    feature_names_in_ = np.array(["Tenure Months", "Monthly Charges", "Total Charges"])

    def predict_proba_positive(self, features: pd.DataFrame) -> np.ndarray:
        cols = [str(x) for x in self.feature_names_in_.tolist()]
        aligned = features.reindex(columns=cols, fill_value=0)
        tenure = pd.to_numeric(aligned["Tenure Months"], errors="coerce").fillna(0.0)
        monthly = pd.to_numeric(aligned["Monthly Charges"], errors="coerce").fillna(0.0)
        total = pd.to_numeric(aligned["Total Charges"], errors="coerce").fillna(0.0)

        score = (0.03 * monthly) + (0.0005 * total) - (0.01 * tenure)
        probability = 1.0 / (1.0 + np.exp(-score / 10.0))
        probability = np.clip(probability, 0.0, 1.0)
        return np.asarray(probability, dtype=np.float64).ravel()


@pytest.fixture
def client_with_model() -> TestClient:
    from churn.api.main import create_app

    app = create_app()
    with TestClient(app) as client:
        client.app.state.model = FakeChurnBundle()
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
