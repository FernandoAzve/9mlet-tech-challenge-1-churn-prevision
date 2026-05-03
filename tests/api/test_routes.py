from __future__ import annotations

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from churn.api.main import create_app
from churn.api.routes import _build_v2_features, _set_binary_enum, _set_if_not_none
from churn.api.schemas import PredictionV2Request


def test_set_if_not_none() -> None:
    features: dict[str, object] = {}

    _set_if_not_none(features, "a", None)
    assert "a" not in features

    _set_if_not_none(features, "a", 1)
    assert features["a"] == 1


def test_set_binary_enum_sets_expected() -> None:
    features: dict[str, object] = {}
    expected = {"Multiple Lines_Yes", "Multiple Lines_No phone service"}

    _set_binary_enum(
        features,
        expected,
        prefix="Multiple Lines",
        value="Yes",
        positive_label="Yes",
        special_label="No phone service",
    )

    assert features["Multiple Lines_Yes"] is True

    features = {}
    _set_binary_enum(
        features,
        expected,
        prefix="Multiple Lines",
        value="No phone service",
        positive_label="Yes",
        special_label="No phone service",
    )

    assert features["Multiple Lines_No phone service"] is True


def test_build_v2_features_builds_expected_columns() -> None:
    payload = PredictionV2Request(
        city="Gotham",
        zip_code=12345,
        latitude=1.23,
        longitude=-4.56,
        tenure_months=10,
        monthly_charges=80.5,
        total_charges=500.0,
        cltv=1000,
        gender="Male",
        senior_citizen=True,
        partner=True,
        dependents=True,
        phone_service=True,
        multiple_lines="Yes",
        internet_service="Fiber optic",
        online_security="No internet service",
        online_backup="Yes",
        device_protection="Yes",
        tech_support="Yes",
        streaming_tv="Yes",
        streaming_movies="Yes",
        contract="Two year",
        paperless_billing=True,
        payment_method="Electronic check",
        threshold=0.7,
    )

    expected_columns = {
        "City_Gotham",
        "Gender_Male",
        "Senior Citizen_Yes",
        "Partner_Yes",
        "Dependents_Yes",
        "Phone Service_Yes",
        "Multiple Lines_Yes",
        "Internet Service_Fiber optic",
        "Online Security_No internet service",
        "Online Backup_Yes",
        "Device Protection_Yes",
        "Tech Support_Yes",
        "Streaming TV_Yes",
        "Streaming Movies_Yes",
        "Contract_Two year",
        "Paperless Billing_Yes",
        "Payment Method_Electronic check",
    }

    features = _build_v2_features(payload, expected_columns)

    assert features["Tenure Months"] == 10
    assert features["Monthly Charges"] == 80.5
    assert features["Total Charges"] == 500.0
    assert features["Zip Code"] == 12345
    assert features["Latitude"] == 1.23
    assert features["Longitude"] == -4.56
    assert features["CLTV"] == 1000
    assert features["City_Gotham"] is True
    assert features["Gender_Male"] is True
    assert features["Senior Citizen_Yes"] is True
    assert features["Partner_Yes"] is True
    assert features["Dependents_Yes"] is True
    assert features["Phone Service_Yes"] is True
    assert features["Multiple Lines_Yes"] is True
    assert features["Internet Service_Fiber optic"] is True
    assert features["Online Security_No internet service"] is True
    assert features["Online Backup_Yes"] is True
    assert features["Device Protection_Yes"] is True
    assert features["Tech Support_Yes"] is True
    assert features["Streaming TV_Yes"] is True
    assert features["Streaming Movies_Yes"] is True
    assert features["Contract_Two year"] is True
    assert features["Paperless Billing_Yes"] is True
    assert features["Payment Method_Electronic check"] is True


def test_build_v2_features_rejects_unknown_city() -> None:
    payload = PredictionV2Request(
        city="Nowhere",
        tenure_months=1,
        monthly_charges=2.0,
        total_charges=3.0,
        threshold=0.5,
    )

    with pytest.raises(HTTPException) as exc:
        _build_v2_features(payload, expected_columns=set())

    assert exc.value.status_code == 422


def test_build_v2_features_other_branches() -> None:
    payload = PredictionV2Request(
        tenure_months=5,
        monthly_charges=40.0,
        total_charges=200.0,
        internet_service="No",
        contract="One year",
        payment_method="Mailed check",
        threshold=0.5,
    )

    expected_columns = {
        "Internet Service_No",
        "Contract_One year",
        "Payment Method_Mailed check",
    }

    features = _build_v2_features(payload, expected_columns)

    assert features["Internet Service_No"] is True
    assert features["Contract_One year"] is True
    assert features["Payment Method_Mailed check"] is True


def test_build_v2_features_credit_card_payment() -> None:
    payload = PredictionV2Request(
        tenure_months=2,
        monthly_charges=30.0,
        total_charges=60.0,
        payment_method="Credit card (automatic)",
        threshold=0.5,
    )

    expected_columns = {"Payment Method_Credit card (automatic)"}

    features = _build_v2_features(payload, expected_columns)

    assert features["Payment Method_Credit card (automatic)"] is True


def test_predict_returns_500_when_feature_columns_missing() -> None:
    app = create_app()

    class DummyModel:
        def predict_proba_positive(self, features):
            return np.array([0.1])

    with TestClient(app) as client:
        client.app.state.model = DummyModel()
        response = client.post(
            "/predict",
            json={
                "tenure_months": 1,
                "monthly_charges": 10.0,
                "total_charges": 10.0,
            },
        )

    assert response.status_code == 500


def test_predict_rejects_unknown_city() -> None:
    app = create_app()

    class DummyModel:
        feature_names_in_ = np.array(["Tenure Months", "Monthly Charges", "Total Charges"], dtype=object)

        def predict_proba_positive(self, features):
            return np.array([0.1])

    with TestClient(app) as client:
        client.app.state.model = DummyModel()
        response = client.post(
            "/predict",
            json={
                "city": "Nowhere",
                "tenure_months": 1,
                "monthly_charges": 10.0,
                "total_charges": 10.0,
            },
        )

    assert response.status_code == 422
