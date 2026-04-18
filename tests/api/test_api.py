from __future__ import annotations


def test_health_returns_service_status_and_model_flag(client_with_model):
    response = client_with_model.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["model_path"] == "fake://in-memory-model"


def test_predict_returns_probability_and_class(client_with_model):
    response = client_with_model.post(
        "/predict",
        json={
            "tenure_months": 12,
            "monthly_charges": 90.5,
            "total_charges": 1086.0,
            "contract": "Month-to-month",
            "internet_service": "Fiber optic",
            "threshold": 0.5,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] in (0, 1)
    assert 0.0 <= payload["probability_churn"] <= 1.0
    assert payload["threshold"] == 0.5


def test_predict_returns_503_when_model_not_loaded(client_without_model):
    response = client_without_model.post(
        "/predict",
        json={
            "tenure_months": 3,
            "monthly_charges": 120,
            "total_charges": 360,
        },
    )

    assert response.status_code == 503


def test_predict_rejects_invalid_payload(client_with_model):
    response = client_with_model.post("/predict", json={})

    assert response.status_code == 422


def test_predict_rejects_invalid_threshold(client_with_model):
    response = client_with_model.post(
        "/predict",
        json={
            "tenure_months": 8,
            "monthly_charges": 55,
            "total_charges": 440,
            "threshold": 1.5,
        },
    )

    assert response.status_code == 422


def test_predict_accepts_minimum_required_fields(client_with_model):
    response = client_with_model.post(
        "/predict",
        json={
            "tenure_months": 10,
            "monthly_charges": 75,
            "total_charges": 750,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] in (0, 1)
    assert 0.0 <= payload["probability_churn"] <= 1.0


def test_predict_rejects_unknown_city(client_with_model):
    response = client_with_model.post(
        "/predict",
        json={
            "city": "Cidade Inexistente",
            "tenure_months": 12,
            "monthly_charges": 90.5,
            "total_charges": 1086.0,
        },
    )

    assert response.status_code == 422
