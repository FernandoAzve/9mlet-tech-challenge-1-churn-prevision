from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from churn.api.schemas import (
    HealthResponse,
    PredictionResponse,
    PredictionV2Request,
)
from churn.models.predict import get_expected_feature_columns, predict_with_pipeline

router = APIRouter(tags=["churn"])


def _set_if_not_none(features: dict[str, object], key: str, value: object | None) -> None:
    if value is not None:
        features[key] = value


def _set_one_hot(features: dict[str, object], expected_columns: set[str], key: str) -> None:
    if key in expected_columns:
        features[key] = True


def _set_binary_enum(
    features: dict[str, object],
    expected_columns: set[str],
    prefix: str,
    value: str | None,
    positive_label: str = "Yes",
    special_label: str | None = None,
) -> None:
    if value is None:
        return
    if value == positive_label:
        _set_one_hot(features, expected_columns, f"{prefix}_{positive_label}")
    if special_label is not None and value == special_label:
        _set_one_hot(features, expected_columns, f"{prefix}_{special_label}")


def _build_v2_features(payload: PredictionV2Request, expected_columns: set[str]) -> dict[str, object]:
    features: dict[str, object] = {
        "Tenure Months": payload.tenure_months,
        "Monthly Charges": payload.monthly_charges,
        "Total Charges": payload.total_charges,
    }

    _set_if_not_none(features, "Zip Code", payload.zip_code)
    _set_if_not_none(features, "Latitude", payload.latitude)
    _set_if_not_none(features, "Longitude", payload.longitude)
    _set_if_not_none(features, "CLTV", payload.cltv)

    if payload.city:
        city_key = f"City_{payload.city}"
        if city_key not in expected_columns:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"City '{payload.city}' is not available in this trained model. "
                    "Use a city present in the training data."
                ),
            )
        features[city_key] = True

    if payload.gender == "Male":
        _set_one_hot(features, expected_columns, "Gender_Male")

    if payload.senior_citizen:
        _set_one_hot(features, expected_columns, "Senior Citizen_Yes")
    if payload.partner:
        _set_one_hot(features, expected_columns, "Partner_Yes")
    if payload.dependents:
        _set_one_hot(features, expected_columns, "Dependents_Yes")
    if payload.phone_service:
        _set_one_hot(features, expected_columns, "Phone Service_Yes")

    _set_binary_enum(features, expected_columns, "Multiple Lines", payload.multiple_lines, "Yes", "No phone service")
    if payload.internet_service == "Fiber optic":
        _set_one_hot(features, expected_columns, "Internet Service_Fiber optic")
    if payload.internet_service == "No":
        _set_one_hot(features, expected_columns, "Internet Service_No")

    _set_binary_enum(features, expected_columns, "Online Security", payload.online_security, "Yes", "No internet service")
    _set_binary_enum(features, expected_columns, "Online Backup", payload.online_backup, "Yes", "No internet service")
    _set_binary_enum(features, expected_columns, "Device Protection", payload.device_protection, "Yes", "No internet service")
    _set_binary_enum(features, expected_columns, "Tech Support", payload.tech_support, "Yes", "No internet service")
    _set_binary_enum(features, expected_columns, "Streaming TV", payload.streaming_tv, "Yes", "No internet service")
    _set_binary_enum(features, expected_columns, "Streaming Movies", payload.streaming_movies, "Yes", "No internet service")

    if payload.contract == "One year":
        _set_one_hot(features, expected_columns, "Contract_One year")
    if payload.contract == "Two year":
        _set_one_hot(features, expected_columns, "Contract_Two year")

    if payload.paperless_billing:
        _set_one_hot(features, expected_columns, "Paperless Billing_Yes")

    if payload.payment_method == "Credit card (automatic)":
        _set_one_hot(features, expected_columns, "Payment Method_Credit card (automatic)")
    if payload.payment_method == "Electronic check":
        _set_one_hot(features, expected_columns, "Payment Method_Electronic check")
    if payload.payment_method == "Mailed check":
        _set_one_hot(features, expected_columns, "Payment Method_Mailed check")

    return features


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    model = getattr(request.app.state, "model", None)
    model_path = getattr(request.app.state, "model_path", "")
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        model_path=model_path,
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionV2Request, request: Request) -> PredictionResponse:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check CHURN_MODEL_PATH.")

    expected_columns = get_expected_feature_columns(model)
    if expected_columns is None:
        raise HTTPException(
            status_code=500,
            detail="Unable to resolve expected feature columns from the loaded model.",
        )

    features = _build_v2_features(payload=payload, expected_columns=set(expected_columns))
    features_df = pd.DataFrame([features])
    prediction_df = predict_with_pipeline(
        pipeline=model,
        features=features_df,
        threshold=payload.threshold,
    )

    first_row = prediction_df.iloc[0]
    return PredictionResponse(
        prediction=int(first_row["prediction"]),
        probability_churn=float(first_row["probability_churn"]),
        threshold=payload.threshold,
    )
