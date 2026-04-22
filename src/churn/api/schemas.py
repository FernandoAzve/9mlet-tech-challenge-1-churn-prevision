from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str


class PredictionResponse(BaseModel):
    prediction: int
    probability_churn: float
    threshold: float


class PredictionV2Request(BaseModel):
    model_config = ConfigDict(extra="forbid")

    city: str | None = Field(default=None, description="Customer city used by the trained city one-hot columns.")
    zip_code: int | None = Field(default=None, ge=0)
    latitude: float | None = None
    longitude: float | None = None
    tenure_months: int = Field(..., ge=0)
    monthly_charges: float = Field(..., ge=0.0)
    total_charges: float = Field(..., ge=0.0)
    cltv: int | None = Field(default=None, ge=0)

    gender: Literal["Female", "Male"] | None = None
    senior_citizen: bool | None = None
    partner: bool | None = None
    dependents: bool | None = None
    phone_service: bool | None = None

    multiple_lines: Literal["No", "Yes", "No phone service"] | None = None
    internet_service: Literal["DSL", "Fiber optic", "No"] | None = None
    online_security: Literal["No", "Yes", "No internet service"] | None = None
    online_backup: Literal["No", "Yes", "No internet service"] | None = None
    device_protection: Literal["No", "Yes", "No internet service"] | None = None
    tech_support: Literal["No", "Yes", "No internet service"] | None = None
    streaming_tv: Literal["No", "Yes", "No internet service"] | None = None
    streaming_movies: Literal["No", "Yes", "No internet service"] | None = None

    contract: Literal["Month-to-month", "One year", "Two year"] | None = None
    paperless_billing: bool | None = None
    payment_method: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ] | None = None

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
