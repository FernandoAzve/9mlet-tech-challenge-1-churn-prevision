"""FastAPI application package for churn inference."""

from churn.api.main import app, create_app

__all__ = ["app", "create_app"]
