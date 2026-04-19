from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from churn.api.logging_config import configure_logging
from churn.api.middleware import LatencyLoggingMiddleware
from churn.api.routes import router
from churn.models.registry import load_pipeline_artifact

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/sklearn/churn_pipeline.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.getenv("CHURN_MODEL_PATH", DEFAULT_MODEL_PATH))
    app.state.model_path = str(model_path)
    app.state.model = None

    if model_path.exists():
        app.state.model = load_pipeline_artifact(model_path)
        logger.info("Model loaded successfully from %s", model_path)
    else:
        logger.warning("Model file not found at %s. /predict will return 503.", model_path)

    yield


def create_app() -> FastAPI:
    configure_logging()

    app = FastAPI(
        title="Churn Prediction API",
        version="0.1.0",
        description=(
            "API de inferencia para previsao de churn com pipeline sklearn treinado no projeto."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(LatencyLoggingMiddleware)
    app.include_router(router)
    return app


app = create_app()
