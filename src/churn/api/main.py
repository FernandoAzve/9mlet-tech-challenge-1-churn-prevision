from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from churn.api.routes import router
from churn.config import DEFAULT_MLP_BUNDLE_DIR
from churn.models.registry import load_churn_mlp_bundle

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle_dir = Path(os.getenv("CHURN_MODEL_BUNDLE_DIR", str(DEFAULT_MLP_BUNDLE_DIR)))
    app.state.model_path = str(bundle_dir.resolve())
    app.state.model = None

    meta = bundle_dir / "metadata.json"
    if meta.is_file():
        app.state.model = load_churn_mlp_bundle(bundle_dir)
        logger.info("Bundle MLP carregado de %s", bundle_dir)
    else:
        logger.warning(
            "Bundle não encontrado em %s (esperado metadata.json). /predict retornará 503.",
            bundle_dir,
        )

    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Churn Prediction API",
        version="0.2.0",
        description=(
            "API de inferência de churn usando o bundle MLP "
            "(pré-processador sklearn serializado + rede PyTorch)."
        ),
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
