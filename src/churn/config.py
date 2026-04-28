from __future__ import annotations

from pathlib import Path

DEFAULT_SEED = 42
TARGET_COLUMN = "Churn Value"

DEFAULT_DATA_PATH_CANDIDATES = (
    Path("data/Telco_customer_churn_ready.csv"),
    Path("../data/Telco_customer_churn_ready.csv"),
)

# Pasta padrão do bundle oficial: preprocessor.joblib + mlp_state.pt + metadata.json
DEFAULT_MLP_BUNDLE_DIR = Path("models/mlp_bundle")
