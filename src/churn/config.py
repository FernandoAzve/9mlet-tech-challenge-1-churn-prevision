from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_SEED = 42
TARGET_COLUMN = "Churn Value"

DEFAULT_DATA_PATH_CANDIDATES = (
    Path("data/Telco_customer_churn_ready.csv"),
    Path("../data/Telco_customer_churn_ready.csv"),
)


@dataclass(frozen=True)
class TrainingConfig:
    test_size: float = 0.2
    random_state: int = DEFAULT_SEED
    estimator_name: str = "logistic_regression"
    output_dir: Path = Path("models/sklearn")

