from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from churn.config import DEFAULT_DATA_PATH_CANDIDATES, DEFAULT_SEED, TARGET_COLUMN
from churn.data.schemas import split_features_target


def resolve_data_path(path: str | Path | None = None) -> Path:
    if path is not None:
        p = Path(path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Dataset not found at '{p}'.")

    for candidate in DEFAULT_DATA_PATH_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Dataset not found in default candidates.")


def load_ready_dataset(path: str | Path | None = None) -> pd.DataFrame:
    dataset_path = resolve_data_path(path)
    return pd.read_csv(dataset_path)


def load_features_target(
    path: str | Path | None = None,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    df = load_ready_dataset(path=path)
    return split_features_target(df, target_column=target_column)


def stratified_train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def stratified_kfold_split(
    features: pd.DataFrame,
    target: pd.Series,
    n_splits: int = 5,
    random_state: int = DEFAULT_SEED,
    shuffle: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2.")

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    return [(train_idx, val_idx) for train_idx, val_idx in splitter.split(features, target)]

