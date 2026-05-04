from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandera as pa
import pytest

from churn.config import TARGET_COLUMN
from churn.data import io as data_io
from churn.data.pandera_schemas import (
    validate_churn_inference_features,
    validate_churn_training_dataset,
)
from churn.data.schemas import split_features_target, validate_ready_dataset


def _write_csv(path: Path, df: pd.DataFrame) -> Path:
    df.to_csv(path, index=False)
    return path


def test_resolve_data_path_explicit_file(tmp_path: Path) -> None:
    df = pd.DataFrame({TARGET_COLUMN: [0], "a": [1]})
    path = _write_csv(tmp_path / "data.csv", df)

    assert data_io.resolve_data_path(path) == path


def test_resolve_data_path_missing_raises() -> None:
    with pytest.raises(FileNotFoundError):
        data_io.resolve_data_path("missing.csv")


def test_resolve_data_path_uses_default_candidates(monkeypatch, tmp_path: Path) -> None:
    df = pd.DataFrame({TARGET_COLUMN: [0], "a": [1]})
    path = _write_csv(tmp_path / "data.csv", df)

    monkeypatch.setattr(data_io, "DEFAULT_DATA_PATH_CANDIDATES", (path,))

    assert data_io.resolve_data_path() == path


def test_resolve_data_path_default_candidates_missing(monkeypatch, tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    monkeypatch.setattr(data_io, "DEFAULT_DATA_PATH_CANDIDATES", (missing,))

    with pytest.raises(FileNotFoundError):
        data_io.resolve_data_path()


def test_load_ready_dataset_reads_csv(tmp_path: Path) -> None:
    df = pd.DataFrame({TARGET_COLUMN: [1], "a": [2]})
    path = _write_csv(tmp_path / "data.csv", df)

    loaded = data_io.load_ready_dataset(path)

    pd.testing.assert_frame_equal(loaded, df)


def test_load_features_target_splits_dataset(tmp_path: Path) -> None:
    df = pd.DataFrame({TARGET_COLUMN: [1, 0], "a": [2, 3]})
    path = _write_csv(tmp_path / "data.csv", df)

    features, target = data_io.load_features_target(path)

    assert list(features.columns) == ["a"]
    assert target.tolist() == [1, 0]


def test_stratified_train_test_split_keeps_classes() -> None:
    features = pd.DataFrame({"a": list(range(20))})
    target = pd.Series([0, 1] * 10)

    x_train, x_test, y_train, y_test = data_io.stratified_train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=0,
    )

    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}
    assert len(x_train) + len(x_test) == len(features)


def test_stratified_kfold_split_keeps_classes() -> None:
    features = pd.DataFrame({"a": list(range(20))})
    target = pd.Series([0, 1] * 10)

    splits = data_io.stratified_kfold_split(features, target, n_splits=4, random_state=0)

    assert len(splits) == 4
    for _train_idx, val_idx in splits:
        assert set(target.iloc[val_idx].unique()) == {0, 1}


def test_validate_ready_dataset_errors() -> None:
    df_missing = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError):
        validate_ready_dataset(df_missing)

    df_null = pd.DataFrame({TARGET_COLUMN: [1, np.nan]})
    with pytest.raises(ValueError):
        validate_ready_dataset(df_null)

    df_non_numeric = pd.DataFrame({TARGET_COLUMN: ["yes", "no"]})
    with pytest.raises(ValueError):
        validate_ready_dataset(df_non_numeric)


def test_split_features_target_returns_expected() -> None:
    df = pd.DataFrame({TARGET_COLUMN: [1, 0], "a": [2, 3]})

    features, target = split_features_target(df)

    assert list(features.columns) == ["a"]
    assert target.dtype == int
    assert target.tolist() == [1, 0]


def test_validate_churn_training_dataset_accepts_valid_df() -> None:
    df = pd.DataFrame(
        {
            TARGET_COLUMN: [0, 1],
            "Tenure Months": [1, 2],
            "Monthly Charges": [20.0, 30.0],
            "Total Charges": [100.0, 200.0],
            "CLTV": [1000.0, 2000.0],
        }
    )

    validated = validate_churn_training_dataset(df)

    pd.testing.assert_frame_equal(validated, df)


def test_validate_churn_training_dataset_rejects_empty() -> None:
    df = pd.DataFrame({TARGET_COLUMN: []})

    with pytest.raises(pa.errors.SchemaError):
        validate_churn_training_dataset(df)


def test_validate_churn_inference_features_rejects_empty() -> None:
    df = pd.DataFrame({"Tenure Months": []})

    with pytest.raises(pa.errors.SchemaError):
        validate_churn_inference_features(df)
