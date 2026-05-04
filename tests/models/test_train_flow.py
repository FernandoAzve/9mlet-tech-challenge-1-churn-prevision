from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from churn.config import TARGET_COLUMN
from churn.models import train as train_module


def _make_dataset(n_rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * (n_rows // 2))
    return pd.DataFrame(
        {
            "Tenure Months": rng.integers(0, 24, size=n_rows),
            "Monthly Charges": rng.uniform(20.0, 100.0, size=n_rows),
            "Total Charges": rng.uniform(0.0, 2000.0, size=n_rows),
            TARGET_COLUMN: y,
        }
    )


def _install_mlflow_stub(monkeypatch) -> dict[str, object]:
    state: dict[str, object] = {
        "params": [],
        "metrics": [],
        "tracking_uri": None,
        "experiment": None,
        "run_name": None,
        "model_artifact": None,
    }

    class DummyRun:
        def __init__(self, run_id: str = "run-123") -> None:
            self.info = SimpleNamespace(run_id=run_id)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def set_tracking_uri(uri: str) -> None:
        state["tracking_uri"] = uri

    def set_experiment(name: str) -> None:
        state["experiment"] = name

    def start_run(run_name: str | None = None):
        state["run_name"] = run_name
        return DummyRun()

    def log_param(key: str, value: object) -> None:
        state["params"].append((key, value))

    def log_metric(key: str, value: object) -> None:
        state["metrics"].append((key, value))

    def log_model(model, artifact_path: str) -> None:
        state["model_artifact"] = artifact_path

    mlflow_module = ModuleType("mlflow")
    mlflow_module.set_tracking_uri = set_tracking_uri
    mlflow_module.set_experiment = set_experiment
    mlflow_module.start_run = start_run
    mlflow_module.log_param = log_param
    mlflow_module.log_metric = log_metric

    mlflow_pytorch = ModuleType("mlflow.pytorch")
    mlflow_pytorch.log_model = log_model
    mlflow_module.pytorch = mlflow_pytorch

    monkeypatch.setitem(sys.modules, "mlflow", mlflow_module)
    monkeypatch.setitem(sys.modules, "mlflow.pytorch", mlflow_pytorch)

    return state


def test_file_sha256(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("abc", encoding="utf-8")

    digest = train_module._file_sha256(path)

    assert len(digest) == 64


def test_split_holdout_validates_target() -> None:
    df = pd.DataFrame({"a": [1, 2]})

    with pytest.raises(ValueError):
        train_module._split_holdout_test(df, TARGET_COLUMN, random_state=0)


def test_split_holdout_returns_shapes() -> None:
    df = _make_dataset(40)

    x_temp, x_test, y_temp, y_test, feature_columns = train_module._split_holdout_test(
        df,
        TARGET_COLUMN,
        random_state=0,
    )

    assert feature_columns == ["Tenure Months", "Monthly Charges", "Total Charges"]
    assert len(x_temp) + len(x_test) == len(df)
    assert len(y_temp) + len(y_test) == len(df)


def test_train_mlp_flow_skips_mlflow(tmp_path: Path, monkeypatch) -> None:
    df = _make_dataset(40)
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(train_module, "NUM_EPOCHS_MAX", 2)
    monkeypatch.setattr(train_module, "PATIENCE", 1)
    monkeypatch.setattr(train_module, "BATCH_SIZE", 8)

    output = train_module.train_mlp_flow(
        data_path=data_path,
        random_state=0,
        bundle_dir=tmp_path / "bundle",
        mlflow_run_name="unit_test",
        skip_mlflow=True,
        n_cv_splits=2,
    )

    assert Path(output["bundle_dir"]).exists()
    assert output["mlflow_run_id"] is None
    assert "test_accuracy" in output["metrics"]


def test_train_mlp_flow_with_mlflow_stub(tmp_path: Path, monkeypatch) -> None:
    df = _make_dataset(40)
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    state = _install_mlflow_stub(monkeypatch)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlflow.db"))
    monkeypatch.setattr(train_module, "NUM_EPOCHS_MAX", 1)
    monkeypatch.setattr(train_module, "PATIENCE", 1)
    monkeypatch.setattr(train_module, "BATCH_SIZE", 8)

    output = train_module.train_mlp_flow(
        data_path=data_path,
        random_state=0,
        bundle_dir=tmp_path / "bundle_mlflow",
        mlflow_run_name="stub-run",
        skip_mlflow=False,
        n_cv_splits=2,
    )

    assert output["mlflow_run_id"] == "run-123"
    assert state["run_name"] == "stub-run"
    assert state["model_artifact"] == "model"
    assert "test_accuracy" in output["metrics"]


def test_train_parse_args_reads_cli(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--data-path",
            "data.csv",
            "--bundle-dir",
            "bundle",
            "--random-state",
            "7",
            "--mlflow-run-name",
            "unit-test",
            "--skip-mlflow",
        ],
    )

    args = train_module._parse_args()

    assert args.data_path == "data.csv"
    assert args.bundle_dir == "bundle"
    assert args.random_state == 7
    assert args.mlflow_run_name == "unit-test"
    assert args.skip_mlflow is True
    assert args.cv_folds == 5
    assert args.log_level == "INFO"


def test_train_main_prints_output(monkeypatch, capsys) -> None:
    dummy_out = {
        "bundle_dir": "bundle",
        "mlflow_run_id": None,
        "metrics": {"test_accuracy": 0.9, "test_f1_score": 0.8, "test_roc_auc": 0.7},
    }

    monkeypatch.setattr(
        train_module,
        "_parse_args",
        lambda: SimpleNamespace(
            data_path="data.csv",
            random_state=0,
            bundle_dir="bundle",
            mlflow_run_name="unit-test",
            skip_mlflow=True,
            cv_folds=5,
            log_level="INFO",
        ),
    )
    monkeypatch.setattr(train_module, "train_mlp_flow", lambda **kwargs: dummy_out)

    train_module.main()

    line = capsys.readouterr().out.strip().splitlines()[-1]
    payload = json.loads(line)
    assert "bundle_dir" in payload


def test_train_module_runs_as_main(monkeypatch, tmp_path: Path, capsys) -> None:
    df = _make_dataset(24)
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    bundle_dir = tmp_path / "bundle"

    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--data-path",
            str(data_path),
            "--bundle-dir",
            str(bundle_dir),
            "--random-state",
            "0",
            "--mlflow-run-name",
            "unit-test",
            "--skip-mlflow",
        ],
    )

    sys.modules.pop("churn.models.train", None)
    runpy.run_module("churn.models.train", run_name="__main__")

    line = capsys.readouterr().out.strip().splitlines()[-1]
    payload = json.loads(line)
    assert "bundle_dir" in payload
