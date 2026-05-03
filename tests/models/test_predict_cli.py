from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from churn.models import predict as predict_module
from churn.features.preprocessing import build_mlp_preprocessing_pipeline
from churn.models.mlp_bundle import ChurnMLPBundle
from churn.models.mlp_torch import MLPChurn


class DummyBundle:
    def predict_proba_positive(self, features: pd.DataFrame) -> np.ndarray:
        return np.array([0.1] * len(features), dtype=float)


class DummyNumpyFeatures:
    feature_names_in_ = np.array(["a", "b"], dtype=object)


class DummyListFeatures:
    feature_names_in_ = ["a", "b"]


class DummyNoFeatures:
    pass


def _create_bundle(bundle_dir: Path) -> Path:
    features = pd.DataFrame(
        {
            "Tenure Months": [1, 2],
            "Monthly Charges": [20.0, 30.0],
            "Total Charges": [100.0, 200.0],
        }
    )
    preprocessor = build_mlp_preprocessing_pipeline()
    preprocessor.fit(features)

    model = MLPChurn(input_dim=features.shape[1], hidden_dim_1=64, hidden_dim_2=32)
    for param in model.parameters():
        torch.nn.init.constant_(param, 0.0)

    bundle = ChurnMLPBundle(
        preprocessor=preprocessor,
        model=model,
        feature_columns=list(features.columns),
        threshold_prediction=0.5,
        threshold_otimo_f1_validacao=0.6,
        device=torch.device("cpu"),
    )
    bundle.save(bundle_dir)
    return bundle_dir


def test_get_expected_feature_columns_variants() -> None:
    assert predict_module.get_expected_feature_columns(DummyNumpyFeatures()) == ["a", "b"]
    assert predict_module.get_expected_feature_columns(DummyListFeatures()) == ["a", "b"]
    assert predict_module.get_expected_feature_columns(DummyNoFeatures()) is None


def test_predict_from_csv_uses_bundle(monkeypatch, tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    input_path = tmp_path / "input.csv"
    df.to_csv(input_path, index=False)

    dummy = DummyBundle()
    monkeypatch.setattr(predict_module, "load_churn_mlp_bundle", lambda _: dummy)

    output = predict_module.predict_from_csv(tmp_path, input_path, threshold=0.2)

    assert output["prediction"].tolist() == [0, 0]
    assert output["probability_churn"].tolist() == [0.1, 0.1]


def test_main_writes_output(monkeypatch, tmp_path: Path, capsys) -> None:
    output_path = tmp_path / "predictions.csv"
    output_df = pd.DataFrame({"prediction": [1], "probability_churn": [0.9]})

    monkeypatch.setattr(predict_module, "predict_from_csv", lambda *args, **kwargs: output_df)
    monkeypatch.setattr(
        predict_module,
        "_parse_args",
        lambda: argparse.Namespace(
            bundle_dir="bundle",
            input_path="input.csv",
            output_path=str(output_path),
            threshold=0.5,
        ),
    )

    predict_module.main()

    assert output_path.exists()

    summary = json.loads(capsys.readouterr().out)
    assert summary["rows_scored"] == 1


def test_parse_args_reads_cli(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "predict",
            "--bundle-dir",
            "bundle",
            "--input-path",
            "input.csv",
            "--output-path",
            "out.csv",
            "--threshold",
            "0.7",
        ],
    )

    args = predict_module._parse_args()

    assert args.bundle_dir == "bundle"
    assert args.input_path == "input.csv"
    assert args.output_path == "out.csv"
    assert args.threshold == 0.7


def test_module_runs_as_main(monkeypatch, tmp_path: Path, capsys) -> None:
    bundle_dir = _create_bundle(tmp_path / "bundle")
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    pd.DataFrame(
        {
            "Tenure Months": [3],
            "Monthly Charges": [50.0],
            "Total Charges": [150.0],
        }
    ).to_csv(input_path, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "predict",
            "--bundle-dir",
            str(bundle_dir),
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--threshold",
            "0.5",
        ],
    )

    sys.modules.pop("churn.models.predict", None)
    runpy.run_module("churn.models.predict", run_name="__main__")

    assert output_path.exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows_scored"] == 1
