from __future__ import annotations

import argparse
import json
from pathlib import Path

from churn.config import DEFAULT_SEED, TARGET_COLUMN, TrainingConfig
from churn.data.io import load_features_target, stratified_train_test_split
from churn.models.evaluate import classification_metrics
from churn.models.registry import save_pipeline_artifact
from churn.pipelines.churn_pipeline import build_churn_pipeline


def train_pipeline(
    data_path: str | Path | None = None,
    estimator_name: str = "logistic_regression",
    output_dir: str | Path = "models/sklearn",
    test_size: float = 0.2,
    random_state: int = DEFAULT_SEED,
) -> dict:
    features, target = load_features_target(path=data_path, target_column=TARGET_COLUMN)
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        features=features,
        target=target,
        test_size=test_size,
        random_state=random_state,
    )

    pipeline = build_churn_pipeline(features=X_train, estimator_name=estimator_name)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_true=y_test, y_pred=y_pred, y_proba=y_proba)

    model_path, metadata_path = save_pipeline_artifact(
        pipeline=pipeline,
        output_dir=output_dir,
        metadata={
            "estimator_name": estimator_name,
            "target_column": TARGET_COLUMN,
            "test_size": test_size,
            "random_state": random_state,
            "n_train_rows": int(X_train.shape[0]),
            "n_test_rows": int(X_test.shape[0]),
            "n_features_input": int(X_train.shape[1]),
            "metrics": metrics,
        },
    )

    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "metrics": metrics,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train reproducible sklearn churn pipeline.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to ready churn CSV.")
    parser.add_argument(
        "--estimator-name",
        type=str,
        default=TrainingConfig().estimator_name,
        choices=["logistic_regression", "random_forest"],
        help="Estimator to use inside the pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(TrainingConfig().output_dir),
        help="Directory to save pipeline artifact and metadata.",
    )
    parser.add_argument("--test-size", type=float, default=TrainingConfig().test_size)
    parser.add_argument("--random-state", type=int, default=TrainingConfig().random_state)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = train_pipeline(
        data_path=args.data_path,
        estimator_name=args.estimator_name,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

