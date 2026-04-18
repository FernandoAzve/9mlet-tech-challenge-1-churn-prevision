from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from churn.models.registry import load_pipeline_artifact


def _resolve_expected_feature_columns(pipeline: Any) -> list[str] | None:
    if hasattr(pipeline, "feature_names_in_"):
        return list(pipeline.feature_names_in_)

    preprocessor = getattr(pipeline, "named_steps", {}).get("preprocessor")
    if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
        return list(preprocessor.feature_names_in_)

    return None


def get_expected_feature_columns(pipeline: Any) -> list[str] | None:
    return _resolve_expected_feature_columns(pipeline)


def predict_with_pipeline(
    pipeline: Any,
    features: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    expected_columns = _resolve_expected_feature_columns(pipeline)
    features_aligned = features.copy()

    # Keeps API inputs simple by filling absent training columns with zero.
    if expected_columns is not None:
        features_aligned = features_aligned.reindex(columns=expected_columns, fill_value=0)

    probabilities = pipeline.predict_proba(features_aligned)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return pd.DataFrame(
        {
            "prediction": predictions,
            "probability_churn": probabilities,
        }
    )


def predict_from_csv(
    model_path: str | Path,
    input_path: str | Path,
    threshold: float = 0.5,
) -> pd.DataFrame:
    pipeline = load_pipeline_artifact(model_path)
    features = pd.read_csv(input_path)
    return predict_with_pipeline(pipeline=pipeline, features=features, threshold=threshold)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run churn predictions from a saved pipeline artifact.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="predictions.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_df = predict_from_csv(
        model_path=args.model_path,
        input_path=args.input_path,
        threshold=args.threshold,
    )
    output_df.to_csv(args.output_path, index=False)
    print(
        json.dumps(
            {
                "rows_scored": int(output_df.shape[0]),
                "output_path": str(Path(args.output_path).resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

