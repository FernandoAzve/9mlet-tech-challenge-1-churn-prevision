from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from churn.models.registry import load_pipeline_artifact


def predict_from_csv(
    model_path: str | Path,
    input_path: str | Path,
    threshold: float = 0.5,
) -> pd.DataFrame:
    pipeline = load_pipeline_artifact(model_path)
    features = pd.read_csv(input_path)

    probabilities = pipeline.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return pd.DataFrame(
        {
            "prediction": predictions,
            "probability_churn": probabilities,
        }
    )


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

