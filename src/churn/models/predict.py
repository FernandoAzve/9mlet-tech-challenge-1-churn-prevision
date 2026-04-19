# -*- coding: utf-8 -*-
"""
Inferência em lote ou na API usando apenas o **bundle MLP** oficial.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from churn.models.mlp_bundle import ChurnMLPBundle
from churn.models.registry import load_churn_mlp_bundle


def get_expected_feature_columns(model: Any) -> list[str] | None:
    """
    Devolve os nomes das colunas na ordem em que o modelo foi treinado.
    A API usa isso para saber quais “dummies” (ex.: cidade) existem no modelo.
    """
    if hasattr(model, "feature_names_in_"):
        fn = model.feature_names_in_
        if isinstance(fn, np.ndarray):
            return [str(x) for x in fn.tolist()]
        return list(fn)
    return None


def predict_with_churn_bundle(
    bundle: ChurnMLPBundle,
    features: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Recebe várias linhas de features, devolve probabilidade de churn e decisão 0/1
    comparando a probabilidade com o limiar (ex.: 0,5 = metade).
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    # P(churn) entre 0 e 1 por linha.
    probabilities = bundle.predict_proba_positive(features)
    # Acima do limiar = prevê churn (1); abaixo = não churn (0).
    predictions = (probabilities >= threshold).astype(int)

    return pd.DataFrame(
        {
            "prediction": predictions,
            "probability_churn": probabilities,
        }
    )


def predict_from_csv(
    bundle_dir: str | Path,
    input_path: str | Path,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Atalho: carrega o bundle da pasta, lê um CSV com as mesmas colunas de features
    e devolve um DataFrame com predição e probabilidade.
    """
    bundle = load_churn_mlp_bundle(bundle_dir)
    features = pd.read_csv(input_path)
    return predict_with_churn_bundle(bundle=bundle, features=features, threshold=threshold)


def _parse_args() -> argparse.Namespace:
    """Argumentos para rodar predição em lote pelo terminal."""
    parser = argparse.ArgumentParser(
        description="Predição em CSV usando o bundle MLP (pré-processador sklearn + rede).",
    )
    parser.add_argument(
        "--bundle-dir",
        type=str,
        required=True,
        help="Pasta com preprocessor.joblib, mlp_state.pt e metadata.json.",
    )
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="predictions.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    """CLI: gera arquivo CSV com colunas prediction e probability_churn e imprime um resumo JSON."""
    args = _parse_args()
    output_df = predict_from_csv(
        bundle_dir=args.bundle_dir,
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
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
