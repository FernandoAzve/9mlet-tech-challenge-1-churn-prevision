
"""
Persistência do **único** artefato de produção: pasta do bundle MLP.

Não usamos mais ``churn_pipeline.joblib`` com classificador sklearn.
"""

from __future__ import annotationsfrom pathlib import Pathfrom churn.models.mlp_bundle import ChurnMLPBundledef load_churn_mlp_bundle(bundle_dir: str | Path) -> ChurnMLPBundle:
    """
    Função única para “abrir” o modelo em produção: lê JSON + joblib + pesos PyTorch.

    ``bundle_dir`` é a pasta (ex.: ``models/mlp_bundle``) que contém os três arquivos.
    """
    return ChurnMLPBundle.load(bundle_dir)
