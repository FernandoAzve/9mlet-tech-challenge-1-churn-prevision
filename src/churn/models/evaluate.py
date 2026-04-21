from __future__ import annotationsfrom typing import Anyfrom sklearn.metrics import (    accuracy_score,    f1_score,    precision_score,    recall_score,    roc_auc_score,)def classification_metrics(y_true, y_pred, y_proba=None) -> dict[str, Any]:
    """
    Métricas clássicas de classificação binária em linguagem simples:

    - **accuracy**: acertos em geral;
    - **precision**: entre os que previmos como positivos, quantos eram realmente positivos;
    - **recall**: entre os positivos reais, quantos pegamos;
    - **f1**: equilíbrio entre precision e recall;
    - **roc_auc** (opcional): quão bem a probabilidade ordena positivos antes dos negativos.
    """
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return metrics
