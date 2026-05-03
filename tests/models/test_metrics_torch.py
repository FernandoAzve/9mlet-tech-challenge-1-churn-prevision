from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from churn.models.evaluate import classification_metrics
from churn.models.mlp_torch import MLPChurn, evaluate, train_one_epoch


def test_classification_metrics_without_proba() -> None:
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]

    metrics = classification_metrics(y_true, y_pred)

    assert "roc_auc" not in metrics
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}


def test_classification_metrics_with_proba() -> None:
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    y_proba = [0.1, 0.9, 0.3, 0.2]

    metrics = classification_metrics(y_true, y_pred, y_proba=y_proba)

    assert "roc_auc" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_mlp_forward_shape() -> None:
    model = MLPChurn(input_dim=3)
    x = torch.randn(2, 3)

    out = model(x)

    assert out.shape == (2, 1)


def test_train_one_epoch_and_evaluate() -> None:
    torch.manual_seed(0)

    model = MLPChurn(input_dim=3)
    x = torch.randn(4, 3)
    y = torch.tensor([[0.0], [1.0], [0.0], [1.0]])

    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_loss = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))

    assert train_loss >= 0.0

    eval_loss, y_true, y_proba = evaluate(model, loader, criterion, torch.device("cpu"))

    assert eval_loss >= 0.0
    assert y_true.shape == (4,)
    assert y_proba.shape == (4,)
