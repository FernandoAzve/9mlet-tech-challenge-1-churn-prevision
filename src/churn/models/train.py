
"""
Treino reprodutível: **pré-processador sklearn serializado** + **MLP PyTorch** + MLflow + bundle.

Baseado no notebook ``04_mlp_training_early_stopping.ipynb`` e no CSV do EDA
(``01_eda.ipynb``). O bundle grava ``preprocessor.joblib`` (Pipeline sklearn),
``mlp_state.pt`` e ``metadata.json``.

O conjunto de desenvolvimento (80%) é avaliado com **StratifiedKFold**; o modelo
final retreina em todo esse bloco com número de épocas = mediana das épocas dos
folds. O holdout de 20% permanece só para métricas finais.

Como executar (``venv`` na raiz do repo):

- Windows: ``$env:PYTHONPATH="src"; .\\venv\\Scripts\\python.exe -m churn.models.train``
- Linux/macOS: ``PYTHONPATH=src ./venv/bin/python -m churn.models.train``

Variável: ``MLFLOW_TRACKING_URI`` (padrão ``sqlite:///mlflow.db``).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from churn.config import DEFAULT_MLP_BUNDLE_DIR, DEFAULT_SEED, TARGET_COLUMN
from churn.data.io import resolve_data_path
from churn.data.pandera_schemas import validate_churn_training_dataset
from churn.features.preprocessing import build_mlp_preprocessing_pipeline
from churn.logging_config import configure_logging
from churn.models.mlp_bundle import ChurnMLPBundle
from churn.models.mlp_torch import MLPChurn, evaluate, train_one_epoch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valores fixos do experimento (fáceis de ajustar num único lugar)
# ---------------------------------------------------------------------------
# Quantas linhas de dados a rede vê por vez antes de atualizar os pesos.
BATCH_SIZE = 64
# Passo do treino: quanto maior, mais “agressivo” (pode oscilar).
LEARNING_RATE = 0.001
# Limite superior de voltas completas nos dados (épocas).
NUM_EPOCHS_MAX = 50
# Early stopping: parar se a validação não melhorar por PATIENCE épocas seguidas.
PATIENCE = 10
# Limiar padrão (50%) para transformar probabilidade em “sim/não” nos relatórios.
THRESHOLD_PREDICTION = 0.5
# Neurônios nas camadas ocultas da MLP (mesma ideia do notebook 04).
HIDDEN_DIM_1 = 64
HIDDEN_DIM_2 = 32
# Nome do “caderno” do MLflow onde aparecem todos os runs deste treino.
MLFLOW_EXPERIMENT_NAME = "MLP-Churn-EarlyStoppingBatching"
# Pasta padrão onde salvamos preprocessor + pesos + JSON de metadados.
DEFAULT_BUNDLE_DIR = DEFAULT_MLP_BUNDLE_DIR
# Dobras estratificadas no conjunto treino+validação (80% do total).
CV_N_SPLITS_DEFAULT = 5


def _file_sha256(path: Path) -> str:
    """
    Calcula uma “impressão digital” do arquivo CSV para sabermos qual versão
    dos dados foi usada no treino (reprodutibilidade).
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        # Lê em pedaços para não estourar memória em CSV grandes.
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_holdout_test(
    df: pd.DataFrame,
    target_column: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """
    Reserva 20% estratificado para teste final; o restante (80%) vai para
    validação cruzada e retreino final.
    """
    if target_column not in df.columns:
        raise ValueError(f"Coluna alvo '{target_column}' ausente.")

    x_all = df.drop(columns=[target_column])
    y_all = df[target_column].astype(int)

    x_temp, x_test, y_temp, y_test = train_test_split(
        x_all,
        y_all,
        test_size=0.2,
        random_state=random_state,
        stratify=y_all,
    )
    feature_columns = x_temp.columns.tolist()
    return x_temp, x_test, y_temp, y_test, feature_columns


def _effective_cv_splits(y: pd.Series, requested: int) -> int:
    """Garante que cada classe tenha amostras suficientes para o StratifiedKFold."""
    min_class = int(y.value_counts().min())
    return max(2, min(requested, min_class))


def _best_threshold_f1(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    threshold_grid = np.linspace(0.05, 0.95, 37)
    best_f1_val = -1.0
    threshold_otimo = THRESHOLD_PREDICTION
    for thr in threshold_grid:
        y_pred_thr = (y_proba >= thr).astype(int)
        f1_val_thr = f1_score(y_true, y_pred_thr, zero_division=0)
        if f1_val_thr > best_f1_val:
            best_f1_val = f1_val_thr
            threshold_otimo = float(thr)
    return threshold_otimo


def _train_mlp_early_stopping(
    x_train_s: np.ndarray,
    y_train_np: np.ndarray,
    x_val_s: np.ndarray,
    y_val_np: np.ndarray,
    n_features: int,
    device: torch.device,
    random_seed: int,
) -> dict[str, Any]:
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    train_ds = TensorDataset(
        torch.from_numpy(x_train_s),
        torch.from_numpy(y_train_np).unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val_s),
        torch.from_numpy(y_val_np).unsqueeze(1),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = MLPChurn(
        input_dim=n_features,
        hidden_dim_1=HIDDEN_DIM_1,
        hidden_dim_2=HIDDEN_DIM_2,
    ).to(device)

    neg_count = float((y_train_np == 0).sum())
    pos_count = float((y_train_np == 1).sum())
    pos_weight_value = neg_count / max(pos_count, 1.0)
    pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for _epoch in range(NUM_EPOCHS_MAX):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    n_epochs = len(history["train_loss"])
    _val_loss, y_val_true, y_val_proba = evaluate(model, val_loader, criterion, device)
    threshold_f1_val = _best_threshold_f1(y_val_true, y_val_proba)

    roc_auc_val: float | None = None
    if len(np.unique(y_val_true)) >= 2:
        roc_auc_val = float(roc_auc_score(y_val_true, y_val_proba))

    return {
        "n_epochs": n_epochs,
        "best_val_loss": float(best_val_loss),
        "threshold_f1_val": threshold_f1_val,
        "roc_auc_val": roc_auc_val,
        "history": history,
    }


def _train_mlp_fixed_epochs(
    x_train_s: np.ndarray,
    y_train_np: np.ndarray,
    n_features: int,
    device: torch.device,
    random_seed: int,
    n_epochs: int,
) -> nn.Module:
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    train_ds = TensorDataset(
        torch.from_numpy(x_train_s),
        torch.from_numpy(y_train_np).unsqueeze(1),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = MLPChurn(
        input_dim=n_features,
        hidden_dim_1=HIDDEN_DIM_1,
        hidden_dim_2=HIDDEN_DIM_2,
    ).to(device)

    neg_count = float((y_train_np == 0).sum())
    pos_count = float((y_train_np == 1).sum())
    pos_weight_value = neg_count / max(pos_count, 1.0)
    pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for _ in range(n_epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)

    return model


def train_mlp_flow(
    data_path: Path | None = None,
    random_state: int = DEFAULT_SEED,
    bundle_dir: Path = DEFAULT_BUNDLE_DIR,
    mlflow_run_name: str = "train_cli",
    skip_mlflow: bool = False,
    n_cv_splits: int = CV_N_SPLITS_DEFAULT,
) -> dict[str, Any]:
    """
    Pipeline completo: ler CSV → validar → dividir dados → CV estratificada →
    retreinar em 80% → medir no teste → salvar bundle → MLflow opcional.
    """
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    resolved = resolve_data_path(data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(resolved)
    df = validate_churn_training_dataset(df)

    dataset_version = _file_sha256(resolved)
    dataset_name = resolved.name

    x_temp, x_test, y_temp, y_test, feature_columns = _split_holdout_test(
        df,
        TARGET_COLUMN,
        random_state,
    )
    n_features = len(feature_columns)

    n_splits = _effective_cv_splits(y_temp, n_cv_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_val_losses: list[float] = []
    fold_epochs: list[int] = []
    fold_thresholds: list[float] = []
    fold_roc_aucs: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x_temp, y_temp)):
        x_tr = x_temp.iloc[train_idx]
        x_va = x_temp.iloc[val_idx]
        y_tr = y_temp.iloc[train_idx]
        y_va = y_temp.iloc[val_idx]

        preproc_fold = build_mlp_preprocessing_pipeline()
        preproc_fold.fit(x_tr)
        x_tr_s = preproc_fold.transform(x_tr).astype(np.float32)
        x_va_s = preproc_fold.transform(x_va).astype(np.float32)
        y_tr_np = y_tr.to_numpy(dtype=np.float32)
        y_va_np = y_va.to_numpy(dtype=np.float32)

        fold_out = _train_mlp_early_stopping(
            x_tr_s,
            y_tr_np,
            x_va_s,
            y_va_np,
            n_features=n_features,
            device=device,
            random_seed=random_state + fold_idx * 31,
        )
        fold_val_losses.append(fold_out["best_val_loss"])
        fold_epochs.append(fold_out["n_epochs"])
        fold_thresholds.append(fold_out["threshold_f1_val"])
        if fold_out["roc_auc_val"] is not None:
            fold_roc_aucs.append(fold_out["roc_auc_val"])

    median_epochs = max(1, int(round(float(np.median(fold_epochs)))))
    threshold_otimo_f1 = float(np.median(fold_thresholds))

    cv_mean_val_loss = float(np.mean(fold_val_losses))
    cv_std_val_loss = float(np.std(fold_val_losses))
    cv_mean_epochs = float(np.mean(fold_epochs))
    cv_mean_roc_auc: float | None
    if fold_roc_aucs:
        cv_mean_roc_auc = float(np.mean(fold_roc_aucs))
    else:
        cv_mean_roc_auc = None

    preprocessor = build_mlp_preprocessing_pipeline()
    preprocessor.fit(x_temp)
    x_temp_s = preprocessor.transform(x_temp).astype(np.float32)
    y_temp_np = y_temp.to_numpy(dtype=np.float32)

    model = _train_mlp_fixed_epochs(
        x_temp_s,
        y_temp_np,
        n_features=n_features,
        device=device,
        random_seed=random_state + 999,
        n_epochs=median_epochs,
    )

    x_test_s = preprocessor.transform(x_test).astype(np.float32)
    y_test_np = y_test.to_numpy(dtype=np.float32)

    test_ds = TensorDataset(
        torch.from_numpy(x_test_s),
        torch.from_numpy(y_test_np).unsqueeze(1),
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    neg_count = float((y_temp_np == 0).sum())
    pos_count = float((y_temp_np == 1).sum())
    pos_weight_value = neg_count / max(pos_count, 1.0)
    pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    test_loss, y_test_true, y_test_proba = evaluate(model, test_loader, criterion, device)
    y_test_pred = (y_test_proba >= THRESHOLD_PREDICTION).astype(int)
    y_test_pred_otimo = (y_test_proba >= threshold_otimo_f1).astype(int)

    accuracy = accuracy_score(y_test_true, y_test_pred)
    precision = precision_score(y_test_true, y_test_pred, zero_division=0)
    recall = recall_score(y_test_true, y_test_pred, zero_division=0)
    f1 = f1_score(y_test_true, y_test_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test_true, y_test_proba)
    pr_auc = average_precision_score(y_test_true, y_test_proba)

    precision_otimo = precision_score(y_test_true, y_test_pred_otimo, zero_division=0)
    recall_otimo = recall_score(y_test_true, y_test_pred_otimo, zero_division=0)
    f1_otimo = f1_score(y_test_true, y_test_pred_otimo, zero_division=0)

    cm = confusion_matrix(y_test_true, y_test_pred)
    tn, fp, fn, tp = cm.ravel()

    bundle = ChurnMLPBundle(
        preprocessor=preprocessor,
        model=model.cpu(),
        feature_columns=feature_columns,
        threshold_prediction=THRESHOLD_PREDICTION,
        threshold_otimo_f1_validacao=threshold_otimo_f1,
        device=torch.device("cpu"),
    )

    extra_meta = {
        "dataset_name": dataset_name,
        "dataset_version_sha256": dataset_version,
        "random_state": random_state,
        "n_train_val_total": int(len(x_temp)),
        "n_test": int(len(x_test)),
        "cv_n_splits": n_splits,
        "pos_weight": pos_weight_value,
        "epochs_trained_final": median_epochs,
        "cv_mean_best_val_loss": cv_mean_val_loss,
        "cv_std_best_val_loss": cv_std_val_loss,
        "cv_mean_epochs": cv_mean_epochs,
        "cv_mean_val_roc_auc": cv_mean_roc_auc,
        "best_val_loss": cv_mean_val_loss,
    }
    bundle_path = bundle.save(bundle_dir, extra_metadata=extra_meta)

    mlflow_run_id: str | None = None
    if not skip_mlflow:
        import mlflow
        import mlflow.pytorch

        tracking = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(tracking)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run(run_name=mlflow_run_name) as run:
            mlflow_run_id = run.info.run_id
            mlflow.log_param("seed", random_state)
            mlflow.log_param("dataset_name", dataset_name)
            mlflow.log_param("dataset_version_sha256", dataset_version)
            mlflow.log_param("n_amostras_total", int(df.shape[0]))
            mlflow.log_param("n_features", int(n_features))
            mlflow.log_param("batch_size", BATCH_SIZE)
            mlflow.log_param("learning_rate", LEARNING_RATE)
            mlflow.log_param("num_epochs_max", NUM_EPOCHS_MAX)
            mlflow.log_param("patience_early_stopping", PATIENCE)
            mlflow.log_param("threshold_prediction", THRESHOLD_PREDICTION)
            mlflow.log_param("threshold_otimo_f1_cv_median", threshold_otimo_f1)
            mlflow.log_param("hidden_dim_1", HIDDEN_DIM_1)
            mlflow.log_param("hidden_dim_2", HIDDEN_DIM_2)
            mlflow.log_param("loss_function", "BCEWithLogitsLoss")
            mlflow.log_param("optimizer", "Adam")
            mlflow.log_param("pos_weight", float(pos_weight_value))
            mlflow.log_param("cv_n_splits", int(n_splits))
            mlflow.log_param("epochs_trained_final_median_cv", float(median_epochs))

            mlflow.log_metric("test_loss", float(test_loss))
            mlflow.log_metric("test_accuracy", float(accuracy))
            mlflow.log_metric("test_precision", float(precision))
            mlflow.log_metric("test_recall", float(recall))
            mlflow.log_metric("test_f1_score", float(f1))
            mlflow.log_metric("test_precision_thr_otimo", float(precision_otimo))
            mlflow.log_metric("test_recall_thr_otimo", float(recall_otimo))
            mlflow.log_metric("test_f1_thr_otimo", float(f1_otimo))
            mlflow.log_metric("test_roc_auc", float(roc_auc))
            mlflow.log_metric("test_pr_auc", float(pr_auc))
            mlflow.log_metric("cv_mean_best_val_loss", cv_mean_val_loss)
            mlflow.log_metric("cv_std_best_val_loss", cv_std_val_loss)
            mlflow.log_metric("cv_mean_epochs", cv_mean_epochs)
            mlflow.log_metric("epochs_trained_final", float(median_epochs))
            if cv_mean_roc_auc is not None:
                mlflow.log_metric("cv_mean_val_roc_auc", float(cv_mean_roc_auc))
            mlflow.log_metric("confusion_tn", float(tn))
            mlflow.log_metric("confusion_fp", float(fp))
            mlflow.log_metric("confusion_fn", float(fn))
            mlflow.log_metric("confusion_tp", float(tp))

            mlflow.pytorch.log_model(model, "model")
            mlflow.log_param("local_bundle_path", str(bundle_path.resolve()))

    metrics_out = {
        "test_loss": float(test_loss),
        "test_accuracy": float(accuracy),
        "test_f1_score": float(f1),
        "test_roc_auc": float(roc_auc),
        "threshold_otimo_f1_validacao": threshold_otimo_f1,
        "cv_n_splits": n_splits,
        "cv_mean_best_val_loss": cv_mean_val_loss,
        "cv_std_best_val_loss": cv_std_val_loss,
        "cv_mean_epochs": cv_mean_epochs,
        "epochs_trained_final": median_epochs,
    }
    if cv_mean_roc_auc is not None:
        metrics_out["cv_mean_val_roc_auc"] = cv_mean_roc_auc

    return {
        "bundle_dir": str(bundle_path.resolve()),
        "mlflow_run_id": mlflow_run_id,
        "metrics": metrics_out,
    }


def _parse_args() -> argparse.Namespace:
    """Lê argumentos da linha de comando quando você roda ``python -m churn.models.train``."""
    parser = argparse.ArgumentParser(
        description="Treina MLP + pré-processador sklearn e salva bundle + MLflow.",
    )
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--bundle-dir", type=str, default=str(DEFAULT_BUNDLE_DIR))
    parser.add_argument("--random-state", type=int, default=DEFAULT_SEED)
    parser.add_argument("--mlflow-run-name", type=str, default="train_cli")
    parser.add_argument("--skip-mlflow", action="store_true")
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=CV_N_SPLITS_DEFAULT,
        help="Número de dobras estratificadas no conjunto de treino (80%%).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Nível de log (DEBUG, INFO, WARNING, ...).",
    )
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada do script: emite JSON estruturado via logging."""
    args = _parse_args()
    configure_logging(level=args.log_level)
    data_path = Path(args.data_path) if args.data_path else None
    out = train_mlp_flow(
        data_path=data_path,
        random_state=args.random_state,
        bundle_dir=Path(args.bundle_dir),
        mlflow_run_name=args.mlflow_run_name,
        skip_mlflow=args.skip_mlflow,
        n_cv_splits=args.cv_folds,
    )
    logger.info(
        "train_cli_completed",
        extra={
            "event": "train_cli_completed",
            "bundle_dir": out["bundle_dir"],
            "mlflow_run_id": out["mlflow_run_id"],
            "metrics": out["metrics"],
        },
    )


if __name__ == "__main__":
    main()
