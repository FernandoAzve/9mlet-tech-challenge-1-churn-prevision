
"""
Treino reprodutível: **pré-processador sklearn serializado** + **MLP PyTorch** + MLflow + bundle.

Baseado no notebook ``04_mlp_training_early_stopping.ipynb`` e no CSV do EDA
(``01_eda.ipynb``). O bundle grava ``preprocessor.joblib`` (Pipeline sklearn),
``mlp_state.pt`` e ``metadata.json``.

Como executar (``venv`` na raiz do repo):

- Windows: ``$env:PYTHONPATH="src"; .\\venv\\Scripts\\python.exe -m churn.models.train``
- Linux/macOS: ``PYTHONPATH=src ./venv/bin/python -m churn.models.train``

Variável: ``MLFLOW_TRACKING_URI`` (padrão ``sqlite:///mlflow.db``).
"""

from __future__ import annotationsimport argparseimport hashlibimport jsonimport osfrom copy import deepcopyfrom pathlib import Pathimport numpy as npimport pandas as pdimport torchimport torch.nn as nnfrom sklearn.metrics import (    accuracy_score,    average_precision_score,    confusion_matrix,    f1_score,    precision_score,    recall_score,    roc_auc_score,)from sklearn.model_selection import train_test_splitfrom torch.utils.data import DataLoader, TensorDatasetfrom churn.config import DEFAULT_MLP_BUNDLE_DIR, DEFAULT_SEED, TARGET_COLUMNfrom churn.data.io import resolve_data_pathfrom churn.features.preprocessing import build_mlp_preprocessing_pipelinefrom churn.models.mlp_bundle import ChurnMLPBundlefrom churn.models.mlp_torch import MLPChurn, evaluate, train_one_epoch# ---------------------------------------------------------------------------
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


def _prepare_splits(
    df: pd.DataFrame,
    target_column: str,
    random_state: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    list[str],
]:
    """
    Separa X (features) e y (Churn Value) e faz dois cortes aleatórios estratificados:
    primeiro 80% treino+validação vs 20% teste final; depois do 80%, 75% treino vs 25% validação.
    Na prática: proporções próximas de ~64% treino, ~16% validação, ~20% teste.
    """
    if target_column not in df.columns:
        raise ValueError(f"Coluna alvo '{target_column}' ausente.")

    # Todas as colunas exceto o alvo = o que o modelo pode “enxergar”.
    x_all = df.drop(columns=[target_column])
    # 0 = ficou, 1 = churn (inteiro para classificação binária).
    y_all = df[target_column].astype(int)

    # Primeiro corte: reserva ~20% só para avaliação final (nunca usada no treino).
    x_temp, x_test, y_temp, y_test = train_test_split(
        x_all,
        y_all,
        test_size=0.2,
        random_state=random_state,
        stratify=y_all,  # mantém a proporção de churn similar em cada pedaço
    )
    # Segundo corte: do pedaço restante, uma parte valida durante o treino (early stopping).
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=0.25,
        random_state=random_state,
        stratify=y_temp,
    )
    # Ordem e nomes das colunas que o scaler e a rede vão usar.
    feature_columns = x_train.columns.tolist()
    return x_train, x_val, x_test, y_train, y_val, y_test, feature_columns


def train_mlp_flow(
    data_path: Path | None = None,
    random_state: int = DEFAULT_SEED,
    bundle_dir: Path = DEFAULT_BUNDLE_DIR,
    mlflow_run_name: str = "train_cli",
    skip_mlflow: bool = False,
) -> dict:
    """
    Pipeline completo: ler CSV → dividir dados → pré-processar → treinar MLP →
    medir no teste → salvar bundle em disco → opcionalmente registrar no MLflow.
    """
    # Mesma “semente” em NumPy e PyTorch para resultados repetíveis.
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # Descobre o caminho do CSV (argumento ou padrão em config).
    resolved = resolve_data_path(data_path)
    # GPU se existir; senão CPU (funciona em qualquer máquina).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(resolved)
    # Identifica exatamente qual arquivo foi treinado.
    dataset_version = _file_sha256(resolved)
    dataset_name = resolved.name

    x_train, x_val, x_test, y_train, y_val, y_test, feature_columns = _prepare_splits(
        df, TARGET_COLUMN, random_state
    )
    # Uma entrada da rede = um vetor com esse número de números.
    n_features = len(feature_columns)

    # Pipeline sklearn: limpa Total Charges, alinha colunas, escala (fit só no treino).
    preprocessor = build_mlp_preprocessing_pipeline()
    preprocessor.fit(x_train)
    # Após transform, tudo vira matriz numérica normalizada para a rede.
    x_train_s = preprocessor.transform(x_train).astype(np.float32)
    x_val_s = preprocessor.transform(x_val).astype(np.float32)
    x_test_s = preprocessor.transform(x_test).astype(np.float32)

    y_train_np = y_train.to_numpy(dtype=np.float32)
    y_val_np = y_val.to_numpy(dtype=np.float32)
    y_test_np = y_test.to_numpy(dtype=np.float32)

    # PyTorch espera tensores; TensorDataset emparelha X e y por linha.
    train_ds = TensorDataset(
        torch.from_numpy(x_train_s),
        torch.from_numpy(y_train_np).unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val_s),
        torch.from_numpy(y_val_np).unsqueeze(1),
    )
    test_ds = TensorDataset(
        torch.from_numpy(x_test_s),
        torch.from_numpy(y_test_np).unsqueeze(1),
    )
    # shuffle=True no treino: mistura linhas a cada época (ajuda a generalizar).
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Rede neural: entrada = n_features, duas camadas ocultas, saída 1 logit.
    model = MLPChurn(input_dim=n_features, hidden_dim_1=HIDDEN_DIM_1, hidden_dim_2=HIDDEN_DIM_2).to(device)

    # Classes desbalanceadas: damos mais “peso” ao erro quando o rótulo é churn (1).
    neg_count = float((y_train_np == 0).sum())
    pos_count = float((y_train_np == 1).sum())
    pos_weight_value = neg_count / max(pos_count, 1.0)
    pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    # BCEWithLogitsLoss = combina sigmoid + perda binária; recebe logits, não probabilidade.
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Guarda curvas de loss para inspeção; early stopping usa val_loss.
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for _epoch in range(NUM_EPOCHS_MAX):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Se a validação melhorou, guardamos uma cópia dos pesos “melhores até agora”.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = deepcopy(model.state_dict())
        else:
            patience_counter += 1

        # Parar cedo evita treinar à toa quando o modelo já não generaliza melhor.
        if patience_counter >= PATIENCE:
            break

    # Volta aos pesos do melhor ponto da validação (não ao último da última época).
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    n_epochs_trained = len(history["train_loss"])

    # Escolhe um limiar alternativo no conjunto de validação: maximiza F1 entre 0,05 e 0,95.
    val_loss, y_val_true, y_val_proba = evaluate(model, val_loader, criterion, device)
    threshold_grid = np.linspace(0.05, 0.95, 37)
    best_f1_val = -1.0
    threshold_otimo_f1 = THRESHOLD_PREDICTION
    for thr in threshold_grid:
        y_val_pred_thr = (y_val_proba >= thr).astype(int)
        f1_val_thr = f1_score(y_val_true, y_val_pred_thr, zero_division=0)
        if f1_val_thr > best_f1_val:
            best_f1_val = f1_val_thr
            threshold_otimo_f1 = float(thr)

    # Avaliação final no conjunto de teste (só para métricas honestas).
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

    # Matriz de confusão: tn, fp, fn, tp — útil para ver erros tipo I e II.
    cm = confusion_matrix(y_test_true, y_test_pred)
    tn, fp, fn, tp = cm.ravel()

    # Empacota pré-processador + rede + limiares para usar na API/CLI sem retreinar.
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
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "n_test": int(len(x_test)),
        "pos_weight": pos_weight_value,
        "epochs_trained": n_epochs_trained,
        "best_val_loss": float(best_val_loss),
    }
    bundle_path = bundle.save(bundle_dir, extra_metadata=extra_meta)

    mlflow_run_id: str | None = None
    if not skip_mlflow:
        import mlflow        import mlflow.pytorch

        # URI do servidor ou arquivo SQLite onde o MLflow guarda histórico.
        tracking = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(tracking)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run(run_name=mlflow_run_name) as run:
            mlflow_run_id = run.info.run_id
            # Parâmetros = o que você escolheu antes de treinar.
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
            mlflow.log_param("threshold_otimo_f1_validacao", threshold_otimo_f1)
            mlflow.log_param("hidden_dim_1", HIDDEN_DIM_1)
            mlflow.log_param("hidden_dim_2", HIDDEN_DIM_2)
            mlflow.log_param("loss_function", "BCEWithLogitsLoss")
            mlflow.log_param("optimizer", "Adam")
            mlflow.log_param("pos_weight", float(pos_weight_value))

            # Métricas = números medidos depois do treino (principalmente no teste).
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
            mlflow.log_metric("best_val_loss", float(best_val_loss))
            mlflow.log_metric("epochs_trained", float(n_epochs_trained))
            mlflow.log_metric("confusion_tn", float(tn))
            mlflow.log_metric("confusion_fp", float(fp))
            mlflow.log_metric("confusion_fn", float(fn))
            mlflow.log_metric("confusion_tp", float(tp))

            # Modelo PyTorch artifact dentro do run (o pré-processador continua só no bundle local).
            mlflow.pytorch.log_model(model, "model")
            mlflow.log_param("local_bundle_path", str(bundle_path.resolve()))

    return {
        "bundle_dir": str(bundle_path.resolve()),
        "mlflow_run_id": mlflow_run_id,
        "metrics": {
            "test_loss": float(test_loss),
            "test_accuracy": float(accuracy),
            "test_f1_score": float(f1),
            "test_roc_auc": float(roc_auc),
            "threshold_otimo_f1_validacao": threshold_otimo_f1,
        },
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
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada do script: imprime um JSON com pasta do bundle e métricas-chave."""
    args = _parse_args()
    data_path = Path(args.data_path) if args.data_path else None
    out = train_mlp_flow(
        data_path=data_path,
        random_state=args.random_state,
        bundle_dir=Path(args.bundle_dir),
        mlflow_run_name=args.mlflow_run_name,
        skip_mlflow=args.skip_mlflow,
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
