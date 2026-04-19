# Documentação — Etapa 3 (Itens 1 e 2): pipeline MLP + `src/` + API

Este documento substitui a versão antiga baseada em **classificador sklearn** (`LogisticRegression` em `churn_pipeline.joblib`). O estado atual do repositório é **um único fluxo de produção**: **pré-processador sklearn serializado** + **MLP PyTorch serializada** + **MLflow** (opcional no CLI) + **API FastAPI** que carrega o **bundle** em disco.

---

## 1. Objetivo e escopo (Etapa 3 — itens 1 e 2)

| Item | O que o projeto entrega hoje |
|------|------------------------------|
| **1 — Refatoração em módulos** | Pacote `churn` sob `src/churn/` com separação por responsabilidade (`features`, `models`, `api`, `data`, `config`). |
| **2 — Pipeline reprodutível** | `sklearn.pipeline.Pipeline` com transformadores custom (`TotalChargesCleaner`, `FeatureColumnAligner`) + `StandardScaler`; modelo final **somente** a MLP (`MLPChurn`); persistência em pasta **bundle** (`preprocessor.joblib`, `mlp_state.pt`, `metadata.json`). |

**Fora do escopo deste documento (mas existem no repo):** notebooks de EDA/baselines (`01`, `02`), MLP didática (`03`), comparação (`05`), trade-off de custo (`06`) — continuam como **experimentação**; o **contrato de deploy** é o descrito aqui.

---

## 2. Visão geral do fluxo atual

```text
Telco_customer_churn_ready.csv (EDA notebook 01)
        │
        ▼
┌───────────────────────────────────────┐
│  churn.models.train (train_mlp_flow) │
│  • split 80/20 + 25/75 (≈64/16/20)   │
│  • fit preprocessor só em X_train      │
│  • treino MLP + early stopping        │
│  • MLflow (opcional)                  │
│  • salva bundle em models/mlp_bundle  │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Inferência                            │
│  • CLI: churn.models.predict           │
│  • API: FastAPI + bundle carregado      │
└───────────────────────────────────────┘
```

**Artefatos do bundle (pasta, ex.: `models/mlp_bundle/`):**

| Arquivo | Conteúdo |
|---------|----------|
| `preprocessor.joblib` | `Pipeline` sklearn: limpeza → alinhamento de colunas → `StandardScaler` |
| `mlp_state.pt` | `state_dict` da rede PyTorch |
| `metadata.json` | Colunas, `input_dim`, limiares, hash do CSV, métricas auxiliares, etc. |

---

## 3. Comparação com os notebooks 01, 04 e 06

### 3.1 Notebook `01_eda.ipynb` (dados prontos)

| Aspecto | Notebook 01 | Código em `src/` |
|--------|-------------|-------------------|
| Entrada bruta | Excel/CSV cru → limpeza e `get_dummies` | Não repetido em `train.py`; espera-se **`Telco_customer_churn_ready.csv`** já gerado pelo EDA. |
| Saída | `data/Telco_customer_churn_ready.csv` | `resolve_data_path()` / `DEFAULT_DATA_PATH_CANDIDATES` em `config.py` apontam para o mesmo arquivo. |
| **Alinhamento** | O treino **depende** do 01 para existir o CSV “ready”. | **Igual em intenção**; o `src` não substitui o EDA, consome o artefato. |

### 3.2 Notebook `04_mlp_training_early_stopping.ipynb` (MLP canônica)

| Aspecto | Notebook 04 | `src/churn/models/train.py` |
|--------|-------------|------------------------------|
| Split | `80/20` depois `75/25` do bloco maior, `stratify`, `random_state=42` | Mesma lógica em `_prepare_splits` (DataFrames em vez de só NumPy). |
| Escala | `StandardScaler` com **fit só no treino** | Equivalente: o passo `scale` do `Pipeline` é ajustado com `preprocessor.fit(x_train)`. |
| Arquitetura | MLP `input → 64 → 32 → 1` (logits) | `HIDDEN_DIM_1=64`, `HIDDEN_DIM_2=32` em `MLPChurn`. |
| Treino | `BCEWithLogitsLoss`, `pos_weight`, `Adam`, batch 64 | Mesmos hiperparâmetros no módulo de treino. |
| Early stopping | Paciência sobre `val_loss` | `PATIENCE = 10`, restaura melhor `state_dict`. |
| Threshold F1 | Grade em validação | `np.linspace(0.05, 0.95, 37)` + melhor F1 na validação. |
| MLflow | Experimento `MLP-Churn-EarlyStoppingBatching`, `log_model` PyTorch | Mesmo nome de experimento; `--skip-mlflow` para CI/ambientes sem DB. |
| **Consolidação extra no `src`** | Só `StandardScaler` solto no notebook | **`TotalChargesCleaner`** + **`FeatureColumnAligner`** + **`StandardScaler`** em um único `Pipeline` serializado; hash SHA-256 do CSV em metadados. |

### 3.3 Notebook `06_tradeoff_custo_fp_fn.ipynb` (negócio / threshold)

| Aspecto | Notebook 06 | `src/` atual |
|--------|-------------|--------------|
| Ideia | Ajustar decisão conforme custo FP/FN e reuso de modelo | O **payload** da API expõe `threshold` (0–1); o bundle guarda `threshold_prediction` e `threshold_otimo_f1_validacao` no JSON. |
| Carga do modelo | Via MLflow URI / caminho local | Produção: **bundle local** (`load_churn_mlp_bundle`); MLflow continua opcional no **treino**. |
| **Alinhamento** | Mesma **probabilidade** (`sigmoid(logits)`). | **Não** há segundo pipeline só para “custo”; a **consolidação** é: um threshold configurável na API/CLI + metadados do treino. |

---

## 4. Configuração (`src/churn/config.py`)

```python
DEFAULT_SEED = 42
TARGET_COLUMN = "Churn Value"

DEFAULT_DATA_PATH_CANDIDATES = (
    Path("data/Telco_customer_churn_ready.csv"),
    Path("../data/Telco_customer_churn_ready.csv"),
)

DEFAULT_MLP_BUNDLE_DIR = Path("models/mlp_bundle")
```

---

## 5. Pré-processamento sklearn (`src/churn/features/preprocessing.py`)

Montagem do **único** `Pipeline` usado antes da MLP:

```python
def build_mlp_preprocessing_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("clean_total_charges", TotalChargesCleaner()),
            ("align_features", FeatureColumnAligner()),
            ("scale", StandardScaler()),
        ]
    )
```

---

## 6. Transformadores custom (`src/churn/features/custom_transformers.py`)

Trechos essenciais:

```python
class TotalChargesCleaner(BaseEstimator, TransformerMixin):
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        if self.column_name in X_copy.columns:
            X_copy[self.column_name] = pd.to_numeric(
                X_copy[self.column_name],
                errors="coerce",
            ).fillna(0.0)
        return X_copy


class FeatureColumnAligner(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: Any = None) -> "FeatureColumnAligner":
        self.columns_ = X.columns.tolist()
        self.feature_names_in_ = np.array(self.columns_, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        aligned = X.reindex(columns=self.columns_, fill_value=0)
        return aligned.to_numpy(dtype=np.float64)
```

---

## 7. Treino e MLflow (`src/churn/models/train.py`)

Docstring do módulo (comandos e `MLFLOW_TRACKING_URI`):

```python
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
```

### 7.1 Constantes e splits

```python
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS_MAX = 50
PATIENCE = 10
THRESHOLD_PREDICTION = 0.5
MLFLOW_EXPERIMENT_NAME = "MLP-Churn-EarlyStoppingBatching"
```

```python
x_temp, x_test, y_temp, y_test = train_test_split(
    x_all, y_all, test_size=0.2, random_state=random_state, stratify=y_all,
)
x_train, x_val, y_train, y_val = train_test_split(
    x_temp, y_temp, test_size=0.25, random_state=random_state, stratify=y_temp,
)
```

### 7.2 Pré-processador + tensores

```python
preprocessor = build_mlp_preprocessing_pipeline()
preprocessor.fit(x_train)
x_train_s = preprocessor.transform(x_train).astype(np.float32)
x_val_s = preprocessor.transform(x_val).astype(np.float32)
x_test_s = preprocessor.transform(x_test).astype(np.float32)
```

### 7.3 Loss, otimizador e early stopping (resumo)

```python
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# ... loop: train_one_epoch / evaluate; restaurar best_model_state; patience ...
```

### 7.4 Bundle + MLflow

```python
bundle = ChurnMLPBundle(
    preprocessor=preprocessor,
    model=model.cpu(),
    feature_columns=feature_columns,
    threshold_prediction=THRESHOLD_PREDICTION,
    threshold_otimo_f1_validacao=threshold_otimo_f1,
    device=torch.device("cpu"),
)
bundle_path = bundle.save(bundle_dir, extra_metadata=extra_meta)
```

```python
tracking = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(tracking)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
# ... log_param / log_metric ...
mlflow.pytorch.log_model(model, "model")
```

### 7.5 CLI

```python
parser.add_argument("--data-path", type=str, default=None)
parser.add_argument("--bundle-dir", type=str, default=str(DEFAULT_BUNDLE_DIR))
parser.add_argument("--random-state", type=int, default=DEFAULT_SEED)
parser.add_argument("--mlflow-run-name", type=str, default="train_cli")
parser.add_argument("--skip-mlflow", action="store_true")
```

---

## 8. Bundle e inferência (`src/churn/models/mlp_bundle.py`)

Nomes dos arquivos no disco:

```python
_PREPROCESSOR_NAME = "preprocessor.joblib"
_WEIGHTS_NAME = "mlp_state.pt"
_METADATA_NAME = "metadata.json"
```

Carregamento (API e CLI usam `load_churn_mlp_bundle` em `registry.py`, que delega para `ChurnMLPBundle.load`):

```python
preprocessor = joblib.load(pre_path)
model = MLPChurn(input_dim=input_dim, hidden_dim_1=hidden_1, hidden_dim_2=hidden_2)
state = torch.load(state_path, map_location="cpu", weights_only=True)
model.load_state_dict(state)
```

Persistência após o treino:

```python
joblib.dump(self.preprocessor, root / _PREPROCESSOR_NAME)
torch.save(self.model.state_dict(), root / _WEIGHTS_NAME)
(root / _METADATA_NAME).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
```

Probabilidade por linha (entrada já alinhada às colunas do treino):

```python
def predict_proba_positive(self, features: pd.DataFrame) -> np.ndarray:
    aligned = features.reindex(columns=self.feature_columns, fill_value=0)
    x_s = self.preprocessor.transform(aligned)
    x_t = torch.from_numpy(x_s.astype(np.float32, copy=False)).to(self._device)
    self.model.eval()
    with torch.no_grad():
        logits = self.model(x_t)
        proba = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
    return np.asarray(proba, dtype=np.float64)
```

O carregamento na API/CLI passa por `src/churn/models/registry.py`: `load_churn_mlp_bundle(bundle_dir)` → `ChurnMLPBundle.load(bundle_dir)`.

---

## 9. Predição em lote (`src/churn/models/predict.py`)

```python
def predict_with_churn_bundle(
    bundle: ChurnMLPBundle,
    features: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    probabilities = bundle.predict_proba_positive(features)
    predictions = (probabilities >= threshold).astype(int)
    return pd.DataFrame({"prediction": predictions, "probability_churn": probabilities})
```

```python
parser.add_argument("--bundle-dir", type=str, required=True, ...)
parser.add_argument("--input-path", type=str, required=True)
parser.add_argument("--output-path", type=str, default="predictions.csv")
parser.add_argument("--threshold", type=float, default=0.5)
```

---

## 10. API FastAPI

### 10.1 Carregamento do bundle (`src/churn/api/main.py`)

```python
bundle_dir = Path(os.getenv("CHURN_MODEL_BUNDLE_DIR", str(DEFAULT_MLP_BUNDLE_DIR)))
meta = bundle_dir / "metadata.json"
if meta.is_file():
    app.state.model = load_churn_mlp_bundle(bundle_dir)
```

### 10.2 Rotas (`src/churn/api/routes.py`)

- `GET /health` — `model_loaded`, `model_path`.
- `POST /predict` — monta `DataFrame` de uma linha a partir do Pydantic, chama `predict_with_churn_bundle`.

```python
features = _build_v2_features(payload=payload, expected_columns=set(expected_columns))
features_df = pd.DataFrame([features])
prediction_df = predict_with_churn_bundle(
    bundle=model,
    features=features_df,
    threshold=payload.threshold,
)
```

Validação de entrada: `src/churn/api/schemas.py` (`PredictionV2Request`, campos obrigatórios `tenure_months`, `monthly_charges`, `total_charges`, etc.).

---

## 11. Como usar (passo a passo)

### 11.1 Variáveis de ambiente

| Variável | Função | Padrão |
|----------|--------|--------|
| `PYTHONPATH` | Deve incluir `src` para `import churn`. | — |
| `MLFLOW_TRACKING_URI` | Onde o MLflow grava runs (treino). | `sqlite:///mlflow.db` (no código, se não setada) |
| `CHURN_MODEL_BUNDLE_DIR` | Pasta do bundle para a API. | `models/mlp_bundle` |

### 11.2 Treinar (Windows PowerShell, `venv` na raiz)

```powershell
$env:PYTHONPATH="src"
.\venv\Scripts\python.exe -m churn.models.train --bundle-dir models/mlp_bundle
```

Sem MLflow:

```powershell
.\venv\Scripts\python.exe -m churn.models.train --skip-mlflow --bundle-dir models/mlp_bundle
```

Outro CSV:

```powershell
.\venv\Scripts\python.exe -m churn.models.train --data-path caminho\arquivo.csv
```

### 11.3 Predição em CSV

```powershell
$env:PYTHONPATH="src"
.\venv\Scripts\python.exe -m churn.models.predict `
  --bundle-dir models/mlp_bundle `
  --input-path entrada.csv `
  --output-path saida.csv `
  --threshold 0.5
```

### 11.4 Subir a API

```powershell
$env:PYTHONPATH="src"
$env:CHURN_MODEL_BUNDLE_DIR="models/mlp_bundle"
uvicorn churn.api.main:app --app-dir src --host 127.0.0.1 --port 8000 --reload
```

- Swagger: `http://127.0.0.1:8000/docs`
- Se não existir `metadata.json` no diretório do bundle, `/predict` retorna **503**.

---

## 12. Iterações futuras (documentação viva)

- Se o notebook **06** passar a fixar regras de custo no JSON do bundle, documente aqui os novos campos.
- Se o pré-processamento ganhar novos passos, atualize a seção 5–6 e o diagrama da seção 2.

---

## 13. Documentos relacionados

| Arquivo | Conteúdo |
|---------|----------|
| [RELATORIO_MUDANCAS_PIPELINE_MLP_MLFLOW.md](RELATORIO_MUDANCAS_PIPELINE_MLP_MLFLOW.md) | Relatório de mudanças em linguagem acessível. |
| [API.md](API.md) | Contrato resumido da API. |
| [METRICAS.md](METRICAS.md) | Métricas técnicas e de negócio (contexto do desafio). |

---

*Documento alinhado ao código em `src/churn/` na versão com bundle MLP + pré-processador sklearn serializado. Atualize este arquivo quando o contrato de artefatos ou os notebooks de referência mudarem.*
