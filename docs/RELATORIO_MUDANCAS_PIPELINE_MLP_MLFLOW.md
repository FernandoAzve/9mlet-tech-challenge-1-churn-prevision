# Relatório — Pipeline único: pré-processador sklearn + MLP + MLflow

Texto em linguagem simples sobre **o fluxo oficial atual** do repositório após a unificação.

---

## 1. O que é o “pipeline” agora?

Existe **um só** caminho de produção:

1. **Dados** — CSV pronto do EDA (`Telco_customer_churn_ready.csv`, gerado no notebook `01_eda.ipynb`).
2. **Pré-processamento sklearn** — salvo em `preprocessor.joblib`:
   - limpeza da coluna `Total Charges` (`TotalChargesCleaner`);
   - alinhamento de colunas com zeros onde faltar (`FeatureColumnAligner`);
   - padronização `StandardScaler` ajustada **só no treino** (como no notebook `04_mlp_training_early_stopping.ipynb`).
3. **Modelo** — rede MLP em PyTorch, pesos em `mlp_state.pt`.
4. **Metadados** — `metadata.json` (colunas, dimensões, limiares, hash do CSV, etc.).
5. **MLflow** (opcional no CLI) — mesmo experimento do notebook 04: `MLP-Churn-EarlyStoppingBatching`.

O **classificador sklearn** antigo (`LogisticRegression` dentro de `churn_pipeline.joblib`) **foi removido** do código e dos artefatos de exemplo.

---

## 2. Arquivos principais em `src/churn/`

| Arquivo | Função |
|---------|--------|
| `src/churn/features/custom_transformers.py` | Transformadores custom (`TotalChargesCleaner`, `FeatureColumnAligner`). |
| `src/churn/features/preprocessing.py` | Monta o `Pipeline` sklearn oficial (`build_mlp_preprocessing_pipeline`). |
| `src/churn/models/train.py` | Treino CLI: split, fit do pré-processador, MLP, early stopping, MLflow, gravação do bundle. |
| `src/churn/models/mlp_torch.py` | Arquitetura `MLPChurn` e loop de época. |
| `src/churn/models/mlp_bundle.py` | `ChurnMLPBundle`: carrega/salva pré-processador + rede + metadados. |
| `src/churn/models/predict.py` | Predição em CSV a partir da pasta do bundle. |
| `src/churn/models/registry.py` | `load_churn_mlp_bundle` — ponto único de carregamento. |
| `src/churn/api/main.py` | Sobe a API carregando só o bundle (`CHURN_MODEL_BUNDLE_DIR`). |

**Removidos:** `pipelines/churn_pipeline.py`, treino sklearn antigo em `models/train.py` (substituído), `predict_mlp.py` / `train_mlp.py` (unificados em `train.py` e `predict.py`).

---

## 3. Como treinar (Windows, `venv` na raiz)

```powershell
$env:PYTHONPATH="src"
.\venv\Scripts\python.exe -m churn.models.train --bundle-dir models/mlp_bundle
```

- Sem MLflow: `--skip-mlflow`
- Outro CSV: `--data-path caminho/arquivo.csv`

---

## 4. Como pontuar CSV

```powershell
$env:PYTHONPATH="src"
.\venv\Scripts\python.exe -m churn.models.predict --bundle-dir models/mlp_bundle --input-path entrada.csv --output-path saida.csv
```

---

## 5. Como subir a API

```powershell
$env:PYTHONPATH="src"
$env:CHURN_MODEL_BUNDLE_DIR="models/mlp_bundle"
uvicorn churn.api.main:app --app-dir src --host 127.0.0.1 --port 8000 --reload
```

---

## 6. O que instalar

Ver `requirements.txt` (`torch`, `scikit-learn`, `pandas`, `mlflow`, `fastapi`, etc.).

---

## 7. Riscos e limitações

- **GPU vs CPU:** pequenas diferenças numéricas são normais.
- **Pickle/joblib:** carregue apenas bundles gerados por este projeto.
- **Notebooks:** continuam úteis para estudo; o **contrato de produção** é o descrito acima.

---

*Última atualização: unificação do pipeline (pré-processador sklearn serializado + MLP serializada).*
