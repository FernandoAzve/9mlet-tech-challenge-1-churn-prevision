# 9mlet-tech-challenge-1-churn-prevision

Tech Challenge Fase 1 - Pós Machine Learning Engineering FIAP

## Visão geral

Este repositório implementa um fluxo completo de previsão de churn em telecom, com:

- EDA e preparação dos dados.
- Baselines clássicos (Dummy e Regressão Logística).
- MLP em PyTorch.
- Treinamento com early stopping.
- Comparação entre MLP e modelos lineares/árvores.
- Análise de trade-off de custo (FP vs FN).
- Rastreabilidade de experimentos no MLflow.

Status atual da Etapa 2:

- Itens 1 a 5 concluídos (incluindo consolidação de runs no MLflow).

## Estrutura do projeto

```text
.
├── data/
│   ├── Telco_customer_churn.xlsx
│   └── Telco_customer_churn_ready.csv
├── docs/
│   ├── METRICAS.md
│   ├── TODO.md
│   ├── MELHORIAS_CONTINUAS_ETAPA2.md
│   └── CHANGELOG.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_dummy_logreg.ipynb
│   ├── 03_mlp_pytorch.ipynb
│   ├── 04_mlp_training_early_stopping.ipynb
│   ├── 05_compare_mlp_baselines.ipynb
│   ├── 06_tradeoff_custo_fp_fn.ipynb
│   ├── mlflow.db
│   ├── mlruns/
│   └── mlflow_resumo_experimentos_etapa2.csv
├── requirements.txt
└── README.md
```

## Pré-requisitos

Instale localmente:

1. Python 3.11.x
2. Git
3. Ambiente virtual (venv)

Dependências do projeto estão em requirements.txt e incluem (versões fixadas):

- torch==2.6.0
- pandas==2.3.3
- numpy==1.26.4
- scikit-learn==1.7.2
- openpyxl==3.1.5
- jupyter==1.1.1
- mlflow==3.10.1
- matplotlib==3.10.8
- seaborn==0.13.2
- fastapi==0.115.12
- uvicorn==0.34.2
- pydantic==2.11.4
- httpx==0.28.1
- pytest==8.3.5
- ruff==0.11.8
- pandera==0.22.1

## Setup rápido

### 1) Clonar o repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd 9mlet-tech-challenge-1-churn-prevision
```

### 2) Criar e ativar o ambiente virtual

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows (Git Bash):

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Instalar dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Abrir os notebooks

```bash
jupyter lab
```

ou

```bash
jupyter notebook
```

## Ordem recomendada de execução

Para reproduzir o estado atual do projeto, execute os notebooks nesta ordem:

1. notebooks/01_eda.ipynb
2. notebooks/02_baseline_dummy_logreg.ipynb
3. notebooks/03_mlp_pytorch.ipynb
4. notebooks/04_mlp_training_early_stopping.ipynb
5. notebooks/05_compare_mlp_baselines.ipynb
6. notebooks/06_tradeoff_custo_fp_fn.ipynb

## API de inferência (Etapa 3)

Esta etapa disponibiliza o modelo sklearn treinado no projeto via FastAPI, com:

- `GET /health`
- `POST /predict`
- documentação Swagger automática em `/docs`

### 1) Treinar e salvar o artefato do pipeline

Execute na raiz do projeto, com o ambiente virtual ativo:

```bash
python -m churn.models.train --output-dir models/sklearn
```

Isso gera, no mínimo:

- `models/sklearn/churn_pipeline.joblib`
- `models/sklearn/metadata.json`

### 2) Iniciar a API

Com o ambiente virtual ativo:

```bash
uvicorn churn.api.main:app --app-dir src --host 127.0.0.1 --port 8000 --reload
```

Se quiser apontar para outro artefato de modelo, defina a variável de ambiente `CHURN_MODEL_PATH` antes de subir a API.

PowerShell:

```powershell
$env:CHURN_MODEL_PATH = "models/sklearn/churn_pipeline.joblib"
uvicorn churn.api.main:app --app-dir src --host 127.0.0.1 --port 8000 --reload
```

### 3) Abrir documentação interativa (Swagger)

- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

### 4) Testar endpoints com curl

Health check:

```bash
curl -X GET "http://127.0.0.1:8000/health"
```

Predição com payload de negócio (endpoint único `/predict`):

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
	-H "Content-Type: application/json" \
	-d '{
		"city": "Tipton",
		"tenure_months": 2,
		"monthly_charges": 49.25,
		"total_charges": 91.1,
		"internet_service": "DSL",
		"contract": "Month-to-month",
		"payment_method": "Electronic check",
		"threshold": 0.5
	}'
```

### 5) Rodar testes da API

```bash
pytest tests/api -v
```

Para rodar todos os testes do projeto:

```bash
pytest
```

## O que cada notebook entrega

### 01_eda.ipynb

- Carrega e analisa os dados brutos.
- Gera o dataset tratado em data/Telco_customer_churn_ready.csv.

### 02_baseline_dummy_logreg.ipynb

- Treina DummyClassifier e Regressão Logística.
- Registra experimentos base no MLflow.

### 03_mlp_pytorch.ipynb

- Implementa MLP em PyTorch (arquitetura + treino básico).

### 04_mlp_training_early_stopping.ipynb

- Treino da MLP com validação e early stopping.
- Logging de treino no MLflow.

### 05_compare_mlp_baselines.ipynb

- Compara MLP vs modelos lineares/árvores com múltiplas métricas.
- Carrega MLP do MLflow (do notebook 04).
- Consolida o item 5: registra todos os modelos em experimento único no MLflow.
- Gera o resumo de auditoria em notebooks/mlflow_resumo_experimentos_etapa2.csv.

### 06_tradeoff_custo_fp_fn.ipynb

- Avalia trade-off de custo com varredura de threshold.
- Reutiliza modelo registrado no MLflow.

## Como rodar o MLflow local

Com o ambiente virtual ativo, execute na raiz:

```bash
mlflow ui --backend-store-uri sqlite:///notebooks/mlflow.db --default-artifact-root ./notebooks/mlruns --host 127.0.0.1 --port 5000
```

Acesse:

- http://127.0.0.1:5000

Experimentos relevantes atualmente:

- churn-baselines-etapa1
- MLP-Churn-EarlyStoppingBatching
- Churn-Etapa2-Comparacao-Modelos

## Resultado consolidado já disponível

O arquivo notebooks/mlflow_resumo_experimentos_etapa2.csv contém o consolidado de auditoria do comparativo da Etapa 2:

- modelo
- família
- run_id
- model_uri
- pr_auc
- roc_auc
- f1
- recall
- precision

## Validação mínima (obrigatória)

Antes de considerar alterações como prontas:

```bash
ruff check
pytest
```

## Troubleshooting

### 1) MLflow UI abre, mas sem runs esperadas

Causa comum:

- comando da UI apontando para outro backend/artifact store.

Correção:

1. Reexecutar os notebooks que registram runs (principalmente 04 e 05).
2. Confirmar uso do comando do MLflow exatamente como neste README.
3. Atualizar a página e selecionar o experimento correto.

### 2) Erro de dataset não encontrado

Mensagem típica:

- arquivo Telco_customer_churn_ready.csv não encontrado.

Correção:

1. Executar notebooks/01_eda.ipynb.
2. Confirmar geração de data/Telco_customer_churn_ready.csv.
3. Executar novamente o notebook que falhou.

### 3) Problema de paralelismo no Windows (BrokenProcessPool)

Situação:

- Pode ocorrer em alguns cenários com n_jobs=-1 em modelos sklearn.

Status atual:

- O notebook 05 já está ajustado para n_jobs=1 nos modelos relevantes, visando estabilidade.

### 4) Avisos de serialização no MLflow

Situação:

- O MLflow pode emitir warnings sobre pickle/cloudpickle ao salvar modelos.

Status atual:

- São avisos informativos; o registro das runs e artefatos ocorre normalmente.

## Documentação de apoio

- docs/METRICAS.md
- docs/TODO.md
- docs/MELHORIAS_CONTINUAS_ETAPA2.md
- docs/CHANGELOG.md
- docs/API.md

## Observações para trabalho em grupo

- Para sincronização técnica, priorize o arquivo notebooks/mlflow_resumo_experimentos_etapa2.csv e os experimentos no MLflow.
- O item 5 (consolidação de experimentos MLP + ensembles) já está concluído no estado atual do projeto.
