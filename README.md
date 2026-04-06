# 9mlet-tech-challenge-1-churn-prevision

Tech Challenge Fase 1 - Pós Machine Learning Engineering FIAP

## Visão geral

Este repositório contém a etapa inicial do projeto de previsão de churn em telecom, com:

- Análise exploratória e preparação dos dados.
- Treinamento de baselines (DummyClassifier e Regressão Logística).
- Registro de experimentos no MLflow (parâmetros, métricas e versão do dataset).

## Estrutura principal

- data/
  - Telco_customer_churn.xlsx
  - Telco_customer_churn_ready.csv (gerado pelo notebook de EDA)
- notebooks/
  - 01_eda.ipynb
  - 02_baseline_dummy_logreg.ipynb
  - mlflow.db
  - mlruns/

## Pré-requisitos

Para executar localmente, tenha instalado:

1. Python 3.10 ou superior.
2. Git.
3. Ambiente virtual (venv).
4. Dependências Python abaixo:
	- pandas
	- numpy
	- scikit-learn
	- openpyxl
	- jupyter
	- mlflow
	- matplotlib
	- seaborn

## Setup local

### 1) Clonar o repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd 9mlet-tech-challenge-1-churn-prevision
```

### 2) Criar e ativar ambiente virtual

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
pip install pandas numpy scikit-learn openpyxl jupyter mlflow matplotlib seaborn
```

## Como executar os notebooks

### 1) Subir o Jupyter

Na raiz do projeto:

```bash
jupyter lab
```

ou

```bash
jupyter notebook
```

### 2) Executar o notebook de EDA

Arquivo:

- notebooks/01_eda.ipynb

Passos:

1. Abrir o notebook.
2. Executar as células em ordem (Run All).
3. Confirmar que o arquivo abaixo foi gerado:
	- data/Telco_customer_churn_ready.csv

Esse arquivo é a saída tratada/codificada usada pelo notebook de baseline.

### 3) Executar o notebook de baseline

Arquivo:

- notebooks/02_baseline_dummy_logreg.ipynb

Passos:

1. Abrir o notebook.
2. Executar as células em ordem.
3. Verificar:
	- tabela de métricas dos modelos
	- confirmação de registro dos experimentos no MLflow

Resultado esperado:

- Regressão Logística deve superar o DummyClassifier nas métricas principais (especialmente ROC-AUC e F1).

## Como executar o MLflow localmente

### 1) Iniciar a interface visual

Na raiz do projeto, execute:

```bash
mlflow ui --backend-store-uri sqlite:///notebooks/mlflow.db --default-artifact-root ./notebooks/mlruns --host 127.0.0.1 --port 5000
```

Abra no navegador:

- http://127.0.0.1:5000

### 2) Onde visualizar os resultados

Na UI do MLflow:

1. Selecione o experimento:
	- churn-baselines-etapa1
2. Vá para a aba Runs para ver as execuções.
3. Clique em uma run para inspecionar:
	- parâmetros
	- métricas
	- artefatos (modelo e dataset)

Importante:

- A aba Models pode aparecer vazia se não houver modelo registrado no registry.
- Para acompanhar os treinamentos desta etapa, use a aba Runs.

## Troubleshooting (problemas comuns)

### Problema: UI abre, mas não mostra execuções

Causa mais comum:

- O notebook e o comando da UI estão apontando para bancos diferentes.

Como resolver:

1. Execute o notebook de baseline novamente para registrar as runs no banco local correto.
2. Garanta que o comando da UI seja exatamente:

```bash
mlflow ui --backend-store-uri sqlite:///notebooks/mlflow.db --default-artifact-root ./notebooks/mlruns --host 127.0.0.1 --port 5000
```

3. Recarregue a página e selecione o experimento churn-baselines-etapa1.

### Problema: arquivo tratado não encontrado

Mensagem típica:

- Arquivo tratado do EDA não encontrado.

Solução:

1. Execute o notebook notebooks/01_eda.ipynb.
2. Confirme a criação de data/Telco_customer_churn_ready.csv.
3. Rode novamente notebooks/02_baseline_dummy_logreg.ipynb.

## Observações finais

- O projeto está alinhado com a stack do desafio (Python, sklearn, MLflow, notebooks).
- Para evolução das próximas etapas, mantenha rastreabilidade no MLflow e reprodutibilidade com seed fixa.
