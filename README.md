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

## Estrutura do projeto

```text
.
├── data/
│   ├── Telco_customer_churn.xlsx
│   ├── Telco_customer_churn.csv
│   └── Telco_customer_churn_ready.csv
├── docs/
│   ├── API.md
│   ├── API_EXEMPLOS_TESTE.md
│   ├── ARQUITETURA_DEPLOY.md
│   ├── ML_Canvas_TC1_FIAP_v2.png
│   ├── METRICAS.md
│   ├── MODEL_CARD_FINAL.md
│   ├── OBSERVABILIDADE.md
│   ├── PLANO_MONITORAMENTO.md
│   └── payloads/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_dummy_logreg.ipynb
│   ├── 03_mlp_pytorch.ipynb
│   ├── 04_mlp_training_early_stopping.ipynb
│   ├── 05_compare_mlp_baselines.ipynb
│   └── 06_tradeoff_custo_fp_fn.ipynb
├── models/
│   └── mlp_bundle/
├── src/
│   └── churn/
│       ├── api/
│       ├── models/
│       └── ...
├── tests/
│   └── api/
├── requirements.txt
└── README.md
```

Observação: o diagrama acima resume as pastas principais. A estrutura completa inclui arquivos auxiliares de documentação e automação.

## ML Canvas

Veja o canvas de negócio e métricas em [docs/ML_Canvas_TC1_FIAP_v3.png](docs/ML_Canvas_TC1_FIAP_v3.png).

## Contexto para agentes de IA (`.github/`)

A pasta **`.github/`** destina-se a **orientar assistentes de código e fluxos automatizados** (GitHub Copilot, Cursor e integrações semelhantes): reúne **contexto fixo do projeto**, **regras**, **políticas de dependências** e **prompts** reutilizáveis, para que intervenções da IA permaneçam alinhadas ao Tech Challenge, à stack acordada e às práticas do repositório. Não substitui a documentação de negócio em `docs/`; complementa o que o agente deve “ler primeiro” antes de sugerir mudanças.

Estrutura interna:

- **`copilot-instructions.md`** (raiz de `.github/`): instruções gerais para o Copilot no VS Code — ordem de leitura (contexto → regras → libs), stack oficial, regras obrigatórias (validação com `ruff` e `pytest`, simplicidade) e referências cruzadas às subpastas abaixo.

- **`context/`** — contexto de produto e engenharia, em Markdown, para alinhar escopo antes de implementar:
  - **`project-goals.md`**: objetivos de negócio, enunciado do problema, restrições e stakeholders.
  - **`architecture.md`**: visão de arquitetura da solução no repositório.
  - **`tech-stack.md`**: tecnologias e ferramentas adotadas.
  - **`deployment.md`**: aspectos relevantes de deploy e execução em ambiente real ou homologação.

- **`rules/`** — normas que os agentes devem seguir ao propor código ou processos:
  - **`ai-usage-rules.md`**: uso responsável da IA (limites, revisão humana, o que automatizar ou não).
  - **`business-rules.md`**: regras de domínio e decisões de negócio que afetam o modelo ou a API.
  - **`code-style.md`**: convenções de estilo e organização de código.
  - **`git-rules.md`**: branches, commits e fluxo de colaboração em Git.
  - **`security-rules.md`**: práticas de segurança (dados, segredos, superfície da API).
  - **`testing-rules.md`**: expectativas sobre testes, cobertura mínima de comportamento e qualidade.

- **`libs/`** — política de dependências e modelos:
  - **`allowed-libs.md`**: bibliotecas permitidas, com justificativa e uso típico.
  - **`forbidden-libs.md`**: dependências ou categorias a evitar e motivos.
  - **`ai-models.md`**: diretrizes sobre modelos de IA / LLMs quando aplicável ao fluxo do projeto.

- **`prompts/`** — prompts em formato `.prompt.md` (metadados YAML + corpo) para **comandos ou fluxos nomeados** (revisão, refactor, documentação, etc.):
  - **`architecture-review.prompt.md`**: revisão de arquitetura.
  - **`challenge-solution.prompt.md`**: encaminhamento da solução do desafio.
  - **`extract-business-rules.prompt.md`**: extração ou consolidação de regras de negócio a partir do material existente.
  - **`generate-boilerplate.prompt.md`**: geração de boilerplate alinhado ao projeto.
  - **`generate-docs.prompt.md`**: produção ou atualização de documentação.
  - **`kickoff-project.prompt.md`**: alinhamento inicial (negócio, requisitos, riscos) antes de codificar.
  - **`pre-deploy-validation.prompt.md`**: checagens antes de deploy.
  - **`refactor-controlled.prompt.md`**: refatoração com escopo e risco controlados.
  - **`review-pr.prompt.md`**: revisão de pull request.
  - **`test-strategy.prompt.md`**: definição ou revisão de estratégia de testes.

Em conjunto, esses arquivos definem **o “manual do agente”** para este repositório: prioridade de leitura, limites e comandos padronizados, sem acoplar essa lógica ao código em `src/`.

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
git clone https://github.com/FernandoAzve/9mlet-tech-challenge-1-churn-prevision.git
cd 9mlet-tech-challenge-1-churn-prevision
```

### 2) Criar e ativar o ambiente virtual

Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Windows (Git Bash):

```bash
python -m venv venv
source venv/Scripts/activate
```

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
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

A API FastAPI carrega **o bundle MLP produtizado** (pré-processador sklearn serializado + rede PyTorch):

- `GET /health`
- `POST /predict`
- documentação Swagger em `/docs`

Comportamento importante:

- Se o bundle não estiver carregado na inicialização, `GET /health` responde com `model_loaded=false`.
- Nesse cenário, `POST /predict` retorna `503` com mensagem orientando treino/configuração do diretório de bundle.

### Acesso direto no Render (sem rodar localmente)

A API está publicada no Render. Acesse a documentação interativa (Swagger):

- https://ninemlet-tech-challenge-1-churn-prevision.onrender.com/docs

Observação: como estamos no plano gratuito, a primeira inicialização pode demorar até 5 minutos. Após esse período, a API funciona com performance excelente.

### 1) Treinar e gerar o bundle

Na raiz do projeto, com `PYTHONPATH=src` e o `venv` ativo:

```powershell
$env:PYTHONPATH="src"
.\venv\Scripts\python.exe -m churn.models.train --bundle-dir models/mlp_bundle
```

Opcional: `--skip-mlflow` para não registrar no MLflow.

Saída em `models/mlp_bundle/`:

- `preprocessor.joblib`
- `mlp_state.pt`
- `metadata.json`

### 2) Iniciar a API

```bash
uvicorn churn.api.main:app --app-dir src --host 127.0.0.1 --port 8000 --reload
```

Outra pasta de bundle (opcional):

```powershell
$env:CHURN_MODEL_BUNDLE_DIR = "models/mlp_bundle"
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

## Comandos rápidos para qualidade e automação

### Lint (verificação de estilo)

```bash
ruff check .
```

### Rodar todos os testes

```bash
pytest
```

### Rodar a API localmente

```bash
uvicorn src.churn.api.main:app --reload
```

> **Nota:** No Windows, use os comandos acima diretamente. O Makefile é opcional e funciona nativamente apenas em Linux/macOS.

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

### 06_tradeoff_custo_fp_fn.ipynb

- Avalia trade-off de custo com varredura de threshold.
- Reutiliza modelo registrado no MLflow.

## Decisão do modelo no pipeline (com base nos notebooks)

O modelo adotado no pipeline de inferência da API é o **MLP em PyTorch (bundle `models/mlp_bundle`)**.

Essa escolha é reflexo direto das comparações dos notebooks:

- No `05_compare_mlp_baselines.ipynb`, o **RandomForest** apareceu como mais consistente no ranking global de métricas.
- No mesmo comparativo, a **MLP** se destacou em **F1** e **recall** após calibração de threshold.
- No `06_tradeoff_custo_fp_fn.ipynb`, a decisão por limiar mostrou o impacto econômico de **FN vs FP**, reforçando a importância de ajustar o ponto de operação.

Racional da escolha no pipeline atual:

- **Objetivo de negócio do projeto:** reduzir perda por churn real (sensível a recall/F1 e calibração de threshold).
- **Objetivo acadêmico e de engenharia:** consolidar um fluxo completo com PyTorch + bundle versionado + API + observabilidade.
- **Deploy reprodutível:** o bundle (`preprocessor.joblib`, `mlp_state.pt`, `metadata.json`) empacota pré-processamento e modelo de forma explícita para execução estável.

Resumo: o pipeline usa MLP por alinhamento com o recorte técnico e de negócio do projeto, sem ignorar que outros modelos tiveram ótimo desempenho agregado no comparativo.

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

Observação: em ambientes Windows/migração de máquina, você pode ver variantes `-local` desses experimentos (ex.: `MLP-Churn-EarlyStoppingBatching-local`) para garantir `artifact_location` gravável.

## Resultado consolidado já disponível

Os runs consolidados da etapa de comparação ficam no MLflow. A auditoria principal pode ser consultada via UI e notebooks.

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

## Conclusão acadêmica da equipe

Como equipe em início de trajetória em Machine Learning Engineering, a principal conclusão desta etapa foi que **escolha de modelo não deve ser feita por uma métrica isolada**.

No comparativo técnico (`05_compare_mlp_baselines.ipynb`), o **RandomForest** apresentou melhor consistência global. Ao mesmo tempo, a **MLP** mostrou vantagem em métricas associadas ao objetivo de retenção (especialmente após calibração de threshold), e o notebook de trade-off (`06_tradeoff_custo_fp_fn.ipynb`) evidenciou que a decisão final depende do custo de **FN vs FP**.

Por isso, o pipeline atual usa a MLP bundleada na API como decisão de engenharia e de escopo do projeto, mantendo a análise comparativa registrada no MLflow para transparência e auditoria.

Próximos passos acadêmicos recomendados:

- repetir o comparativo em novos recortes temporais para validar estabilidade;
- avaliar calibração de probabilidade e fairness com dados adicionais;
- testar estratégia híbrida de seleção de modelo por contexto de negócio (ex.: operação orientada a recall vs operação orientada a custo).

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


### 3) Avisos de serialização no MLflow

Situação:

- O MLflow pode emitir warnings sobre pickle/cloudpickle ao salvar modelos.

Status atual:

- São avisos informativos; o registro das runs e artefatos ocorre normalmente.

## Documentação de apoio

- docs/METRICAS.md
- docs/API.md
- docs/API_EXEMPLOS_TESTE.md
- docs/ARQUITETURA_DEPLOY.md
- docs/OBSERVABILIDADE.md
- docs/PLANO_MONITORAMENTO.md
- docs/MODEL_CARD_FINAL.md

## Observações para trabalho em grupo

- Para sincronização técnica, priorize os experimentos consolidados no MLflow (incluindo possíveis variantes `-local`).
- O item 5 (consolidação de experimentos MLP + ensembles) já está concluído no estado atual do projeto.
