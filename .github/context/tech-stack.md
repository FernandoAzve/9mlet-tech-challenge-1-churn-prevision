# Stack tecnológica

## Objetivo

Documentar a stack oficial do projeto, versões de referência, ferramentas de qualidade e política de dependências alinhadas ao Tech Challenge de previsão de churn em telecom.

## Escopo

- Linguagem e runtime
- Bibliotecas obrigatórias e recomendadas
- Ferramentas de desenvolvimento, testes e lint
- Restrições técnicas explícitas

---

## Linguagem e runtime

- **Linguagem:** Python 3.11 ou superior (fixar versão mínima no `pyproject.toml`).
- **Gerenciamento de ambiente:** ambiente virtual (`venv`) ou ferramenta equivalente; dependências declaradas no **`pyproject.toml`**.
- **Execução local da API:** **Uvicorn** (ASGI) para servir a aplicação FastAPI.

---

## Stack oficial (obrigatória)

| Componente | Tecnologia | Uso no projeto |
|------------|------------|----------------|
| Modelo principal | **PyTorch** | MLP para classificação binária de churn |
| Baselines | **Scikit-Learn** | Modelos de referência e pipelines de pré-processamento |
| Experimentos | **MLflow** | Rastreamento de métricas, parâmetros e artefatos |
| API | **FastAPI** | Endpoints **`/predict`** e **`/health`** |
| Contratos e validação | **Pydantic** | Schemas de entrada/saída da API e configurações |
| Qualidade de código | **ruff** | Lint e formatação (obrigatório sem erros) |
| Testes | **pytest** | Testes unitários, de API e de contrato |
| Cliente HTTP em testes | **httpx** | Compatível com `TestClient` do FastAPI |
| Dados tabulares | **pandas** / **numpy** | Carga, feature engineering e tensores |
| Validação de dados (pipeline) | **pandera** | Schemas e checagens no pipeline, quando aplicável |
| Observabilidade | **logging** (módulo `logging`) | Logs estruturados (JSON ou formato consistente) |

---

## Estrutura de repositório (obrigatória)

- `src/` — código-fonte (treino, inferência, pipeline, utilitários).
- `data/` — dados brutos e processados (não versionar dados sensíveis volumosos sem política clara).
- `models/` — artefatos serializados (pesos, checkpoints exportados conforme convenção do projeto).
- `tests/` — suíte pytest espelhando módulos em `src/`.
- `notebooks/` — exploração e experimentação reprodutível.
- `docs/` — documentação complementar (incluindo referência ao Model Card).

Arquivos de orquestração local:

- **`Makefile`** — alvos para `lint`, `test`, `train`, `serve`, etc.
- **`pyproject.toml`** — projeto, dependências, configuração do **ruff** e, quando aplicável, do pytest.

---

## Ferramentas de desenvolvimento

### Qualidade de código

- **Lint e formatação:** **ruff** (única ferramenta obrigatória de lint; usar também `ruff format` se o time padronizar).
- **Tipagem:** anotações de tipo em código novo; opcionalmente **mypy** ou checagem via IDE, sem substituir o ruff como gate principal.

### Testes

- **Framework:** **pytest**.
- **Convenção de arquivos:** `test_*.py` ou `*_test.py` conforme padrão definido no `pyproject.toml`.
- **Cobertura:** **pytest-cov** recomendado para relatórios em CI.

### CI/CD

- **GitHub Actions:** pipelines para instalar dependências, executar **ruff** e **pytest** em pull requests e na branch principal.
- **Container:** **Docker** opcional para empacotar API e dependências; não é requisito obrigatório do desafio, mas é bem-vindo para reprodutibilidade.

---

## Política de dependências

### Permitido

- Consultar **`.github/libs/allowed-libs.md`** para a lista detalhada e justificativas.

### Proibido ou sujeito a aprovação

- Consultar **`.github/libs/forbidden-libs.md`**.

### Atualizações

- Corrigir vulnerabilidades conhecidas com prioridade.
- Alterações de versão major de `torch`, `sklearn` ou `mlflow` exigem validação de reprodutibilidade e regressão de testes.

---

## Documentação relacionada

- **Arquitetura:** `.github/context/architecture.md`
- **Implantação:** `.github/context/deployment.md`
- **Estilo de código:** `.github/rules/code-style.md`
- **Bibliotecas permitidas:** `.github/libs/allowed-libs.md`
