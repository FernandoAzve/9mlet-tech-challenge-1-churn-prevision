# TODO — Etapa 3 (Engenharia e API)

## Objetivo deste documento

Organizar a execução da **Etapa 3 — Engenharia e API** com foco em:

- Refatoração profissional do projeto para estrutura modular em `src/`.
- Criação de pipeline reprodutível para treino e inferência.
- Entrega de API FastAPI funcional, validada e testada.
- Padronização de qualidade com lint, testes e comandos de automação.

Este documento segue o padrão do `docs/TODO.md` e separa os TODOs por tarefa, para execução síncrona até o fechamento da etapa.

---

## Itens

### 1. Refatorar código em módulos (`src/`) com estrutura limpa

| Campo          | Valor |
| -------------- | ----- |
| **Status**     | Concluído |
| **Entregável** | Estrutura base criada em `src/churn/` com separação de responsabilidades para dados, features, pipelines e modelos (`data/`, `features/`, `pipelines/`, `models/`, `config.py`). |
| **Escopo**     | Lógica principal de carregamento, contrato de dados, treino e inferência foi extraída para módulos reutilizáveis sem remover notebooks existentes. |

---

### 2. Criar pipeline reprodutível (`sklearn` + transformadores custom)

| Campo          | Valor |
| -------------- | ----- |
| **Status**     | Concluído |
| **Entregável** | Pipeline reprodutível implementado com `Pipeline` + `ColumnTransformer`, incluindo transformador custom (`TotalChargesCleaner`) e suporte a serialização via `joblib` em `src/churn/models/registry.py`. |
| **Escopo**     | Pipeline modularizado em `src/churn/pipelines/churn_pipeline.py` com consistência entre treino e inferência e ponto único de montagem para uso em script/notebook/API. |

---

### Observações de fechamento técnico (Itens 1 e 2)

- Contrato de dados e utilitários de carga/split centralizados em `src/churn/data/`.
- Treino e inferência desacoplados de notebook em `src/churn/models/train.py` e `src/churn/models/predict.py`.
- Persistência padronizada com `joblib` e metadados JSON em `src/churn/models/registry.py`.
- Notebook `notebooks/03_mlp_pytorch.ipynb` mantido e estendido com célula de integração para consumir o pipeline de `src/`.

---

### 3. Escrever testes (`pytest`): unitários, schema (`pandera`) e smoke test

| Campo          | Valor |
| -------------- | ----- |
| **Status**     | Pendente |
| **Entregável** | Suíte de testes em `tests/` cobrindo funções críticas, contrato de dados e teste rápido de execução da aplicação. |
| **Escopo**     | Criar testes unitários dos módulos de negócio, validação de schema de entrada/saída com `pandera`, e smoke test para garantir que a aplicação/pipeline inicializa e responde no fluxo mínimo esperado. |

---

### 4. Construir API FastAPI (`/predict`, `/health`, validação Pydantic)

| Campo          | Valor |
| -------------- | ----- |
| **Status**     | Pendente |
| **Entregável** | API FastAPI operacional com rotas `/health` e `/predict`, incluindo validação de payload e resposta. |
| **Escopo**     | Definir modelos Pydantic de request/response, tratamento de erros de validação, carregamento de pipeline/modelo em ciclo de vida da aplicação e contrato estável para consumo da inferência. |

---

### 5. Adicionar logging estruturado e middleware de latência

| Campo          | Valor |
| -------------- | ----- |
| **Status**     | Pendente |
| **Entregável** | Logging estruturado configurado + middleware para medir latência por requisição. |
| **Escopo**     | Padronizar logs com contexto mínimo (rota, método, status, duração, correlação de request) e incluir medição de tempo de resposta para observabilidade básica da API. |

---

### 6. Configurar `pyproject.toml`, `ruff`, `Makefile` (lint, test, run)

| Campo          | Valor |
| -------------- | ----- |
| **Status**     | Pendente |
| **Entregável** | Projeto com configuração centralizada de ferramentas e comandos padronizados para desenvolvimento. |
| **Escopo**     | Criar `pyproject.toml` com configuração mínima necessária, centralizar regras do `ruff`, e adicionar `Makefile` com alvos ao menos para `lint`, `test` e `run` da API. |

---

## Critério de pronto da Etapa 3

- Estrutura modular em `src/` substituindo dependência direta de notebooks para inferência.
- Pipeline reprodutível implementado e reutilizado pela API.
- Testes unitários, de schema e smoke test executando com sucesso.
- API FastAPI funcional com `/health` e `/predict`.
- Logging estruturado e middleware de latência ativos.
- `pyproject.toml`, `ruff` e `Makefile` configurados e operacionais.

---

## Entregável final esperado

**Repositório refatorado + API funcional + testes passando.**

---

## Ordem sugerida de execução (síncrona)

1 → 2 → 4 → 5 → 3 → 6

Observação: o item 3 (testes) deve começar junto dos itens 1/2 e ser completado antes do fechamento; a ordem acima prioriza fluxo técnico de construção da API e depois fechamento formal de qualidade.

---

## Documentação relacionada

- `docs/TODO.md` — padrão de acompanhamento de execução  
- `docs/METRICAS.md` — métricas técnicas e de negócio  
- `.github/context/tech-stack.md` — stack oficial  
- `.github/rules/testing-rules.md` — diretrizes de testes  
- `.github/rules/code-style.md` — padronização de código  
