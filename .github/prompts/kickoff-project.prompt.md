---
agent: ask
description: "Use when: executar o fluxo kickoff-project no projeto de churn"
---

# Comando: início de projeto (kickoff)

## Objetivo

Alinhar **entendimento de negócio, requisitos técnicos e riscos** antes de implementar código no Tech Challenge de **previsão de churn** em telecom.

## Contexto do projeto

- **Nome:** `9mlet-tech-challenge-1-churn-prevision`
- **Stack:** Python, PyTorch (MLP), Scikit-Learn, MLflow, FastAPI, pytest, ruff, Pydantic
- **Estrutura:** `src/`, `data/`, `models/`, `tests/`, `notebooks/`, `docs/`
- **Problema:** antecipar clientes com alto risco de cancelamento para ações de **retenção** e redução de **custo de churn**.

## Entrada sugerida

- Descrição resumida do dataset (origem, tamanho, features, alvo).
- Restrições do orientador ou do enunciado FIAP.
- Dúvidas abertas sobre métricas de negócio (custo de falso positivo vs falso negativo).

## Instruções para a IA

Atue como **arquiteta ou arquiteto de ML sênior**.

Com base no contexto oficial do Tech Challenge:

1. **Liste requisitos funcionais**
   - O que o sistema deve fazer (treino, tracking, API `/predict` e `/health`, documentação).
   - Fluxos principais: experimentação → artefato → inferência.

2. **Liste requisitos não funcionais**
   - Reprodutibilidade (seeds), desempenho aceitável da API, manutenibilidade, rastreabilidade (MLflow).

3. **Identifique riscos**
   - Técnicos (desbalanceamento, vazamento treino-serviço, complexidade do pipeline).
   - De negócio (interpretação do threshold, uso indevido da probabilidade).
   - De dados (qualidade, vieses, LGPD se houver dados reais).

4. **Proponha arquitetura inicial de alto nível**
   - Camadas: dados → features → treino → MLflow → artefato → FastAPI.
   - Componentes principais e responsabilidades em `src/`.

## Restrições

- **Não** gerar código neste comando (apenas texto estruturado).
- Responder em **português brasileiro**, objetivo e estruturado.
- Respeitar stack obrigatória (PyTorch para MLP, baselines sklearn, MLflow, FastAPI).

## Saída esperada

Documento estruturado contendo:

- Requisitos funcionais
- Requisitos não funcionais
- Riscos identificados
- Arquitetura proposta (alto nível)

## Documentação relacionada

- **Objetivos:** `.github/context/project-goals.md`
- **Arquitetura:** `.github/context/architecture.md`
- **Stack:** `.github/context/tech-stack.md`
