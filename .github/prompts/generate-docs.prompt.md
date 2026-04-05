---
agent: ask
description: "Use when: executar o fluxo generate-docs no projeto de churn"
---

# Comando: gerar documentação técnica

## Objetivo

Produzir ou atualizar **documentação em português brasileiro** alinhada ao Tech Challenge: README, Model Card, docstrings e guias em `docs/`.

## Contexto do projeto

- Projeto de **churn** com MLP **PyTorch**, baselines **sklearn**, **MLflow**, API **FastAPI**.
- Entregáveis típicos: README completo, Model Card, instruções de reprodutibilidade (`Makefile`, `pyproject.toml`).

## Entrada

- Tipo de documento (README, Model Card, ADR, guia de API).
- Público-alvo (avaliador FIAP, colega de time).
- Estado atual do modelo e métricas (valores reais ou placeholders claramente marcados).

## Instruções para a IA

1. **Estruture** o documento com seções claras (visão geral, dados, treino, avaliação, uso da API, limitações).
2. **Inclua** comandos reais sugeridos (`make`, `uvicorn`, `pytest`, `ruff`) coerentes com o repositório.
3. **Destaque** limitações, vieses e riscos de uso indevido da probabilidade de churn.
4. **Não** inventar números de desempenho; usar placeholders **EXPLICITAMENTE** rotulados se faltarem medições.
5. **Alinhe** terminologia com `.github/rules/business-rules.md` (falso positivo, falso negativo, retenção).

## Restrições

- Texto principal em **português brasileiro**.
- Não documentar stack Node.js ou ferramentas removidas do escopo.

## Saída esperada

- Markdown pronto para colar em `README.md`, `docs/model_card.md` ou similar
- Lista de figuras/tabelas opcionais a produzir depois
- Itens “pendentes de preenchimento humano” claramente marcados

## Documentação relacionada

- **Objetivos:** `.github/context/project-goals.md`
- **Implantação:** `.github/context/deployment.md`
- **Arquitetura:** `.github/context/architecture.md`
