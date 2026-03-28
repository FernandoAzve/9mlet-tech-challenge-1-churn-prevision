# Comando: revisão de arquitetura

## Objetivo

Validar se decisões arquiteturais do projeto de **churn** estão **coerentes** com o Tech Challenge, boas práticas de **MLOps** e a stack **Python** acordada.

## Contexto do projeto

- Fluxo esperado: **dados → pipeline de features → treino (PyTorch + baselines sklearn) → MLflow → artefato em `models/` → inferência FastAPI** (`/predict`, `/health`).
- Repositório com `pyproject.toml`, `Makefile`, testes **pytest** e lint **ruff**.

## Entrada sugerida

- Trecho ou resumo da arquitetura atual (texto, diagrama ou lista de módulos).
- Dúvidas específicas (ex.: onde serializar o pré-processador, como versionar experimentos).

## Instruções para a IA

1. **Compare** a proposta com `.cursor/context/architecture.md` e `.cursor/context/tech-stack.md`.
2. **Aponte lacunas:** treino acoplado à API, ausência de separação de features, MLflow só no notebook, etc.
3. **Avalie riscos:** reprodutibilidade, skew treino-serviço, testabilidade, logging e erros da API.
4. **Sugira melhorias** priorizadas (alto / médio / baixo impacto).
5. **Indique inconsistências** com regras de negócio (probabilidade vs decisão de campanha).

## Restrições

- Resposta em **português brasileiro**.
- Não assumir banco de dados relacional ou filas se não estiverem no escopo.
- Não recomendar substituir PyTorch como modelo principal da MLP.

## Saída esperada

- Parecer estruturado: pontos fortes, problemas, riscos, recomendações priorizadas, checklist de ações.

## Documentação relacionada

- **Arquitetura:** `.cursor/context/architecture.md`
- **Regras de negócio:** `.cursor/rules/business-rules.md`
- **Uso de IA:** `.cursor/rules/ai-usage-rules.md`
