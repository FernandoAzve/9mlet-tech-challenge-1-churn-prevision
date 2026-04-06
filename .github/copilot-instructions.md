# Copilot Instructions - Projeto Churn (FIAP)

## Objetivo
Este repositorio usa GitHub Copilot no VS Code com regras nativas na pasta .github.

## Como atuar neste projeto
1. Ler primeiro o contexto em .github/context.
2. Respeitar regras em .github/rules.
3. Respeitar politicas de dependencias em .github/libs.
4. Tratar a stack oficial como Python + PyTorch + sklearn + MLflow + FastAPI + pytest + ruff.

## Regras obrigatorias
- IA propoe; humanos aprovam decisoes que impactam negocio, compliance e producao.
- Codigo sugerido so e valido se passar lint e testes.
- Nao sugerir stacks fora de escopo do desafio para substituir a stack principal.
- Priorizar simplicidade e reprodutibilidade.

## Validacao minima antes de considerar pronto
- ruff check
- pytest

## Referencias de contexto
- .github/context/project-goals.md
- .github/context/architecture.md
- .github/context/tech-stack.md
- .github/context/deployment.md

## Referencias de regras
- .github/rules/ai-usage-rules.md
- .github/rules/business-rules.md
- .github/rules/code-style.md
- .github/rules/git-rules.md
- .github/rules/security-rules.md
- .github/rules/testing-rules.md

## Referencias de bibliotecas e modelos
- .github/libs/allowed-libs.md
- .github/libs/forbidden-libs.md
- .github/libs/ai-models.md

## Comandos de prompt (slash prompts)
Os comandos estruturados ficam em .github/prompts como arquivos *.prompt.md.
Use esses prompts para tarefas estruturadas.
