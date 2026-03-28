# Comando: validação pré-deploy (ou pré-demonstração)

## Objetivo

Montar um **checklist** para validar que o projeto está pronto para **demonstração**, entrega acadêmica ou **deploy opcional** (Docker / nuvem).

## Contexto do projeto

- API **FastAPI** com `/predict` e `/health`.
- **ruff** sem erros, **pytest** passando, **MLflow** com runs rastreáveis, **Model Card** e **README** atualizados.
- **Makefile** e **pyproject.toml** como fonte de comandos padronizados.

## Entrada

- Ambiente alvo (local, container, nuvem).
- Versão ou tag Git candidata à entrega.

## Instruções para a IA

Gere checklist agrupado em seções:

1. **Código e qualidade:** `ruff check`, `ruff format --check` (se usado), estrutura `src/` e `tests/`.
2. **Testes:** `pytest`, cobertura mínima acordada, testes de API com `TestClient`.
3. **Modelo e MLflow:** artefato referenciado corretamente, seeds documentadas, comparação MLP vs baselines registrada.
4. **API:** contratos Pydantic, códigos HTTP, mensagens de erro seguras (sem stack trace ao cliente).
5. **Documentação:** README (instalação, uso, reprodutibilidade), Model Card (limitações, métricas, dados).
6. **Deploy opcional:** Dockerfile, variáveis de ambiente, healthcheck em `/health`.
7. **Segurança básica:** sem segredos no Git, sem dados pessoais em logs.

## Restrições

- Texto em **português brasileiro**.
- Não exigir npm, `package.json` ou ferramentas fora da stack Python do desafio.

## Saída esperada

- Lista de verificação com caixas `[ ]`
- Comandos sugeridos (`make …`, `pytest …`, `docker build …`)
- Itens que **devem** ser validados manualmente por humano (ex.: vídeo STAR, narrativa de negócio)

## Documentação relacionada

- **Implantação:** `.cursor/context/deployment.md`
- **Segurança:** `.cursor/rules/security-rules.md`
- **Testes:** `.cursor/rules/testing-rules.md`
