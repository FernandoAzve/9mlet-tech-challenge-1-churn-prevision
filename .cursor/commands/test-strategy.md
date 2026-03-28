# Comando: estratégia de testes

## Objetivo

Definir ou revisar a **estratégia de testes automatizados** (pytest) para o projeto de churn: unitários, contrato, fumaça e API.

## Contexto do projeto

- **pytest**, arquivos `test_*.py`, **TestClient** / **httpx** para FastAPI.
- Endpoints obrigatórios: **`/health`** e **`/predict`**.
- Dados sensíveis **não** entram nos testes versionados.

## Entrada sugerida

- Estado atual da pasta `tests/` (resumo ou lista de arquivos).
- Prioridades (ex.: “garantir contrato Pydantic antes do treino pesado”).

## Instruções para a IA

1. **Proponha pirâmide de testes** adequada ao repositório (unitário → integração leve → API).
2. **Liste casos de teste** para `/health` e `/predict` (sucesso, validação 422, modelo ausente se aplicável).
3. **Sugira fixtures** em `conftest.py` (app FastAPI, dados sintéticos com seed fixa).
4. **Defina marcações** `slow` para testes que treinam rede ou rodam MLflow real.
5. **Indique métricas de cobertura** alvo e o que deve ter prioridade em `src/`.

## Restrições

- Resposta em **português brasileiro**.
- Compatível com `.cursor/rules/testing-rules.md`.
- Não propor Jest, Vitest ou outros frameworks de outros ecossistemas.

## Saída esperada

- Plano de testes em seções (unitário, schema, fumaça, API)
- Lista de casos com nome sugerido `test_*`
- Notas sobre CI (`ruff` + `pytest`)

## Documentação relacionada

- **Regras de testes:** `.cursor/rules/testing-rules.md`
- **Estilo de código:** `.cursor/rules/code-style.md`
