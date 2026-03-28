# Configuração da pasta `.cursor`

## Objetivo

Centralizar **contexto**, **regras** e **comandos** para uso consistente de IA e desenvolvimento do Tech Challenge de **previsão de churn** (Python, PyTorch, MLflow, FastAPI).

## Estrutura

### Pasta `context/`

Contexto persistente do projeto — responde “como este repositório funciona?”.

- `architecture.md` — decisões arquiteturais (dados → treino → MLflow → API).
- `tech-stack.md` — stack oficial e ferramentas.
- `project-goals.md` — objetivos de negócio e restrições do desafio.
- `deployment.md` — CI/CD, Makefile, Docker opcional, ambientes.

### Pasta `rules/`

Regras duras — contrato com humanos e com a IA.

- `code-style.md` — Python, PEP 8, ruff, organização modular.
- `testing-rules.md` — pytest, TestClient, cobertura.
- `git-rules.md` — commits, branches, pull requests.
- `security-rules.md` — API, dados, segredos, logs.
- `ai-usage-rules.md` — onde a IA pode e não pode atuar.
- `business-rules.md` — churn, retenção, FP/FN, escopo da API.

### Pasta `libs/`

Política de dependências e de modelos LLM.

- `allowed-libs.md` — bibliotecas Python aprovadas.
- `forbidden-libs.md` — restrições e exceções.
- `ai-models.md` — orientação de escolha de LLM por fase.

### Pasta `commands/`

Prompts reutilizáveis (comandos do Cursor ou cópia manual).

- `kickoff-project.md` — alinhar requisitos antes de codar.
- `architecture-review.md` — revisar arquitetura ML/API.
- `extract-business-rules.md` — extrair regras de texto-fonte.
- `test-strategy.md` — planejar pytest e testes de API.
- `generate-boilerplate.md` — gerar esqueleto de código.
- `refactor-controlled.md` — refatorar sem mudar comportamento.
- `generate-docs.md` — README, Model Card, docs.
- `review-pr.md` — revisão de PR.
- `challenge-solution.md` — contestar solução proposta.
- `pre-deploy-validation.md` — checklist pré-demonstração ou deploy.

### Pasta `.setai/`

Configuração de referência do gerador **SetAI** (não versionar segredos reais).

## Uso

### Para desenvolvedoras e desenvolvedores

1. Configurar **`pyproject.toml`**, **ruff** e **pytest** antes de evoluir o código.
2. Ler `context/` para alinhar visão.
3. Seguir `rules/` em todo PR.
4. Respeitar `libs/allowed-libs.md` ao adicionar dependências.
5. Usar `commands/` quando precisar de um roteiro estruturado de prompt.

### Para a IA

1. Verificar que **ruff** e **pytest** fazem parte do fluxo antes de sugerir “pronto para merge”.
2. Ler `rules/` primeiro.
3. Usar `context/` para decisões de arquitetura e stack.
4. Consultar `libs/` antes de propor novas bibliotecas.
5. Tratar `commands/` como templates de instrução.

## Princípios

- **Contexto explícito é melhor que prompt vazio.**
- **Regras duras, não sugestões soltas.**
- **Documentação viva:** atualizar quando o repositório mudar.
- **A IA propõe; humanos aprovam** o que afeta negócio, dados sensíveis e entrega.

## Manutenção

- Atualizar arquivos quando a stack ou o enunciado mudarem.
- Manter consistência entre `context/`, `rules/` e `libs/`.
- Registrar decisões importantes em `docs/` quando necessário.

---

*Alinhado às práticas de desenvolvimento assistido por IA e ao Tech Challenge de ML — 2026.*
