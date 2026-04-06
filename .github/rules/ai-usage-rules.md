# Regras de uso de inteligência artificial

## Objetivo

Definir **onde** a IA pode apoiar o desenvolvimento deste projeto de **ML para churn**, **como** deve ser validada a saída gerada e quais **pré-requisitos** de qualidade são obrigatórios antes de confiar em código automático.

## Escopo

- Princípios gerais
- Permissões e limites por fase (arquitetura, código, testes, documentação)
- Requisitos de padronização (Python, ruff, pytest)
- Riscos e mitigação

---

## Princípios fundamentais

1. **A IA não decide sozinha** sobre arquitetura de negócio, privacidade de dados, segurança sensível ou critérios finais de aceite do modelo.
2. **Processo antes da ferramenta:** contexto em `.github/context` e regras em `.github/rules` devem estar atualizados; prompts genéricos geram dívidas técnicas mais rápido.
3. **Contexto explícito vale mais que prompt bonito:** incluir stack (PyTorch, MLflow, FastAPI) e estrutura de pastas nas interações.
4. **Código gerado é código de produção:** passa por revisão humana, testes e lint como qualquer outro commit.

---

## Onde a IA pode atuar

### 1. Arquitetura e planejamento

**Permitido:** sim.

- Propor divisão de módulos em `src/` (dados, treino, API).
- Discutir trade-offs (latência vs complexidade do modelo, baseline vs MLP).
- Sugerir fluxo MLflow e convenções de artefatos.

**Obrigatório:**

- Decisões finais documentadas por humanos (README, comentários de PR ou ADR curto em `docs/`).
- Não aceitar desenhos que violem requisitos do Tech Challenge (PyTorch para MLP, `/predict` e `/health`, etc.).

---

### 2. Implementação de código

**Permitido:** sim.

- Pipelines de features, treino PyTorch, baselines sklearn, integração MLflow.
- API FastAPI com Pydantic, carregamento de modelo, logging.
- Scripts auxiliares e alvos do Makefile.

**Obrigatório:**

- Respeitar **`.github/libs/allowed-libs.md`** e **`.github/rules/code-style.md`**.
- Saída deve passar em **`ruff check`** e em **`pytest`** após integração.
- Não introduzir dependências proibidas sem revisão explícita.

---

### 3. Testes

**Permitido:** sim.

- Geração de testes unitários, testes de schema e testes de API com `TestClient` / **httpx**.

**Obrigatório:**

- Seguir **`.github/rules/testing-rules.md`** (pytest, nomenclatura `test_*.py`).
- Testes devem ser determinísticos (seeds, dados sintéticos pequenos).

---

### 4. Refatoração

**Permitido:** sim, com cautela.

- Melhorar legibilidade sem alterar comportamento acordado.

**Obrigatório:**

- Testes existentes continuam verdes; para bugs, adicionar teste que falhava antes do fix.

---

### 5. Documentação

**Permitido:** sim.

- README, Model Card, docstrings, diagramas em Mermaid.

**Obrigatório:**

- Revisão humana para exatidão (métricas, limitações do modelo, viés).

---

## Onde a IA não deve atuar sozinha

- **Políticas de privacidade e LGPD** (uso de dados reais, anonimização, bases de terceiros).
- **Configuração de segredos e produção** (chaves, URLs de tracking com credenciais).
- **Remoção ou alteração de dados reais** sem processo aprovado.
- **Definição unilateral de threshold de negócio** (ex.: ponto de corte de probabilidade para campanha) sem alinhamento com stakeholders.

**Regra de ouro:** a IA **propõe**; humanos **aprovam** o que impacta negócio, compliance e produção.

---

## Padronização obrigatória antes de confiar na IA para código

Antes de usar IA como “autopiloto” em funcionalidades grandes, o repositório deve ter:

- [ ] **`pyproject.toml`** com dependências e **ruff** configurado.
- [ ] **`Makefile`** (ou equivalente) com lint e testes documentados.
- [ ] Pasta **`tests/`** com pelo menos um teste de fumaça da API ou do pipeline, quando a API existir.
- [ ] Estrutura **`src/`**, **`data/`**, **`models/`**, **`notebooks/`**, **`docs/`** conforme o desafio.

**Código que não passa no ruff ou nos testes do CI é considerado inválido**, independentemente de ter sido gerado por IA.

---

## Modelos de IA sugeridos

Consulte **`.github/libs/ai-models.md`** para orientação por fase (arquitetura, implementação, depuração). Os nomes comerciais dos modelos mudam com o tempo; priorize o modelo disponível na sua assinatura com maior contexto e melhor desempenho em código Python.

---

## Riscos comuns e mitigação

| Risco                                               | Mitigação                                                               |
| --------------------------------------------------- | ----------------------------------------------------------------------- |
| Alucinação de API ou assinaturas                    | Conferir documentação oficial (FastAPI, PyTorch, MLflow) e rodar testes |
| Vazamento de dados sintéticos confundidos com reais | Nunca colar datasets reais no chat; usar amostras mínimas fictícias     |
| Dependência não permitida                           | Checar `allowed-libs` / `forbidden-libs` antes do merge                 |
| Overengineering                                     | Relembrar escopo do Tech Challenge e non-goals em `project-goals.md`    |

---

## Documentação relacionada

- **Estilo de código:** `.github/rules/code-style.md`
- **Testes:** `.github/rules/testing-rules.md`
- **Modelos de IA:** `.github/libs/ai-models.md`
- **Objetivos do projeto:** `.github/context/project-goals.md`
