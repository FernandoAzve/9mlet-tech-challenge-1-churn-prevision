# Bibliotecas permitidas

## Objetivo

Listar bibliotecas **aprovadas** para o projeto de previsão de churn, com justificativa e casos de uso típicos.

## Escopo

- Núcleo de ML e dados
- API e validação
- Testes e qualidade
- Utilitários opcionais

---

## Núcleo obrigatório (stack oficial)

| Biblioteca | Motivo | Uso principal |
|------------|--------|----------------|
| **torch** | Requisito do desafio | MLP, treino e inferência PyTorch |
| **scikit-learn** | Requisito do desafio | Baselines, métricas, pipelines de pré-processamento |
| **mlflow** | Requisito do desafio | Tracking de experimentos, registro de artefatos |
| **fastapi** | Requisito do desafio | API REST com `/predict` e `/health` |
| **pydantic** | Requisito do desafio | Schemas de request/response e configurações |
| **uvicorn** | Servidor ASGI padrão | Execução local e deploy da API |
| **pytest** | Requisito do desafio | Suíte de testes automatizados |
| **httpx** | Cliente HTTP moderno | `TestClient` / testes assíncronos da API |
| **ruff** | Requisito do desafio | Lint e formatação |

---

## Dados e validação em pipeline

| Biblioteca | Motivo | Uso principal |
|------------|--------|----------------|
| **pandas** | Manipulação tabular | ETL leve, feature engineering |
| **numpy** | Arrays numéricos | Tensores, álgebra linear, interop com PyTorch |
| **pandera** | Schemas de DataFrame | Validação de dados no pipeline offline |

---

## Utilitários recomendados (opcionais)

| Biblioteca | Motivo | Uso principal |
|------------|--------|----------------|
| **pydantic-settings** | Config tipada | Carregar variáveis de ambiente com validação |
| **pytest-cov** | Cobertura | Relatórios no CI |
| **joblib** | Serialização sklearn | Persistir pipelines e modelos sklearn |
| **matplotlib** / **seaborn** | Visualização | EDA em notebooks (não obrigatório na API) |
| **tqdm** | Progresso | Loops de treino em terminal ou notebook |

Adicionar outras dependências **leves** é permitido desde que:

- não dupliquem função de bibliotecas já adotadas sem motivo;
- sejam mantidas no `pyproject.toml` com versão compatível;
- passem na revisão do PR.

---

## Política de atualização

- **Patches** de segurança: aplicar com prioridade.
- **Major** de `torch`, `scikit-learn` ou `mlflow`: exigir validação de reprodutibilidade e regressão de testes antes do merge.

---

## Documentação relacionada

- **Bibliotecas proibidas:** `.github/libs/forbidden-libs.md`
- **Stack:** `.github/context/tech-stack.md`
