# Regras de testes

## Objetivo

Definir estratégia e regras **obrigatórias** de testes automatizados para o projeto de previsão de churn, usando **pytest**, com foco em reprodutibilidade e qualidade da API.

## Escopo

- Metodologia (incluindo TDD quando aplicável)
- Tipos de teste (unitário, contrato, API)
- Estrutura de arquivos e cobertura
- Integração com CI

---

## 1. Framework e convenções

- **Framework:** **pytest** exclusivamente para testes Python do projeto.
- **Nomes de arquivo:** `test_*.py` ou `*_test.py` (padronizar um estilo no `pyproject.toml` e manter).
- **Nomes de funções de teste:** `test_<comportamento_esperado>`.
- **Fixtures compartilhadas:** arquivo `tests/conftest.py`.

---

## 2. Pirâmide de testes

- **Base (maior volume):** testes **unitários** — funções puras de features, métricas auxiliares, utilitários de pré-processamento (com dados sintéticos pequenos).
- **Meio:** testes de **integração leve** — pipeline de transformação + modelo mock ou modelo mínimo treinado em memória quando viável.
- **Topo:** testes de **API** — endpoints `/health` e `/predict` usando **TestClient** do FastAPI com **httpx** como backend assíncrono quando necessário.

---

## 3. TDD (recomendado como disciplina)

Para funcionalidades novas de lógica de negócio ou contrato da API:

1. **Vermelho:** escrever teste que falha.
2. **Verde:** implementar o mínimo para passar.
3. **Refatorar:** melhorar o código mantendo os testes verdes.

**Regras duras:**

- Não integrar funcionalidade nova sem testes que a cubram, salvo **exceção explícita** registrada (spike descartável).
- Não fazer merge com testes ignorados (`skip`, `xfail`) sem justificativa no PR.

---

## 4. Tipos de teste detalhados

### 4.1 Testes unitários

- Um comportamento principal por teste (facilita diagnóstico).
- Padrão **AAA** (Arrange, Act, Assert).
- Dados de entrada **fabricados** (fixtures, `numpy.random.Generator` com seed fixa, `pandas` DataFrame mínimo).

### 4.2 Testes de schema / contrato

- Validar que payloads inválidos retornam **422** (Pydantic).
- Validar formato de resposta de `/predict` (campos obrigatórios, tipos).
- Quando usar **pandera** no offline, testar que schemas rejeitam dados fora do esperado.

### 4.3 Testes de fumaça (smoke)

- Subir app de teste com modelo **stub** ou arquivo mínimo em `tmp_path`.
- Chamar `/health` e um `/predict` com payload válido simples.
- Garantir que a aplicação inicia e responde sem erro interno.

### 4.4 Testes da API com TestClient

Exemplo de ideia (não copiar literalmente se o projeto usar outra estrutura de app):

```python
# tests/test_api_predict.py
from fastapi.testclient import TestClient


def test_health_retorna_200(app):
    client = TestClient(app)
    resposta = client.get("/health")
    assert resposta.status_code == 200


def test_predict_rejeita_payload_invalido(app):
    client = TestClient(app)
    resposta = client.post("/predict", json={})
    assert resposta.status_code == 422
```

*(A fixture `app` deve ser definida em `conftest.py` importando a factory da aplicação FastAPI.)*

---

## 5. ML e testes

- **Treinos longos** não rodam em cada `pytest` por padrão; marcar com `@pytest.mark.slow` ou equivalente e excluir no CI rápido.
- Testes de treino podem usar **dataset minúsculo** e **poucas épocas** só para verificar que o loop roda e que MLflow é chamado com mocks, se aplicável.
- **Seeds:** fixar seeds nos testes que dependem de aleatoriedade para evitar flakiness.

---

## 6. Cobertura

- **Meta geral sugerida:** **≥ 70%** de cobertura de linhas em `src/`, ajustável pelo time.
- **Prioridade absoluta:** camada da API, validação de entrada, utilitários de pós-processamento de predição.
- Medir com **pytest-cov** no CI e inspecionar diff de cobertura no PR.

---

## 7. Proibições

- Usar dados reais sensíveis nos testes versionados.
- Depender da ordem de execução entre arquivos de teste.
- Testes que fazem chamadas de rede externas sem mock (exceto integração explícita e isolada).
- Commitar testes que falham intermitentemente sem correção.

---

## 8. CI

- Todo pull request deve executar `pytest` (conjunto rápido padrão).
- Falha de teste **bloqueia merge**.

---

## Documentação relacionada

- **Estilo de código:** `.github/rules/code-style.md`
- **Regras Git:** `.github/rules/git-rules.md`
- **Arquitetura:** `.github/context/architecture.md`
