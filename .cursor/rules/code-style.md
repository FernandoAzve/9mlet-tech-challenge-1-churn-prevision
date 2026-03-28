# Regras de estilo de código

## Objetivo

Definir padrões **obrigatórios** de escrita de código Python no projeto de previsão de churn, alinhados a **PEP 8**, **ruff** e boas práticas de legibilidade e manutenção.

## Contexto do projeto

- **Projeto:** `9mlet-tech-challenge-1-churn-prevision`
- **Linguagem:** Python
- **Domínio:** Machine Learning (PyTorch, Scikit-Learn, MLflow) e API FastAPI

---

## Escopo

- Configuração de lint e formatação
- Convenções de nomenclatura e estrutura de módulos
- Idioma do código, comentários e documentação inline
- Relação com testes e Git

---

## 1. Lint e formatação (obrigatório)

### Antes de desenvolver

O repositório **deve** conter:

1. **`pyproject.toml`** com dependências do projeto e seção de configuração do **ruff** (regras e, se desejado, `ruff format`).
2. **Makefile** (ou scripts documentados) com alvos equivalentes a:
   - `lint` / `ruff check`
   - `format` / `ruff format` (se adotado)
   - `test` / `pytest`

### Regras

- **Nenhum merge** deve introduzir erros reportados por `ruff check`.
- Se o time adotar **`ruff format`**, o código novo deve estar formatado; validar com `ruff format --check` no CI.
- Não desabilitar regras do ruff sem **justificativa em comentário** (`# noqa: …`) próxima ao caso excepcional.

### O que não usar como padrão deste projeto

- Ferramentas de lint/formatação de outros ecossistemas que substituam o ruff como fonte da verdade.

---

## 2. Python e PEP 8

- Seguir **PEP 8**, complementada pelas regras do **ruff** (que incorporam boa parte das convenções).
- Preferir funções e métodos **pequenos** e com responsabilidade única.
- Evitar `import *`; imports ordenados conforme configuração do ruff (ex.: isort integrado).
- Usar **pathlib** para caminhos quando fizer sentido.

---

## 3. Tipagem e docstrings

- **Anotações de tipo** em funções e métodos públicos de `src/` (parâmetros e retornos).
- **Docstrings** em estilo claro (Google ou NumPy, **uma convenção só** no repositório) para:
  - módulos de treino,
  - pipeline de features,
  - camada da API (handlers principais).
- Tipos complexos podem usar `TypedDict` ou modelos **Pydantic** onde couber.

---

## 4. Organização modular (`src/`)

- Separar, quando possível:
  - **dados e features** (carregamento, transformação),
  - **treino** (PyTorch, sklearn),
  - **integração MLflow**,
  - **API FastAPI** (rotas, dependências, schemas),
  - **configuração** (constantes, leitura de env).
- Evitar lógica pesada de treino dentro de arquivos que só deveriam servir a API.
- Constantes mágicas nomeadas (`UPPER_SNAKE_CASE`).

---

## 5. Idioma: código vs comentários

- **Código (inglês):** nomes de variáveis, funções, classes, módulos, mensagens de log e exceções técnicas **em inglês** (padrão da indústria e das bibliotecas).
- **Comentários e docstrings voltadas ao time (pt-BR):** uso de **português brasileiro** é aceito e incentivado onde facilitar o entendimento da equipe FIAP.
- **README, Model Card e documentação de usuário:** **português brasileiro**, salvo exigência contrária da instituição.

*(Se o time decidir 100% inglês inclusive em comentários, documente essa decisão no README e alinhe todos os arquivos.)*

---

## 6. Testes e estilo

- Consultar **`.cursor/rules/testing-rules.md`**.
- Código de teste segue as mesmas regras de lint; fixtures em `tests/conftest.py` quando útil.

---

## 7. Git

- Consultar **`.cursor/rules/git-rules.md`**.
- Mensagens de commit em inglês ou português, **de forma consistente** com o que for definido no README do repositório.

---

## 8. Padrões proibidos (resumo)

- Pular configuração do **ruff** antes de subir código novo significativo.
- Commitar código que falhe no `ruff check` configurado no CI.
- Misturar sem necessidade lógica de notebook dentro de módulos de produção em `src/` (notebooks ficam em `notebooks/`).
- Hardcode de caminhos absolutos pessoais; preferir variáveis de ambiente ou caminhos relativos ao projeto.

---

## Documentação relacionada

- **Testes:** `.cursor/rules/testing-rules.md`
- **Stack:** `.cursor/context/tech-stack.md`
- **Uso de IA:** `.cursor/rules/ai-usage-rules.md`
- **Bibliotecas permitidas:** `.cursor/libs/allowed-libs.md`
