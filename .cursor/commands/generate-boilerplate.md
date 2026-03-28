# Comando: gerar boilerplate

## Objetivo

Gerar **esqueleto de código** repetitivo mas alinhado aos padrões do repositório (ML + FastAPI), sem introduzir abstrações desnecessárias.

## Contexto do projeto

- Python em `src/`, testes em `tests/`, configuração em `pyproject.toml` e tarefas em `Makefile`.
- Stack: PyTorch, sklearn, MLflow, FastAPI, Pydantic, pytest, ruff.

## Entrada

- Tipo de boilerplate desejado (ex.: “módulo de treino MLP”, “router FastAPI”, “schema Pydantic de predição”, “teste de API”).
- Padrões já existentes no repo (citar arquivos de referência se possível).

## Instruções para a IA

1. **Siga** estrutura e nomenclatura já usada em `src/` e `tests/`.
2. **Não** crie camadas extras (ex.: repository pattern) sem necessidade demonstrada.
3. **Respeite** `.cursor/rules/code-style.md` e **ruff**.
4. **Use apenas** dependências de `.cursor/libs/allowed-libs.md`.
5. **Inclua** testes pytest mínimos quando gerar lógica ou endpoints novos.
6. **Comentários** explicativos em **português brasileiro** quando agregarem valor.

## Restrições

- Resposta com código deve ser **idiomática Python 3.11+** e passível de passar no ruff após ajustes finos locais.
- Mensagens de explicação ao redor do código em **português brasileiro**.

## Saída esperada

- Código proposto (blocos bem delimitados)
- Breve explicação da estrutura criada
- Lista de arquivos afetados ou novos
- Próximos passos (comandos `make` ou `pytest` sugeridos)

## Documentação relacionada

- **Arquitetura:** `.cursor/context/architecture.md`
- **Bibliotecas permitidas:** `.cursor/libs/allowed-libs.md`
- **Estilo:** `.cursor/rules/code-style.md`
