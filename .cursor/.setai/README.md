# Configuração SetAI

Esta pasta contém uma **cópia de referência** da configuração da CLI **SetAI** usada na geração inicial da estrutura do projeto.

## Conteúdo

- `config.json` — parâmetros da CLI (chaves de API, idioma de perguntas/arquivos, etc.).
- `.gitignore` — evita versionar `config.json` com dados sensíveis.

## Avisos importantes

- **Não commite** chaves de API reais. O arquivo local pode conter apenas placeholders; confira antes de qualquer `git push`.
- A configuração “oficial” da CLI costuma ficar fora do repositório:
  - **Windows:** `%USERPROFILE%\.setai\config.json`
  - **macOS / Linux:** `~/.setai/config.json`
- Para alterar opções da ferramenta, use o comando `setai config` no terminal (conforme documentação da CLI).

## Relação com este repositório

O projeto em si é **Python / ML** (churn); esta pasta é apenas **metadado do gerador** e não faz parte da stack de execução da API ou do treino.
