# Comando: refatoração controlada

## Objetivo

Melhorar **legibilidade, organização e manutenção** do código Python do projeto de churn **sem alterar comportamento** observável já coberto por testes.

## Contexto do projeto

- Módulos em `src/` (dados, features, treino, MLflow, API).
- Qualidade garantida por **pytest** e **ruff**.

## Entrada

- Arquivos ou trechos a refatorar (referência por caminho).
- Objetivo da refatoração (ex.: “extrair função de pré-processamento”, “separar config da app FastAPI”).

## Instruções para a IA

1. **Leia** o código existente e os testes relacionados.
2. **Proponha** passos pequenos e reversíveis (um comportamento por commit ideal).
3. **Preserve** assinaturas públicas estáveis ou liste **breaking changes** se inevitáveis.
4. **Mantenha** reprodutibilidade (não alterar seeds ou ordem de dados sem motivo).
5. **Após cada mudança conceitual**, indique quais testes devem continuar verdes e se novos testes são necessários.

## Restrições

- **Não** mudar métricas de ML ou hiperparâmetros “de passagem”; refatoração não é retreino.
- **Não** adicionar dependências proibidas.
- Explicações em **português brasileiro**; nomes de código podem seguir inglês técnico.

## Saída esperada

- Plano numerado de passos
- Diff ou blocos de código sugeridos
- Checklist de testes (`pytest`, possivelmente marcador `slow`)
- Riscos residuais

## Documentação relacionada

- **Estilo:** `.cursor/rules/code-style.md`
- **Testes:** `.cursor/rules/testing-rules.md`
- **Uso de IA:** `.cursor/rules/ai-usage-rules.md`
