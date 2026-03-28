# Comando: revisão de pull request

## Objetivo

Revisar um PR como **revisor técnico sênior** de projeto **ML + API Python**, priorizando correção funcional, reprodutibilidade, testes e conformidade com o Tech Challenge.

## Contexto do projeto

- Stack: PyTorch (MLP), sklearn, MLflow, FastAPI, Pydantic, pytest, ruff.
- Endpoints `/predict` e `/health`; pipeline reprodutível; seeds e validação cruzada estratificada quando aplicável.

## Entrada

- Descrição do PR ou diff resumido.
- Objetivo da mudança e riscos percebidos pelo autor.

## Instruções para a IA

1. **Resuma** a mudança em até cinco frases.
2. **Verifique aderência** aos requisitos do desafio (stack, estrutura de pastas, endpoints).
3. **Avalie qualidade:** ruff, pytest, clareza de módulos, logging adequado.
4. **Identifique riscos de ML:** vazamento de dados, skew treino-serviço, métricas enganosas, seeds não fixadas.
5. **Classifique** comentários como **bloqueante**, **importante** ou **sugestão**.
6. **Sugira** mensagem de commit ou título de PR alinhada ao **padrão Conventional Commits** (pode ser em português se o time padronizar assim).

## Restrições

- Resposta em **português brasileiro**.
- Não exigir JWT ou ORM se não fizer parte do escopo.
- Ser direto; evitar elogios genéricos.

## Saída esperada

- Resumo
- Pontos positivos
- Problemas e riscos (com severidade)
- Checklist pré-merge
- Comentários sugeridos para linhas específicas (se o usuário tiver colado trechos)

## Documentação relacionada

- **Regras Git:** `.cursor/rules/git-rules.md`
- **Testes:** `.cursor/rules/testing-rules.md`
- **Segurança:** `.cursor/rules/security-rules.md`
