# Comando: extrair regras de negócio

## Objetivo

A partir de texto livre (enunciado, ata, e-mail do orientador, descrição do dataset), extrair e organizar **regras de negócio** relevantes para o sistema de **previsão de churn**.

## Contexto do projeto

- Foco em **retenção**, **custo de churn evitado** (conceitual) e trade-off **falso positivo / falso negativo**.
- O software entrega **predição** (idealmente probabilidade); **campanhas** ficam fora do escopo técnico mínimo.

## Entrada

- Texto-fonte colado pelo usuário (enunciado, requisitos escritos à mão, etc.).

## Instruções para a IA

1. **Identifique** objetivos de negócio explícitos e implícitos.
2. **Extraia** regras verificáveis (ex.: “API deve expor `/predict`”, “usar validação cruzada estratificada”).
3. **Classifique** cada item como: obrigatório do desafio, derivado do domínio telecom, ou suposição (marcar claramente suposições).
4. **Relacione** com impacto no modelo (métricas, threshold, segmentação).
5. **Liste non-goals** detectados no texto (o que o sistema não precisa fazer).

## Restrições

- Resposta em **português brasileiro**.
- Não inventar regras que não tenham base no texto; quando inferir, rotular como **suposição**.
- Alinhar com `.cursor/rules/business-rules.md` e apontar conflitos se houver.

## Saída esperada

- Lista numerada de regras de negócio
- Suposições claramente marcadas
- Non-goals
- Sugestão de onde documentar cada item (README, Model Card, código)

## Documentação relacionada

- **Regras de negócio:** `.cursor/rules/business-rules.md`
- **Objetivos:** `.cursor/context/project-goals.md`
