# Comando: contestar solução proposta

## Objetivo

**Desafiar** uma solução já proposta (modelo, arquitetura de código, pipeline ou API) para reduzir viés de confirmação e **complexidade desnecessária** no projeto de churn.

## Contexto do projeto

- Tech Challenge com MLP **PyTorch**, baselines **sklearn**, **MLflow** e **FastAPI**.
- Foco em valor acadêmico e reprodutibilidade, não em produto enterprise completo.

## Entrada

- Descrição da solução atual (ou trecho de design).
- Premissas que o time considera fixas.

## Instruções para a IA

1. **O que pode dar errado?** (dados, generalização, latência, manutenção, falhas na API.)
2. **Existem alternativas mais simples?** (menos camadas, menos dependências, baseline mais forte antes da MLP.)
3. **Onde há engenharia em excesso?** (abstrações sem teste, microserviços, funcionalidades desnecessárias.)
4. **A solução respeita o escopo?** (não prometer campanhas automáticas nem CRM.)
5. **Trade-off FP/FN** está explícito para o negócio?

## Restrições

- Tom **crítico e construtivo**, em **português brasileiro**.
- Não sugerir trocar **PyTorch** como modelo principal da MLP por outro framework de deep learning apenas por preferência pessoal.
- Não sugerir stack **Node.js** para substituir a API do desafio.

## Saída esperada

- Riscos e pontos cegos
- Alternativas mais simples (se houver)
- Pontos de overengineering
- Recomendações priorizadas

## Documentação relacionada

- **Arquitetura:** `.cursor/context/architecture.md`
- **Uso de IA:** `.cursor/rules/ai-usage-rules.md`
- **Regras de negócio:** `.cursor/rules/business-rules.md`
