---
name: Melhoria README acadêmico
overview: Reestruturar o README para ficar mais aderente ao contexto acadêmico e ao estado real do projeto, destacando comparação entre modelos nos notebooks, justificativa do modelo escolhido no pipeline, rationale de bundle e melhor documentação de estrutura, logs e API.
todos:
  - id: mapear-readme-atual
    content: Mapear no README atual os trechos que serão reescritos (abertura, estrutura, API, observabilidade e conclusão).
    status: completed
  - id: redigir-comparativo-modelos
    content: Redigir seção de comparação entre modelos com conclusão alinhada aos notebooks comparativos como fonte oficial e conexão explícita com a decisão no pipeline.
    status: completed
  - id: justificar-modelo-pipeline
    content: Explicitar no README que o modelo produtizado no pipeline é consequência direta dos resultados de PR-AUC, ROC-AUC, F1, recall e precision observados nos notebooks.
    status: completed
  - id: redigir-bundle-didatico
    content: Redigir explicação didática do bundle e rationale de uso em engenharia de ML.
    status: completed
  - id: reforcar-api-logs
    content: Reescrever seções de API e logs estruturados com foco em entendimento e operação.
    status: completed
  - id: incluir-conclusao-academica
    content: Incluir conclusão em tom acadêmico de equipe iniciante, destacando aprendizado e próximos passos.
    status: completed
  - id: revisar-coerencia-final
    content: Fazer revisão final de consistência, clareza e aderência ao contexto de trabalho de faculdade.
    status: completed
isProject: false
---

# Plano de melhoria do README acadêmico

## Objetivo
Atualizar o README para refletir com clareza o projeto como trabalho de pós-graduação, combinando documentação técnica de execução com análise crítica da equipe baseada no comparativo dos notebooks (fonte oficial definida) e justificando por que o modelo adotado no pipeline foi escolhido.

## Princípio de edição (importante)
- Não remover conteúdo útil já existente no README.
- Atualizar as seções atuais para o estado real da versão mais recente do projeto (ex.: estrutura de pastas, fluxo de notebooks, API, observabilidade e modelo em uso no pipeline).
- Incluir novas seções apenas quando houver lacuna de informação relevante para entendimento técnico/acadêmico.
- Priorizar continuidade textual do documento atual, evitando reescrita total desnecessária.

## Escopo de edição
- Arquivo principal: [D:/Projetos/FIAP/Tech Challenge 01/9mlet-tech-challenge-1-churn-prevision/README.md](D:/Projetos/FIAP/Tech%20Challenge%2001/9mlet-tech-challenge-1-churn-prevision/README.md)
- Fontes de apoio para consistência:
  - [D:/Projetos/FIAP/Tech Challenge 01/9mlet-tech-challenge-1-churn-prevision/src/churn/api/main.py](D:/Projetos/FIAP/Tech%20Challenge%2001/9mlet-tech-challenge-1-churn-prevision/src/churn/api/main.py)
  - [D:/Projetos/FIAP/Tech Challenge 01/9mlet-tech-challenge-1-churn-prevision/src/churn/api/routes.py](D:/Projetos/FIAP/Tech%20Challenge%2001/9mlet-tech-challenge-1-churn-prevision/src/churn/api/routes.py)
  - [D:/Projetos/FIAP/Tech Challenge 01/9mlet-tech-challenge-1-churn-prevision/src/churn/api/logging_config.py](D:/Projetos/FIAP/Tech%20Challenge%2001/9mlet-tech-challenge-1-churn-prevision/src/churn/api/logging_config.py)
  - [D:/Projetos/FIAP/Tech Challenge 01/9mlet-tech-challenge-1-churn-prevision/src/churn/api/middleware.py](D:/Projetos/FIAP/Tech%20Challenge%2001/9mlet-tech-challenge-1-churn-prevision/src/churn/api/middleware.py)
  - [D:/Projetos/FIAP/Tech Challenge 01/9mlet-tech-challenge-1-churn-prevision/docs/MELHORIAS_CONTINUAS_ETAPA2.md](D:/Projetos/FIAP/Tech%20Challenge%2001/9mlet-tech-challenge-1-churn-prevision/docs/MELHORIAS_CONTINUAS_ETAPA2.md)

## Estrutura proposta para o novo README
- Reescrever a abertura com contexto acadêmico e objetivo de negócio (predição de churn com viés de decisão).
- Atualizar a seção de estrutura do projeto com visão por camadas (dados, notebooks, código-fonte, testes, documentação, artefatos) e descrição de propósito de cada pasta-chave.
- Criar seção explícita de comparação de modelos (MLP e baselines), priorizando métricas e conclusão derivada dos notebooks comparativos.
- Adicionar subseção “Modelo escolhido no pipeline e justificativa”, conectando decisão de engenharia aos resultados do comparativo (desempenho global e trade-offs de negócio).
- Incluir seção “Bundle do modelo: o que é e por que usamos”, explicando em linguagem acessível o pacote (`preprocessor.joblib`, `mlp_state.pt`, `metadata.json`) e ganhos de reprodutibilidade/deploy.
- Melhorar seção de API com descrição funcional de `/health` e `/predict`, contrato de entrada/saída e comportamento quando bundle não está carregado (503).
- Melhorar seção de observabilidade com resumo de logs estruturados, `X-Request-ID`, latência e utilidade para troubleshooting.
- Adicionar seção de conclusão acadêmica (tom de estudantes de 1º semestre) explicando o que foi aprendido com os trade-offs e por que o modelo selecionado para o pipeline foi considerado o mais adequado ao objetivo do projeto.

## Critérios de conteúdo (alinhamento com seu pedido)
- Preservar a base do README atual: atualizar o que já existe e adicionar somente o que estiver faltando.
- Comparação entre modelos com narrativa interpretativa (não só listagem de métricas).
- Justificativa explícita da escolha do modelo no pipeline, mostrando que a decisão vem dos resultados observados nos notebooks e dos objetivos de negócio (não de preferência arbitrária).
- Explicação didática do bundle e justificativa prática de engenharia.
- Conclusão crítica e honesta da equipe sobre resultado final do comparativo.
- Descrição mais clara da estrutura do repositório para novos integrantes/professor avaliador.
- Logs estruturados explicados em termos de operação real (correlação de requisições e latência).
- Rotas da API documentadas com foco em uso e validação.

## Validação após edição
- Revisão de coerência interna do README (evitar contradições entre seções).
- Revisão de consistência com o comportamento real da API e do bundle no código.
- Revisão de linguagem (equilíbrio entre clareza para leigo e rigor técnico acadêmico).