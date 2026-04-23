# Melhorias Contínuas - Etapa 2 (Churn)

## Objetivo deste documento

Consolidar as análises críticas dos notebooks da Etapa 2 e transformar os achados em um plano prático de evolução, com prioridades e critérios de conclusão.

Escopo avaliado:
- notebooks/03_mlp_pytorch.ipynb
- notebooks/04_mlp_training_early_stopping.ipynb
- notebooks/05_compare_mlp_baselines.ipynb
- notebooks/06_tradeoff_custo_fp_fn.ipynb

## Resumo executivo

Pontos fortes:
- Pipeline de modelagem completo (MLP, comparação com baselines e trade-off de threshold).
- Reprodutibilidade básica com SEED e split consistente.
- Integracao com MLflow presente e funcional para MLP e analise de trade-off.

Pontos de atencao:
- Modelo MLP atual apresenta recall baixo em threshold 0.5.
- RandomForest teve desempenho técnico superior no comparativo.
- Consolidação de auditoria de runs (item 5 da Etapa 2) já foi concluída.
- Ajustes de calibração de probabilidade e threshold ainda podem melhorar valor de negócio.

## Analise critica por notebook

### Notebook 03 - MLP introdutorio

Status:
- Adequado como baseline didático.

Melhorias sugeridas:
- Adicionar métricas orientadas à classe minoritária (PR-AUC e recall) já neste notebook.
- Incluir avaliação por mais de um threshold para evitar dependência exclusiva de 0.5.

### Notebook 04 - MLP com early stopping

Status:
- Fluxo principal concluído e logging no MLflow funcionando.

Achados relevantes:
- MLP com boa capacidade de ranking (ROC-AUC), mas baixo recall no threshold padrão.

Melhorias sugeridas:
- Testar class weighting na loss para reduzir falsos negativos.
- Testar variações de arquitetura e regularização (dropout, hidden dims, learning rate).
- Registrar também PR-AUC no tracking para comparabilidade direta com notebook 05.

### Notebook 05 - Comparacao com baselines

Status:
- Comparação concluída e estável.

Achados relevantes:
- RandomForest mais consistente no conjunto atual.
- MLP ficou atrás no ranking geral sem ajuste fino de threshold/otimização.

Melhorias sugeridas:
- Rodar busca de hiperparâmetros para MLP e ensembles em protocolo único.
- Incluir intervalo de confiança (bootstrap) para diferenças de métricas.
- Consolidar uma tabela final de comparacao com run_id e model_uri.

### Notebook 06 - Trade-off de custo

Status:
- Análise de custo concluída com reutilização de modelo via MLflow.

Achados relevantes:
- Threshold técnico (F1) e threshold financeiro divergem, o que é esperado.
- Melhor threshold por valor ocorreu em faixa de maior recall (mais agressiva).

Melhorias sugeridas:
- Validar sensibilidade do resultado para diferentes premissas de custo.
- Adicionar analise de cenarios (otimista, base, conservador).
- Fixar política de decisão de threshold por objetivo de negócio.

## Backlog de melhorias priorizado

### Prioridade alta (proximos ciclos)

1. Consolidacao MLflow de todos os modelos (concluída)
- Entregável: tabela única com modelo, run_id, model_uri, parâmetros e métricas finais.
- Critério de pronto: 100% dos modelos do comparativo rastreados e auditáveis.

2. Calibração de threshold orientada a negócio
- Entregável: recomendação oficial de threshold com justificativa técnica e financeira.
- Critério de pronto: decisão documentada com cenários de custo.

3. Validação final do repositório
- Entregável: execução limpa de ruff check e pytest.
- Critério de pronto: sem erros bloqueantes.

### Prioridade media

1. Otimização de hiperparâmetros
- Escopo: MLP e principais ensembles.
- Criterio de pronto: melhoria estatisticamente sustentada em PR-AUC/recall/valor.

2. Robustez estatística da comparação
- Escopo: bootstrap ou repeticao com seeds diferentes.
- Critério de pronto: métricas com faixa de variação reportada.

3. Calibração de probabilidade
- Escopo: Platt scaling ou isotonic (quando aplicavel).
- Critério de pronto: melhora em qualidade de score para decisão de threshold.

### Prioridade baixa

1. Refinamento de visualizações para apresentação executiva.
2. Padronizacao de templates de conclusao em todos os notebooks.
3. Automacao de relatorio final em um artefato unico.

## Plano incremental sugerido

Ciclo 1:
- Item 5 fechado (consolidação MLflow).
- Congelar baseline oficial de comparacao.

Ciclo 2:
- Revisar threshold com cenarios de custo.
- Publicar recomendacao operacional.

Ciclo 3:
- Otimização direcionada de hiperparâmetros.
- Reavaliar modelo final (técnico + financeiro).

## Criterios de sucesso

Tecnico:
- PR-AUC e recall competitivos para classe churn.
- Estabilidade de desempenho em repetições.

Negocio:
- Threshold que maximize valor esperado com premissas explícitas.
- Risco de falso negativo controlado em nível aceitável.

Governanca:
- Runs rastreaveis no MLflow.
- Evidencias de validacao (ruff e pytest) registradas.

## Riscos e mitigacoes

Risco 1: overfitting durante tuning
- Mitigação: validação estratificada, early stopping e comparação em teste cego.

Risco 2: threshold super ajustado a uma premissa única de custo
- Mitigação: análise de sensibilidade por cenários.

Risco 3: divergência entre melhor métrica técnica e melhor valor financeiro
- Mitigação: critério de decisão formal priorizando objetivo de negócio.

## Proxima acao recomendada

Com o item 5 concluído, a próxima ação é revisar threshold com cenários de custo e consolidar a recomendação operacional. Isso reduz retrabalho e direciona as próximas evoluções com rastreabilidade completa.
