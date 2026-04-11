# CHANGELOG da Sessão

Data: 11/04/2026
Projeto: 9mlet-tech-challenge-1-churn-prevision

## Resumo executivo

Nesta sessão, o foco foi estabilizar os notebooks da Etapa 2, corrigir problemas de ambiente/execução, consolidar o registro de experimentos no MLflow (item 5) e atualizar a documentação de acompanhamento.

## Arquivos criados

- docs/MELHORIAS_CONTINUAS_ETAPA2.md
- docs/CHANGELOG.md

## Arquivos alterados

### Dependências

- requirements.txt
  - Removido conflito de versão duplicada do PyTorch.
  - Mantido torch==2.6.0 como versão única.

### Notebooks

- notebooks/04_mlp_training_early_stopping.ipynb
  - Corrigido fluxo de impressão de metadados do run do MLflow (uso consistente do objeto run).
  - Ajustada consistência documental dos splits para 64/16/20.

- notebooks/05_compare_mlp_baselines.ipynb
  - Corrigido problema de execução em Windows (BrokenProcessPool) ajustando n_jobs=1 nos modelos sklearn paralelizáveis.
  - Ajustada narrativa para refletir que a MLP é reutilizada via MLflow a partir do notebook 04.
  - Implementada busca robusta de run no MLflow (por nome preferencial e fallback para run mais recente).
  - Implementada resolução robusta do caminho do mlflow.db.
  - Adicionada etapa de consolidação do item 5 com registro de todos os modelos no experimento unificado:
    - LogisticRegression
    - DecisionTree
    - RandomForest
    - ExtraTrees
    - MLP-PyTorch
  - Gerado consolidado de auditoria com modelo, família, run_id, model_uri e métricas.
  - Ajuste para API atual do MLflow (uso de name em log_model).
  - Reorganização da seção da Etapa 6 para melhor leitura (texto explicativo antes da célula de código).

- notebooks/06_tradeoff_custo_fp_fn.ipynb
  - Ajustada resolução do tracking URI para suportar execução em diretórios diferentes (mlflow.db e ../mlflow.db).

### Documentação

- docs/TODO.md
  - Item 5 atualizado de “em andamento (parcial)” para “concluído”.
  - Escopo atualizado com o experimento consolidado Churn-Etapa2-Comparacao-Modelos.
  - Revisão de texto para refletir manutenção/versionamento do consolidado após conclusão.

- docs/MELHORIAS_CONTINUAS_ETAPA2.md
  - Criado roadmap de melhoria contínua com análise crítica por notebook.
  - Após conclusão do item 5, atualizado para status concluído no tópico de consolidação MLflow.
  - Revisado para pt-BR com acentuação adequada.

## Registros e artefatos de execução

- MLflow
  - Experimento consolidado criado/atualizado: Churn-Etapa2-Comparacao-Modelos.
  - Novos runs registrados para todos os modelos comparados no notebook 05.
  - Artefatos de modelo e metadados gerados automaticamente em notebooks/mlruns/.
  - Banco local atualizado: notebooks/mlflow.db.

## Validações realizadas durante a sessão

- Execução das células críticas dos notebooks com verificação de sucesso.
- Validação de carregamento do modelo MLP via MLflow após correções.
- Validação do registro consolidado do item 5 no MLflow com exibição da tabela final.

## Pendências técnicas (próximo ciclo)

- Executar validação final do repositório:
  - ruff check
  - pytest
- Iniciar melhoria de maior impacto: calibração de threshold orientada a negócio com cenários de custo.

## Observações importantes para sincronização com o grupo

- Houve geração de artefatos automáticos de MLflow em notebooks/mlruns/ e atualização de notebooks/mlflow.db.
- O arquivo notebooks/mlflow_resumo_experimentos_etapa2.csv é o ponto mais direto para auditoria e integração entre os membros.
- O item 5 da Etapa 2 está concluído funcionalmente e documentado.
