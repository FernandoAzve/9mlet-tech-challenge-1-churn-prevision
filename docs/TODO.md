# TODO — Roadmap em notebooks (Churn / PyTorch / MLflow)

## Objetivo deste documento

Registrar o plano de trabalho do projeto em **notebooks Jupyter** (`notebooks/`), mantendo os **mesmos parâmetros e convenções** da fase já desenvolvida (notebook `03_mlp_pytorch.ipynb` e anteriores):

- Dados: `Telco_customer_churn_ready.csv` (mesmos caminhos relativos e checagens de existência do arquivo).
- Estilo: visão de **estudante de ML** — código didático, sem excesso de complexidade.
- **Comentários em pt-BR** em todos os blocos de código novos.
- Ambiente: dependências no **`requirements.txt`**; instalação apenas no **`.venv`** (ativar o virtual e `pip install -r requirements.txt`; criação com `python -m venv .venv` ou `virtualenv .venv` conforme preferência).
- Novas bibliotecas: atualizar `requirements.txt` **e** alinhar com a equipe / política do repositório (`.github/libs/allowed-libs.md`).
- Alinhamento com métricas e negócio descritos em `docs/METRICAS.md`, quando aplicável.

---

## Itens

### 1. Construir MLP em PyTorch: arquitetura, função de ativação, loss function

| Campo          | Valor                                                                                                                                |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Status**     | Concluído (fase atual)                                                                                                               |
| **Entregável** | `notebooks/03_mlp_pytorch.ipynb`                                                                                                     |
| **Conteúdo**   | MLP (`Linear` + **ReLU** + saída logit), **`BCEWithLogitsLoss`**, `DataLoader` com batching básico, treino e avaliação introdutória. |

---

### 2. Implementar loop de treinamento com early stopping e batching

| Campo          | Valor                                                                                                                                                                                                                                                                                                                        |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**     | Concluído                                                                                                                                                                                                                                                                                                                    |
| **Entregável** | Novo notebook Jupyter em `notebooks/` (sugestão de nome: `04_mlp_training_early_stopping.ipynb`)                                                                                                                                                                                                                             |
| **Escopo**     | Refinar ou estender o fluxo do MLP: loop de treino explícito com **validação**, **early stopping** (ex.: paciência sobre métrica ou loss de validação) e **batching** já organizado via `DataLoader` (manter padrão reprodutível: `SEED`, split treino/val/teste ou treino/val a partir do treino, sem vazamento do scaler). |

---

### 3. Comparar MLP vs. baselines (lineares + árvores) usando ≥ 4 métricas

| Campo          | Valor                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**     | Concluído                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| **Entregável** | Novo notebook Jupyter em `notebooks/` (sugestão: `05_compare_mlp_baselines.ipynb`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **Escopo**     | Comparar o MLP (PyTorch) com **baselines lineares** (ex.: regressão logística, como no `02_baseline_dummy_logreg.ipynb`) e **modelos baseados em árvores** (ex.: Random Forest ou Gradient Boosting, desde que permitidos em `allowed-libs`). Calcular e reportar **pelo menos quatro métricas** (ex.: acurácia, precisão, recall, F1, ROC-AUC, PR-AUC — escolher um conjunto coerente com `docs/METRICAS.md`). Mesmo dataset tratado e, quando possível, mesmo split / semente para comparação justa. No estado atual, a MLP é carregada via MLflow a partir do notebook `04`. |

---

### 4. Analisar trade-off de custo (falso positivo vs. falso negativo)

| Campo          | Valor                                                                                                                                                                                                                                               |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**     | Concluído                                                                                                                                                                                                                                           |
| **Entregável** | `notebooks/06_tradeoff_custo_fp_fn.ipynb`                                                                                                                                                                                                           |
| **Escopo**     | Implementado com reuso de modelos já treinados/registrados no MLflow, varredura de limiar, curvas de métricas e simulação financeira de custo FP/FN (alinhado a `docs/METRICAS.md`), incluindo recomendação de limiar orientada a valor de negócio. |

---

### 5. Registrar todos os experimentos (MLP e ensembles) no MLflow

| Campo          | Valor                                                                                                                                                                                                                                                                |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**     | Concluído                                                                                                                                                                                                                                                            |
| **Entregável** | Integrar MLflow nos notebooks relevantes **ou** um notebook dedicado (sugestão: `07_mlflow_experimentos_mlp_ensembles.ipynb`) que reproduza/registre runs de forma consolidada — desde que **todos** os experimentos de MLP e ensembles fiquem rastreados no MLflow. |
| **Escopo**     | Tracking consolidado no notebook `05_compare_mlp_baselines.ipynb`, com registro de MLP, lineares e ensembles no experimento `Churn-Etapa2-Comparacao-Modelos`, incluindo `run_id`, `model_uri`, parâmetros e métricas em resumo único de auditoria.                  |

---

## Revisão técnica final (11/04/2026)

### Situação por notebook

- `03_mlp_pytorch.ipynb`: executa corretamente e cumpre o objetivo didático de arquitetura/loss/treino básico.
- `04_mlp_training_early_stopping.ipynb`: fluxo técnico concluído; ajuste aplicado para evitar erro ao imprimir `run_id` fora do contexto do run MLflow.
- `05_compare_mlp_baselines.ipynb`: comparação concluída e estável (ajuste em `n_jobs=1` no Windows e alinhamento da documentação com o reuso da MLP do notebook 04 via MLflow).
- `06_tradeoff_custo_fp_fn.ipynb`: análise de trade-off concluída com reuso de modelo; ajuste aplicado para resolver o caminho do `mlflow.db` de forma robusta.

### Conclusão da tarefa da Etapa 2

- Itens 1, 2, 3 e 4: **concluídos**.
- Item 5: **concluído** (tracking consolidado dos modelos do comparativo em resumo único de auditoria).

### Melhorias recomendadas (críticas)

1. Evoluir o consolidado final de runs do MLflow com versionamento de dataset e assinatura de schema (modelo, `run_id`, `model_uri`, parâmetros e métricas finais).
2. Padronizar os experimentos do MLflow para evitar dependência de nomes de run específicos.
3. Executar validação final do repositório (`ruff check` e `pytest`) antes do fechamento definitivo.

---

## Regras para desenvolvimento assíncrono

- Cada notebook deve ser capaz de rodar de forma independente, sem exigir execução anterior de outro notebook.
- Usar o mesmo `SEED`, caminhos de dados (`Telco_customer_churn_ready.csv`) e lógica de split treino/val/teste em todos os notebooks.
- Implementar o pré-processamento de forma replicável: ajustar scaler apenas no treino e aplicar em validação/teste.
- Documentar no início de cada notebook as configurações principais usadas (`SEED`, splits, colunas usadas, validação, métricas).
- Se precisar reaproveitar um modelo treinado em outro notebook, exportá-lo e documentar o formato/versão; preferir treinar o mesmo modelo localmente para evitar dependência direta.
- Manter as comparações justas com o mesmo dataset tratado e o mesmo esquema de pré-processamento.

---

## Ordem sugerida

1 → **já feito** · 2 → 3 → 4 → 5 (o item 5 pode ser aplicado incrementalmente a cada notebook novo ou fechado em um notebook único ao final, desde que nada fique de fora do MLflow).

---

## Próximos passos (fechamento da Etapa 2)

Para manter a Etapa 2 operacionalmente robusta após a conclusão, executar os passos abaixo:

1. **Manter e versionar o consolidado MLflow dos modelos comparados**
	- Preservar runs completos e reproduzíveis para os modelos do comparativo do notebook `05` (MLP, lineares e árvores/ensembles).
	- Atualizar o resumo de auditoria sempre que houver novo treinamento ou ajuste de hiperparâmetros.

2. **Validar execução ponta a ponta dos notebooks da Etapa 2**
	- Rodar os notebooks `03`, `04`, `05` e `06` em sequência com ambiente limpo.
	- Confirmar que o notebook `06` reutiliza modelo já registrado no MLflow (sem retreino).

3. **Fechar entregável final para apresentação**
	- Consolidar tabela comparativa de modelos (métricas + referência de run no MLflow).
	- Definir explicitamente modelo final e threshold de negócio recomendado.

4. **Executar validação mínima do repositório (obrigatória)**
	- `ruff check`
	- `pytest`

---

## Documentação relacionada

- `docs/METRICAS.md` — métricas e custo de negócio  
- `.github/libs/allowed-libs.md` — dependências permitidas  
- `.github/context/tech-stack.md` — stack oficial do desafio  
