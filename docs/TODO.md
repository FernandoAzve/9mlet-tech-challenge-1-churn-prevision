# TODO — Roadmap em notebooks (Churn / PyTorch / MLflow)

## Objetivo deste documento

Registrar o plano de trabalho do projeto em **notebooks Jupyter** (`notebooks/`), mantendo os **mesmos parâmetros e convenções** da fase já desenvolvida (notebook `03_mlp_pytorch.ipynb` e anteriores):

- Dados: `Telco_customer_churn_ready.csv` (mesmos caminhos relativos e checagens de existência do arquivo).
- Estilo: visão de **estudante de ML** — código didático, sem excesso de complexidade.
- **Comentários em pt-BR** em todos os blocos de código novos.
- Ambiente: dependências no **`requirements.txt`**; instalação apenas no **`.venv`** (ativar o virtual e `pip install -r requirements.txt`; criação com `python -m venv .venv` ou `virtualenv .venv` conforme preferência).
- Novas bibliotecas: atualizar `requirements.txt` **e** alinhar com a equipe / política do repositório (`.cursor/libs/allowed-libs.md`).
- Alinhamento com métricas e negócio descritos em `docs/METRICAS.md`, quando aplicável.

---

## Itens

### 1. Construir MLP em PyTorch: arquitetura, função de ativação, loss function

| Campo | Valor |
|--------|--------|
| **Status** | Concluído (fase atual) |
| **Entregável** | `notebooks/03_mlp_pytorch.ipynb` |
| **Conteúdo** | MLP (`Linear` + **ReLU** + saída logit), **`BCEWithLogitsLoss`**, `DataLoader` com batching básico, treino e avaliação introdutória. |

---

### 2. Implementar loop de treinamento com early stopping e batching

| Campo | Valor |
|--------|--------|
| **Status** | Pendente |
| **Entregável** | Novo notebook Jupyter em `notebooks/` (sugestão de nome: `04_mlp_training_early_stopping.ipynb`) |
| **Escopo** | Refinar ou estender o fluxo do MLP: loop de treino explícito com **validação**, **early stopping** (ex.: paciência sobre métrica ou loss de validação) e **batching** já organizado via `DataLoader` (manter padrão reprodutível: `SEED`, split treino/val/teste ou treino/val a partir do treino, sem vazamento do scaler). |

---

### 3. Comparar MLP vs. baselines (lineares + árvores) usando ≥ 4 métricas

| Campo | Valor |
|--------|--------|
| **Status** | Pendente |
| **Entregável** | Novo notebook Jupyter em `notebooks/` (sugestão: `05_compare_mlp_baselines.ipynb`) |
| **Escopo** | Comparar o MLP (PyTorch) com **baselines lineares** (ex.: regressão logística, como no `02_baseline_dummy_logreg.ipynb`) e **modelos baseados em árvores** (ex.: Random Forest ou Gradient Boosting, desde que permitidos em `allowed-libs`). Calcular e reportar **pelo menos quatro métricas** (ex.: acurácia, precisão, recall, F1, ROC-AUC, PR-AUC — escolher um conjunto coerente com `docs/METRICAS.md`). Mesmo dataset tratado e, quando possível, mesmo split / semente para comparação justa. |

---

### 4. Analisar trade-off de custo (falso positivo vs. falso negativo)

| Campo | Valor |
|--------|--------|
| **Status** | Pendente |
| **Entregável** | Novo notebook Jupyter em `notebooks/` (sugestão: `06_tradeoff_custo_fp_fn.ipynb`) |
| **Escopo** | Usar matriz de confusão e/ou curvas (ROC ou PR) para discutir **custo de FP vs. FN** no contexto de churn (ex.: custos relativos ou cenários hipotéticos, alinhados à narrativa de negócio em `METRICAS.md`). Pode incluir varredura de **limiar** de decisão e visualizações simples (matplotlib/seaborn já no projeto). |

---

### 5. Registrar todos os experimentos (MLP e ensembles) no MLflow

| Campo | Valor |
|--------|--------|
| **Status** | Pendente |
| **Entregável** | Integrar MLflow nos notebooks relevantes **ou** um notebook dedicado (sugestão: `07_mlflow_experimentos_mlp_ensembles.ipynb`) que reproduza/registre runs de forma consolidada — desde que **todos** os experimentos de MLP e ensembles fiquem rastreados no MLflow. |
| **Escopo** | Seguir o padrão do `02_baseline_dummy_logreg.ipynb` (tracking de parâmetros, métricas, artefatos quando fizer sentido). Garantir rastreabilidade dos modelos comparados nos itens 2–4 (MLP, baselines, árvores/ensembles). |

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

## Documentação relacionada

- `docs/METRICAS.md` — métricas e custo de negócio  
- `.cursor/libs/allowed-libs.md` — dependências permitidas  
- `.cursor/context/tech-stack.md` — stack oficial do desafio  
