# Relatório de mudanças: CV estratificada, validação Pandera e logging nos CLIs

**Contexto:** Tech Challenge — projeto de previsão de churn (`churn-prediction`).  
**Escopo deste documento:** alterações relacionadas a **StratifiedKFold** no fluxo de treino/avaliação, **contratos Pandera** nos dados e **substituição de `print()` por logging estruturado** nos comandos de linha (`train` e `predict`), incluindo reorganização do módulo de logging compartilhado.

---

## 1. Visão geral

| Área | O que mudou |
|------|-------------|
| Treino / avaliação | De um único split treino/validação dentro dos 80% para **validação cruzada estratificada** nesse bloco; modelo final retreinado em todo o conjunto de desenvolvimento (80%). |
| Dados | **Pandera** valida o DataFrame de treino após leitura do CSV e o DataFrame de inferência antes da predição em lote. |
| CLIs | Saída de resultado via **`logging`** com **JSON em linha única** (mesmo `JsonFormatter` usado pela API), em vez de `print()` + JSON manual. |
| Infra de logging | Implementação central movida para **`churn.logging_config`** para evitar importar o pacote **`churn.api`** (FastAPI) ao rodar apenas os módulos CLI. |

---

## 2. Fluxo de treino com StratifiedKFold

### 2.1 Comportamento anterior (resumo)

- Holdout **80% / 20%** (estratificado) para desenvolvimento vs teste final.
- Dentro dos 80%, novo split **75% / 25%** → aproximadamente **64% treino / 16% validação / 20% teste**.
- Um único pré-processador e uma única sequência de épocas com early stopping na validação fixa.

### 2.2 Comportamento atual

1. **Holdout 20%** para **teste final** permanece igual em espírito: não entra no treino nem na CV; só gera métricas finais honestas.
2. Nos **80%** de desenvolvimento (`x_temp`, `y_temp`):
   - **`StratifiedKFold`** com `shuffle=True` e `random_state` alinhado ao experimento.
   - Número de folds solicitado via parâmetro (CLI: `--cv-folds`; padrão definido em código como **5**).
   - O número efetivo de folds é limitado pelo **tamanho da classe minoritária** (requisito do scikit-learn: a menor classe precisa ter pelo menos `n_splits` amostras). A função interna `_effective_cv_splits` faz `max(2, min(requested, min_class_count))`.
3. **Por fold:**
   - Novo **`Pipeline` sklearn** (`build_mlp_preprocessing_pipeline`) ajustado **somente** nas linhas de treino do fold.
   - Transformação das parcelas de treino e validação do fold.
   - Treino da MLP com **early stopping** na validação do fold (mesma lógica de `NUM_EPOCHS_MAX`, `PATIENCE`, `BCEWithLogitsLoss` com `pos_weight`, etc.).
   - Registro por fold: melhor `val_loss`, número de épocas efetivas, limiar F1 otimizado na grade 0,05–0,95 na validação do fold, e ROC-AUC na validação quando existem **duas classes** na fatia.
4. **Agregação da CV:** médias/desvios das métricas de validação dos folds; **mediana das épocas** e **mediana dos limiares F1** dos folds.
5. **Modelo que vai para o bundle (deploy):**
   - Pré-processador **ajustado em todos os 80%** (`fit` em `x_temp`).
   - MLP treinada em **todos os 80%** transformados por **exatamente** a **mediana das épocas** obtida na CV (**sem** segundo early stopping nessa fase).
   - **`threshold_otimo_f1_validacao`** no bundle passa a refletir a **mediana dos limiares** calculados nos folds (não mais um único split treino/val).
6. **Métricas de teste** (holdout 20%): permanecem no mesmo espírito (acurácia, F1, ROC-AUC, matriz de confusão, etc.), agora com o modelo retreinado conforme acima.

### 2.3 MLflow

Novos ou renomeados parâmetros/métricas relevantes incluem, entre outros:

- `cv_n_splits`, `epochs_trained_final_median_cv`, `threshold_otimo_f1_cv_median`.
- Métricas agregadas da CV: `cv_mean_best_val_loss`, `cv_std_best_val_loss`, `cv_mean_epochs`, `cv_mean_val_roc_auc` (quando aplicável).

### 2.4 Metadados do bundle (`metadata.json` via `extra_metadata`)

Campos adicionais ou com novo significado incluem `n_train_val_total`, `cv_n_splits`, `epochs_trained_final`, `cv_mean_best_val_loss`, `cv_std_best_val_loss`, `cv_mean_epochs`, `cv_mean_val_roc_auc`, etc. (conforme implementado em `train_mlp_flow`).

### 2.5 API pública do módulo de treino

- **`_prepare_splits`** foi **substituída** por **`_split_holdout_test`**, que retorna apenas o par desenvolvimento (80%) / teste (20%) e as colunas de features — não há mais o split interno 75/25 para validação única.

---

## 3. Validação com Pandera

### 3.1 Arquivo novo: `src/churn/data/pandera_schemas.py`

- **`ChurnTrainingDatasetSchema`:** schema de DataFrame com `strict=False`, `coerce=True`, coluna obrigatória **`Churn Value`** (`isin([0, 1])`), colunas numéricas centrais (**Tenure Months**, **Monthly Charges**, **Total Charges**, **CLTV**) com `required=False` para permitir conjuntos mínimos em testes automatizados; quando presentes, validações como não negativas onde aplicável.
- **`ChurnInferenceFeaturesSchema`:** validação de inferência (entrada não vazia; colunas opcionais análogas para não impedir CSVs só com parte das colunas — alinhado ao uso de `FeatureColumnAligner` que preenche faltantes).
- Funções encapsuladas: **`validate_churn_training_dataset`**, **`validate_churn_inference_features`**, ambas usando `validate(..., lazy=True)`.

### 3.2 Onde entram no fluxo

- **Treino:** após `pd.read_csv`, o DataFrame passa por **`validate_churn_training_dataset`** antes dos splits.
- **Predição em lote:** em **`predict_from_csv`**, após ler o CSV, chama-se **`validate_churn_inference_features`** antes de carregar o bundle e predizer.

### 3.3 Testes novos

- Arquivo **`tests/data/test_pandera_churn_schema.py`:** cobre aceitação de contrato mínimo, rejeição de alvo inválido, DataFrame vazio na inferência, colunas extras e colunas parciais.

---

## 4. Logging estruturado nos CLIs

### 4.1 Comportamento

- **`churn.models.train`** (`main`): chama **`configure_logging(level=args.log_level)`** e em seguida **`logger.info("train_cli_completed", extra={...})`**, com campos como `event`, `bundle_dir`, `mlflow_run_id`, `metrics` (objeto aninhado serializável).
- **`churn.models.predict`** (`main`): idem com evento **`predict_cli_completed`** e campos `rows_scored`, `output_path`.
- A saída no terminal é **uma linha JSON por registro**, com timestamp, level, logger, message e todos os campos passados em `extra` (implementação do **`JsonFormatter`**).

### 4.2 Novos argumentos de linha de comando

- Treino: **`--log-level`**, **`--cv-folds`**.
- Predição: **`--log-level`**.

### 4.3 Reorganização: `churn.logging_config`

- **Novo módulo:** **`src/churn/logging_config.py`** — contém **`JsonFormatter`** e **`configure_logging`** (código efetivo, antes concentrado em `api/logging_config.py`).
- **`src/churn/api/logging_config.py`** passou a ser apenas **reexportação** (`from churn.logging_config import JsonFormatter, configure_logging`) para compatibilidade de imports antigos.
- **`src/churn/api/main.py`** importa **`configure_logging`** a partir de **`churn.logging_config`**.
- **Motivo:** importar `churn.models.train` ou `predict` não deve disparar o carregamento de **`churn.api`** (que importa FastAPI/`create_app` via `churn.api.__init__`).

### 4.4 Testes ajustados

- **`tests/models/test_train_flow.py`** e **`tests/models/test_predict_cli.py`:** leem a **última linha** do stdout, fazem **`json.loads`**, e assertam chaves como `bundle_dir` ou `rows_scored` no objeto JSON completo (não mais JSON “puro” sem envelope de log).
- Namespaces simulados em testes passaram a incluir **`log_level`** e, no treino, **`cv_folds`** onde necessário.
- Assertivas em **`_parse_args`** do treino incluem **`cv_folds`** e **`log_level`** padrão.

---

## 5. Outros arquivos tocados (referência rápida)

| Caminho | Alteração |
|---------|-----------|
| `src/churn/models/train.py` | CV estratificada, helpers de treino por fold e treino final por épocas fixas, Pandera, logging, constantes `CV_N_SPLITS_DEFAULT`, CLI estendido. |
| `src/churn/models/predict.py` | Validação Pandera em `predict_from_csv`, logging no `main`, `--log-level`, remoção de `print`/uso direto de `json` para saída. |
| `src/churn/logging_config.py` | **Novo:** formatter JSON e configuração de logging. |
| `src/churn/api/logging_config.py` | Reexportação fina. |
| `src/churn/api/main.py` | Import de `configure_logging` atualizado. |
| `src/churn/data/pandera_schemas.py` | **Novo:** schemas e funções de validação. |
| `tests/api/test_middleware_logging.py` | Import de `JsonFormatter` / `configure_logging` via `churn.logging_config`. |
| `tests/models/test_train_flow.py` | Novos nomes de API interna, asserts de log JSON, `n_cv_splits=2` nos testes de fluxo para manter suíte rápida. |
| `tests/data/test_pandera_churn_schema.py` | **Novo.** |

---

## 6. Observação sobre documentação existente

O arquivo **`docs/OBSERVABILIDADE.md`** ainda pode citar o caminho antigo `src/churn/api/logging_config.py` como “implementação” do formatter. A implementação canônica passou a ser **`src/churn/logging_config.py`**; o arquivo em `api/` mantém apenas compatibilidade de import. Vale atualizar esse doc em uma revisão futura se quiser alinhar caminhos e evitar ambiguidade.

---

## 7. Como reproduzir / validar localmente

- Testes automatizados: `pytest tests` (a partir do ambiente virtual do projeto, com `PYTHONPATH` apontando para `src` se necessário).
- Treino com menos folds (exemplo): `python -m churn.models.train --skip-mlflow --cv-folds 3 ...`
- Predição: saída estruturada visível no stderr/stdout conforme configuração do handler (habitualmente última linha JSON com `event`, `rows_scored`, etc.).

---

*Documento gerado para registro das mudanças de CV estratificada, Pandera e logging nos CLIs.*
