# API de predição de churn

A API usa **FastAPI** e carrega um único artefato: a **pasta do bundle MLP** gerada pelo treino (`CHURN_MODEL_BUNDLE_DIR`, padrão `models/mlp_bundle`).

Dentro da pasta esperamos:

- `preprocessor.joblib` — `sklearn.pipeline.Pipeline` (limpeza + alinhamento de colunas + `StandardScaler`)
- `mlp_state.pt` — pesos da rede PyTorch
- `metadata.json` — metadados e limiares

## Endpoints

### GET /health

Retorna status do serviço e se o bundle foi carregado.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "D:/.../models/mlp_bundle"
}
```

### POST /predict

Recebe um payload de negócio compacto e monta internamente as colunas one-hot esperadas pelo modelo (com zeros onde faltar).

Campos obrigatórios: `tenure_months`, `monthly_charges`, `total_charges`.

## Swagger

- `http://127.0.0.1:8000/docs`

## Observações

- `city` precisa existir no conjunto de cidades do treino (coluna `City_<nome>`).
- Colunas não enviadas no JSON são preenchidas com `0` no alinhamento.
