# API de Predicao de Churn

Esta API foi criada com FastAPI e reutiliza o pipeline sklearn salvo em `models/sklearn/churn_pipeline.joblib`.

## Endpoints

### GET /health

Retorna status do servico e se o modelo foi carregado.

Resposta esperada:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "models/sklearn/churn_pipeline.joblib"
}
```

### POST /predict

Unico endpoint de inferencia do projeto.

Recebe payload de negocio (compacto) e converte internamente para o formato esperado pelo modelo treinado.

Request:

```json
{
  "city": "Tipton",
  "tenure_months": 2,
  "monthly_charges": 49.25,
  "total_charges": 91.1,
  "gender": "Male",
  "senior_citizen": true,
  "partner": false,
  "dependents": false,
  "phone_service": true,
  "multiple_lines": "No",
  "internet_service": "DSL",
  "online_security": "Yes",
  "online_backup": "No",
  "device_protection": "No",
  "tech_support": "No",
  "streaming_tv": "No",
  "streaming_movies": "No",
  "contract": "Month-to-month",
  "paperless_billing": true,
  "payment_method": "Electronic check",
  "threshold": 0.5
}
```

Campos obrigatorios:

- `tenure_months`
- `monthly_charges`
- `total_charges`

Response:

```json
{
  "prediction": 1,
  "probability_churn": 0.999,
  "threshold": 0.5
}
```

## Swagger

- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/openapi.json

## Observacao

- `city` precisa existir no conjunto de cidades do modelo treinado.
- Se alguma coluna interna esperada pelo modelo nao estiver no payload de negocio, a API completa com `0` durante o alinhamento de features.
