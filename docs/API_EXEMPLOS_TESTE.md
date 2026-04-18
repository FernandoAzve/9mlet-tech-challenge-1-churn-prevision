# API de Predicao de Churn - Guia de Uso e Payloads de Teste

Este documento explica o contrato atual da API (um unico endpoint de predicao) e traz payloads prontos para testar variacoes entre nao churn e churn.

## 1. O que a API faz

A API disponibiliza o modelo de churn treinado no projeto para inferencia online.

Endpoints:

- `GET /health`: status da API e do carregamento do modelo.
- `POST /predict`: predicao de churn com payload de negocio (compacto).

## 2. Parametros aceitos em `POST /predict`

Campos obrigatorios:

- `tenure_months`
- `monthly_charges`
- `total_charges`

Campos opcionais:

- `city`, `zip_code`, `latitude`, `longitude`, `cltv`
- `gender`, `senior_citizen`, `partner`, `dependents`, `phone_service`
- `multiple_lines`, `internet_service`, `online_security`, `online_backup`, `device_protection`, `tech_support`, `streaming_tv`, `streaming_movies`
- `contract`, `paperless_billing`, `payment_method`
- `threshold` (default 0.5, entre 0 e 1)

Payload exemplo:

```json
{
  "city": "Tipton",
  "tenure_months": 2,
  "monthly_charges": 49.25,
  "total_charges": 91.1,
  "gender": "Male",
  "senior_citizen": true,
  "phone_service": true,
  "internet_service": "DSL",
  "contract": "Month-to-month",
  "paperless_billing": true,
  "payment_method": "Electronic check",
  "threshold": 0.5
}
```

Resposta:

```json
{
  "prediction": 1,
  "probability_churn": 0.9990283518238032,
  "threshold": 0.5
}
```

## 3. Respostas comuns

- `200 OK`: predicao realizada.
- `422 Unprocessable Entity`: payload invalido (tipos, valores ou campos fora do contrato).
- `503 Service Unavailable`: modelo nao carregado.

## 4. Payloads prontos para variacao de churn e nao churn

Payloads gerados para o endpoint `POST /predict`:

- [docs/payloads_v2/payload_v2_01_muito_baixo_nao_churn.json](docs/payloads_v2/payload_v2_01_muito_baixo_nao_churn.json)
- [docs/payloads_v2/payload_v2_02_baixo_nao_churn.json](docs/payloads_v2/payload_v2_02_baixo_nao_churn.json)
- [docs/payloads_v2/payload_v2_03_medio_nao_churn.json](docs/payloads_v2/payload_v2_03_medio_nao_churn.json)
- [docs/payloads_v2/payload_v2_04_alto_churn.json](docs/payloads_v2/payload_v2_04_alto_churn.json)
- [docs/payloads_v2/payload_v2_05_muito_alto_churn.json](docs/payloads_v2/payload_v2_05_muito_alto_churn.json)

Faixa esperada com threshold 0.5:

- `payload_v2_01`: probabilidade muito baixa, classe 0
- `payload_v2_02`: probabilidade baixa, classe 0
- `payload_v2_03`: probabilidade media, classe 0
- `payload_v2_04`: probabilidade alta, classe 1
- `payload_v2_05`: probabilidade muito alta, classe 1

Comandos para testar em lote (Git Bash):

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @docs/payloads_v2/payload_v2_01_muito_baixo_nao_churn.json
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @docs/payloads_v2/payload_v2_02_baixo_nao_churn.json
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @docs/payloads_v2/payload_v2_03_medio_nao_churn.json
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @docs/payloads_v2/payload_v2_04_alto_churn.json
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @docs/payloads_v2/payload_v2_05_muito_alto_churn.json
```

## 5. Notacao cientifica na probabilidade

Valor como `5.119691115895446e-10` e valido.

- Esta em notacao cientifica.
- Em decimal, aproximadamente `0.0000000005119691`.
- Continua no intervalo [0, 1].

## 6. Swagger

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/openapi.json`

- [docs/payloads/payload_01_muito_baixo_nao_churn.json](docs/payloads/payload_01_muito_baixo_nao_churn.json)
- [docs/payloads/payload_02_baixo_nao_churn.json](docs/payloads/payload_02_baixo_nao_churn.json)
- [docs/payloads/payload_03_medio_nao_churn.json](docs/payloads/payload_03_medio_nao_churn.json)
- [docs/payloads/payload_04_alto_churn.json](docs/payloads/payload_04_alto_churn.json)
- [docs/payloads/payload_05_muito_alto_churn.json](docs/payloads/payload_05_muito_alto_churn.json)

Faixa esperada de resultado (threshold = 0.5):

- payload_01: probabilidade muito baixa, `prediction = 0`.
- payload_02: probabilidade baixa, `prediction = 0`.
- payload_03: probabilidade media (~0.18), `prediction = 0`.
- payload_04: probabilidade alta (~0.93), `prediction = 1`.
- payload_05: probabilidade muito alta (~0.999), `prediction = 1`.

Comandos de teste:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @docs/payloads/payload_01_muito_baixo_nao_churn.json
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @docs/payloads/payload_02_baixo_nao_churn.json
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @docs/payloads/payload_03_medio_nao_churn.json
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @docs/payloads/payload_04_alto_churn.json
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @docs/payloads/payload_05_muito_alto_churn.json
```

Se estiver usando PowerShell no Windows:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" --data-binary "@docs/payloads/payload_01_muito_baixo_nao_churn.json"
curl.exe -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" --data-binary "@docs/payloads/payload_02_baixo_nao_churn.json"
curl.exe -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" --data-binary "@docs/payloads/payload_03_medio_nao_churn.json"
curl.exe -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" --data-binary "@docs/payloads/payload_04_alto_churn.json"
curl.exe -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" --data-binary "@docs/payloads/payload_05_muito_alto_churn.json"
```
