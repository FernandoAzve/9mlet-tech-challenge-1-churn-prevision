# API de Predição de Churn - Guia de Uso e Payloads de Teste

Este documento explica o contrato atual da API (um único endpoint de predição) e traz payloads prontos para testar variações entre não churn e churn.

## 1. O que a API faz

A API disponibiliza o modelo de churn treinado no projeto para inferência online.

Endpoints:

- `GET /health`: status da API e do carregamento do modelo.
- `POST /predict`: predição de churn com payload de negócio (compacto).

## 2. Parâmetros aceitos em `POST /predict`

Campos obrigatórios:

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

- `200 OK`: predição realizada.
- `422 Unprocessable Entity`: payload inválido (tipos, valores ou campos fora do contrato).
- `503 Service Unavailable`: modelo não carregado.

## 4. Payloads prontos para variação de churn e não churn

Payloads gerados para o endpoint `POST /predict`:

- [docs/payloads/payload_01_muito_baixo_nao_churn.json](docs/payloads/payload_01_muito_baixo_nao_churn.json)
- [docs/payloads/payload_02_baixo_nao_churn.json](docs/payloads/payload_02_baixo_nao_churn.json)
- [docs/payloads/payload_03_medio_nao_churn.json](docs/payloads/payload_03_medio_nao_churn.json)
- [docs/payloads/payload_04_alto_churn.json](docs/payloads/payload_04_alto_churn.json)
- [docs/payloads/payload_05_muito_alto_churn.json](docs/payloads/payload_05_muito_alto_churn.json)

Faixa esperada de resultado (threshold = 0.5):

- payload_01: probabilidade muito baixa, `prediction = 0`.
- payload_02: probabilidade baixa, `prediction = 0`.
- payload_03: probabilidade média (~0.18), `prediction = 0`.
- payload_04: probabilidade alta (~0.93), `prediction = 1`.
- payload_05: probabilidade muito alta (~0.999), `prediction = 1`.

Comandos de teste (Git Bash):

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

## 5. Notação científica na probabilidade

Valor como `5.119691115895446e-10` é válido.

- Está em notação científica.
- Em decimal, aproximadamente `0.0000000005119691`.
- Continua no intervalo [0, 1].

## 6. Swagger

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/openapi.json`
