# Observabilidade da API — Logging Estruturado e Middleware de Latência

## 1. Objetivo e escopo

Esta documentação descreve a implementação de observabilidade básica da API de predição de churn, cobrindo:

| Componente | Arquivo | Responsabilidade |
|---|---|---|
| **Logging estruturado** | `src/churn/api/logging_config.py` | Formata todos os logs como JSON de linha única |
| **Middleware de latência** | `src/churn/api/middleware.py` | Mede duração, atribui ID de correlação e emite log por requisição |
| **Inicialização** | `src/churn/api/main.py` | Configura logging e registra o middleware na aplicação |

---

## 2. Logging estruturado (`logging_config.py`)

### Como funciona

Ao iniciar a aplicação (`create_app()`), a função `configure_logging()` substitui o handler padrão do Python pelo `JsonFormatter`, que serializa cada `LogRecord` como um objeto JSON de linha única enviado para `stdout`.

### Formato de saída

Todo log emitido pela aplicação segue o esquema:

```json
{
  "timestamp": "2026-04-19T14:00:01.123456+00:00",
  "level": "INFO",
  "logger": "churn.api.middleware",
  "message": "request_finished",
  "request_id": "f0e1d2c3-a1b2-4c3d-8e5f-9a0b1c2d3e4f",
  "method": "POST",
  "route": "/predict",
  "status_code": 200,
  "duration_ms": 12.847
}
```

### Campos do log

| Campo | Origem | Descrição |
|---|---|---|
| `timestamp` | automático | Horário UTC em formato ISO 8601 |
| `level` | automático | Nível do log (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |
| `logger` | automático | Nome do logger Python que emitiu o registro |
| `message` | automático | Texto da mensagem |
| Campos extras | `extra={...}` | Qualquer chave passada via `logger.info(..., extra={...})` |

### Configuração do nível de log

O nível padrão é `INFO`. Para alterar, passe a variável de ambiente `LOG_LEVEL` antes de iniciar a aplicação ou ajuste a chamada em `main.py`:

```bash
LOG_LEVEL=DEBUG uvicorn churn.api.main:app --reload
```

> Loggers ruidosos de terceiros (`uvicorn.access`, `httpx`) são silenciados automaticamente para `WARNING`.

---

## 3. Middleware de latência (`middleware.py`)

### Como funciona

O `LatencyLoggingMiddleware` envolve cada requisição HTTP antes e depois do roteamento FastAPI. Para cada requisição:

1. Lê o header `X-Request-ID` enviado pelo cliente; se ausente, gera um UUID v4 novo.
2. Registra o instante de início com `time.perf_counter()` (alta resolução, sem drift).
3. Chama o próximo handler (rota FastAPI).
4. Calcula a duração em milissegundos.
5. Emite um log `INFO` com os metadados da requisição (veja campos abaixo).
6. Adiciona os headers `X-Request-ID` e `X-Process-Time-Ms` na resposta.

Em caso de exceção não tratada, o middleware ainda emite um log de erro com `status_code: 500` antes de re-lançar a exceção.

### Headers de resposta

| Header | Exemplo | Descrição |
|---|---|---|
| `X-Request-ID` | `f0e1d2c3-...` | ID de correlação da requisição |
| `X-Process-Time-Ms` | `12.847` | Tempo de processamento em ms |

### Campos extras no log de requisição

| Campo | Tipo | Descrição |
|---|---|---|
| `request_id` | `string` | UUID de correlação da requisição |
| `method` | `string` | Método HTTP (`GET`, `POST`, etc.) |
| `route` | `string` | Caminho da URL (ex: `/predict`, `/health`) |
| `status_code` | `int` | Código HTTP da resposta |
| `duration_ms` | `float` | Duração total em milissegundos |

### Rastreabilidade com `X-Request-ID`

O header pode ser enviado pelo cliente para propagar o ID de correlação de sistemas externos:

```bash
curl -X POST http://localhost:8000/predict \
  -H "X-Request-ID: meu-id-de-rastreio-123" \
  -H "Content-Type: application/json" \
  -d @docs/payloads/payload_04_alto_churn.json
```

O mesmo ID aparece no log e na resposta, permitindo correlacionar logs de sistemas distribuídos.

---

## 4. Integração com `main.py`

O fluxo de inicialização da aplicação é:

```
create_app()
  ├── configure_logging()          ← configura JsonFormatter no logger raiz
  ├── FastAPI(lifespan=lifespan)   ← carrega o modelo sklearn
  ├── add_middleware(LatencyLoggingMiddleware)
  └── include_router(router)       ← registra /health e /predict
```

A ordem importa: `configure_logging()` é chamada antes de qualquer outro log para garantir que até os logs do ciclo de vida (`lifespan`) já saiam em formato JSON.

---

## 5. Exemplo de saída completa no terminal

Inicializando a API e fazendo uma predição:

```
{"timestamp": "2026-04-19T14:00:00.001Z", "level": "INFO", "logger": "churn.api.main", "message": "Model loaded successfully from models/sklearn/churn_pipeline.joblib"}
{"timestamp": "2026-04-19T14:00:05.312Z", "level": "INFO", "logger": "churn.api.middleware", "message": "request_finished", "request_id": "f0e1d2c3-a1b2-4c3d-8e5f-9a0b1c2d3e4f", "method": "POST", "route": "/predict", "status_code": 200, "duration_ms": 12.847}
{"timestamp": "2026-04-19T14:00:06.001Z", "level": "INFO", "logger": "churn.api.middleware", "message": "request_finished", "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890", "method": "GET", "route": "/health", "status_code": 200, "duration_ms": 0.531}
```

---

## 6. Testes automatizados

Os testes de observabilidade estão em `tests/api/test_api.py` e cobrem:

| Teste | O que valida |
|---|---|
| `test_middleware_adiciona_headers_de_rastreabilidade` | Toda resposta contém `X-Request-ID` e `X-Process-Time-Ms` |
| `test_middleware_propaga_request_id_do_cliente` | ID enviado pelo cliente é refletido sem alteração na resposta |
| `test_middleware_gera_request_id_quando_ausente` | UUID gerado automaticamente tem formato correto (36 caracteres) |
| `test_middleware_process_time_e_numero_valido` | `X-Process-Time-Ms` é um número `>= 0` |

Para executar:

```bash
PYTHONPATH=src pytest tests/api/test_api.py -v
```

---

# 7. Como testar manualmente (Windows PowerShell)
 
## Opção 1 — Rodar a API localmente e validar manualmente
 
### 1. Entre na pasta do projeto
 
```powershell
cd C:\Users\User\Documents\Github\9mlet-tech-challenge-1-churn-prevision
```
 
### 2. Ative o ambiente virtual
 
```powershell
.\.venv\Scripts\activate
```
 
### 3. Defina o `PYTHONPATH`
 
```powershell
$env:PYTHONPATH="src"
```
 
### 4. Suba a API
 
```powershell
uvicorn churn.api.main:app --reload
```
 
### 5. Em outro terminal, envie uma predição
 
```powershell
cd C:\Users\User\Documents\Github\9mlet-tech-challenge-1-churn-prevision
.\.venv\Scripts\activate
 
curl.exe -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  -H "X-Request-ID: teste-123" `
  --data-binary "@docs/payloads/payload_04_alto_churn.json" |
python -m json.tool
```
 
### Resultado esperado
 
No terminal do `uvicorn`, você verá um log JSON estruturado semelhante a:
 
```json
{
  "message": "request_finished",
  "request_id": "teste-123",
  "method": "POST",
  "route": "/predict",
  "status_code": 200,
  "duration_ms": 12.847
}
```
 
### 6. Validar headers de observabilidade
 
```powershell
curl.exe -i "http://localhost:8000/health"
```
 
Verifique no retorno:
 
```
X-Request-ID: ...
X-Process-Time-Ms: ...
```
 
---
 
# 8. Como rodar os testes automatizados (Windows PowerShell)
 
```powershell
cd C:\Users\User\Documents\Github\9mlet-tech-challenge-1-churn-prevision
.\.venv\Scripts\activate
$env:PYTHONPATH="src"
pytest tests/api/test_api.py -v
```
 
---

## 7. Documentação relacionada

- [docs/API.md](API.md) — Contrato dos endpoints `/health` e `/predict`
- [docs/TODO_ETAPA_3.MD](TODO_ETAPA_3.MD) — Item 5 (logging estruturado e middleware de latência)
- `src/churn/api/logging_config.py` — Implementação do formatter JSON
- `src/churn/api/middleware.py` — Implementação do middleware de latência
- `src/churn/api/middleware.py` — Implementação do middleware de latência

