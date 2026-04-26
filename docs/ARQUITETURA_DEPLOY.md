# Arquitetura de Deploy — Churn Prevision

## Sumário

- [Decisão Arquitetural](#decisão-arquitetural)
- [Arquitetura Escolhida: Real-Time (Inferência Online)](#arquitetura-escolhida-real-time-inferência-online)
- [Por que não Batch?](#por-que-não-batch)
- [Diagrama de Fluxo](#diagrama-de-fluxo)
- [Stack de Deploy](#stack-de-deploy)
- [Componentes Principais](#componentes-principais)
- [Padrão Bundle de Modelo](#padrão-bundle-de-modelo)
- [Observabilidade](#observabilidade)
- [Trade-offs e Considerações](#trade-offs-e-considerações)

---

## Decisão Arquitetural

| Critério                   | Batch Processing             | **Real-Time REST API ✓**      |
|----------------------------|------------------------------|-------------------------------|
| Latência de resposta        | Alta (horas/dias)            | Baixa (< 100 ms)              |
| Caso de uso principal       | Relatórios periódicos        | Integração em sistemas vivos  |
| Acionabilidade              | Demorada                     | Imediata                      |
| Complexidade de integração  | ETL pipelines, agendadores   | HTTP simples (REST)           |
| Custo de infraestrutura     | Escalável em janelas         | Sempre disponível             |
| Feedback loop               | Offline                      | Online / near real-time       |

**Decisão: Real-Time REST API via FastAPI + Uvicorn.**

---

## Arquitetura Escolhida: Real-Time (Inferência Online)

O sistema expõe um serviço HTTP REST que recebe um payload JSON com os atributos de um cliente e retorna, de forma síncrona e imediata, a probabilidade de churn e a decisão binária.

### Justificativa

1. **Acionabilidade imediata**: em prevenção de churn, a janela de oportunidade para reter um cliente é curta. Uma predição obtida minutos após a detecção de um sinal de risco (nova reclamação, redução de consumo) permite que equipes de retenção ou sistemas de automação de CRM ajam enquanto o cliente ainda está engajado.

2. **Integração nativa com sistemas operacionais**: plataformas de CRM, call centers e aplicações web/mobile consomem APIs REST nativamente. Uma API síncrona elimina a necessidade de pipelines ETL intermediários, reduzindo a complexidade operacional.

3. **Threshold configurável por requisição**: o campo `threshold` (padrão `0.5`) é parte do payload da requisição, permitindo que sistemas consumidores ajustem o ponto de corte conforme o contexto (ex.: campanhas de alto custo exigem threshold mais conservador), sem necessidade de redesploy.

4. **Modelo leve o suficiente para latência online**: a rede MLP com arquitetura `input → Linear(64) → ReLU → Linear(32) → ReLU → Linear(1)` e pré-processamento sklearn é pequena e executa inferência em milissegundos, viabilizando o modelo real-time sem hardware especializado.

5. **Escalabilidade horizontal simples**: o serviço é stateless — o bundle do modelo é carregado em memória no startup. Múltiplas instâncias podem ser iniciadas atrás de um load balancer sem coordenação de estado.

---

## Por que não Batch?

O processamento em batch seria adequado se:
- As predições fossem consumidas apenas em relatórios periódicos (diário/semanal).
- O volume de dados por execução fosse muito alto (milhões de registros simultâneos).
- Não houvesse necessidade de integração em tempo real com sistemas operacionais.

Para o contexto deste projeto (prevenção de churn em telecom), nenhum desses critérios se aplica como requisito primário. O utilitário `predict_from_csv` (via CLI) existe como ferramenta auxiliar para inferência em lote, mas não constitui o modo de deploy principal.

---

## Diagrama de Fluxo

```
┌─────────────────────────────────────────────────────────────────┐
│                        Cliente / Sistema CRM                     │
└────────────────────────────┬────────────────────────────────────┘
                             │ POST /predict
                             │ { customer_data, threshold }
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI + Uvicorn                            │
│                                                                  │
│  ┌──────────────────┐    ┌────────────────────────────────────┐  │
│  │  Middleware       │    │  Route: /predict                   │  │
│  │  LatencyLogging  │───▶│  _build_v2_features()              │  │
│  │  X-Request-ID    │    │  Validação Pydantic v2             │  │
│  └──────────────────┘    └──────────────┬─────────────────────┘  │
│                                         │                         │
│                          ┌──────────────▼─────────────────────┐  │
│                          │  ChurnPredictor (mlp_bundle.py)     │  │
│                          │                                     │  │
│                          │  preprocessor.joblib (sklearn)      │  │
│                          │    └─ StandardScaler + OHE         │  │
│                          │                                     │  │
│                          │  mlp_state.pt (PyTorch MLP)         │  │
│                          │    └─ Linear(64)→ReLU→             │  │
│                          │       Linear(32)→ReLU→             │  │
│                          │       Linear(1)→Sigmoid            │  │
│                          │                                     │  │
│                          │  metadata.json                      │  │
│                          │    └─ threshold, features, versão  │  │
│                          └──────────────┬─────────────────────┘  │
└─────────────────────────────────────────┼───────────────────────┘
                                          │
                             ┌────────────▼────────────┐
                             │  Response JSON           │
                             │  {                       │
                             │    prediction: 0 | 1,    │
                             │    probability_churn,    │
                             │    threshold             │
                             │  }                       │
                             └─────────────────────────┘
```

---

## Stack de Deploy

| Camada              | Tecnologia          | Versão    |
|---------------------|---------------------|-----------|
| Servidor HTTP       | Uvicorn (ASGI)      | 0.34      |
| Framework API       | FastAPI             | 0.115     |
| Validação de schema | Pydantic v2         | 2.11.4    |
| Modelo ML           | PyTorch (MLP)       | 2.6.0     |
| Pré-processamento   | scikit-learn        | 1.7.2     |
| Serialização        | joblib              | 1.5.0     |
| Rastreamento        | MLflow              | 3.10.1    |
| Linguagem           | Python              | ≥ 3.9     |

### Comando de execução

```bash
PYTHONPATH=src uvicorn churn.api.main:app --host 0.0.0.0 --port 8000
```

---

## Componentes Principais

### `src/churn/api/main.py`
Ponto de entrada FastAPI com `lifespan` context manager. Carrega o bundle do modelo no startup. Se o bundle não for encontrado, a API sobe mas `/predict` retorna **HTTP 503** (degradação graciosa).

### `src/churn/api/routes.py`
Define os endpoints:
- `GET /health` — verificação de saúde com status do modelo
- `POST /predict` — inferência síncrona por cliente

A função `_build_v2_features()` constrói vetores one-hot a partir de campos de negócio, preenchendo colunas ausentes com `0` para compatibilidade com o espaço de features do treino.

### `src/churn/api/middleware.py`
`LatencyLoggingMiddleware`: injeta `X-Request-ID` em toda requisição e emite logs JSON estruturados com método, rota, status HTTP e latência em ms.

### `src/churn/api/schemas.py`
Schemas Pydantic v2 para request/response. Validação tipada de todos os campos de entrada.

### `src/churn/models/mlp_bundle.py`
`ChurnPredictor`: carrega o bundle do diretório configurado via `CHURN_MODEL_BUNDLE_DIR`, executa a pipeline `preprocessor → MLP` e retorna `(prediction, probability)`.

---

## Padrão Bundle de Modelo

O modelo é versionado e distribuído como um diretório `mlp_bundle/` com três artefatos:

```
models/mlp_bundle/
├── preprocessor.joblib   # sklearn Pipeline (StandardScaler + alinhamento OHE)
├── mlp_state.pt          # state_dict PyTorch da MLP
└── metadata.json         # threshold padrão, lista de features, versão do bundle
```

O caminho do bundle é configurável via variável de ambiente `CHURN_MODEL_BUNDLE_DIR`, permitindo diferentes bundles por ambiente (dev, staging, prod) sem alteração de código.

---

## Observabilidade

- **Logs estruturados (JSON)**: toda requisição emite um log com `request_id`, `method`, `path`, `status_code` e `duration_ms`.
- **`GET /health`**: endpoint dedicado para health checks de liveness/readiness.
- **MLflow**: rastreamento de experimentos de treinamento (métricas, hiperparâmetros, hash do dataset, artefatos), possibilitando auditoria e reprodutibilidade.

---

## Trade-offs e Considerações

| Aspecto                  | Decisão tomada                       | Impacto                                      |
|--------------------------|--------------------------------------|----------------------------------------------|
| Inferência unitária       | Um cliente por request               | Simples, baixa latência, menor throughput    |
| Batch endpoint            | Ausente (apenas CLI auxiliar)        | Limitação para integrações em volume         |
| Sem Docker (atual)        | Deploy direto com Uvicorn            | Facilita desenvolvimento, frágil em produção|
| Threshold no payload      | Configurável por chamada             | Flexibilidade máxima sem redesploy           |
| Estado do modelo          | Carregado em memória no startup      | Latência zero na inferência, uso fixo de RAM |
| Escalabilidade horizontal | Stateless — múltiplas instâncias OK  | Necessita load balancer externo              |
