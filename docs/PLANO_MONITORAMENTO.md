# Plano de Monitoramento — API e Modelo de Churn

## 1. Finalidade do documento

Este documento estabelece o padrão de monitoramento para a API de predição de churn e para o ciclo de vida do modelo em produção. O objetivo é garantir previsibilidade operacional, resposta rápida a incidentes e governança sobre mudanças que impactam decisões de retenção.

O plano cobre três frentes:

- Métricas de serviço, modelo e qualidade técnica.
- Política de alertas com critérios de severidade.
- Procedimento de resposta e recuperação (playbook).

Premissas de referência:

- Stack oficial do projeto: Python, PyTorch, sklearn, MLflow, FastAPI, pytest e ruff.
- Operação orientada por evidências, com aprovação humana para decisões de negócio e produção.

---

## 2. Diretrizes operacionais

1. O monitoramento deve priorizar risco de indisponibilidade, degradação de desempenho e regressão de qualidade preditiva.
2. Sinais nativos da solução são a primeira fonte de observabilidade: logs estruturados, endpoint de saúde, métricas de execução e rastreabilidade em MLflow.
3. Limites e thresholds são vivos: começam conservadores e evoluem com histórico operacional.
4. Segurança de dados é mandatória: logs e alertas não devem expor dados pessoais.
5. A automação apoia o diagnóstico, mas decisões que alteram política de negócio permanecem sob governança humana.

---

## 3. Escopo monitorado

### 3.1 Serviço de inferência (FastAPI)

- Disponibilidade dos endpoints `/health` e `/predict`.
- Latência de ponta a ponta por rota.
- Taxa de erro por classe de status HTTP (4xx e 5xx).
- Ocorrências de exceção não tratada (`unhandled_exception`).

### 3.2 Comportamento do modelo

- Estado de carregamento do bundle MLP no ciclo de vida da aplicação.
- Distribuição de probabilidade predita de churn.
- Taxa de classificação positiva.
- Perfil de uso do threshold informado nas requisições.

### 3.3 Qualidade de entrada

- Taxa de payload inválido (422).
- Principais padrões de rejeição de contrato.
- Sinais de desvio de distribuição (drift) entre entrada atual e perfil de treino.

### 3.4 Qualidade contínua de engenharia

- Conformidade com `ruff check`.
- Estabilidade de `pytest`.
- Acompanhamento das métricas técnicas do modelo no MLflow (PR-AUC, F1 e ROC-AUC).

---

## 4. Catálogo de métricas

### 4.1 Métricas de serviço (online)

| Métrica | Tipo | Fonte | Objetivo |
|---|---|---|---|
| `api_health_status` | Gauge | `GET /health` | Confirmar disponibilidade do serviço |
| `api_model_loaded` | Gauge | `GET /health` (`model_loaded`) | Detectar indisponibilidade de artefato |
| `api_request_count_total` | Counter | Logs `request_finished` | Medir volume por rota/método |
| `api_request_latency_ms_p50` | Percentil | Logs `duration_ms` | Acompanhar latência mediana |
| `api_request_latency_ms_p95` | Percentil | Logs `duration_ms` | Monitorar degradação de cauda |
| `api_request_latency_ms_p99` | Percentil | Logs `duration_ms` | Identificar extremos de latência |
| `api_http_4xx_rate` | Taxa | Logs `status_code` | Sinalizar problemas de contrato de entrada |
| `api_http_5xx_rate` | Taxa | Logs `status_code` + exceções | Sinalizar falhas de serviço |
| `api_unhandled_exception_count` | Counter | Logs `unhandled_exception` | Quantificar falhas críticas |

### 4.2 Métricas de inferência e dados (online)

| Métrica | Tipo | Fonte | Objetivo |
|---|---|---|---|
| `model_prediction_positive_rate` | Taxa | Resposta `/predict` (`prediction`) | Detectar mudança abrupta de comportamento |
| `model_probability_churn_mean` | Média | Resposta `/predict` (`probability_churn`) | Monitorar deslocamento de risco médio |
| `model_probability_churn_std` | Desvio padrão | Resposta `/predict` | Detectar colapso ou saturação de score |
| `model_threshold_usage` | Distribuição | Request `/predict` (`threshold`) | Validar coerência de uso de limiar |
| `api_validation_error_rate_422` | Taxa | Logs de resposta 422 | Medir integridade de integração com consumidores |
| `api_city_unknown_rate` | Taxa | HTTP 422 por cidade não suportada | Identificar incompatibilidade com base de treino |

### 4.3 Métricas de qualidade do modelo (offline)

| Métrica | Tipo | Fonte | Objetivo |
|---|---|---|---|
| `train_pr_auc` | Score | MLflow | Indicador principal para classe desbalanceada |
| `train_f1` | Score | MLflow | Balancear precisão e recall |
| `train_roc_auc` | Score | MLflow | Medir separabilidade global |
| `threshold_business_score` | Score | Notebook/relatório de custo | Apoiar decisão de limiar por custo FP/FN |
| `model_bundle_freshness_days` | Idade em dias | `metadata.json` + data de release | Controlar envelhecimento de artefato |

### 4.4 Métricas de engenharia (pipeline de qualidade)

| Métrica | Tipo | Fonte | Objetivo |
|---|---|---|---|
| `ci_ruff_check_status` | Binária | CI local/GitHub Actions | Garantir padrão de qualidade de código |
| `ci_pytest_status` | Binária | CI local/GitHub Actions | Evitar regressões funcionais |
| `ci_test_duration_seconds` | Tempo | CI | Detectar aumento anormal no tempo de validação |

---

## 5. SLOs e limites iniciais

As metas abaixo representam a linha de base operacional para a fase atual e devem ser recalibradas após consolidação de histórico.

| Indicador | Meta (SLO) | Janela |
|---|---|---|
| Disponibilidade de `/health` | >= 99,0% | 7 dias |
| `model_loaded=true` em produção | >= 99,5% | 7 dias |
| Latência p95 de `/predict` | <= 300 ms | 1 dia |
| Taxa de erro 5xx em `/predict` | <= 1,0% | 1 hora |
| Taxa de erro 422 em `/predict` | <= 10,0% | 1 dia |

Nota operacional: crescimento de 422 indica, em regra, desvio de integração de consumidores com o contrato da API e deve ser tratado como prioridade de interface.

---

## 6. Política de alertas

### 6.1 Classificação de severidade

- `SEV1`: indisponibilidade ou falha crítica com impacto direto no atendimento.
- `SEV2`: degradação relevante de desempenho ou estabilidade.
- `SEV3`: anomalia de qualidade ou tendência de risco sem impacto imediato.

### 6.2 Regras de acionamento

| Alerta | Condição | Severidade | Resposta inicial |
|---|---|---|---|
| Modelo não carregado | `/health.model_loaded=false` por 5 min | SEV1 | Validar diretório de bundle e restaurar disponibilidade |
| Queda de disponibilidade | `/health` abaixo de 99% na janela de 1h | SEV1 | Acionar plano de recuperação do serviço |
| Explosão de 5xx | `api_http_5xx_rate > 3%` por 10 min | SEV1 | Isolar causa, conter impacto e avaliar rollback |
| Degradação de latência | p95 de `/predict` > 600 ms por 15 min | SEV2 | Identificar gargalo de processamento |
| Aumento de 422 | `api_validation_error_rate_422 > 20%` por 30 min | SEV2 | Revisar contrato e orientar consumidores |
| Variação anômala de predição positiva | variação > 30% vs média móvel de 7 dias | SEV3 | Avaliar drift de dados de entrada |
| Colapso de distribuição de score | desvio padrão de `probability_churn` muito baixo por 1 dia | SEV3 | Verificar consistência de dados e artefato |
| Bundle desatualizado | `model_bundle_freshness_days > 30` | SEV3 | Planejar retreino e ciclo de homologação |

### 6.3 Canais de comunicação

- Canal primário: time técnico responsável pela operação.
- Canal secundário: registro formal de incidente no repositório.
- Canal de negócio: acionamento da liderança funcional quando houver impacto em decisões de retenção.

---

## 7. Playbook de resposta a incidentes

### 7.1 Fluxo padrão de atendimento

1. Receber e classificar o alerta.
2. Confirmar impacto e abrangência.
3. Aplicar contenção para estabilização do serviço.
4. Executar diagnóstico da causa raiz.
5. Corrigir, restaurar ou efetuar rollback.
6. Validar recuperação por métricas e testes.
7. Consolidar lições aprendidas e ações preventivas.

### 7.2 Procedimentos por cenário

#### Cenário A: `/predict` retorna 503 (modelo indisponível)

1. Validar `CHURN_MODEL_BUNDLE_DIR` no ambiente em execução.
2. Confirmar presença de `metadata.json`, `preprocessor.joblib` e `mlp_state.pt`.
3. Reiniciar aplicação e revalidar `/health`.
4. Persistindo falha, restaurar último bundle íntegro e homologado.
5. Registrar incidente com evidências e tempo de recuperação.

#### Cenário B: pico de 5xx ou `unhandled_exception`

1. Correlacionar `X-Request-ID` para isolar requisições afetadas.
2. Identificar rota, payload e janela de ocorrência.
3. Em caso de regressão recente, executar rollback do deploy.
4. Validar baseline técnico antes de normalizar (`ruff check` e `pytest`).
5. Comunicar status e plano de estabilização ao time.

#### Cenário C: crescimento de 422 em `/predict`

1. Mapear padrões de rejeição de payload.
2. Confirmar alterações de contrato e compatibilidade entre versões.
3. Atualizar documentação de integração e exemplos de request.
4. Acompanhar redução da taxa de erro após correções.

#### Cenário D: anomalia de distribuição (suspeita de drift)

1. Comparar distribuição atual com baseline de 7 e 30 dias.
2. Identificar variáveis com maior deslocamento estatístico.
3. Reavaliar desempenho offline com dados recentes (PR-AUC, F1, ROC-AUC).
4. Se houver degradação material, iniciar ciclo de retreino controlado.
5. Promover novo modelo somente após validação técnica e aprovação de negócio.

---

## 8. Governança de decisão

| Situação | Responsável técnico | Responsável de negócio | Diretriz de decisão |
|---|---|---|---|
| Falha de API (SEV1) | Engenharia/API | N/A | Recuperação imediata e possível rollback |
| Regressão de métrica técnica | Engenharia/ML | Produto/Negócio | Reavaliar modelo e limiar operacional |
| Alteração de threshold | Engenharia/ML | Negócio (aprovação) | Executar apenas com aprovação formal |
| Drift persistente | Engenharia/ML | Negócio (aprovação) | Planejar retreino e publicação versionada |

---

## 9. Cadência operacional

### Diário

- Verificar disponibilidade, latência e erro 5xx.
- Revisar taxa de 422 e principais causas.

### Semanal

- Analisar distribuição de `probability_churn` e taxa de positivos.
- Revisar alertas, incidentes e ruído operacional.

### Quinzenal

- Revisar desempenho offline no MLflow e necessidade de retreino.
- Revalidar premissas de threshold com stakeholders.

### Mensal

- Consolidar post-mortem e atualizar este plano.
- Auditar conformidade com diretrizes de segurança e privacidade de logs.

---

## 10. Roadmap de evolução

1. Instrumentação imediata a partir dos logs estruturados e de `/health`.
2. Painel operacional com disponibilidade, latência, erros e distribuição de score.
3. Automação de alertas por janela e severidade.
4. Rotina de detecção de drift com gatilho formal para retreino.

---

## 11. Indicadores de efetividade

- Detecção de incidentes críticos em até 10 minutos.
- Redução progressiva do MTTR ao longo dos ciclos mensais.
- Estabilidade sustentada de latência p95 e taxa de 5xx.
- Retreino acionado por critério objetivo e não por percepção humana.