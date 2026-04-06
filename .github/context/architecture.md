# Visão de arquitetura

## Objetivo

Registrar decisões arquiteturais do sistema de **Machine Learning end-to-end para previsão de churn** em telecom: do dado ao experimento, ao artefato e à inferência via API.

## Escopo

- Visão geral do sistema e estilo arquitetural
- Fluxo de dados e componentes principais
- Fronteiras entre treino, rastreamento e serviço de inferência
- Segurança e observabilidade em nível adequado ao Tech Challenge

---

## 1. Visão geral do sistema

**Nome do projeto:** `9mlet-tech-challenge-1-churn-prevision`

**Descrição:**  
Solução que prevê o **risco de churn** de clientes de operadora a partir de dados tabulares, permitindo ações de retenção. O fluxo cobre **preparação de dados**, **treinamento de uma MLP em PyTorch**, **baselines em Scikit-Learn**, **rastreamento em MLflow**, **persistência de artefatos** e **exposição de inferência** via **FastAPI** (`/predict`, `/health`).

**Usuários principais:**  
Áreas de negócio, CRM, marketing e retenção consomem as previsões; times de dados operam treino, experimentos e a API.

**Estilo arquitetural:**  
Arquitetura em **camadas** com separação clara entre **pipeline de ML**, **código de treino**, **rastreamento** e **camada de API** (sem acoplamento forte entre treino e servidor em tempo de execução).

---

## 2. Fluxo de alto nível

O fluxo canônico é:

**dados → features → treino → tracking (MLflow) → artefato → inferência (FastAPI)**

```text
[data/] → pipeline de features → treino (PyTorch + baselines sklearn)
                ↓
         MLflow (métricas, params, artefatos)
                ↓
         models/ + metadados (versão, assinatura de features)
                ↓
         FastAPI carrega artefato → /predict , /health
```

### Componentes principais

1. **Ingestão e preparação**  
   Leitura a partir de `data/`; validação de schema (Pydantic e/ou **pandera** no offline); tratamento de valores ausentes, codificação e normalização alinhados ao modelo em produção.

2. **Pipeline de features**  
   Transformações reprodutíveis (preferência por objetos serializáveis do sklearn ou módulos dedicados em `src/`) para garantir **mesma lógica** no treino e na inferência.

3. **Treinamento**  
   - **Modelo principal:** MLP em **PyTorch** (classificação binária).  
   - **Baselines:** modelos **Scikit-Learn** comparáveis (mesmo split ou mesma validação cruzada).  
   - **Reprodutibilidade:** **seeds fixadas** (Python, NumPy, PyTorch quando aplicável).  
   - **Validação:** **validação cruzada estratificada** para métricas estáveis em classes desbalanceadas.

4. **MLflow**  
   Registro de experimentos, parâmetros, métricas (ex.: ROC-AUC, F1, precisão/recall) e artefatos (modelo, pré-processadores). Facilita comparação MLP vs baselines.

5. **Persistência**  
   Artefatos em `models/` (ou URI registrada no MLflow), com convenção de nomes e metadados suficientes para carregar na API.

6. **API FastAPI**  
   - **`/health`:** disponibilidade do serviço e, opcionalmente, checagem de carregamento do modelo.  
   - **`/predict`:** entrada validada por **Pydantic**, aplicação do mesmo pipeline de features e retorno de probabilidade ou classe + metadados úteis (sem vazar informação interna).

---

## 3. Decisões e padrões

- **Separação treino / inferência:** o servidor não treina em requisição; apenas carrega artefato e executa forward pass (ou equivalente sklearn para baseline servido, se previsto).
- **Contratos estáveis:** alterações no schema de entrada/saída exigem versionamento ou documentação explícita (README, Model Card).
- **Erros HTTP:** usar códigos adequados (ex.: 422 para validação, 503 se modelo indisponível); mensagens claras para o cliente sem expor stack trace.
- **Logging estruturado:** correlacionar requisições quando houver `request_id`; não registrar dados pessoais completos nem features sensíveis em claro.

---

## 4. Dados e estado

- **Fonte de verdade dos dados brutos:** diretório `data/` e políticas documentadas no README (origem, licença, anonimização se houver).
- **Estado da API:** **stateless** entre requisições; configuração via variáveis de ambiente (caminho do modelo, nível de log).
- **Sem banco de dados obrigatório:** o escopo do desafio não exige persistência relacional para predição; se no futuro houver feature store ou fila, documentar em ADR.

---

## 5. Segurança (nível do desafio)

- Validação rigorosa de entrada com **Pydantic**.
- Limites de tamanho de payload e timeouts conforme configuração do servidor ASGI.
- Segredos apenas em variáveis de ambiente ou segredos de CI; nunca no repositório.
- Autenticação forte (tokens, OAuth) é **opcional** neste Tech Challenge; se ausente, documentar que a API é para ambiente controlado ou demonstração.

---

## 6. Escalabilidade e confiabilidade

- **MVP:** uma instância Uvicorn é suficiente para entrega acadêmica.
- **Crescimento:** horizontalizar réplicas atrás de proxy, garantindo que o carregamento do modelo seja eficiente (carregar uma vez na inicialização).
- **Monitoramento:** logs + métricas básicas; integração com APM é opcional.

---

## 7. Diagramas

Recomenda-se manter em `docs/` diagramas (Mermaid ou equivalente) do fluxo acima e da estrutura de pacotes em `src/`, atualizados quando a implementação evoluir.

---

## 8. Documentação relacionada

- **Stack:** `.github/context/tech-stack.md`
- **Objetivos do projeto:** `.github/context/project-goals.md`
- **Implantação:** `.github/context/deployment.md`
- **Regras de segurança:** `.github/rules/security-rules.md`
- **Regras de negócio:** `.github/rules/business-rules.md`
