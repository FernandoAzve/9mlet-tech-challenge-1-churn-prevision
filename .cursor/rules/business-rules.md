# Regras de negócio

## Objetivo

Registrar regras de **negócio e qualidade** que o sistema de previsão de churn deve respeitar, alinhadas ao problema da operadora, ao Tech Challenge e à API de inferência.

## Contexto do projeto

- **Projeto:** `9mlet-tech-challenge-1-churn-prevision`
- **Tipo:** solução de **Machine Learning** com API **FastAPI**
- **Problema:** reduzir **churn** (cancelamento de clientes) em telecom, priorizando **retenção** e uso eficiente de budget de campanha.

---

## Escopo

- Objetivos de negócio refletidos no software
- Comportamento esperado da predição e da API
- Trade-offs entre falso positivo e falso negativo
- O que o sistema **não** faz (non-goals)

---

## 1. Objetivos de negócio

- **Reduzir churn** identificando clientes com maior probabilidade de cancelamento **antes** do evento.
- **Preservar receita recorrente** ao direcionar ações de retenção de forma mais eficiente.
- **Otimizar custo de retenção:** evitar gastar com todos os clientes de forma uniforme; priorizar quem apresenta risco relevante (definição operacional de “relevante” é decisão de negócio, não só do modelo).
- **Apoiar decisões** com indicadores claros (ex.: probabilidade de churn, faixas de risco), sem substituir o julgamento humano na campanha.

**Métrica de valor (conceitual):** “custo de churn evitado” ou “valor retido” pode ser usado na apresentação (vídeo STAR, documentação), mesmo que o código entregue apenas a **probabilidade** ou **classe**.

---

## 2. Escopo funcional da solução

- O sistema **entrega predição de risco de churn** (tipicamente probabilidade da classe positiva ou rótulo com threshold).
- A solução é **end-to-end** no sentido acadêmico: dados → treino (MLP PyTorch + baselines sklearn) → MLflow → artefato → API com **`/predict`** e **`/health`**.
- **Model Card** e **README** descrevem limitações, dados, métricas e uso responsável.

---

## 3. Non-goals (fora de escopo)

- **Não** executar automaticamente campanhas (e-mail, SMS, descontos) a partir da API neste desafio.
- **Não** substituir CRM completo, billing ou canais de atendimento.
- **Não** garantir causalidade (“por que o cliente churnou”) em profundidade; o foco é **predição supervisionada** para ação de retenção informada.
- **Deploy em nuvem** é **opcional**; a solução pode ser validada localmente ou em container.

---

## 4. Regras da API

- **`/health`:** deve indicar se o serviço está vivo e, quando possível, se o modelo foi carregado com sucesso.
- **`/predict`:**  
  - aceitar apenas payloads que passem validação **Pydantic**;  
  - retornar estrutura JSON **consistente** documentada no README;  
  - em caso de falha interna, não expor detalhes técnicos ao cliente final.
- Uso de **REST** e JSON alinhado às boas práticas HTTP (status 422 para erro de validação, etc.).

---

## 5. Trade-off: falso positivo vs falso negativo

- **Falso positivo (prever churn quando não ocorreria):** pode gerar **custo** de retenção desperdiçado (desconto, tempo de call center).
- **Falso negativo (não prever churn que ocorreria):** pode gerar **perda de receita** por não agir a tempo.
- O **threshold** de decisão (ex.: 0,5 vs 0,3 na probabilidade) é **decisão de negócio**, dependendo do custo relativo assumido pela operadora; o modelo deve expor **probabilidade** quando possível para flexibilizar esse ajuste.
- Documentar no Model Card **limitações** e **desempenho** por segmento se houver análise (ex.: grupos com poucos dados).

---

## 6. Qualidade de dados e modelo

- Dataset tabular de **classificação binária** com requisitos mínimos do desafio (**≥ 5.000 registros** e **≥ 10 features**), salvo orientação contrária formal.
- **Validação cruzada estratificada** para estimar desempenho de forma mais estável com desbalanceamento.
- **Seeds fixadas** para reprodutibilidade dos experimentos principais.
- **Baselines sklearn** devem ser comparados à **MLP PyTorch** no **MLflow** (métricas e, se possível, curvas).

---

## 7. Integridade e consistência

- O **mesmo pipeline de features** usado no treino deve ser aplicado na inferência (ou mecanismo equivalente serializado), evitando **skew** treino-serviço.
- Versão do modelo e metadados relevantes devem ser rastreáveis (MLflow + convenção em `models/`).

---

## 8. Documentação relacionada

- **Objetivos do projeto:** `.cursor/context/project-goals.md`
- **Arquitetura:** `.cursor/context/architecture.md`
- **Segurança:** `.cursor/rules/security-rules.md`
- **Comando extrair regras:** `.cursor/commands/extract-business-rules.md`
