# Objetivos do projeto

## Objetivo deste documento

Registrar o **contexto de negócio**, **objetivos** e **restrições** do Tech Challenge de previsão de churn, alinhando a equipe antes de qualquer implementação.

## Escopo

- Problema que estamos resolvendo
- Usuários e partes interessadas
- Objetivos de negócio
- Restrições técnicas e de negócio

---

## 1. Enunciado do problema

**Problema que estamos resolvendo:**

O projeto resolve o problema de perda acelerada de clientes da operadora, prevendo quais clientes têm maior risco de churn antes do cancelamento acontecer. Com isso, a empresa pode agir preventivamente para retenção, reduzindo perda de receita. A solução foi construída end-to-end, desde a preparação dos dados até a disponibilização do modelo via API, usando uma MLP em PyTorch e comparando com baselines rastreados no MLflow.

**Por que esse problema importa:**

O churn impacta diretamente a receita recorrente da operadora e a sustentabilidade do negócio: a perda de clientes reduz o faturamento e aumenta os custos com aquisição de novos usuários, em geral mais altos que os de retenção. Ao prever com antecedência quais clientes têm maior risco de cancelamento, a empresa pode agir de forma preventiva com estratégias de retenção (ofertas personalizadas, melhoria no atendimento), reduzindo perdas financeiras e aumentando fidelização, além de apoiar decisões com dados.

---

## 2. Público e usuários

### Usuários principais

Áreas de **negócio e operação da operadora**, em especial retenção, marketing, CRM e atendimento, que usam as previsões para identificar alto risco de churn e direcionar ações preventivas (campanhas, ofertas, contato proativo). Gestores e diretoria utilizam os insights para decisões sobre receita, experiência do cliente e planejamento comercial. A equipe de dados e engenharia usa a API para monitoramento, manutenção e integração aos sistemas da empresa.

---

## 3. Objetivos de negócio

### Metas principais

Reduzir a taxa de churn, preservar receita recorrente e aumentar retenção, permitindo identificar antecipadamente clientes com maior risco de cancelamento e agir com ações direcionadas. Otimizar o custo de retenção priorizando esforços em clientes com maior probabilidade de saída e maior valor, apoiando decisões estratégicas com previsões confiáveis.

---

## 4. Restrições

### Restrições técnicas

Uso obrigatório de **PyTorch** para a MLP, **Scikit-Learn** para pipelines e baselines, **MLflow** para rastreamento de experimentos e **FastAPI** para a API de inferência. Boas práticas de engenharia de ML: **seeds fixadas**, **validação cruzada estratificada**, **testes automatizados**, **logging estruturado**, **ruff sem erros** e estrutura (`src/`, `data/`, `models/`, `tests/`, `notebooks/`, `docs/`). **Makefile** e **pyproject.toml** para reprodutibilidade e padronização.

Endpoints da API: **`/predict`** e **`/health`**. Documentação: **README** completo e **Model Card**. Dataset tabular binário com pelo menos **5.000 registros** e **10 features** (conforme enunciado).

### Restrições de negócio

Reduzir churn e apoiar a diretoria com modelo **confiável e acionável**, permitindo ações preventivas de retenção e foco em receita recorrente e custo de retenção. Considerar explicitamente o trade-off entre **falsos positivos** (custo de ação desnecessária) e **falsos negativos** (perda de receita por não agir). No contexto acadêmico, a entrega inclui comunicação clara (vídeo método STAR, documentação, indicadores como **custo de churn evitado** quando aplicável).

---

## 5. Fora de escopo (non-goals)

**O que este projeto não faz:**

Não explica causas individuais de churn em profundidade nem **executa automaticamente** campanhas, descontos ou contato com clientes: o escopo central é **predição do risco** e **disponibilização via API**. Não inclui CRM completo, dashboards executivos nem integrações operacionais em tempo real com sistemas da operadora. **Deploy em nuvem é opcional**, não obrigatório para a solução principal do desafio.

---

## Documentação relacionada

- **Arquitetura:** `.cursor/context/architecture.md`
- **Regras de negócio:** `.cursor/rules/business-rules.md`
- **Stack:** `.cursor/context/tech-stack.md`
