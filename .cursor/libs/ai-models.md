# Modelos de IA (LLM) recomendados

## Objetivo

Orientar a escolha de **modelos de linguagem** (ferramentas de IA generativa) por fase do trabalho neste repositório **Python / ML / MLOps**, sem confundir com o modelo de **churn** (MLP PyTorch).

## Contexto do projeto

- **Projeto:** `9mlet-tech-challenge-1-churn-prevision`
- **Tipo:** Machine Learning aplicado — churn em telecom
- **Stack de código:** Python, PyTorch, Scikit-Learn, MLflow, FastAPI, pytest, ruff, Pydantic

---

## Escopo

- Uso recomendado por fase (arquitetura, implementação, depuração, documentação)
- Regras de uso responsável
- Nota sobre mudança de nomes comerciais

---

## Princípios

- Sempre **revisar** código e texto gerados.
- **Validar** com documentação oficial (PyTorch, FastAPI, MLflow) quando houver dúvida de API.
- **Não** colar dados reais sensíveis nos prompts; usar amostras **fictícias mínimas**.
- Decisões de **negócio**, **LGPD** e **segurança** permanecem com humanos.

---

## Fases e modelos sugeridos

Os nomes abaixo referem-se a famílias de modelos; use a variante **mais capaz** disponível na sua assinatura (contexto longo, boa performance em Python).

### Arquitetura e desenho do sistema

- **Prioridade:** modelo com forte raciocínio estrutural e contexto amplo.
- **Uso:** desenhar módulos em `src/`, fluxo MLflow, limites treino vs API, estratégia de testes.
- **Alternativa:** outro modelo top de linha do mesmo fornecedor.

### Implementação de código (dia a dia)

- **Prioridade:** modelo especializado ou otimizado para **código Python**.
- **Uso:** implementar treino, API, testes pytest, configuração ruff.
- **Alternativa:** modelo multimodal da mesma classe, se o fluxo de trabalho for integrado ao IDE.

### Refatoração e leitura de muitos arquivos

- **Prioridade:** modelo com **grande janela de contexto** e boa fidelidade a diffs.
- **Uso:** refatorar mantendo comportamento, mover funções entre pacotes.

### Depuração e análise de erro

- **Prioridade:** modelo bom em rastrear stack traces e hipóteses.
- **Uso:** falhas de shape em tensores, erros Pydantic, problemas de carregamento de artefato.

### Documentação (README, Model Card, comentários)

- **Prioridade:** modelo equilibrado em **português** técnico e clareza.
- **Uso:** redigir seções, glossário de métricas, limitações do modelo.

### Tarefas rápidas (boilerplate pequeno)

- **Prioridade:** modelo **rápido e barato**.
- **Uso:** esqueleto de arquivo, renomeações simples, ajustes de formatação já guiados pelo ruff.

---

## Uso proibido ou de alto risco sem supervisão

- Gerar **pip install** de pacotes não listados sem checar **allowed-libs**.
- Aceitar **exclusão de testes** ou **desligamento de regras do ruff** sem justificativa no PR.
- Inventar **métricas** ou **resultados de experimento** sem base em runs reais do MLflow ou do código.

---

## Configuração no Cursor

- Incluir pastas **`.cursor/context`** e **`.cursor/rules`** no contexto relevante.
- Para tarefas longas, anexar trechos **mínimos** de `src/` e `tests/` em vez do repositório inteiro sem filtro.

---

## Mudança de nomes comerciais

Os provedores renomeiam modelos com frequência. Se um nome deste arquivo estiver desatualizado, aplique a regra: **escolha o modelo mais recente da mesma categoria** (top para arquitetura, código para implementação, etc.).

---

## Documentação relacionada

- **Regras de uso de IA:** `.cursor/rules/ai-usage-rules.md`
- **Comandos:** `.cursor/commands/`
