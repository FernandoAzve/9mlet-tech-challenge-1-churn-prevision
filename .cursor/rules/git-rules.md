# Regras Git

## Objetivo

Padronizar **mensagens de commit**, **branches** e **pull requests** no repositório `9mlet-tech-challenge-1-churn-prevision`.

## Contexto do projeto

- Projeto **Python** de Machine Learning (churn) com API **FastAPI**.
- Qualidade verificada por **ruff** e **pytest** no fluxo de contribuição.

---

## Escopo

- Formato de commits
- Nomenclatura de branches
- Processo de PR e revisão
- Boas práticas proibidas

---

## Mensagens de commit

### Formato

Usar **Conventional Commits**:

```text
<tipo>(<escopo>): <assunto curto>

<corpo opcional>

<rodapé opcional>
```

### Tipos comuns

- **feat:** nova funcionalidade (ex.: endpoint, script de treino).
- **fix:** correção de bug.
- **docs:** documentação apenas.
- **style:** formatação / estilo sem mudança de lógica (ex.: `ruff format`).
- **refactor:** refatoração sem mudança de comportamento pretendida.
- **test:** adição ou correção de testes.
- **chore:** tarefas de manutenção (dependências, Makefile, CI).

### Exemplos

```text
feat(api): adiciona endpoint /predict com schema Pydantic

fix(train): corrige seed não fixa no loop de validação

docs: atualiza Model Card com métricas de validação cruzada
```

**Idioma:** manter **um idioma** escolhido pelo time (português ou inglês) de forma consistente no histórico.

---

## Estratégia de branches

### Nomes

- **Funcionalidade:** `feature/descricao-curta`
- **Correção:** `fix/descricao-curta`
- **Hotfix:** `hotfix/descricao-curta`
- **Release:** `release/versao` (se usado)

### Branches principais

- **main:** código estável entregue ou candidato à entrega.
- **develop:** opcional, integração contínua de features antes da main.

---

## Pull requests

### Antes de abrir o PR

- [ ] Código segue `.cursor/rules/code-style.md` (`ruff check` ok).
- [ ] Testes passam (`pytest`).
- [ ] Novas funcionalidades têm testes adequados.
- [ ] Documentação atualizada se mudou uso ou contrato da API.
- [ ] Autorrevisão feita (diff lido uma vez).

### Título

- Mesmo estilo que commit: `tipo(escopo): assunto`.

### Descrição

Incluir:

- **O quê:** mudanças realizadas.
- **Por quê:** motivação ou issue associada.
- **Como testar:** comandos (`make test`, exemplo de `curl` na API, etc.).
- **Riscos:** impacto em artefatos, métricas ou compatibilidade.

### Tamanho

- Preferir PRs **pequenos** e focados (&lt; ~400 linhas de diff úteis).
- PRs grandes exigem justificativa e podem ser divididos (treino vs API vs docs).

---

## Revisão

### Revisora ou revisor

- Verificar aderência ao desafio (stack, endpoints, reprodutibilidade).
- Checar testes e possíveis vieses ou vazamento treino-serviço.
- Aprovar ou solicitar mudanças com clareza.

### Autor

- Responder comentários e reabrir revisão após ajustes.

### Regras

- Pelo menos **uma aprovação** antes do merge (conforme política do time).
- Resolver conversas bloqueantes antes de integrar.

---

## Merge

- **Squash and merge:** preferível para branches de feature (histórico linear e legível).
- **Merge commit:** opcional para releases.
- **Rebase and merge:** apenas se o time padronizar e todos dominarem o fluxo.

Antes do merge:

- [ ] CI verde (ruff + pytest).
- [ ] Aprovação obtida.
- [ ] Sem conflitos com a branch alvo.

---

## Práticas proibidas

- Commit direto na `main` sem processo acordado.
- `git push --force` em branches compartilhadas.
- Commitar segredos, `.env` com credenciais ou datasets sensíveis grandes.
- Pular hooks intencionalmente (`--no-verify`) sem motivo documentado.
- Mensagens vagas (“ajustes”, “fix stuff”).

---

## Hooks (opcional)

- **pre-commit:** `ruff check` e, se desejado, `ruff format --check`.
- **pre-push:** `pytest` (suíte rápida).

Configuração fica a cargo do time (ferramentas Python, não dependem de ecossistema Node).

---

## Documentação relacionada

- **Estilo de código:** `.cursor/rules/code-style.md`
- **Testes:** `.cursor/rules/testing-rules.md`
- **Uso de IA:** `.cursor/rules/ai-usage-rules.md`
