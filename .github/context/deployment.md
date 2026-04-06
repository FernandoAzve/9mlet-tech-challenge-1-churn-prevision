# Implantação e operações

## Objetivo

Descrever como configurar ambientes, executar tarefas locais via **Makefile**, integrar **CI/CD** com **GitHub Actions** e, opcionalmente, empacotar a solução com **Docker**, mantendo alinhamento com o pipeline de ML e a API FastAPI.

## Escopo

- Ambientes (desenvolvimento e demonstração)
- Comandos padronizados (`Makefile`, `pyproject.toml`)
- Pipeline de integração contínua
- Variáveis de ambiente relevantes
- Container opcional

---

## 1. Visão geral

- **Plataforma de hospedagem:** a definir pelo time (nuvem ou máquina local); o desafio trata deploy em produção como **opcional**.
- **Entregáveis mínimos:** código reprodutível, **README** completo, **Model Card**, API funcional localmente ou containerizada.
- **Orquestração local:** **Makefile** com alvos claros (`install`, `lint`, `test`, `train`, `serve`, etc.).

---

## 2. Ambientes

### Desenvolvimento

- **Local:** clone do repositório, criação de venv, instalação com `pip install -e ".[dev]"` ou comando equivalente documentado no README.
- **Dados:** uso de `data/` conforme instruções do projeto; não commitar datasets grandes sem **Git LFS** ou política explícita.

### Demonstração / staging (opcional)

- Mesma base de código com variáveis de ambiente apontando para artefato de modelo validado.
- Pode ser uma VM, container único ou ambiente gerenciado pelo aluno.

### Produção (opcional)

- Imagem **Docker** com Uvicorn, healthcheck apontando para **`/health`**.
- Secrets via variáveis de ambiente do provedor ou GitHub Secrets no CI.

---

## 3. Processo de implantação (referência)

### Checklist pré-merge (mínimo)

- [ ] `ruff check` sem erros (e `ruff format --check` se adotado).
- [ ] `pytest` passando.
- [ ] README atualizado se houver mudança de uso ou dependências.
- [ ] Model Card revisado se o comportamento do modelo mudar.
- [ ] Seeds e instruções de reprodutibilidade coerentes com o último treino relevante.

### Passos típicos para liberar nova versão da API (exemplo)

1. Treinar ou atualizar modelo; registrar run no **MLflow**.
2. Exportar artefato para `models/` (ou URI acordada).
3. Executar testes e lint.
4. Atualizar documentação e, se aplicável, tag de versão no Git.
5. Publicar imagem Docker ou instruções de execução atualizadas.

### Rollback

- Reverter para commit/tag anterior com artefato de modelo conhecido.
- Manter no MLflow runs identificáveis para recuperar métricas e artefatos.

---

## 4. Variáveis de ambiente (exemplos)

Documentar no README os nomes exatos; exemplos comuns:

| Variável | Finalidade |
|----------|------------|
| `MODEL_PATH` ou `ARTIFACT_URI` | Caminho ou referência ao modelo carregado pela API |
| `LOG_LEVEL` | Nível de log (INFO, WARNING, …) |
| `MLFLOW_TRACKING_URI` | URI do servidor MLflow em experimentos (se remoto) |

**Regra:** nunca commitar arquivos `.env` com segredos reais.

---

## 5. CI/CD com GitHub Actions

### Objetivos do pipeline

1. Instalar Python na versão definida no projeto.
2. Instalar dependências a partir do **`pyproject.toml`**.
3. Executar **ruff** (`check` e, se aplicável, `format --check`).
4. Executar **pytest** (com cobertura opcional via **pytest-cov**).
5. (Opcional) build de imagem Docker em tags ou na branch principal.

### Gatilhos sugeridos

- **Pull request:** lint + testes obrigatórios.
- **Push na branch principal:** idem; publicação de artefatos opcional.

### Segredos

- Tokens ou credenciais de MLflow remoto, registry de container, etc., apenas em **GitHub Secrets**.

---

## 6. Docker (opcional)

- **Dockerfile** multi-stage ou simples: base Python, cópia do código, instalação de dependências, comando padrão `uvicorn` apontando para o módulo FastAPI.
- **`.dockerignore`:** excluir `data/` volumoso, caches e ambientes virtuais.
- Expor porta da API; documentar mapeamento no README.

---

## 7. Monitoramento e logs

- Logs estruturados no stdout para coleta por plataforma de hospedagem.
- Métricas de negócio (taxa de predição positiva, latência) podem ser adicionadas depois; não são obrigatórias no escopo mínimo do desafio.

---

## 8. Backup e recuperação

- **Código:** histórico no Git.
- **Modelos:** versionados por convenção (`models/` + MLflow); recuperação via run ou tag de release.

---

## Documentação relacionada

- **Arquitetura:** `.github/context/architecture.md`
- **Stack:** `.github/context/tech-stack.md`
- **Regras de segurança:** `.github/rules/security-rules.md`
