# Bibliotecas proibidas ou restritas

## Objetivo

Evitar dependências **incompatíveis** com o escopo do Tech Challenge, **redundantes** ou que induzam stack errada (ecossistemas não Python para a aplicação principal).

## Escopo

- Lista de tecnologias e padrões **não desejados**
- Exceções e processo de aprovação

---

## 1. Ecossistemas alheios ao núcleo da aplicação

As seguintes tecnologias **não** devem ser usadas como base da API, do treino ou do pipeline principal deste repositório:

- **Node.js** como runtime do serviço de predição
- Frameworks web **JavaScript/TypeScript** substituindo **FastAPI** para os endpoints do desafio
- **TensorFlow / Keras** como substituto da **MLP em PyTorch** (o requisito é PyTorch)

**Motivo:** o desafio exige explicitamente **Python + PyTorch + FastAPI + MLflow** na arquitetura entregue.

---

## 2. Frameworks Python concorrentes (sem aprovação)

- **Flask** ou **Django** como framework principal da API de inferência — **não permitido** salvo decisão documentada do time e aceite do orientador **e** manutenção dos endpoints exigidos.
- **LightGBM / XGBoost / CatBoost** como **modelo principal** — o modelo principal deve ser **MLP PyTorch**; usar esses pacotes apenas como **baseline adicional** se o enunciado permitir e estiver documentado no Model Card.

**Motivo:** evitar divergência dos requisitos obrigatórios e da narrativa de comparação MLP vs sklearn.

---

## 3. Dependências de qualidade substitutas

- **flake8**, **black**, **isort** como **substituto total** do **ruff** — não permitido enquanto o projeto padronizar ruff.
- Ferramentas que exijam **outro ecossistema** (ex.: linters de TypeScript) para validar o código Python do repositório.

**Motivo:** uma única fonte da verdade de lint (`ruff`) simplifica CI e contribuições.

---

## 4. Práticas de dependência

- Pacotes com **última manutenção muito antiga** ou sem licença clara — evitar; exigir aprovação explícita.
- Dependências que puxam **binários não auditáveis** de origem duvidosa — rejeitar.
- **Copiar código proprietário** ou de licença incompatível — proibido.

---

## 5. Dados e segurança

- Carregar em produção **pickle** de fontes externas não confiáveis — proibido.
- Versionar no Git **datasets grandes** ou **credenciais** — proibido (usar `.gitignore`, LFS ou storage externo conforme política).

---

## Processo de exceção

Se for **estritamente necessário** usar item restrito:

1. Documentar motivo técnico no PR.
2. Obter acordo do time/orientador.
3. Atualizar este arquivo com a **exceção datada** ou registrar em `docs/adr-*.md`.

---

## Documentação relacionada

- **Bibliotecas permitidas:** `.github/libs/allowed-libs.md`
- **Stack:** `.github/context/tech-stack.md`
