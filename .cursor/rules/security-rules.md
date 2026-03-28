# Regras de segurança

## Objetivo

Estabelecer práticas **obrigatórias** de segurança para o projeto de ML e API de churn, **proporcionais** ao Tech Challenge (demonstração, ambiente acadêmico e eventual deploy opcional), sem impor padrões de aplicação enterprise não requeridos pelo escopo.

## Escopo

- Dados sensíveis e privacidade
- API FastAPI e validação de entrada
- Dependências e segredos
- Erros e logs

---

## 1. Princípios

- **Validar tudo que entra** na API com **Pydantic** (tipos, ranges, listas permitidas).
- **Não confiar** em validação apenas no cliente (notebook ou front-end externo).
- **Minimizar exposição:** mensagens de erro claras para o usuário, sem stack trace nem caminhos internos em produção.
- **Segredos fora do código:** variáveis de ambiente ou gerenciador de secrets do provedor; nunca commitar credenciais.

---

## 2. Dados pessoais e LGPD

- Tratar dados de clientes como **sensíveis** quando a origem for real ou semelhante a produção.
- **Não** logar em claro: nome completo, CPF, telefone, e-mail, endereço ou combinações que identifiquem pessoa natural.
- Preferir **IDs anonimizados** ou hashes irreversíveis quando necessário para rastreio técnico.
- Documentar no README a **base legal simulada** ou o uso de dados sintéticos/públicos, conforme o caso do trabalho acadêmico.

---

## 3. API (FastAPI / Uvicorn)

### Validação

- Schemas Pydantic para corpo e, se houver, query params.
- Limitar tamanho de payload via configuração do servidor ou middleware, quando aplicável.

### Autenticação

- **Não é obrigatória** no escopo mínimo do desafio para a nota acadêmica.
- Se a API for exposta à internet, **recomenda-se** pelo menos **API key** em header ou rede privada; implementação fica a critério do time e do orientador.
- Se não houver autenticação, documentar claramente que o serviço é **somente para laboratório** ou rede restrita.

### CORS e taxa de uso

- Se houver cliente web, restringir **origens** confiáveis em produção.
- Considerar limite de requisições (middleware ou proxy reverso) em deploy público.

### HTTPS

- Em produção real, servir atrás de TLS (proxy ou plataforma gerenciada).

---

## 4. Pickle e artefatos

- **Pickle** de modelos ou transformadores pode executar código arbitrário ao carregar. Preferir formatos mais seguros quando possível; se usar pickle, **somente artefatos gerados pelo próprio pipeline confiável** e verificação de integridade (hash) opcional.
- Não carregar artefatos de fontes não confiáveis.

---

## 5. Dependências

- Manter `pyproject.toml` com versões razoáveis; revisar alertas de vulnerabilidade (`pip audit` ou dependabot).
- Não adicionar pacotes abandonados ou de procedência duvidosa.

---

## 6. Erros e logging

- Em respostas HTTP, usar mensagens genéricas para erros internos (ex.: “falha ao processar predição”) e registrar detalhes apenas no **log** do servidor.
- Logs estruturados sem dados pessoais; incluir `request_id` quando implementado.

---

## 7. CI/CD

- **GitHub Secrets** para tokens (MLflow remoto, registry, etc.).
- Não imprimir segredos em logs de pipeline.

---

## 8. Checklist antes de exposição pública (opcional)

- [ ] README descreve superfície de ataque e limitações.
- [ ] `/predict` valida entrada e não vaza informação interna.
- [ ] Variáveis sensíveis apenas no ambiente de hospedagem.
- [ ] Dependências sem vulnerabilidades críticas conhecidas.

---

## Documentação relacionada

- **Arquitetura:** `.cursor/context/architecture.md`
- **Implantação:** `.cursor/context/deployment.md`
- **Regras de negócio:** `.cursor/rules/business-rules.md`
