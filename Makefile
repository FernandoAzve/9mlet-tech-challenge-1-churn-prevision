# Makefile para projeto de churn prediction
# Configuração acadêmica simples e reprodutível

.PHONY: help lint test run clean install-dev

# Alvo padrão
help:
	@echo "Comandos disponíveis:"
	@echo "  make lint     - Executa linting com ruff"
	@echo "  make test     - Executa testes com pytest"
	@echo "  make run      - Executa API FastAPI localmente"
	@echo "  make clean    - Remove arquivos temporários"
	@echo "  make install-dev - Instala dependências de desenvolvimento"

# Linting com ruff
lint:
	ruff check .
	ruff format --check .

# Testes com pytest
test:
	pytest tests/ -v

# Executar API FastAPI
run:
	uvicorn src.churn.api.main:app --reload --host 0.0.0.0 --port 8000

# Limpar arquivos temporários
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Instalar dependências de desenvolvimento
install-dev:
	pip install -e ".[dev]"