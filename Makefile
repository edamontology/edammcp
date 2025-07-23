.PHONY: help install install-dev test format lint type-check clean run example docs-build docs-serve docs-deploy

help: ## Show this help message
	@echo "EDAM MCP Server - Development Commands"
	@echo "======================================"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	uv sync

install-dev: ## Install development dependencies and package
	uv sync --dev
	uv pip install -e .

test: ## Run tests
	uv run pytest

test-verbose: ## Run tests with verbose output
	uv run pytest -v

test-coverage: ## Run tests with coverage
	uv run pytest --cov=edam_mcp --cov-report=html

format: ## Format code with black and isort
	uv run black edam_mcp/ tests/ examples/
	uv run isort edam_mcp/ tests/ examples/

lint: ## Run linting with ruff
	uv run ruff check edam_mcp/ tests/ examples/

lint-fix: ## Fix linting issues automatically
	uv run ruff check --fix edam_mcp/ tests/ examples/

type-check: ## Run type checking with mypy
	uv run mypy edam_mcp/

check-all: ## Run all checks (format, lint, type-check, test)
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) test

run: ## Run the MCP server
	uv run python -m edam_mcp.main

example: ## Run the basic usage example
	uv run python examples/basic_usage.py

docs-build: ## Build documentation
	uv run mkdocs build

docs-serve: ## Serve documentation locally
	uv run mkdocs serve

docs-deploy: ## Deploy documentation to GitHub Pages
	uv run mkdocs gh-deploy

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage

setup: ## Initial setup of development environment
	uv sync --dev
	uv pip install -e .
	$(MAKE) format
	$(MAKE) check-all 