.PHONY: help install dev test lint format run docker-build docker-up docker-down clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -e .

dev: ## Install development dependencies
	pip install -e ".[dev]"

test: ## Run test suite
	pytest tests/ -v --cov=ragx --cov-report=term-missing

test-phase1: ## Run Phase 1 (ingestion) tests
	pytest tests/test_ingestion/ -v

test-phase2: ## Run Phase 2 (embeddings) tests
	pytest tests/test_embeddings/ -v

test-phase3: ## Run Phase 3 (retrieval) tests
	pytest tests/test_retrieval/ -v

test-phase4: ## Run Phase 4 (generation) tests
	pytest tests/test_generation/ -v

test-phase5: ## Run Phase 5 (API) tests
	pytest tests/test_api/ -v

lint: ## Run linter
	ruff check ragx/ tests/

format: ## Format code
	ruff format ragx/ tests/

run: ## Start the API server (development)
	uvicorn ragx.api.main:app --reload --host 0.0.0.0 --port 8000

run-prod: ## Start the API server (production)
	gunicorn ragx.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

docker-build: ## Build Docker image
	docker build -f deployment/Dockerfile -t ragx:latest .

docker-up: ## Start all services with Docker Compose
	docker compose -f deployment/docker-compose.yml up -d

docker-down: ## Stop all Docker Compose services
	docker compose -f deployment/docker-compose.yml down

clean: ## Clean build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name *.egg-info -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache .mypy_cache dist build
