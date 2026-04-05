# GlassBox RAG Build System
.PHONY: help clean install test lint build check publish docker-build docker-run all

# Default target
help: ## Show this help message
	@echo "GlassBox RAG Build System"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	@rm -rf dist/ build/ *.egg-info/ .pytest_cache/ .coverage
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@echo "✓ Build artifacts cleaned"

install: ## Install dependencies
	@echo "Installing dependencies..."
	@if command -v uv >/dev/null 2>&1; then \
		uv pip install -e ".[dev,embeddings,vector-stores,databases,multimodal,telemetry,auth,reranking]"; \
	else \
		pip install -e ".[dev,embeddings,vector-stores,databases,multimodal,telemetry,auth,reranking]"; \
	fi
	@echo "✓ Dependencies installed"

test: ## Run tests
	@echo "Running tests..."
	@python -m pytest tests/ -v --cov=glassbox_rag --cov-report=html --cov-report=term-missing
	@echo "✓ Tests completed"

test-unit: ## Run unit tests only
	@echo "Running unit tests..."
	@python -m pytest tests/unit/ -v
	@echo "✓ Unit tests completed"

test-integration: ## Run integration tests only
	@echo "Running integration tests..."
	@python -m pytest tests/integration/ -v
	@echo "✓ Integration tests completed"

lint: ## Run linting
	@echo "Running linters..."
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check src/ tests/; \
		ruff format --check src/ tests/; \
	else \
		echo "⚠️  Ruff not found, skipping linting"; \
	fi
	@echo "✓ Linting completed"

format: ## Format code
	@echo "Formatting code..."
	@if command -v ruff >/dev/null 2>&1; then \
		ruff format src/ tests/; \
	else \
		echo "⚠️  Ruff not found, skipping formatting"; \
	fi
	@echo "✓ Code formatted"

build: clean ## Build package
	@echo "Building package..."
	@python -m build
	@echo "✓ Package built successfully"
	@ls -la dist/

check: build ## Check package for PyPI
	@echo "Checking package..."
	@python -m twine check dist/*
	@echo "✓ Package check completed"

publish: check ## Publish to PyPI
	@echo "⚠️  Publishing to PyPI..."
	@python -m twine upload dist/*
	@echo "✓ Package published to PyPI!"

docker-build: ## Build Docker image
	@echo "Building Docker image..."
	@docker build -t glassbox-rag:latest .
	@echo "✓ Docker image built"

docker-run: ## Run Docker container
	@echo "Starting Docker container..."
	@docker run -p 8000:8000 --name glassbox-rag-container glassbox-rag:latest

docker-dev: ## Run Docker container in development mode
	@echo "Starting Docker container in development mode..."
	@docker run -p 8000:8000 -v $(PWD):/app --name glassbox-rag-dev glassbox-rag:latest

docker-compose-up: ## Start all services with docker-compose
	@echo "Starting services with docker-compose..."
	@docker-compose up -d
	@echo "✓ Services started"

docker-compose-down: ## Stop all services with docker-compose
	@echo "Stopping services with docker-compose..."
	@docker-compose down
	@echo "✓ Services stopped"

docker-compose-prod-up: ## Start production services
	@echo "Starting production services..."
	@docker-compose -f docker-compose.prod.yml --profile monitoring up -d
	@echo "✓ Production services started"

docker-compose-prod-down: ## Stop production services
	@echo "Stopping production services..."
	@docker-compose -f docker-compose.prod.yml down
	@echo "✓ Production services stopped"

serve: ## Start the development server
	@echo "Starting development server..."
	@python -m glassbox_rag serve --host 0.0.0.0 --port 8000

dev: install ## Setup development environment
	@echo "Setting up development environment..."
	@pre-commit install 2>/dev/null || echo "⚠️  pre-commit not available"
	@echo "✓ Development environment ready"

all: clean install lint test build check ## Run full build pipeline
	@echo "✓ All checks passed!"

# Development workflow targets
ci: install lint test ## Run CI pipeline (install, lint, test)

release: clean all publish ## Full release pipeline</content>
<parameter name="filePath">c:\Users\Amaan\OneDrive\Desktop\Glassbox\glassbox-rag\Makefile