.PHONY: help install install-dev install-gpu format lint type-check test test-cov clean all

help:
	@echo "Warsaw Econometric Challenge 2026 - Makefile Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install core dependencies"
	@echo "  make install-dev   Install with development tools"
	@echo "  make install-gpu   Install with GPU support"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format        Format code with ruff"
	@echo "  make lint          Lint code with ruff"
	@echo "  make type-check    Run mypy type checking"
	@echo "  make test          Run tests"
	@echo "  make test-cov      Run tests with coverage report"
	@echo "  make all           Run format, lint, type-check, and test"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         Remove cache and build artifacts"

install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev]"

install-gpu:
	uv pip install -e ".[gpu]"

format:
	@echo "Formatting code with ruff..."
	ruff format src/ tests/
	ruff check --fix src/ tests/

lint:
	@echo "Linting code with ruff..."
	ruff check src/ tests/

type-check:
	@echo "Type checking with mypy..."
	mypy src/

test:
	@echo "Running tests..."
	pytest tests/ -v

test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

all: format lint type-check test

clean:
	@echo "Cleaning up..."
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov/ .coverage
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "Clean complete!"
