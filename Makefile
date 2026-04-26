ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null || pwd)
PYTHON_PATHS := src tests
UV_RUN := uv run --no-sync
NB := $(shell test -d "$(ROOT)/notebooks" && find "$(ROOT)/notebooks" -type f -name "*.ipynb" 2>/dev/null)

.PHONY: help sync-core sync-dev install install-dev install-gpu format fmt lint type-check types test test-cov all nb-format nb-lint nb-types

help:
	@echo "Warsaw Econometric Challenge 2026 - Makefile Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make sync-core     Sync core dependencies with uv"
	@echo "  make sync-dev      Sync development dependencies with uv"
	@echo "  make install       Alias for sync-core"
	@echo "  make install-dev   Alias for sync-dev"
	@echo "  make install-gpu   Sync with GPU extras"
	@echo ""
	@echo "Code Quality:"
	@echo "  make fmt           Format and auto-fix Python code"
	@echo "  make lint          Check Ruff linting and formatting"
	@echo "  make types         Run mypy on src and tests"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run pytest"
	@echo "  make test-cov      Run pytest with coverage report"
	@echo "  make all           Run lint, types, and test"
	@echo ""
	@echo "Notebooks:"
	@echo "  make nb-format     Format notebooks via nbQA"
	@echo "  make nb-lint       Lint notebooks via nbQA"
	@echo "  make nb-types      Type-check notebooks via nbQA"

sync-core:
	cd "$(ROOT)" && uv sync

sync-dev:
	cd "$(ROOT)" && uv sync --extra dev

install: sync-core

install-dev: sync-dev

install-gpu:
	cd "$(ROOT)" && uv sync --extra gpu

format fmt:
	cd "$(ROOT)" && $(UV_RUN) ruff check --fix $(PYTHON_PATHS)
	cd "$(ROOT)" && $(UV_RUN) ruff format $(PYTHON_PATHS)

lint:
	cd "$(ROOT)" && $(UV_RUN) ruff check $(PYTHON_PATHS)
	cd "$(ROOT)" && $(UV_RUN) ruff format --check $(PYTHON_PATHS)

type-check types:
	cd "$(ROOT)" && $(UV_RUN) mypy src tests

test:
	cd "$(ROOT)" && $(UV_RUN) pytest

test-cov:
	cd "$(ROOT)" && $(UV_RUN) pytest --cov=src --cov-report=term-missing --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

all: lint types test

nb-format:
	@if [ -z "$(NB)" ]; then echo "No notebooks to format."; else \
		cd "$(ROOT)" && uv run --no-sync --with nbqa nbqa ruff $(NB) -- check --fix; \
		cd "$(ROOT)" && uv run --no-sync --with nbqa nbqa ruff $(NB) -- format; \
	fi

nb-lint:
	@if [ -z "$(NB)" ]; then echo "No notebooks to lint."; else \
		cd "$(ROOT)" && uv run --no-sync --with nbqa nbqa ruff $(NB) -- check; \
		cd "$(ROOT)" && uv run --no-sync --with nbqa nbqa ruff $(NB) -- format --check; \
	fi

nb-types:
	@if [ -z "$(NB)" ]; then echo "No notebooks to type-check."; else \
		cd "$(ROOT)" && uv run --no-sync --with nbqa nbqa mypy $(NB); \
	fi
