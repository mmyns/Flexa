.PHONY: help install sync lint format test coverage check forecast battery report \
        docker-build docker-forecast docker-battery docker-report clean

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies and project in editable mode
	uv sync

sync:  ## Sync dependencies from lockfile
	uv sync --frozen

lint:  ## Run code style and typing checks
	uv run ruff check .
	uv run mypy src tests

format:  ## Format code with ruff
	uv run ruff format .
	uv run ruff check --fix .

test:  ## Run pytest test suite
	uv run pytest

coverage:  ## Run test suite with coverage report
	uv run pytest --cov=src/flexa --cov-report=term-missing --cov-report=html

check: lint test  ## Run all linting and tests

forecast:  ## Run the forecasting pipeline (champion GBDT Delta T)
	uv run python scripts/run_forecast.py

battery:  ## Run the battery steering optimizer (1h vs 2h arbitrage)
	uv run python scripts/run_battery.py

report:  ## Build the combined PDF report (codebase guide + forecast + BESS)
	uv run python scripts/generate_full_report.py

docker-build:  ## Build the Flexa docker image
	docker build -t flexa:latest .

docker-forecast:  ## Run the forecasting pipeline inside docker
	docker run --rm -v "$$(pwd)/data:/app/data" flexa:latest forecast

docker-battery:  ## Run the battery steering optimizer inside docker
	docker run --rm -v "$$(pwd)/data:/app/data" flexa:latest battery

docker-report:  ## Build the combined PDF report inside docker
	docker run --rm -v "$$(pwd)/data:/app/data" flexa:latest report

notebook:  ## Launch interactive marimo EDA notebook
	uv run marimo edit notebooks/eda_raw_data.py

clean:  ## Clean temporary caches and build files
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
