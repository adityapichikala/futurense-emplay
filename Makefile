.PHONY: help install test lint typecheck check verify run api docker-build docker-run clean

PYTHON ?= python
SRC := src

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime + dev dependencies
	$(PYTHON) -m pip install -r requirements.txt

test: ## Run the test suite
	$(PYTHON) -m pytest tests -v

lint: ## Lint and auto-fix
	$(PYTHON) -m ruff check $(SRC) --fix
	$(PYTHON) -m ruff format $(SRC)

typecheck: ## Static type check
	$(PYTHON) -m mypy $(SRC)

cov: ## Test coverage report
	$(PYTHON) -m pytest tests -q --cov=rfp_extractor --cov-report=term-missing

check: lint typecheck test ## All quality gates

verify: check ## Quality gates + a real pipeline run over data/raw
	$(PYTHON) scripts/run_pipeline.py

run: ## Run the pipeline over data/raw and write artifacts
	$(PYTHON) scripts/run_pipeline.py

api: ## Start the FastAPI service with autoreload
	PYTHONPATH=$(SRC) $(PYTHON) -m uvicorn rfp_extractor.api.main:app --reload --port 8000

docker-build: ## Build the container image
	docker build -t rfp-extractor:1.0.0 .

docker-run: ## Run the container image
	docker run --rm -p 8000:8000 rfp-extractor:1.0.0

clean: ## Remove generated artifacts and caches
	rm -rf artifacts/* .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
