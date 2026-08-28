.DEFAULT_GOAL := help
PY ?= python
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: help venv install synthetic splits smoke train evaluate explain export serve test lint type fmt clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

venv: ## Create a virtualenv
	$(PY) -m venv $(VENV)

install: ## Install package + dev deps
	$(BIN)/pip install -U pip
	$(BIN)/pip install -e ".[serve,dev]"

synthetic: ## Generate synthetic MURA-style data for smoke tests
	$(BIN)/python scripts/make_synthetic_data.py --out data/mura --patients 60

splits: ## Compute patient-disjoint folds + dataset summary
	$(BIN)/python scripts/prepare_splits.py --data-dir data/mura

smoke: synthetic ## End-to-end smoke run on synthetic data (fast_dev)
	$(BIN)/python scripts/train.py --experiment fast_dev
	$(BIN)/python scripts/evaluate.py --checkpoint artifacts/best.ckpt --experiment fast_dev --n-bootstrap 200

train: ## Full training run (strong recipe)
	$(BIN)/python scripts/train.py --experiment strong

evaluate: ## Evaluate a checkpoint (CKPT=path)
	$(BIN)/python scripts/evaluate.py --checkpoint $(CKPT)

explain: ## Grad-CAM overlays (CKPT=path)
	$(BIN)/python scripts/explain.py --checkpoint $(CKPT) --images data/mura --limit 24

export: ## Export TorchScript + ONNX (CKPT=path)
	$(BIN)/python scripts/export_model.py --checkpoint $(CKPT) --onnx

serve: ## Run the FastAPI service (FRACTURE_CKPT=path)
	$(BIN)/uvicorn fracture.serve.api:app --host 0.0.0.0 --port 8000

test: ## Run the test suite
	$(BIN)/pytest

lint: ## Ruff lint
	$(BIN)/ruff check src scripts tests

type: ## mypy
	$(BIN)/mypy src/fracture

fmt: ## Ruff format + autofix
	$(BIN)/ruff format src scripts tests
	$(BIN)/ruff check --fix src scripts tests

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} +
