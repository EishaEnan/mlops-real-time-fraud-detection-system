# ===== Variables =====
SHELL := /bin/bash
MLFLOW_PORT ?= 5500
MLFLOW_HOST_URI := http://localhost:$(MLFLOW_PORT)
MLFLOW_DOCKER_URI := http://mlflow:5000

# Prefer venv python; fallback to python3/python
PYTHON := $(if $(wildcard venv/bin/python),venv/bin/python,$(shell command -v python3 2>/dev/null || command -v python))

# Default to host MLflow unless overridden
export MLFLOW_TRACKING_URI ?= $(MLFLOW_HOST_URI)

# ===== Targets =====
.PHONY: help mlflow-up train eval promote api-up smoke-api check-python check-mlflow-version

help:
	@echo "Available commands:"
	@echo "  make mlflow-up     - Start MLflow server (http://localhost:$(MLFLOW_PORT))"
	@echo "  make train         - Run training pipeline, log to MLflow"
	@echo "  make eval          - Run evaluation pipeline, log metrics"
	@echo "  make promote       - Promote best model to 'Staging'"
	@echo "  make api-up        - Start FastAPI service (http://localhost:8000)"
	@echo "  make smoke-api     - Verify /healthz"

check-python:
	@test -n "$(PYTHON)" || (echo "No python found. Create venv or install python3." && exit 1)

mlflow-up:
	docker compose up -d mlflow
	@echo "MLflow UI -> $(MLFLOW_HOST_URI)"

train: check-python
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
	$(PYTHON) training/pipelines/train_xgb.py

eval: check-python
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
	$(PYTHON) training/pipelines/evaluate.py

promote: check-python
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
	$(PYTHON) scripts/promote_best.py

api-up:
	docker compose up -d api
	@echo ">>> API running at http://localhost:8000/healthz"

smoke-api:
	$(PYTHON) scripts/smoke_api.py

check-mlflow-version:
	@echo "Host:"; \
	$(PYTHON) -c 'import mlflow; print(mlflow.__version__)'
	@echo "API container:"; \
	docker compose run --rm api python -c 'import mlflow; print(mlflow.__version__)'
	@echo "MLflow server:"; \
	docker compose exec mlflow python -c 'import mlflow; print(mlflow.__version__)' || docker compose exec mlflow mlflow --version