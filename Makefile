.PHONY: setup data smoke train evaluate test lint typecheck serve ui docker-up docker-down benchmark docs env

PYTHON ?= uv run python
UV ?= uv
API_PORT ?= 8101
UI_PORT ?= 3101
MLFLOW_PORT ?= 5101
ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

setup:
	$(UV) python pin 3.11
	$(UV) sync --extra dev --extra train --no-cache || $(UV) sync --extra dev --no-cache
	@echo "setup complete"

data:
	$(PYTHON) scripts/download_data.py --out data/raw --allow-synthetic
	$(PYTHON) scripts/prepare_dataset.py --config configs/data.yaml

smoke:
	$(PYTHON) scripts/prepare_dataset.py --config configs/local/data.yaml --smoke
	$(PYTHON) scripts/train_sft.py --config configs/local/sft.yaml --smoke
	$(PYTHON) scripts/train_dpo.py --config configs/local/dpo.yaml --smoke
	$(PYTHON) scripts/evaluate.py --config configs/local/evaluation.yaml --smoke
	$(PYTHON) scripts/collect_environment.py
	$(UV) run pytest tests/unit tests/data tests/model -q --tb=short
	@echo "smoke complete"

train:
	$(PYTHON) scripts/train_sft.py --config configs/sft.yaml
	$(PYTHON) scripts/train_dpo.py --config configs/dpo.yaml

evaluate:
	$(PYTHON) scripts/evaluate.py --config configs/evaluation.yaml

test:
	$(UV) run pytest tests -q --tb=short

lint:
	$(UV) run ruff check src scripts tests
	$(UV) run ruff format --check src scripts tests

typecheck:
	$(UV) run mypy src/resolvelm

serve:
	$(UV) run uvicorn resolvelm.api.app:app --host 0.0.0.0 --port $(API_PORT)

ui:
	cd dashboard && npm install && npm run dev -- --port $(UI_PORT)

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

benchmark:
	$(PYTHON) scripts/benchmark.py --config configs/local/evaluation.yaml

docs:
	$(PYTHON) scripts/collect_environment.py
	$(PYTHON) scripts/evaluate.py --config configs/local/evaluation.yaml --docs-only || true

env:
	$(PYTHON) scripts/collect_environment.py
