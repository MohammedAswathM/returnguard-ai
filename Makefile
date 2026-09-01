PYTHON ?= python3

.PHONY: setup data train-baselines test lint typecheck verify clean-artifacts

setup:
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) scripts/generate_benchmark.py --config configs/data.yaml

train-baselines:
	$(PYTHON) scripts/train_baselines.py --data-dir artifacts/data --config configs/model_baseline.yaml

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src scripts tests

typecheck:
	$(PYTHON) -m mypy src/returnguard

verify: data train-baselines test lint typecheck

clean-artifacts:
	$(PYTHON) scripts/clean_artifacts.py

