PYTHON ?= python3

.PHONY: setup download-uci validate-uci data data-uci data-robustness train-baselines train-baselines-uci train-primary select-policy robustness bundle metric-integrity publish-evidence test lint typecheck verify verify-phase-1-5 verify-phase-2 verify-phase-3 verify-phase-4 preflight preflight-repository api dashboard clean-artifacts

setup:
	$(PYTHON) -m pip install -e ".[dev]"

download-uci:
	$(PYTHON) scripts/download_uci.py --output data/raw/online_retail_ii.zip

validate-uci:
	$(PYTHON) scripts/validate_uci.py --archive data/raw/online_retail_ii.zip

data:
	$(PYTHON) scripts/generate_benchmark.py --config configs/data.yaml

data-uci: validate-uci
	$(PYTHON) scripts/generate_benchmark.py --config configs/data_uci.yaml

data-robustness: validate-uci
	$(PYTHON) scripts/generate_benchmark.py --config configs/data_uci_seed_20260901.yaml
	$(PYTHON) scripts/generate_benchmark.py --config configs/data_uci_seed_20260902.yaml

train-baselines:
	$(PYTHON) scripts/train_baselines.py --data-dir artifacts/data --config configs/model_baseline.yaml

train-baselines-uci:
	$(PYTHON) scripts/train_baselines.py --data-dir artifacts/data_uci --config configs/model_baseline_uci.yaml

train-primary:
	$(PYTHON) scripts/train.py

select-policy:
	$(PYTHON) scripts/select_policy.py

robustness: data-robustness
	$(PYTHON) scripts/evaluate_robustness.py \
		--alternate-seed-dir artifacts/robustness/data_uci_seed_20260901 \
		--alternate-seed-dir artifacts/robustness/data_uci_seed_20260902

bundle:
	$(PYTHON) scripts/export_bundle.py

metric-integrity:
	$(PYTHON) scripts/build_metric_integrity_correction.py

publish-evidence:
	$(PYTHON) scripts/publish_evidence.py

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src scripts tests

typecheck:
	$(PYTHON) -m mypy src/returnguard

verify: data train-baselines test lint typecheck

verify-phase-1-5: data-uci train-baselines-uci test lint typecheck

verify-phase-2: train-primary select-policy robustness bundle test lint typecheck

verify-phase-3:
	$(PYTHON) -m pytest -q tests/integration/test_api_workflow.py
	$(MAKE) lint typecheck PYTHON=$(PYTHON)

verify-phase-4:
	$(PYTHON) -m pytest -q tests/integration/test_api_workflow.py tests/unit/test_razorpay_adapter.py
	$(MAKE) lint typecheck PYTHON=$(PYTHON)

preflight:
	$(PYTHON) scripts/preflight.py

preflight-repository:
	$(PYTHON) scripts/preflight.py --repository-only

api:
	$(PYTHON) scripts/run_api.py

dashboard:
	$(PYTHON) -m streamlit run scripts/run_dashboard.py

clean-artifacts:
	$(PYTHON) scripts/clean_artifacts.py
