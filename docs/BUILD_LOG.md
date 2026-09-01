# Build Log

## 2026-08-31 - Phase 0/1

- Treated the supplied parent-level `BRAIN1.md` as the `BRAIN.md` authority referenced elsewhere.
- Initialized a Python `src/` package with strict Pydantic contracts, UTC timestamps, and integer paise.
- Chose a deterministic fixture as the offline default; the official UCI downloader remains separate.
- Added a deterministic UCI workbook transformer; it is tested with a tiny local workbook fixture but
  the official archive was not downloaded during this phase run.
- Kept UCI cancellations as transaction events, never labels.
- Kept scenario and outcome fields outside the feature registry and enforced a leakage-name denylist.
- Defined four chronological partitions. Only `train` is accepted by the Logistic fit API.
- Created cold-start as a separate mapping, leaving primary final-test identities and rows unchanged.
- Used a fixed baseline threshold from config; no threshold or policy selection is performed in Phase 1.
- Redacted final-test prevalence and excluded final features/predictions/metrics until the Phase 2
  model bundle and policy freeze.
- Commands: `make data`, `make train-baselines`, `pytest -q`, `make verify`.

### Gate evidence

- Environment: local temporary virtualenv, Python 3.12.3; package declares Python 3.11/3.12.
- `python -m pip install -e ".[dev]"`: passed inside the isolated virtualenv.
- `pytest -q`: 21 passed.
- `ruff check src scripts tests`: passed.
- `mypy src/returnguard`: passed, 23 source files checked.
- `make verify PYTHON=/tmp/returnguard-venv/bin/python`: passed in 13 seconds.
- Split support: train 4,800; calibration 800; policy selection 800; final test 1,600.
- Policy-selection prevalence: 9.625%; label-permutation AP: 9.596%.
- One-level decision-tree AP: 19.494%; no single numeric split trivially solves the label.
- Development-period metrics are fixture-only simulation diagnostics, not performance targets or
  production evidence. No final-test metric was computed.

The initial full-size replay exposed quadratic history scans. The point-in-time engine was changed
to sorted timestamp bisection and per-customer maturity heaps without changing cutoff semantics.
