# Evaluator Guide

## Fast path

1. Create a Python 3.11+ virtual environment and run `python -m pip install -e ".[dev]"`.
2. Run `pytest -q`, `make lint`, `make typecheck`, and `make preflight`.
3. Run `python scripts/run_api.py` and `streamlit run scripts/run_dashboard.py`.
4. Open **Golden workflows** for the two-action legitimate rescue, inconsistent evidence, and safe-failure paths.

Automated tests always use the labelled mock adapter. One genuine Razorpay Test Mode refund has already been completed and must not be repeated; inspect only the sanitized `evidence/razorpay_test_closure.json` record.

## Evidence map

- `evidence/v2/preregistration.json`: v2 protocol fixed before generation.
- `artifacts/v2/data/metadata.json`: source, transformed fingerprints, split support, provenance, and lane checks.
- `artifacts/v2/development/shortcut_audit.json`: univariate, stump, mutual-information, and permutation diagnostics.
- `artifacts/v2/training/selection.json`: rolling-origin model comparison and calibration-only selection.
- `artifacts/v2/policy/policy.json`: frozen likelihood ratios, thresholds, constraints, and selection results.
- `artifacts/v2/freeze_manifest.json`: pre-final hashes and operating contract.
- `artifacts/v2/final_results/results.lock.json`: immutable one-time v2 final output.
- `results.v2.metric_integrity.v2.0.1.json`: authoritative corrected bootstrap and policy-cost reporting.
- `evidence/v2/final_report.json`: reliability, slices, robustness, and public metrics derived from case predictions.
- `results.lock.json` and `results.metric_integrity.v1.1.json`: preserved historical v1 evidence and correction.

## What v2 proves

The benchmark separates 200 deterministic payment-integrity cases from 12,000 technically valid ambiguous ML claims. The final ML split contains 2,400 chronologically later cases at 10% simulated prevalence. Raw AP is 0.1400, calibrated Brier is 0.0897, and the adaptive policy rescues 255 legitimate cases while using 45.83 manual reviews per 1,000.

This is modest simulation evidence, not production validation. Return-first-all has the lowest simulated cost under the declared assumptions but imposes returns on all legitimate customers; adaptive verification demonstrates a customer-rescue tradeoff, not universal economic dominance.

The dashboard's evidence views are v2-backed. The interactive API demo deliberately retains the already validated v1 serving bundle; this work does not claim v2 online serving parity.

## Integrity checks

The preflight validates v1 and v2 hashes, v2 confusion arithmetic, authoritative correction linkage, policy counts, README claims, genuine Razorpay evidence, secret patterns, private-file references, and bundle integrity. The final result cannot be regenerated in place: opening code rejects an existing lock.

No environment credentials, webhook signatures, full API payloads, payment IDs, or refund IDs are required for evaluation.
