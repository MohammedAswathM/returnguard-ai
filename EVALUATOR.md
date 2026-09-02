# Evaluator Guide

## Fast path

1. Create and activate a Python 3.11 or 3.12 virtual environment.
2. Run `python -m pip install -e ".[dev]"`.
3. Run `pytest -q`, `make lint`, `make typecheck`, and `make preflight-repository`.
4. With generated artifacts present, run `make preflight` and `python scripts/run_api.py`.
5. Run `streamlit run scripts/run_dashboard.py` and open the **Legitimate rescue** view.

## What to inspect

- `results.lock.json` preserves the frozen v1 output. `results.metric_integrity.v1.1.json` is authoritative for corrected public reporting and binds a case-level derivative.
- `submission_manifest.json` binds the result lock, UCI archive, transformed data, and local bundle manifest.
- `configs/features.yaml` declares ordered point-in-time features and maturity rules.
- `src/returnguard/verification/` contains the single verifier and Bayesian update.
- `src/returnguard/policy/` contains deterministic cost enumeration and the frozen operating contract.
- `src/returnguard/backend/` separates risk, verification, policy, persistence, and gateway effects.
- `evidence/v1_policy_transitions.csv` traces every frozen case through stage A, evidence, posterior band, final action, and the unavailable batch execution state.

## Three demonstrations

**Legitimate rescue:** A suspicious case receives one consistent order-integrity result, its posterior falls, and approval is restored before a labelled mock refund and full audit.

**Inconsistent evidence:** A structured mismatch routes to named human review. The verifier does not accuse, reject, or call the gateway.

**Safe failure:** Invalid bundle health or unavailable verification produces `FAILED_SAFE`, `INCONCLUSIVE`, or `RETURN_FIRST`; no unauthorized refund occurs.

## Evidence boundary

UCI Online Retail II supplies transaction distributions. Refund operations, evidence, abuse labels, costs, and outcomes are simulated. Results are held-out benchmark evidence, not production validation. Genuine Razorpay test closure remains an external action.

V1 cannot reconstruct point-in-time refundable balance. Its previous monetary headline is withdrawn; the evaluator should use the correction artifact for customer-impact definitions and frozen subgroup evidence.
