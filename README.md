# ReturnGuard AI

ReturnGuard separates deterministic payment validity from probabilistic return-abuse risk, then uses one structured evidence check to rescue legitimate customers before a bounded refund action.

The submission has two deliberately separate frozen surfaces. **V2 is the primary clean offline ML benchmark**; the interactive API and golden workflows use the previously validated **v1 operational demonstration bundle**. The completed genuine Razorpay Test Mode refund validates the human-gated, webhook-confirmed execution layer, not v2 online deployment.

## Legitimate-customer rescue

In the locked v2 benchmark, 365 of 2,160 legitimate claims were initially challenged. Consistent evidence restored auto-approval for 255: an **11.81% legitimate rescue rate** and **69.86% challenged-legitimate rescue rate**. The remaining **5.09% terminal legitimate intervention rate** is customer friction, not elapsed delay. No workflow-latency metric is claimed.

The system never autonomously accuses or rejects a customer. Payment-invalid requests stop at deterministic integrity rules. Technically valid, ambiguous claims receive a calibrated behavioral score, at most one verifier, a transparent Bayesian posterior, and a bounded action with named human authorization where required.

## Workflow

```text
refund request
  -> authoritative payment snapshot + atomic balance reservation
  -> deterministic payment-integrity lane (invalid requests stop here)
  -> causal LightGBM score for technically valid ambiguous claims
  -> sigmoid-calibrated probability
  -> one structured order-integrity verification when useful
  -> published likelihood ratio + immutable posterior
  -> approve / return-first / named human review
  -> idempotent Razorpay Test Mode adapter
  -> append-only SQLite audit
```

Within the v2 offline benchmark, training, replay, and evaluation use the same ordered point-in-time feature engine. Histories exclude the current request, admit only matured outcomes, apply neutral cold-start priors, and exclude payment-invalid reason codes and balance-ratio shortcuts. Migrating that feature contract into the operational API is intentionally outside this frozen submission.

## Locked v2 evidence

V2 was preregistered before generation. Its final test was opened once after the data, feature schema, model, calibrator, likelihoods, policy, evaluator, costs, dependencies, and hashes were frozen.

| Metric | Locked v2 result |
|---|---:|
| Support / simulated prevalence | **2,400 / 10.00%** |
| Raw average precision (PR-AUC) | **0.1400** (95% CI **0.1178-0.1691**) |
| Precision / recall at frozen capacity threshold | **15.38% / 2.50%** |
| False-positive rate | **1.53%** |
| Counts | **6 TP, 33 FP, 2,127 TN, 234 FN** |
| Calibrated Brier / log loss | **0.0897 / 0.3237** |
| Manual reviews | **45.83 per 1,000** |
| Simulated abuse-case intervention recall | **25.00%** |
| Simulated abuse-amount intervention recall | **24.91%** |

The corrected v2 bootstrap and cost definitions are in `results.v2.metric_integrity.v2.0.1.json`. The original v2 result lock remains unchanged for provenance. V2 discrimination is modest; these results are not production-performance evidence.

## Policy tradeoff

All values below are simulated policy costs under the same frozen assumptions, not realized savings or recovered money.

| V2 policy | Modeled cost | Initial legitimate challenge | Terminal legitimate intervention | Legitimate return-first burden | Reviews / 1,000 | Abuse case intervention | Abuse-amount intervention |
|---|---:|---:|---:|---:|---:|---:|---:|
| Approve all | INR 7,828,077.72 | 0.00% | 0.00% | 0 | 0.00 | 0.00% | 0.00% |
| Return-first all | **INR 3,405,216.85** | 100.00% | 100.00% | 2,160 | 0.00 | 100.00% | 100.00% |
| Frozen fixed bands | INR 7,622,836.75 | 1.53% | 1.53% | 0 | 16.25 | 2.50% | 2.71% |
| Adaptive verification | INR 6,559,069.27 | 16.90% | **5.09%** | **16** | 45.83 | 25.00% | 24.91% |

Return-first-all is cheapest under the declared simulation, but imposes a return on every legitimate customer. Adaptive verification is not presented as economically dominant; its measured advantage is rescuing 255 legitimate customers while avoiding blanket return-first burden.

## Shortcut audit and robustness

V1 mixed 55 certainly payment-invalid requests into ML evaluation, and `amount_paid_ratio > 1` identified all 55. V2 moves payment validity into a separate 200-case deterministic suite and prohibits captured amount, refundable balance, amount ratios, integrity codes, future outcomes, and generator metadata from model inputs.

On development data, maximum single-feature AP was **0.1273**, maximum decision-stump AP was **0.1131**, and permuted-label AP was **0.1029** at 10% prevalence. Five complete non-final generator replays produced mean raw AP **0.1495**, standard deviation **0.0088**, and worst-seed AP **0.1361**. These are synthetic stress tests, not additional real-world held-out evidence.

The cold-start neutral-prior counterfactual achieved AP **0.1336**. Missing evidence and verifier unavailability remain weak customer-friction slices; technical unavailability has LR exactly `1.0` and cannot increase risk.

## Razorpay Test Mode

One genuine bounded INR 1 partial refund was completed in Razorpay Test Mode. A fresh authoritative captured-payment check preceded the local reservation; signed raw-body `refund.created` and `refund.processed` webhooks were validated; refunded amount increased by exactly 100 paise; replay returned the same effect and created no second refund. Sanitized evidence is in `evidence/razorpay_test_closure.json`. Credentials, signatures, full payloads, and identifiers are not published.

The reproducible test adapter remains explicitly labelled `MOCK_RAZORPAY_TEST_ADAPTER`. Genuine execution requires ignored environment credentials and never runs during automated tests.

## Run locally

Use Python 3.11 or newer in a virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/run_api.py
```

In another terminal:

```bash
. .venv/bin/activate
streamlit run scripts/run_dashboard.py
```

Open `http://127.0.0.1:8501`. The golden rescue workflow is reachable from **Golden workflows** in two actions.

## Reproducibility

Public non-secret checks for the locked repository:

```bash
pytest -q
make lint
make typecheck
make verify-v2-repository
make preflight-repository
```

`make verify-v2-locked` additionally validates the ignored local v1 model bundle when that development artifact is present.

A clean generation run starts with the official [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) archive (CC BY 4.0), SHA-256 `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb`. V2 contains 7,200 train, 1,200 calibration, 1,200 policy-selection, and 2,400 locked future cases. UCI provides transaction distributions; refund operations, evidence, scenarios, verifier outcomes, policy costs, and simulated abuse labels are benchmark constructs.

## Metric glossary

- **Initial legitimate challenge rate:** legitimate claims whose stage-A action is not immediate auto-approval, divided by all legitimate claims.
- **Legitimate rescue rate:** challenged legitimate claims later auto-approved, divided by all legitimate claims.
- **Challenged-legitimate rescue rate:** rescued legitimate claims divided by challenged legitimate claims.
- **Terminal legitimate intervention rate:** legitimate claims with a final non-auto action, divided by all legitimate claims. It is not a latency measure.
- **Classifier false positive:** a legitimate case above the frozen probability threshold; it is separate from policy-level customer friction.
- **Simulated abuse-amount intervention recall:** submitted amount on simulated-abuse cases receiving a non-auto final action divided by submitted amount on all simulated-abuse cases. It is not executable or protected money.

## Historical v1 evidence

V1 remains byte-identical and auditable. Its corrected definitions are 184/1,451 initial legitimate challenges (12.68%), 168/1,451 legitimate rescues (11.58%), 168/184 challenged-legitimate rescues (91.30%), and 16/1,451 terminal legitimate interventions (1.10%). Its prior monetary headline is withdrawn because point-in-time refundable balance cannot be reconstructed. See `results.metric_integrity.v1.1.json`.

## Limitations

- Abuse labels, operational fields, verifier results, and costs are simulated; no production savings or fraud-prevention guarantee is claimed.
- UCI is UK-centric retail history converted into an INR-denominated benchmark, not India-wide validation.
- V2 raw AP is modest and several hard-legitimate slices are underpowered.
- The frozen capacity threshold has low recall; adaptive policy recall comes mainly from bounded verification, not classifier positives at that threshold.
- Technical verifier failure is neutral in risk but can still route a case to human review, creating customer friction.
- The dashboard reports v2 evidence, while the interactive API demo retains the separately validated v1 serving bundle; v2 serving-bundle migration is not claimed in this pass.
- Production use requires merchant-specific prospective validation, fairness review, monitoring, access controls, retention controls, incident response, and appeal operations.

See [EVALUATOR.md](EVALUATOR.md), [DATA_CARD.md](DATA_CARD.md), [MODEL_CARD.md](MODEL_CARD.md), [POLICY_CARD.md](POLICY_CARD.md), and [LIMITATIONS.md](LIMITATIONS.md).
