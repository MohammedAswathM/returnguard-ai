# ReturnGuard AI

ReturnGuard rescues legitimate customers who look risky by using one structured evidence check before a bounded refund decision. In the frozen v1 benchmark, 168 of 184 initially challenged legitimate cases returned to auto-approval after consistent evidence.

## Product difference

This is not a generic fraud dashboard and it does not autonomously accuse or reject customers. A causal, calibrated stage-A score selects one bounded action: approve, verify one structured order-integrity fact, require a return, or route to a named operator. Verification updates risk through a published Bayesian likelihood ratio, preserving the original score and every material state in an append-only audit.

## 60-second workflow

```text
refund request
  -> point-in-time LightGBM score + isotonic calibration
  -> deterministic expected-cost policy
  -> at most one order-integrity verification
  -> transparent Bayesian posterior
  -> approve / return-first / named human review
  -> idempotent Razorpay adapter
  -> append-only SQLite audit
```

Training, replay, API inference, and golden validation share the same ordered feature registry. The trusted bundle validates hashes, dependency versions, schema order, object types, and golden probability/action/reason parity before the API reports healthy.

## Live demo

Use Python 3.11 or 3.12 in a virtual environment:

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

Open `http://127.0.0.1:8501`, select **Golden workflows**, open **Legitimate rescue**, then use the two numbered controls. Without Razorpay credentials, execution is explicitly returned as `MOCK_RAZORPAY_TEST_ADAPTER`.

## Corrected customer-impact metrics

The versioned [metric-integrity correction](results.metric_integrity.v1.1.json) is authoritative for public reporting. It preserves the original lock and predictions while correcting one reporting definition.

| Metric | Numerator / denominator | Corrected result |
|---|---:|---:|
| Initial legitimate challenge rate | 184 / 1,451 | **12.68%** |
| Legitimate rescue rate | 168 / 1,451 | **11.58%** |
| Challenged-legitimate rescue rate | 168 / 184 | **91.30%** |
| Terminal legitimate intervention rate | 16 / 1,451 | **1.10%** |
| Manual reviews per 1,000 | 77 / 1,600 x 1,000 | **48.125** |

The original lock mislabeled the terminal legitimate intervention rate as a legitimate-delay rate. V1 contains no elapsed workflow latency, so no delay metric is claimed. Classifier false positives remain separate: the calibrated model produced 27 false positives at the frozen `0.5` threshold.

## Held-out model results

The locked chronological future test contains **1,600** refund requests at **9.3125% simulated abuse prevalence**. At the frozen 0.5 reporting threshold, calibrated LightGBM produced:

| Metric | Locked value |
|---|---:|
| PR-AUC | **0.7615** |
| Precision | **77.31%** |
| Recall | **61.74%** |
| FPR | **1.86%** |
| Brier score | **0.0381** |
| Counts | **92 TP, 27 FP, 1,424 TN, 57 FN** |

Customer-cluster bootstrap PR-AUC was 0.7640 median with a 95% interval of **0.7005–0.8169** across 1,000 resamples. Raw LightGBM PR-AUC was 0.8132; isotonic calibration selected on the calibration period improved probability reliability there but created tied scores and reduced final ranking. The model and calibrator were not changed after this result.

Rules achieved 0.3065 PR-AUC and Logistic Regression achieved 0.6615 PR-AUC on the same locked cases. These results use an official UCI transaction-derived foundation with simulated abuse labels and operational fields. They are benchmark evidence, not production performance.

The separate 128-case cold-start challenge contained 17 simulated positives and achieved 0.7916 PR-AUC. Every named hard-legitimate final slice contained only 29–32 cases and no simulated positives, so those slice results are explicitly underpowered. Pre-final prevalence stress ranged from 0.4614 PR-AUC at 2% to 0.8569 at 15%; 5% and 10% label-noise stress produced 0.5778 and 0.4887. Scaling false-positive friction from 0.5× to 2× moved frozen-policy expected cost from ₹808,772 to ₹816,051 per 1,000 policy-selection requests. Removing the request/order feature group reduced PR-AUC to 0.0827.

## Legitimate-customer rescue

The locked policy produced **168 legitimate rescues**. These are 11.58% of all legitimate cases and 91.30% of the 184 legitimate cases initially challenged. The repeatable demo moved probability from 0.1290 to 0.01590 before approval, refund execution, and audit. Both recorded local runs used the labelled mock adapter.

## V1 policy and value boundary

The frozen adaptive policy produced **48.125 reviews per 1,000** and **90.12% simulated abuse-amount intervention recall** over submitted request amounts. Total attempted simulated requested value was ₹56,892,240.71. This is context only, not executable exposure.

V1 records original captured amount but has no point-in-time prior-refund ledger. Fifty-five requests exceeded even the original captured amount; the remaining 1,545 requests are only **requests not exceeding original captured payment**, not verified executable refunds. The prior ₹4,037,255-per-1,000 estimate is withdrawn and retained only in the correction artifact for provenance. No certified gateway-executable or incremental monetary result is claimed.

Under the declared simulation assumptions, fixed return-first bands had slightly lower modeled cost than adaptive verification. Adaptive verification's demonstrated advantage is customer rescue and reduced return-first burden, not a certified monetary win.

| Frozen policy replay | Initial legitimate challenge | Terminal legitimate intervention | Legitimate return-first burden | Reviews | Simulated abuse case intervention | Simulated abuse-amount intervention |
|---|---:|---:|---:|---:|---:|---:|
| Approve all | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 0.00% |
| Return-first all | 100.00% | 100.00% | 100.00% | 0 | 100.00% | 100.00% |
| Fixed risk bands | 12.68% | 12.68% | 12.47% | 77 | 89.26% | 91.16% |
| Adaptive verification | 12.68% | 1.10% | 0.90% | 77 | 86.58% | 90.12% |

## Frozen subgroup audit

The audit below uses existing predictions only; it is not model selection.

| Population | Support | Prevalence | TP / FP / TN / FN | Raw AP | Brier | Precision | Recall | FPR | Reviews |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All final cases | 1,600 | 9.31% | 92 / 27 / 1,424 / 57 | 0.8132 | 0.0381 | 77.31% | 61.74% | 1.86% | 77 |
| Requests not exceeding original captured payment | 1,545 | 6.08% | 37 / 27 / 1,424 / 57 | 0.5935 | 0.0395 | 57.81% | 39.36% | 1.86% | 25 |

All 55 over-captured cases were simulated positives and all crossed the classifier threshold. The model directly received `amount_paid_ratio`; `amount_paid_ratio > 1` exactly identifies this subgroup and has a large positive SHAP contribution. This is a possible shortcut and is reserved for a preregistered future benchmark redesign, not a post-test v1 retrain.

## Metric glossary

- **Initial legitimate challenge rate:** legitimate requests whose stage-A action was not immediate auto-approval, divided by all legitimate requests.
- **Legitimate rescue rate:** legitimate requests initially challenged and later auto-approved after consistent evidence, divided by all legitimate requests.
- **Challenged-legitimate rescue rate:** rescued legitimate requests divided by initially challenged legitimate requests.
- **Terminal legitimate intervention rate:** initially challenged legitimate requests whose final batch action remained non-auto-approved, divided by all legitimate requests. This is not elapsed delay.
- **Classifier false positive:** a legitimate case with calibrated probability at or above the frozen `0.5` reporting threshold; it is not synonymous with a policy intervention.
- **Simulated abuse-amount intervention rate:** submitted amount on simulated abuse cases with a non-auto final action, divided by submitted amount on all simulated abuse cases. It is not executable or protected money.

## Razorpay test integration

The adapter accepts only `rzp_test_` keys from environment variables, verifies a captured payment and refundable balance, sends integer paise, uses `X-Refund-Idempotency`, sanitizes stored metadata, verifies webhook HMAC-SHA256 over the raw body, and deduplicates webhook events.

No genuine Razorpay test refund was executed in this repository run because credentials and a captured test payment were unavailable. See `docs/external_action_checklist.json`. Mock success is never presented as Razorpay API success.

## Reproducibility

```bash
make setup
make download-uci
make verify-phase-1-5
make train-primary
make select-policy
make robustness
make bundle
make metric-integrity
make test
make lint
make typecheck
make preflight
```

The official [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) archive is CC BY 4.0. The validated archive contains 1,067,371 rows and has SHA-256 `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb`. The transformed benchmark fingerprint is `7c13c6d3aa853a7d0c0f914fcdc099785b376d4caaf719a8edfc4c6ede65715c`.

The tracked `results.lock.json` is preserved for v1 provenance. Public reporting corrections are in `results.metric_integrity.v1.1.json`, supported by the case-level derivative under `evidence/`. Raw data, generated datasets, model binaries, databases, credentials, and local artifacts are excluded from version control.
The hash-bound `slice_results.json` contains post-open cold-start and hard-legitimate slice summaries; its labels were never available during fitting or policy selection.

## Limitations

- Abuse outcomes, refund operations, evidence, and costs are simulated; UCI supplies transaction structure, not fraud labels.
- The benchmark is UK-centric retail history converted to an INR-denominated simulation, not India-wide validation.
- Hard-legitimate slices such as shared households and defect bursts are underpowered.
- Isotonic score ties reduced final PR-AUC relative to raw LightGBM ranking.
- V1 cannot establish point-in-time refundable balance or certify an incremental monetary result.
- The verifier checks structured order integrity only; missing or unavailable evidence is inconclusive, never inconsistent.
- A production rollout requires merchant-specific prospective validation, fairness review, monitoring, appeal operations, access controls, retention controls, and genuine Razorpay test closure.

See [EVALUATOR.md](EVALUATOR.md), [DATA_CARD.md](DATA_CARD.md), [MODEL_CARD.md](MODEL_CARD.md), [POLICY_CARD.md](POLICY_CARD.md), and [LIMITATIONS.md](LIMITATIONS.md).
