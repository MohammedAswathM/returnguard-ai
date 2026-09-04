# Build Log

## 2026-08-31 - Phase 0/1

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
- Development-period metrics are fixture-only simulation diagnostics, not performance targets or
  production evidence. No final-test metric was computed.

The initial full-size replay exposed quadratic history scans. The point-in-time engine was changed
to sorted timestamp bisection and per-customer maturity heaps without changing cutoff semantics.

## 2026-09-01 - Official Transaction Foundation

- Downloaded the official UCI Online Retail II archive from the documented source.
- Validated archive identity, worksheets, schema, dates, missingness, duplicates, customer coverage,
  cancellations, and invalid quantity/price rows.
- Added invoice items, timestamped cancellation events, invoice-scoped guest customers, and complete
  field-level source/derived/simulated provenance.
- Added a physical final-truth seal. Development tables redact final labels, scenarios, outcomes,
  verifier results, and feature replay.
- Converted difficult legitimate scenarios into observable conditions including shared households,
  defect batches, carrier incidents, high-value damage, first-order returns, and missing evidence.
- Repeated the full generation with the same seed; every canonical table fingerprint matched.
- UCI-backed shallow-tree AP was 0.1898. Five-permutation mean AP was 0.1193 against 0.08875
  policy-selection prevalence. These are development diagnostics, not final results.
- Source SHA-256: `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb`.
- Generator configuration SHA-256: `901bde00e7d343b9a348150c7c2948a4431777cdd468918b0bb253335b7712b3`.
- Transformed data SHA-256: `7c13c6d3aa853a7d0c0f914fcdc099785b376d4caaf719a8edfc4c6ede65715c`.

### Gate evidence

- `make verify-phase-1-5 PYTHON=/tmp/returnguard-venv/bin/python`: passed.
- `pytest -q`: 26 passed at the Phase 1.5 checkpoint.
- Ruff and strict mypy passed across 27 source files.
- Final support and timestamp boundaries were exported while labels and verifier truth remained sealed.

## 2026-09-01 - Model, Verifier, Policy, and Locked Evaluation

- Fit LightGBM on train only and selected isotonic calibration on the calibration period only.
- Added versioned point-in-time feature parity, TreeSHAP reason codes, one deterministic order-integrity verifier, Laplace-smoothed likelihood ratios, and separate Bayesian posterior decisions.
- Selected the expected-cost operating contract on policy-selection data with 5% review capacity and a 20% legitimate-delay ceiling.
- Regenerated two independent official-UCI benchmarks for seed robustness; their policy PR-AUC values were 0.7645 and 0.8248.
- Completed 1,000-resample customer-cluster bootstrap, prevalence, label-noise, missing-feature, unseen-category, cold-start, hard-legitimate, and verifier-unavailable stress evidence.
- Froze source, data, split, feature, model, calibrator, likelihood, threshold, policy, and cost hashes before final access.
- Opened the final test once. The preserved v1 lock binds immutable case results. A v2 lock adds reconstructable policy and confidence-interval fields without changing the model, predictions, cases, or operating contract.
- Exported and validated a complete bundle with file hashes and golden probability, action, and reason-code parity.

### Locked evidence

- Support 1,600; simulated prevalence 9.3125%; TP 92, FP 27, TN 1,424, FN 57.
- Calibrated PR-AUC 0.7615; precision 77.31%; recall 61.74%; FPR 1.86%; Brier 0.0381.
- Customer-cluster bootstrap PR-AUC 95% interval 0.7005–0.8169.
- Originally reported 168 legitimate rescues, 48.125 reviews per 1,000, and a 1.10% value that was later identified as terminal legitimate intervention rather than elapsed delay.
- Raw PR-AUC exceeded calibrated PR-AUC because isotonic probability ties reduced ranking; no post-final tuning was performed.

## 2026-09-02 - API, Audit, Razorpay Boundary, and Dashboard

- Added FastAPI endpoints, SQLite repositories, validated lifecycle transitions, immutable decisions, one-active-verifier enforcement, named operator gates, append-only audit triggers, and idempotent execution.
- Added startup bundle validation and fail-closed health behavior.
- Added a labelled mock Razorpay adapter for tests and a test-mode-only HTTP adapter with captured-payment, refundable-balance, idempotency, sanitized metadata, and raw-body webhook checks.
- Added Portfolio, Review Queue, Case Detail, Evidence, and two-click Legitimate Rescue Streamlit views backed by locked artifacts and backend state.
- Ran the rescue workflow twice without database edits. Both runs moved probability from 0.1290 to 0.01590 and completed through `MOCK_RAZORPAY_TEST_ADAPTER`.
- Genuine Razorpay test closure remains an external action because no credentials or captured test payment were supplied.

### Gate evidence

- API integration and Razorpay adapter suite: 7 passed.
- Trusted-bundle API health returned `healthy`.
- Streamlit returned HTTP 200 on the local smoke port.

## 2026-09-02 - Submission Hardening

- Published the sanitized locked result and submission manifest while excluding raw data, generated datasets, model binaries, SQLite databases, credentials, and caches.
- Added evaluator, architecture, data, model, policy, limitations, responsible-use, and five-minute demo documentation.
- Added preflight checks for result hashes, confusion and denominator arithmetic, policy-value arithmetic, README claims, dashboard artifact loading, bundle/golden validation, simulation wording, test-mode wording, secrets, and local-file privacy.
- Confirmed all local development context remained ignored and untracked.
- Created a fresh repository copy without ignored content, installed into a new virtual environment, and ran the complete clean-repository gate.

### Final gate evidence

- Main worktree: 40 tests passed; Ruff passed; strict mypy passed across 51 source files.
- `make verify`: passed with deterministic fixture regeneration, baseline training, 40 tests, Ruff, and mypy.
- `make verify-phase-3`: passed; API integration 4/4.
- `make verify-phase-4`: passed; API/Razorpay integration 7/7.
- Full artifact preflight: passed with valid bundle and 112 public files scanned for secrets.
- Fresh repository copy: clean installation passed; 40 tests, Ruff, mypy, and repository-only preflight passed.
- API startup validated the trusted bundle; dashboard smoke returned HTTP 200.
- Genuine Razorpay test closure remains external and is not claimed.
- Completed previously omitted friction-cost and missing-feature-group calculations using policy-selection data only; model and policy artifacts were unchanged.
- Exported a hash-bound post-open slice summary from the immutable case file: cold-start support 128 with PR-AUC 0.7916; named hard-legitimate slices were underpowered.

## 2026-09-02 - Metric Integrity Correction and Payment Boundary

- Preserved the frozen v1 locks, case predictions, model, threshold, likelihoods, and policy byte-for-byte.
- Corrected the customer-impact definitions from the case export: 184 of 1,451 legitimate cases were initially challenged, 168 were rescued, and 16 retained a terminal intervention.
- Separated 55 simulated requests above original captured payment. Removing this deterministic subgroup reduced frozen raw AP from 0.8132 to 0.5935 and threshold recall from 61.74% to 39.36%.
- Identified `amount_paid_ratio` as a direct deterministic signal for that subgroup and documented the resulting shortcut risk without retraining.
- Withdrew the previous monetary headline because v1 has no point-in-time prior-refund ledger and cannot certify executable or incremental value.
- Added prospective payment snapshots, refund-ledger entries, deterministic pre-model integrity decisions, and atomic balance reservations.

## 2026-09-03 - Razorpay Test-Mode Closure Attempt

- Confirmed the local environment file is ignored and untracked without reading or displaying its contents.
- Confirmed that required credentials and webhook configuration are present and that the API key uses the test-mode prefix.
- After the identifier was corrected, an authoritative test-mode lookup verified credential-scoped access, captured status, INR currency, current refund state, and sufficient balance for a bounded partial refund.
- Confirmed that the externally managed HTTPS tunnel reaches the healthy backend.
- The running Windows API used `MOCK_RAZORPAY_TEST_ADAPTER`; its ₹1 local execution replayed idempotently, while authoritative Razorpay follow-up confirmed zero refunds and no genuine effect.
- Updated the identifier-free closure record with status `BLOCKED_EXTERNAL` and the exact API restart action required.
- Hardened the genuine adapter so the configured payment ID is not serialized into local snapshots or response metadata and signed webhook confirmation is required before a genuine reservation is finalized.
- Restarted the public backend with `RAZORPAY_TEST_MODE` and `SIGNED_WEBHOOK_REQUIRED` health capabilities.
- Executed one bounded 100-paise test-mode refund after fresh authoritative payment checks.
- Observed signed `refund.created` and `refund.processed` events through the external HTTPS tunnel; `refund.processed` finalized the reservation.
- Replayed the identical execute request and confirmed the same effect, one authoritative refund-count increase, and an exact 100-paise refunded-amount increase.

## 2026-09-03 - Benchmark V2 Two-Lane Redesign

- Preregistered the v2 generator, splits, feature allowlist, forbidden fields, temporal model search, calibration rule, policy constraints, bootstrap protocol, and final-opening conditions before data generation.
- Generated 12,000 technically valid ambiguous ML claims from the official UCI transaction foundation and a separate 200-case deterministic payment-integrity suite.
- Enforced explicit simulated payment snapshots and prior-refund ledger entries, chronological `7,200/1,200/1,200/2,400` splits, compute-before-update, matured outcomes, and a sealed final truth.
- Passed the development shortcut audit: maximum single-feature AP 0.1273, maximum stump AP 0.1131, and permuted-label AP 0.1029 at 10% prevalence.
- Compared rules, Logistic Regression, default LightGBM, and 32 bounded LightGBM trials on three rolling-origin train folds. Selected regularized LightGBM by the preregistered penalized objective.
- Selected sigmoid calibration using calibration data only. Froze verifier likelihoods with technical unavailability and timeout fixed at LR 1.0.
- Selected the adaptive policy on policy-selection only at 48.33 reviews per 1,000 and 18.98% initial legitimate challenge.
- Ran five complete non-final generator seeds; mean development raw AP was 0.1495, standard deviation 0.0088, and worst seed 0.1361. All replay final partitions remained sealed.
- Passed 63 tests, Ruff, strict mypy, and preflight before freezing. Preregistration SHA-256 is `f58e5ee470832c8819aea6ceb81106794c7b0558b86029f2d7667ba59e347a8e`; freeze SHA-256 is `4038c6d22fd5eb8dc7dffb982e5b6d12823fe96e3d6e499dedb8f1f54b95b30a`.
- Opened v2 final once. Support is 2,400 at 10% simulated prevalence; raw AP 0.1400, Brier 0.0897, and adaptive abuse-case intervention recall 25.00%.
- Preserved the original v2 result lock after identifying a reporting defect in bootstrap score scale and policy-cost attribution. Published a separate `v2.0.1` correction; no model, probability, threshold, likelihood, action, label, or cost assumption changed.
- Kept all historical v1 artifacts and the completed genuine Razorpay Test Mode closure evidence byte-identical.

### Final gate evidence

- Main worktree: 65 tests passed with three known SHAP dependency warnings; Ruff passed; strict mypy passed across 62 source files.
- Full preflight passed with the v2 lock, correction, freeze, public evidence, v1 provenance, and genuine Razorpay closure hashes validated.
- Phase 3 API integration passed 8 tests; Phase 4 API and Razorpay integration passed 11 tests without making another genuine refund.
- A fresh public repository copy excluding ignored files installed successfully into a new virtual environment. Its repository-only gate passed 65 tests, Ruff, strict mypy, secret scanning, and artifact-independent preflight.
- Confirmed `.env`, raw data, local environments, caches, and private development context were absent from the fresh public copy.

## 2026-09-04 - Submission Freeze

- Retained the v1 operational demonstration bundle after confirming that v2 cannot be wired into the current API without a serving-feature migration. The v2 model expects nineteen point-in-time features and is not packaged in the validated v1 serving-bundle contract; silently defaulting unavailable history/context would violate golden parity.
- Kept v2 as the primary clean offline ML benchmark and made the version boundary explicit in the README, architecture, model/policy cards, evaluator guide, and dashboard.
- Clarified that the completed genuine Razorpay Test Mode evidence validates the shared integrity, reservation, webhook, and exactly-once execution layer rather than v2 online deployment.
- Removed the ignored local demo guide from public test dependencies; it is not required by preflight or the submission manifest.
- Final verification: 65 tests passed with three known SHAP dependency warnings; Ruff passed; strict mypy passed across 62 source files; phase 3 passed 8 tests; phase 4 passed 11 tests; full preflight and secret scan passed across 182 files.
- No benchmark generation, training, calibration, policy selection, final evaluation, or genuine refund execution occurred during this freeze pass.
