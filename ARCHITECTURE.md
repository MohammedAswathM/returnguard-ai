# Architecture

## Decision path

```text
authoritative payment state
  -> deterministic integrity boundary
  -> technically valid ambiguous claim
  -> shared point-in-time feature engine
  -> calibrated LightGBM prior
  -> deterministic policy
  -> at most one structured verifier
  -> likelihood-ratio posterior
  -> bounded action / named operator gate
  -> idempotent Razorpay adapter
  -> append-only audit
```

Payment existence, ownership, captured status, currency, positive integer-paise amount, refundable balance, idempotency, and atomic reservation are enforced before ML. Missing authoritative balance fails closed and does not alter fraud probability.

## Benchmark lanes

Lane A is a 200-case deterministic invariant suite and is excluded from ML metrics. Lane B contains 12,000 technically valid ambiguous claims partitioned chronologically into train, calibration, policy selection, and locked final. Simulated labels represent noisy behavioral combinations, not payment invalidity.

The v2 feature engine lives under `src/returnguard/v2/` and applies compute-before-update, strict event cutoffs, mature-outcome admission, neutral cold-start defaults, and unseen-category handling. Payment balances and invalidity outputs are forbidden model inputs.

## Model and policy ownership

Train fits preprocessing and candidate models. Calibration alone selects sigmoid versus isotonic. Policy selection alone fixes thresholds, likelihoods, costs, and capacity. The one-time final evaluator loads only frozen artifacts. Case predictions support independent metric reconstruction.

The risk model cannot call the gateway. The verifier cannot select actions or call the gateway. The policy is deterministic. SQLite transactions reserve balance before dispatch, and signed webhook reconciliation finalizes or releases reservations. Audit rows are append-only.

## Failure behavior

Invalid payment facts return explicit deterministic reason codes. Evidence absence is distinct from verifier unavailability and timeout. Technical failures use LR 1.0, never increase risk, and route to bounded human handling when required. Bundle/hash/schema failures degrade health and block money movement.
