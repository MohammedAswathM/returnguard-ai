# Architecture

## Decision path

```text
authoritative payment snapshot + atomic balance ledger
             |
             v
deterministic payment integrity boundary
             |
             v
canonical request + prior matured history
             |
             v
shared feature engine -> calibrated LightGBM -> immutable stage-A decision
                                                |
                         +----------------------+
                         | VERIFY at most once
                         v
              deterministic order-integrity verifier
                         |
                         v
               smoothed likelihood ratio
                         |
                         v
              immutable Bayesian posterior
                         |
                         v
       expected-cost policy + named operator where required
                         |
                         v
       idempotent gateway adapter + append-only SQLite audit
```

## Boundaries

- The feature engine alone computes training, replay, and serving features.
- Payment ownership, currency, captured state, and remaining balance are checked before ML. Invalid or unavailable payment state cannot increase model risk.
- The model returns probability and structured reasons; it cannot move money.
- The verifier returns only consistent, inconsistent, inconclusive, or expired.
- The policy is deterministic and consumes declared costs and capacity constraints.
- The operator gate is mandatory for inconsistent or high-risk adverse decisions.
- The gateway accepts only an already approved request and a deterministic idempotency key.
- Execution refreshes authoritative test-mode payment state where available and atomically reserves local balance before the gateway call.
- SQLite triggers prohibit decision and audit mutation or deletion.

## Failure behavior

Bundle hash, schema, object-type, or golden-parity failure degrades health and blocks scoring. Critical request data blocks a model decision. Missing optional evidence is disclosed and cannot become an inconsistent result. Gateway failure records `FAILED_SAFE` without recording a successful refund.
