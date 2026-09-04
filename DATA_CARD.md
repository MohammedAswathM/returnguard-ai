# Data Card

## Foundation and provenance

V2 uses UCI Online Retail II under CC BY 4.0. The archive contains 1,067,371 rows and has SHA-256 `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb`. UCI supplies invoice, product, quantity, price, customer, country, and timestamp distributions. Cancellations remain transaction events and are never abuse labels.

Operational refund requests, payment/refund ledger state, evidence, device/address relationships, verifier results, scenarios, policy costs, and serial false-claim labels are simulated and identified in `artifacts/v2/data/metadata.json`.

## Two lanes

The 200-case deterministic lane covers over-balance amount, cumulative exhaustion, uncaptured payment, merchant/currency mismatch, missing payment, duplicate idempotency, concurrent balance conflict, gateway failure with reservation release, and webhook replay. These cases never enter ML metrics.

The 12,000-case ML lane contains only technically valid ambiguous claims at decision time: payment exists, merchant and currency match, status is captured, amount is positive and within simulated point-in-time balance, and complete simulated prior-refund history is present.

## Partitions

| Partition | Support | Use |
|---|---:|---|
| Train | 7,200 | Fit models |
| Calibration | 1,200 | Select/refit calibrator only |
| Policy selection | 1,200 | Freeze thresholds and operating policy |
| Locked future test | 2,400 | Opened once after freeze |

The transformed-data fingerprint is `da91c45038f2088d5fd31a9b00b04dfb6f57862013f11d028bbf4f2601586db0`. The preregistration hash is `f58e5ee470832c8819aea6ceb81106794c7b0558b86029f2d7667ba59e347a8e`.

## Hard legitimate cases

V2 includes defect bursts, high-value loyal customers, first refunds, cold starts, unusual purchases, missing evidence, and verifier unavailability. These are simulated stress conditions. Several final slices have only 68-69 cases and are underpowered.

## Limitations

The source is UK-centric and all refund-risk semantics are simulated. The benchmark does not establish merchant-specific prevalence, Indian consumer behavior, production fraud detection, fairness, executable monetary value, or realized policy outcomes.
