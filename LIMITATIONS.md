# Limitations

- UCI Online Retail II provides UK-centric transaction distributions, not return-abuse labels or Indian merchant validation.
- Abuse labels, payment/refund histories, evidence, verifier results, operational scenarios, and policy costs are simulated.
- V2 final raw AP is 0.1400 at 10% prevalence. This modest ranking performance is not sufficient evidence for production deployment.
- The frozen capacity threshold recalls 2.50% of abuse labels; broader adaptive intervention reaches 25.00% by using structured verification.
- Several hard-legitimate final slices have only 68-69 cases and are underpowered.
- Verifier unavailability is probability-neutral but routes challenged cases to review, producing a high terminal-intervention rate in that small slice.
- Policy costs are assumption-dependent. Return-first-all is cheapest under the frozen simulation but creates unacceptable blanket customer burden; no realized or incremental monetary value is claimed.
- The original v2 lock contained invalid precision/recall bootstrap intervals and overbroad policy cost attribution. The original remains preserved; `results.v2.metric_integrity.v2.0.1.json` is authoritative for those fields.
- Historical v1 mixed 55 certainly invalid over-captured requests into ML evaluation. V2 corrects the benchmark design; v1 remains preserved with its own reporting correction.
- The completed genuine Razorpay evidence covers one bounded Test Mode partial refund and exactly-once replay. It is not a production integration, load test, or live-money claim.
- V2 is the primary offline benchmark, but the interactive API demo still uses the preserved v1 trusted bundle. A separate serving-parity migration is required before v2 can power API decisions.
- Production deployment requires merchant-specific prospective validation, fairness analysis, calibrated monitoring, access and retention controls, incident response, reconciliation, and appeal operations.
