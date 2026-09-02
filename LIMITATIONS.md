# Limitations and Responsible Use

- Abuse labels, evidence, refund operations, verifier outcomes, and economics are simulated.
- UCI Online Retail II is UK-centric transaction history, not India-wide or merchant-specific validation.
- The final test contains 149 simulated positive cases; several hard-legitimate slices are underpowered.
- The 128-case cold-start challenge has only 17 simulated positives; every named hard-legitimate final slice has 29–32 cases and no positive labels.
- Isotonic calibration introduced tied probabilities and reduced final PR-AUC versus raw model ranking.
- V1 has no point-in-time prior-refund ledger, so refundable balance and incremental monetary value cannot be certified.
- Fifty-five simulated abuse requests exceeded original captured payment, and the model directly used requested-to-paid ratio. The frozen full-test score therefore includes an easy deterministic subgroup.
- Structured order integrity cannot determine physical product damage or customer intent.
- Missing evidence is not evidence of abuse and must remain inconclusive.
- Genuine Razorpay test execution has not been completed in this run.

## Responsible use

The system should delay adverse outcomes until a named operator reviews the facts, provide neutral reasons and reconsideration, minimize retained evidence, enforce access controls, and measure legitimate-customer impact prospectively. Production use requires merchant-specific validation, legal and fairness review, monitoring, incident response, retention limits, and an appeal workflow.
