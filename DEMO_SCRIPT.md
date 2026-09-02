# Five-Minute Demo

## 0:00–0:25 — Problem and difference

Instant refunds reduce support cost but expose merchants to serial false claims. ReturnGuard adds one least-cost evidence check only when the causal score and expected-cost policy justify it; it is not an autonomous fraud denial system.

## 0:25–0:50 — Architecture

Show the point-in-time score, one deterministic order-integrity verifier, Bayesian posterior, bounded policy, named operator gate, idempotent adapter, and append-only audit.

## 0:50–2:10 — Legitimate rescue

Open **Legitimate rescue**. Create and score the frozen demo case. Note the immutable prior near 0.1290 and `VERIFY`. Complete consistent evidence, show posterior near 0.01590 and approval, execute, and point out whether the adapter is mock or genuine test mode. Open the audit timeline.

## 2:10–2:50 — Inconsistent case

Show a payment/order or amount mismatch. The verifier returns structured inconsistency without accusing the customer. The policy routes to named manual review and the gateway remains untouched.

## 2:50–3:10 — Safe failure

Show degraded bundle health or unavailable verification. The case becomes `FAILED_SAFE`, `INCONCLUSIVE`, or `RETURN_FIRST`; no money moves.

## 3:10–4:20 — Held-out evidence and economics

Show 1,600 locked cases, 9.3125% simulated prevalence, PR-AUC 0.7615, confusion counts, reliability support, and bootstrap interval. Reconcile 184 initially challenged legitimate cases, 168 rescues, 16 terminal interventions, and 48.125 reviews per 1,000. Then show the frozen subgroup audit: raw AP falls from 0.8132 to 0.5935 after separating the 55 requests above original captured payment. State that the prior monetary estimate was withdrawn and fixed return-first had slightly lower modeled simulation cost.

## 4:20–5:00 — Razorpay relevance and limits

Show test-key enforcement, refundable-balance check, idempotency, raw-body webhook verification, and audit. Explicitly distinguish the mock demonstration from genuine Razorpay test closure. Close with simulated-label, UK-foundation, underpowered-slice, and production-validation limitations.
