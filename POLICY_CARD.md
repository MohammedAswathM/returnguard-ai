# Policy Card

## Frozen v2 contract

This contract is evaluated offline. The interactive API retains the v1 operational demonstration bundle; the completed genuine Razorpay Test Mode refund validates execution controls rather than v2 online deployment.

Policy selection used only the 1,200-case policy-selection period. Constraints are at most 5% manual review, at most one verification, at most 20% initial legitimate challenge during selection, and no autonomous rejection. The selected development policy produced 48.33 reviews per 1,000 and 18.98% initial legitimate challenge.

Likelihood ratios use train plus calibration counts with Laplace smoothing: consistent `0.2199`, inconsistent `17.7014`, evidence-inconclusive `1.1916`, verifier-unavailable `1.0`, and verifier-timeout `1.0`. Technical failure cannot raise risk. Initial and posterior decisions remain separate.

## Locked outcomes

On v2 final, adaptive verification produced 45.83 reviews per 1,000, 25.00% simulated abuse-case intervention recall, and 24.91% simulated abuse-amount intervention recall. It challenged 365 of 2,160 legitimate cases, rescued 255, and left 110 with a terminal intervention. Sixteen legitimate cases received return-first.

## Policy comparison

Corrected simulated costs are INR 7,828,077.72 approve-all, INR 3,405,216.85 return-first-all, INR 7,622,836.75 fixed bands, and INR 6,559,069.27 adaptive verification. Return-first-all is cheapest under these assumptions but imposes returns on all 2,160 legitimate cases. Adaptive verification is justified by customer rescue and bounded return burden, not unconditional economic superiority.

The original v2 lock charged verification cost too broadly and omitted unsalvaged value for abusive return-first cases. The versioned correction changes reporting arithmetic only; all model probabilities, thresholds, likelihoods, and actions remain frozen.

## Safety boundary

The verifier checks structured facts but cannot approve, reject, or call the gateway. Missing evidence is `EVIDENCE_INCONCLUSIVE`; infrastructure failure is `VERIFIER_UNAVAILABLE` or `VERIFIER_TIMEOUT`. High-risk and inconsistent cases require a named human, and payment execution remains subject to fresh authoritative balance checks and atomic reservation.
