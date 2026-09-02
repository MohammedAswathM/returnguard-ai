# Policy Card

## Operating contract

The policy enumerates expected cost for auto-approve, one verification, return-first, and manual review using integer paise and declared loss, salvage, logistics, review, verification, friction, and delay assumptions. Selection used only the policy-selection period.

Frozen constraints are at most 5% manual review, at most one customer verification, at most 20% initial legitimate challenge during selection, and no autonomous adverse decision. Isotonic probability ties at the review boundary use a frozen refund-amount tie-breaker to respect capacity.

## Verifier update

The single order-integrity verifier checks payment/order match, captured status, refundable balance, quantity eligibility, delivery/reason compatibility, and allowlisted demo evidence. Missing data or verifier failure is inconclusive, not inconsistent.

Likelihood ratios use train and calibration support with Laplace smoothing. Consistent evidence has LR below one, inconsistent evidence LR above one, inconclusive evidence remains close to one, and expired evidence is neutral. The initial decision remains immutable; posterior risk is a separate record.

## Locked evidence

The final policy produced 48.125 reviews per 1,000 and 90.12% simulated abuse-amount intervention recall. It initially challenged 184 of 1,451 legitimate cases (12.68%), rescued 168 (91.30% of those challenged), and left 16 with a terminal intervention (1.10% of legitimate cases).

The original v1 monetary estimate is withdrawn because v1 lacks point-in-time prior-refund state and included submitted amounts above original captured payment. No certified gateway-executable or incremental monetary result is claimed. Fixed return-first bands had slightly lower modeled simulation cost; adaptive verification's observed advantage is customer rescue and reduced return-first burden.

Adaptive verification materially outperformed score-to-review on policy-selection cost. Fixed return-first bands were slightly cheaper than adaptive verification, so the adaptive strategy is justified by rescue and lower friction rather than an unconditional cost win.
