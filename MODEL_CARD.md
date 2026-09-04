# Model Card

## V2 model

ReturnGuard v2 estimates a simulated serial false-claim probability only after deterministic payment-integrity checks pass. It does not establish guilt, reject customers autonomously, or replace operator judgment.

V2 is the primary clean offline benchmark. The interactive API retains the validated v1 operational demonstration bundle because the existing request path does not construct the full v2 point-in-time feature contract. Genuine Razorpay evidence validates the execution layer, not v2 online serving.

Thirty-two bounded LightGBM trials used three rolling-origin folds within train. Mean temporal AP was 0.1572 for rules, 0.1573 for Logistic Regression, 0.1610 for default LightGBM, and 0.1720 for the selected regularized LightGBM. Selection used the preregistered stability/capacity-penalized objective. No final labels participated.

Sigmoid calibration beat isotonic on the calibration selection half by Brier (0.08465 versus 0.08509) and log loss (0.31039 versus 0.34492), then was refit on the full calibration period. Policy thresholds were selected on policy-selection only.

## Locked performance

The 2,400-case future test has 10% simulated prevalence. Raw AP is 0.1400 (customer-cluster bootstrap 95% CI 0.1178-0.1691). At the frozen capacity threshold: 6 TP, 33 FP, 2,127 TN, 234 FN; precision 15.38%, recall 2.50%, FPR 1.53%. Calibrated Brier is 0.0897 and log loss is 0.3237.

The original lock's precision/recall bootstrap intervals mixed raw-score and calibrated-probability scales. `results.v2.metric_integrity.v2.0.1.json` corrects those intervals without changing predictions or selection.

## Shortcut controls

V2 forbids payment balances, captured amounts, requested-to-paid ratios, invalidity codes, future outcomes, post-verification actions, scenario names, and seeds. Maximum development single-feature AP was 0.1273, stump AP 0.1131, and permuted-label AP 0.1029 at 10% prevalence. The largest mean absolute SHAP share was 0.372; the top three totaled 0.652.

## Feature groups

The ordered schema covers claim/order context, smoothed customer history, time-decayed refund velocity, customer-peer deviation, product defect context, claim-integrity evidence, supporting entities, and missingness. All history is computed before the current event is added and uses only outcomes matured strictly before decision time.

## Interpretation

V2 discrimination is modest. The score is suitable only as one input to bounded verification under this simulation. Technical verifier unavailability has likelihood ratio 1.0 and cannot increase the probability.

## Prohibited use

Do not use this model for autonomous denial, accusation, cross-merchant blacklisting, credit decisions, or claims of production performance or realized savings.
