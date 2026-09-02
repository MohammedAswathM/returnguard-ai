# Model Card

## Intended use

LightGBM estimates the probability of a simulated serial false-claim label at refund-request time. It supports evidence selection and bounded policy actions; it does not establish guilt, autonomously reject a customer, or replace operator review.

## Training and calibration

Only the chronological train period fits preprocessing and LightGBM. The calibration period alone compares sigmoid and isotonic calibration using a chronological half-fit/half-evaluate protocol. Isotonic was selected and refit on the complete calibration period. Policy-selection and final labels were excluded from all model and calibration choices.

## Locked performance

On 1,600 future cases at 9.3125% simulated prevalence: PR-AUC 0.7615, precision 77.31%, recall 61.74%, FPR 1.86%, Brier 0.0381, and counts 92 TP / 27 FP / 1,424 TN / 57 FN. Customer-cluster bootstrap PR-AUC 95% interval was 0.7005–0.8169.

Raw LightGBM PR-AUC was 0.8132. Isotonic ties reduced final ranking while preserving the frozen probability mapping; no post-final tuning occurred.

## Post-lock subgroup audit

The 55 requests exceeding original captured payment were all simulated positives, and `amount_paid_ratio > 1` identifies them exactly. Removing them without changing predictions reduces raw AP from 0.8132 to 0.5935 and frozen-threshold recall from 61.74% to 39.36%. The model directly uses requested-to-paid ratio, which contributed strongly positive SHAP values in this subgroup. This is a possible generator shortcut and a required v2 redesign item; v1 was not retrained or tuned.

## Metric glossary

- **Initial legitimate challenge rate:** stage-A non-auto actions among all legitimate cases: 184 / 1,451 = 12.68%.
- **Legitimate rescue rate:** initially challenged legitimate cases restored to auto-approval among all legitimate cases: 168 / 1,451 = 11.58%.
- **Challenged-legitimate rescue rate:** those rescues among challenged legitimate cases: 168 / 184 = 91.30%.
- **Terminal legitimate intervention rate:** challenged legitimate cases remaining non-auto at the final batch action: 16 / 1,451 = 1.10%. It is not a latency measurement.
- **Classifier false positive:** a legitimate case above the frozen reporting threshold, independent of the policy action.

## Explanations

TreeSHAP supplies no more than three risk-increasing and two mitigating factors. Each reason contains the actual feature, observed value, development reference, signed contribution, deterministic code, and neutral merchant-facing text. Reasons describe model contributions, not causes or guilt.

## Prohibited use

Do not use the score as an autonomous denial, customer accusation, cross-merchant blacklist, credit decision, or production-performance claim.
