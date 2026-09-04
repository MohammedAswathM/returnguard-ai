#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from returnguard.data.fingerprints import file_sha256
from returnguard.evaluation.model_metrics import reliability_bins

result_path = Path("artifacts/v2/final_results/results.lock.json")
correction_path = Path("results.v2.metric_integrity.v2.0.1.json")
case_path = Path("artifacts/v2/final_results/case_predictions.parquet")
output_path = Path("evidence/v2/final_report.json")
cases = pd.read_parquet(case_path)
result = json.loads(result_path.read_text(encoding="utf-8"))
correction = json.loads(correction_path.read_text(encoding="utf-8"))
labels = cases["is_refund_abuse_simulated"].astype(bool).to_numpy()
probability = cases["calibrated_probability"].to_numpy(float)
logit = np.log(np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1))
calibration_fit = LogisticRegression(C=1e6, max_iter=2000).fit(logit.reshape(-1, 1), labels)
cases["posterior_risk_band"] = pd.cut(
    cases["posterior_probability"], [-np.inf, 0.08, 0.15, np.inf],
    labels=["LOW", "MEDIUM", "HIGH"],
)
cases["ground_truth_class"] = np.where(
    cases["is_refund_abuse_simulated"].astype(bool), "SIMULATED_ABUSE", "LEGITIMATE"
)
transitions = cases.groupby([
    "stage_a_action", "verification_result_simulated", "posterior_risk_band",
    "final_action", "ground_truth_class",
], observed=True, dropna=False).size().reset_index(name="count")
transition_path = Path("evidence/v2/policy_transitions.csv")
transitions.to_csv(transition_path, index=False, lineterminator="\n")
report = {
    "schema_version": "2.0.1", "benchmark_version": "returnguard-v2.0",
    "source_result_lock": {"path": result_path.as_posix(), "sha256": file_sha256(result_path)},
    "authoritative_correction": {
        "path": correction_path.as_posix(), "sha256": file_sha256(correction_path)
    },
    "case_export": {"path": case_path.as_posix(), "sha256": file_sha256(case_path)},
    "support": result["support"], "prevalence": result["prevalence"],
    "classifier": {
        **{key: result["classifier"][key] for key in (
            "tn", "fp", "fn", "tp", "precision", "recall", "fpr",
            "raw_average_precision", "calibrated_brier_score", "calibrated_log_loss",
        )},
        "bootstrap_95_percent_ci": correction["corrections"]["bootstrap_intervals"]["values"],
        "reliability_bins": reliability_bins(labels, probability, 10),
        "calibration_slope": float(calibration_fit.coef_[0, 0]),
        "calibration_intercept": float(calibration_fit.intercept_[0]),
    },
    "policy_comparison": correction["corrections"]["policy_cost"]["comparison"],
    "cold_start_counterfactual": result["cold_start_counterfactual"],
    "hard_legitimate_slices": result["hard_legitimate_slices"],
    "development_generator_seed_robustness": json.loads(Path(
        "artifacts/v2/development/generator_seed_robustness.json"
    ).read_text(encoding="utf-8"))["summary"],
    "shortcut_audit": {
        key: json.loads(Path(
            "artifacts/v2/development/shortcut_audit.json"
        ).read_text(encoding="utf-8"))[key]
        for key in (
            "max_single_feature_average_precision",
            "max_decision_stump_average_precision",
            "permuted_label_average_precision_mean",
        )
    },
    "transition_table": {
        "path": transition_path.as_posix(), "sha256": file_sha256(transition_path)
    },
    "disclosure": (
        "Official UCI transaction-derived foundation with simulated abuse labels, operational "
        "fields, verification results, and policy costs; not production performance."
    ),
}
output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"path": output_path.as_posix(), "sha256": file_sha256(output_path)}))
