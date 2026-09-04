#!/usr/bin/env python3
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from returnguard.data.fingerprints import file_sha256
from returnguard.v2.config import load_v2_policy_config
from returnguard.v2.final import bootstrap_intervals
from returnguard.v2.policy import policy_metrics

result_path = Path("artifacts/v2/final_results/results.lock.json")
case_path = Path("artifacts/v2/final_results/case_predictions.parquet")
output_path = Path("results.v2.metric_integrity.v2.0.1.json")
if output_path.exists():
    raise FileExistsError("v2 metric-integrity correction is immutable")
result = json.loads(result_path.read_text(encoding="utf-8"))
cases = pd.read_parquet(case_path)
config = load_v2_policy_config(Path("configs/v2/policy.yaml"))
threshold = float(result["classifier"]["threshold"])
corrected_ci = bootstrap_intervals(
    cases, cases["calibrated_probability"].to_numpy(float), threshold,
    resamples=1000, seed=20260903,
)


def cost_breakdown(frame: pd.DataFrame) -> dict[str, float]:
    legitimate = ~frame["is_refund_abuse_simulated"].astype(bool)
    abuse = ~legitimate
    approved = frame["final_action"].eq("AUTO_APPROVE")
    returned = frame["final_action"].eq("RETURN_FIRST")
    reviewed = frame["final_action"].eq("MANUAL_REVIEW")
    verified = frame["stage_a_action"].eq("VERIFY")
    amount = frame["claimed_refund_amount_paise"].astype(float)
    terms = {
        "approved_abuse_loss_paise": float((
            amount[abuse & approved] * (1 + config.additional_loss_rate)
        ).sum()),
        "returned_abuse_unsalvaged_value_paise": float((
            amount[abuse & returned] * (1 - config.salvage_rate)
        ).sum()),
        "verification_operating_cost_paise": float(verified.sum() * config.verification_cost_paise),
        "manual_review_operating_cost_paise": float(reviewed.sum() * config.review_cost_paise),
        "legitimate_verification_friction_paise": float(
            (legitimate & verified).sum() * config.legitimate_verification_friction_paise
        ),
        "legitimate_review_friction_paise": float(
            (legitimate & reviewed).sum() * config.legitimate_review_friction_paise
        ),
        "legitimate_review_delay_cost_paise": float(
            (legitimate & reviewed).sum() * config.legitimate_delay_cost_paise
        ),
        "return_processing_cost_paise": float(returned.sum() * config.reverse_logistics_cost_paise),
    }
    terms["total_modeled_cost_paise"] = sum(terms.values())
    return terms


strategies: dict[str, pd.DataFrame] = {"adaptive_verification": cases.copy()}
approve = cases.copy()
approve["stage_a_action"] = "AUTO_APPROVE"
approve["final_action"] = "AUTO_APPROVE"
strategies["approve_all"] = approve
returned = cases.copy()
returned["stage_a_action"] = "RETURN_FIRST"
returned["final_action"] = "RETURN_FIRST"
strategies["return_first_all"] = returned
fixed = cases.copy()
fixed["stage_a_action"] = "AUTO_APPROVE"
fixed["final_action"] = "AUTO_APPROVE"
fixed_mask = fixed["calibrated_probability"].ge(threshold)
fixed.loc[fixed_mask, "stage_a_action"] = "MANUAL_REVIEW"
fixed.loc[fixed_mask, "final_action"] = "MANUAL_REVIEW"
strategies["fixed_risk_bands"] = fixed
comparison = {}
for name, frame in strategies.items():
    counts = policy_metrics(frame, config)
    counts.pop("modeled_cost_paise")
    comparison[name] = {**counts, "modeled_cost": cost_breakdown(frame)}

correction = {
    "schema_version": "2.0.1", "benchmark_version": "returnguard-v2.0",
    "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    "correction_type": "post-lock reporting-integrity correction",
    "authoritative_for_public_reporting": True,
    "original_result_lock": {"path": result_path.as_posix(), "sha256": file_sha256(result_path)},
    "case_level_source": {"path": case_path.as_posix(), "sha256": file_sha256(case_path)},
    "corrections": {
        "bootstrap_intervals": {
            "original_error": (
                "Original precision and recall bootstrap intervals compared raw scores to a "
                "threshold defined on calibrated probabilities."
            ),
            "corrected_definition": (
                "Customer-cluster bootstrap with 1,000 resamples; calibrated probabilities and "
                "the frozen calibrated threshold are used for precision and recall."
            ),
            "values": corrected_ci,
        },
        "policy_cost": {
            "original_error": (
                "Original comparison charged verification cost to every non-approve strategy and "
                "did not debit unsalvaged value for abusive return-first cases."
            ),
            "corrected_definition": (
                "Action-specific simulated costs under the already frozen assumptions; no realized "
                "savings or production economics claim."
            ),
            "comparison": comparison,
        },
    },
    "unchanged": {
        "model": True, "features": True, "calibrator": True, "predictions": True,
        "thresholds": True, "likelihood_ratios": True, "policy_actions": True,
        "cost_assumptions": True, "labels": True,
    },
    "tuning_or_selection_performed": False,
}
output_path.write_text(json.dumps(correction, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"path": output_path.as_posix(), "sha256": file_sha256(output_path)}))
