#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from returnguard.data.fingerprints import file_sha256, object_sha256
from returnguard.models.bundle import validate_bundle


def close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if abs(actual - expected) > tolerance:
        raise ValueError(f"arithmetic mismatch: {actual} != {expected}")


def candidate_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], text=True
    )
    return [Path(value) for value in output.splitlines() if Path(value).is_file()]


parser = argparse.ArgumentParser()
parser.add_argument("--repository-only", action="store_true")
args = parser.parse_args()

results_path = Path("results.lock.json")
results: dict[str, Any] = json.loads(results_path.read_text(encoding="utf-8"))
declared_hash = results.pop("results_sha256")
if object_sha256(results) != declared_hash:
    raise ValueError("results content hash mismatch")
results["results_sha256"] = declared_hash
metric = results["metrics"]["lightgbm_calibrated"]
if metric["tn"] + metric["fp"] + metric["fn"] + metric["tp"] != results["support"]:
    raise ValueError("confusion counts do not sum to support")
close(metric["precision"], metric["tp"] / (metric["tp"] + metric["fp"]))
close(metric["recall"], metric["tp"] / (metric["tp"] + metric["fn"]))
close(metric["fpr"], metric["fp"] / (metric["fp"] + metric["tn"]))
close(results["prevalence"], (metric["tp"] + metric["fn"]) / results["support"])
comparison = results["policy"]["comparison"]
expected_value = (
    comparison["approve_all"]["realized_cost_paise"]
    - comparison["returnguard_adaptive"]["realized_cost_paise"]
) * 1000 / results["support"]
close(results["policy"]["estimated_net_value_protected_paise_per_1000"], expected_value)

submission = json.loads(Path("submission_manifest.json").read_text(encoding="utf-8"))
if submission["results_lock_sha256"] != file_sha256(results_path):
    raise ValueError("submission manifest results hash mismatch")
closure_path = Path("evidence/razorpay_test_closure.json")
if submission.get("razorpay_test_closure_evidence_sha256") != file_sha256(closure_path):
    raise ValueError("submission manifest Razorpay closure evidence hash mismatch")
closure: dict[str, Any] = json.loads(closure_path.read_text(encoding="utf-8"))
if closure.get("contains_credentials_or_identifiers") is not False:
    raise ValueError("Razorpay closure evidence is not declared identifier-free")
if submission.get("razorpay_test_closure_status") != closure["status"]:
    raise ValueError("submission manifest Razorpay closure status mismatch")
genuine_closure = closure.get("status") == "SUCCESS"
if submission.get("genuine_razorpay_test_refund") is not genuine_closure:
    raise ValueError("genuine Razorpay status is not supported by closure evidence")
if genuine_closure:
    closure_outcome = closure.get("outcome", {})
    closure_checks = closure.get("checks", {})
    if closure_checks.get("refund_dispatch") != "GENUINE_TEST_MODE_ONE_RUPEE":
        raise ValueError("genuine Razorpay dispatch evidence is incomplete")
    if closure_checks.get("webhook_observation") != "RAW_BODY_SIGNATURE_VALIDATED":
        raise ValueError("genuine Razorpay webhook evidence is incomplete")
    if closure_outcome.get("refund_amount_paise") != 100:
        raise ValueError("genuine Razorpay refund was not the bounded amount")
    if closure_outcome.get("authoritative_refund_count_delta") != 1:
        raise ValueError("genuine Razorpay exactly-once count is unsupported")
    if closure_outcome.get("authoritative_refunded_amount_delta_paise") != 100:
        raise ValueError("genuine Razorpay authoritative amount delta is unsupported")
    if closure_outcome.get("execution_replay_same_effect") is not True:
        raise ValueError("genuine Razorpay replay evidence is incomplete")
    if closure_outcome.get("final_state") != "REFUNDED":
        raise ValueError("genuine Razorpay terminal state is unsupported")
else:
    if closure.get("checks", {}).get("refund_dispatch") not in {
        "NOT_ATTEMPTED", "NO_GENUINE_RAZORPAY_EFFECT",
    }:
        raise ValueError("Razorpay closure evidence conflicts with blocked status")
if submission.get("certified_incremental_monetary_result") is not False:
    raise ValueError("v1 cannot support a certified incremental monetary result")

correction_path = Path("results.metric_integrity.v1.1.json")
if submission.get("metric_integrity_correction_sha256") != file_sha256(correction_path):
    raise ValueError("submission manifest correction hash mismatch")
correction: dict[str, Any] = json.loads(correction_path.read_text(encoding="utf-8"))
correction_hash = correction.pop("correction_sha256")
if object_sha256(correction) != correction_hash:
    raise ValueError("metric-integrity correction content hash mismatch")
case_audit_path = Path(correction["provenance"]["public_case_audit"]["path"])
transition_path = Path(correction["provenance"]["transition_table"]["path"])
if file_sha256(case_audit_path) != correction["provenance"]["public_case_audit"]["sha256"]:
    raise ValueError("public case audit hash mismatch")
if file_sha256(transition_path) != correction["provenance"]["transition_table"]["sha256"]:
    raise ValueError("policy transition table hash mismatch")
if submission.get("public_case_audit_sha256") != file_sha256(case_audit_path):
    raise ValueError("submission manifest case-audit hash mismatch")
if submission.get("policy_transition_table_sha256") != file_sha256(transition_path):
    raise ValueError("submission manifest transition hash mismatch")
case_audit = pd.read_csv(case_audit_path)
legitimate = ~case_audit["label_simulated"].astype(bool)
challenged = legitimate & case_audit["stage_a_action"].ne("AUTO_APPROVE")
rescued = challenged & case_audit["verification_result"].eq("consistent") & case_audit[
    "final_action"
].eq("AUTO_APPROVE")
terminal = challenged & case_audit["final_action"].ne("AUTO_APPROVE")
corrected_metrics = correction["correction"]["authoritative_public_metrics"]
for name, numerator, denominator in (
    ("initial_legitimate_challenge_rate", int(challenged.sum()), int(legitimate.sum())),
    ("legitimate_rescue_rate", int(rescued.sum()), int(legitimate.sum())),
    ("challenged_legitimate_rescue_rate", int(rescued.sum()), int(challenged.sum())),
    ("terminal_legitimate_intervention_rate", int(terminal.sum()), int(legitimate.sum())),
):
    if corrected_metrics[name]["numerator"] != numerator or corrected_metrics[name]["denominator"] != denominator:
        raise ValueError(f"corrected metric denominator mismatch: {name}")
slice_path = Path("slice_results.json")
if submission.get("slice_results_sha256") != file_sha256(slice_path):
    raise ValueError("submission manifest slice-result hash mismatch")
slice_results = json.loads(slice_path.read_text(encoding="utf-8"))
slice_hash = slice_results.pop("slice_results_sha256")
if object_sha256(slice_results) != slice_hash:
    raise ValueError("slice result content hash mismatch")

readme = Path("README.md").read_text(encoding="utf-8")
required_claims = (
    "1,600", "9.3125%", "0.7615", "77.31%", "61.74%", "1.86%",
    "92 TP", "27 FP", "1,424 TN", "57 FN", "simulated abuse labels",
    "not production performance", "MOCK_RAZORPAY_TEST_ADAPTER", "12.68%", "91.30%",
    "Terminal legitimate intervention rate", "withdrawn", "0.5935",
)
missing_claims = [claim for claim in required_claims if claim not in readme]
if missing_claims:
    raise ValueError(f"README is missing locked claims/disclosures: {missing_claims}")
dashboard = Path("src/returnguard/dashboard/app.py").read_text(encoding="utf-8")
if "results.lock.json" not in dashboard or "load_results()" not in dashboard:
    raise ValueError("dashboard is not bound to the locked result artifact")
if "results.metric_integrity.v1.1.json" not in dashboard or "load_correction()" not in dashboard:
    raise ValueError("dashboard is not bound to the authoritative correction artifact")

for file in candidate_files():
    if file.suffix.lower() in {".png", ".jpg", ".jpeg", ".parquet", ".joblib", ".zip", ".xlsx"}:
        continue
    try:
        text = file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        continue
    if re.search(r"rzp_live_[A-Za-z0-9]{8,}", text):
        raise ValueError(f"possible live Razorpay key in {file}")
    if re.search(r"RAZORPAY_KEY_SECRET[ \t]*=[ \t]*[^ \t\r\n]+", text):
        raise ValueError(f"possible Razorpay secret in {file}")
    private_names = (
        "AG" + "ENTS.md", "CON" + "TEXT.md", "BR" + "AIN1.md",
        "MASTER_" + "PROMPT.md", "START_" + "HERE.md",
        "COLAB_TRAINING_" + "HANDOFF.md", "IMPLEMENTATION_" + "GATES.md",
    )
    if any(name in text for name in private_names):
        raise ValueError(f"private development filename referenced by {file}")

bundle_status = "NOT_PRESENT_REPOSITORY_ONLY"
bundle_dir = Path("artifacts/model_bundle")
if not args.repository_only:
    validation = validate_bundle(bundle_dir, Path("artifacts/data_uci"), Path("configs/features.yaml"))
    bundle_status = str(validation["status"])
    if submission["model_bundle_manifest_sha256"] != file_sha256(bundle_dir / "manifest.json"):
        raise ValueError("submission bundle manifest hash mismatch")

print(json.dumps({
    "status": "PASS", "support": results["support"],
    "results_sha256": declared_hash, "bundle": bundle_status,
    "secret_scan_files": len(candidate_files()),
}, sort_keys=True))
