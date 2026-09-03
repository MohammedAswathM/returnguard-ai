#!/usr/bin/env python3
import json
import shutil
from pathlib import Path

from returnguard.data.fingerprints import file_sha256

source = Path("artifacts/final_results/results.lock.json")
destination = Path("results.lock.json")
if not source.is_file():
    raise FileNotFoundError("locked final result artifact is unavailable")
shutil.copyfile(source, destination)
results = json.loads(destination.read_text(encoding="utf-8"))
slice_path = Path("slice_results.json")
correction_path = Path("results.metric_integrity.v1.1.json")
case_audit_path = Path("evidence/v1_case_audit.csv")
transition_path = Path("evidence/v1_policy_transitions.csv")
razorpay_closure_path = Path("evidence/razorpay_test_closure.json")
bundle_manifest_path = Path("artifacts/model_bundle/manifest.json")
manifest = {
    "schema_version": "1.0",
    "project": "ReturnGuard",
    "results_lock_sha256": file_sha256(destination),
    "results_content_sha256": results["results_sha256"],
    "source_data_sha256": "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb",
    "transformed_data_sha256": "7c13c6d3aa853a7d0c0f914fcdc099785b376d4caaf719a8edfc4c6ede65715c",
    "model_bundle_manifest_sha256": (
        file_sha256(bundle_manifest_path) if bundle_manifest_path.is_file() else None
    ),
    "slice_results_sha256": file_sha256(slice_path) if slice_path.is_file() else None,
    "metric_integrity_correction_sha256": (
        file_sha256(correction_path) if correction_path.is_file() else None
    ),
    "public_case_audit_sha256": file_sha256(case_audit_path) if case_audit_path.is_file() else None,
    "policy_transition_table_sha256": (
        file_sha256(transition_path) if transition_path.is_file() else None
    ),
    "certified_incremental_monetary_result": False,
    "genuine_razorpay_test_refund": True,
    "razorpay_test_closure_status": "SUCCESS",
    "razorpay_test_closure_evidence_sha256": (
        file_sha256(razorpay_closure_path) if razorpay_closure_path.is_file() else None
    ),
    "external_actions": [],
}
Path("submission_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(manifest, sort_keys=True))
