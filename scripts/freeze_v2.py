#!/usr/bin/env python3
import importlib.metadata
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from returnguard.data.fingerprints import file_sha256, object_sha256
from returnguard.v2.features import validate_ml_lane

output = Path("artifacts/v2")
freeze_path = output / "freeze_manifest.json"
if freeze_path.exists():
    raise FileExistsError("v2 freeze manifest already exists")
lock = json.loads((output / "locked_final/manifest.json").read_text(encoding="utf-8"))
audit = json.loads((output / "development/shortcut_audit.json").read_text(encoding="utf-8"))
policy = json.loads((output / "policy/policy.json").read_text(encoding="utf-8"))
if lock["state"] != "SEALED" or not audit["passed"]:
    raise ValueError("v2 final cannot freeze: seal or shortcut gate failed")
if policy["selected_metrics"]["manual_reviews_per_1000"] > 50:
    raise ValueError("v2 final cannot freeze: review capacity exceeded")
validate_ml_lane(output / "data")
packages = ["numpy", "pandas", "scikit-learn", "lightgbm", "shap", "pyarrow"]
versions = {name: importlib.metadata.version(name) for name in packages}
requirements_path = output / "requirements-lock.txt"
requirements_path.write_text(
    "".join(f"{name}=={version}\n" for name, version in sorted(versions.items())), encoding="utf-8"
)
paths = {
    "preregistration": Path("evidence/v2/preregistration.json"),
    "data_metadata": output / "data/metadata.json",
    "development_features": output / "development/features.parquet",
    "shortcut_audit": output / "development/shortcut_audit.json",
    "robustness": output / "development/robustness.json",
    "generator_seed_robustness": output / "development/generator_seed_robustness.json",
    "feature_config": Path("configs/v2/features.yaml"),
    "model_config": Path("configs/v2/model.yaml"),
    "policy_config": Path("configs/v2/policy.yaml"),
    "model": output / "training/model.joblib",
    "model_selection": output / "training/selection.json",
    "model_trials": output / "training/lightgbm_trials.json",
    "policy": output / "policy/policy.json",
    "final_truth": output / "locked_final/final_truth.csv",
    "final_evaluator": Path("src/returnguard/v2/final.py"),
    "final_script": Path("scripts/evaluate_v2_final.py"),
    "requirements_lock": requirements_path,
}
hashes = {name: file_sha256(path) for name, path in paths.items()}
manifest = {
    "schema_version": "2.0", "benchmark_version": "returnguard-v2.0",
    "status": "FROZEN_FOR_ONE_TIME_FINAL_EVALUATION",
    "frozen_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "artifact_hashes": hashes, "aggregate_hash": object_sha256(hashes),
    "operating_constraints": {"manual_review_rate_max": 0.05,
                              "initial_legitimate_challenge_rate_max": 0.20,
                              "max_verifications_per_request": 1,
                              "no_autonomous_rejection": True},
    "v1_provenance": {
        "results_lock_sha256": file_sha256(Path("results.lock.json")),
        "metric_correction_sha256": file_sha256(Path("results.metric_integrity.v1.1.json")),
    },
    "genuine_razorpay_evidence_sha256": file_sha256(Path("evidence/razorpay_test_closure.json")),
}
freeze_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
Path(f"{freeze_path}.sha256").write_text(
    f"{file_sha256(freeze_path)}  {freeze_path.name}\n", encoding="utf-8"
)
print(json.dumps({"status": manifest["status"], "freeze_sha256": file_sha256(freeze_path)}))
