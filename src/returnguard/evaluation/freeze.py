from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from returnguard.data.fingerprints import file_sha256, object_sha256
from returnguard.models.bundle import validate_bundle
from returnguard.models.primary import CalibratedRiskModel


def _component_sha256(component: object) -> str:
    stream = io.BytesIO()
    joblib.dump(component, stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def _tree_sha256(paths: tuple[Path, ...]) -> str:
    files = sorted(file for path in paths for file in (path.rglob("*") if path.is_dir() else [path]) if file.is_file())
    payload = [{"path": file.as_posix(), "sha256": file_sha256(file)} for file in files]
    return object_sha256(payload)


def freeze_pre_final(
    data_dir: Path, lock_dir: Path, training_dir: Path, policy_dir: Path,
    bundle_dir: Path, feature_config: Path, policy_config: Path,
    source_commit: str, source_paths: tuple[Path, ...], output_path: Path,
) -> dict[str, Any]:
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["source_validation"]["status"] != "VALIDATED_OFFICIAL_SOURCE":
        raise ValueError("official UCI validation is required before freeze")
    lock = json.loads((lock_dir / "manifest.json").read_text(encoding="utf-8"))
    if lock["state"] != "SEALED" or lock["opened_at"] is not None:
        raise ValueError("final truth must still be sealed")
    validate_bundle(bundle_dir, data_dir, feature_config)
    model = joblib.load(training_dir / "model_pipeline.joblib")
    if not isinstance(model, CalibratedRiskModel):
        raise TypeError("training artifact has unexpected type")
    policy = json.loads((policy_dir / "policy.json").read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "schema_version": "1.0", "state": "FROZEN",
        "source_commit": source_commit, "source_tree_sha256": _tree_sha256(source_paths),
        "data_fingerprint": metadata["transformed_data_fingerprint_sha256"],
        "source_fingerprint": metadata["source_validation"]["archive"]["sha256"],
        "split_fingerprint": metadata["split_fingerprint_sha256"],
        "feature_schema_hash": file_sha256(training_dir / "feature_schema.json"),
        "model_hash": _component_sha256(model.estimator),
        "calibrator_hash": _component_sha256(model.calibrator),
        "pipeline_hash": file_sha256(training_dir / "model_pipeline.joblib"),
        "likelihood_table_hash": file_sha256(policy_dir / "verifier_likelihoods.json"),
        "threshold_hash": file_sha256(bundle_dir / "thresholds.json"),
        "policy_hash": file_sha256(policy_dir / "policy.json"),
        "cost_config_hash": file_sha256(policy_config),
        "bundle_manifest_hash": file_sha256(bundle_dir / "manifest.json"),
        "operating_constraints": {
            "max_manual_review_rate": policy["max_manual_review_rate"],
            "max_legitimate_delay_rate": policy["max_legitimate_delay_rate"],
            "max_customer_verifications": 1,
            "autonomous_adverse_decision": False,
        },
        "freeze_timestamp": datetime.now(UTC).isoformat(),
        "final_truth_sha256": lock["truth_sha256"],
    }
    payload["freeze_manifest_sha256"] = object_sha256(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
