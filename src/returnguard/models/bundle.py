from __future__ import annotations

import importlib.metadata
import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from returnguard.config import load_policy_config
from returnguard.data.fingerprints import file_sha256, object_sha256
from returnguard.features.engine import build_point_in_time_features
from returnguard.features.registry import load_feature_registry
from returnguard.models.explanations import explain_cases
from returnguard.models.primary import CalibratedRiskModel
from returnguard.policy.selection import PolicyContract, apply_contract

BUNDLE_FILES = (
    "model_pipeline.joblib", "feature_schema.json", "verifier_likelihoods.json",
    "policy.json", "thresholds.json", "metrics.json", "requirements-lock.txt",
    "golden_cases.parquet", "golden_predictions.json", "model_card.md", "data_card.md",
)


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copyfile(source, destination)


def export_bundle(
    data_dir: Path, training_dir: Path, policy_dir: Path, feature_config: Path,
    data_card: Path, bundle_dir: Path,
    policy_config_path: Path = Path("configs/policy.yaml"),
    final_results_path: Path = Path("artifacts/final_results/results.lock.json"),
) -> dict[str, Any]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    _copy(training_dir / "model_pipeline.joblib", bundle_dir / "model_pipeline.joblib")
    _copy(training_dir / "feature_schema.json", bundle_dir / "feature_schema.json")
    _copy(policy_dir / "verifier_likelihoods.json", bundle_dir / "verifier_likelihoods.json")
    _copy(policy_dir / "policy.json", bundle_dir / "policy.json")
    model = joblib.load(training_dir / "model_pipeline.joblib")
    if not isinstance(model, CalibratedRiskModel):
        raise TypeError("training model has unexpected type")
    policy = json.loads((policy_dir / "policy.json").read_text(encoding="utf-8"))
    thresholds = {
        key: policy[key] for key in (
            "approve_threshold", "review_threshold", "review_amount_threshold_paise",
        )
    }
    (bundle_dir / "thresholds.json").write_text(
        json.dumps(thresholds, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    final_metrics: dict[str, Any] = {"state": "SEALED"}
    if final_results_path.is_file():
        final_metrics = json.loads(final_results_path.read_text(encoding="utf-8"))
    metrics = {
        "training": json.loads((training_dir / "training_metrics.json").read_text(encoding="utf-8")),
        "policy_selection": json.loads(
            (policy_dir / "selection_metrics.json").read_text(encoding="utf-8")
        ),
        "robustness": json.loads((policy_dir / "robustness.json").read_text(encoding="utf-8")),
        "final_test": final_metrics,
    }
    (bundle_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    packages = ("numpy", "pandas", "scikit-learn", "lightgbm", "shap", "pyarrow", "pydantic")
    lock = "".join(f"{name}=={importlib.metadata.version(name)}\n" for name in packages)
    (bundle_dir / "requirements-lock.txt").write_text(lock, encoding="utf-8")
    predictions = pd.read_parquet(training_dir / "development_predictions.parquet")
    golden = predictions.loc[predictions["partition"] == "calibration"].head(20).copy()
    golden[["refund_request_id"]].to_parquet(bundle_dir / "golden_cases.parquet", index=False)
    registry = load_feature_registry(feature_config)
    features = build_point_in_time_features(data_dir, registry).set_index("refund_request_id")
    golden_features = features.loc[golden["refund_request_id"]]
    explanations = explain_cases(model, golden_features, registry)
    policy_config = load_policy_config(policy_config_path)
    contract = PolicyContract(**{
        key: policy[key] for key in PolicyContract.__dataclass_fields__
    })
    actions = apply_contract(
        model.predict_proba(golden_features),
        np.asarray(
            golden_features["requested_amount_paise"].astype("int64").to_numpy(), dtype=np.int64
        ),
        contract, policy_config,
    )
    golden_payload = []
    for row, action, reasons in zip(golden.to_dict(orient="records"), actions, explanations, strict=True):
        golden_payload.append({
            "refund_request_id": row["refund_request_id"], "raw_score": row["raw_score"],
            "calibrated_probability": row["calibrated_probability"], "action": action.value,
            "reason_codes": [reason["code"] for reason in reasons],
        })
    (bundle_dir / "golden_predictions.json").write_text(
        json.dumps(golden_payload, indent=2) + "\n", encoding="utf-8"
    )
    (bundle_dir / "model_card.md").write_text(
        "# Model Card\n\nThe primary model is LightGBM with calibration selected only on the "
        "chronological calibration period. Labels are simulated; this is not production evidence.\n",
        encoding="utf-8",
    )
    _copy(data_card, bundle_dir / "data_card.md")
    hashes = {name: file_sha256(bundle_dir / name) for name in BUNDLE_FILES}
    (bundle_dir / "hashes.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())), encoding="utf-8"
    )
    schema = json.loads((bundle_dir / "feature_schema.json").read_text(encoding="utf-8"))
    manifest: dict[str, Any] = {
        "schema_version": "1.0", "bundle_version": "returnguard-2026-09-01.1",
        "model_type": "CalibratedRiskModel", "files": hashes,
        "feature_schema_sha256": schema["schema_sha256"],
        "final_test_state": final_metrics["state"], "golden_tolerance": 1e-10,
    }
    manifest["manifest_sha256"] = object_sha256(manifest)
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_bundle(bundle_dir, data_dir, feature_config)
    return manifest


def validate_bundle(bundle_dir: Path, data_dir: Path, feature_config: Path) -> dict[str, Any]:
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    expected_manifest_hash = manifest.pop("manifest_sha256")
    if object_sha256(manifest) != expected_manifest_hash:
        raise ValueError("bundle manifest hash mismatch")
    for name, expected in manifest["files"].items():
        if file_sha256(bundle_dir / name) != expected:
            raise ValueError(f"bundle file hash mismatch: {name}")
    model = joblib.load(bundle_dir / "model_pipeline.joblib")
    if not isinstance(model, CalibratedRiskModel):
        raise TypeError("bundle model has unexpected type")
    schema = json.loads((bundle_dir / "feature_schema.json").read_text(encoding="utf-8"))
    registry = load_feature_registry(feature_config)
    if tuple(schema["ordered_names"]) != model.feature_names:
        raise ValueError("bundle feature order mismatch")
    if tuple(spec.name for spec in registry) != model.feature_names:
        raise ValueError("runtime feature registry mismatch")
    golden_ids = pd.read_parquet(bundle_dir / "golden_cases.parquet")["refund_request_id"]
    features = build_point_in_time_features(data_dir, registry).set_index("refund_request_id")
    cases = features.loc[golden_ids]
    expected = pd.DataFrame(json.loads(
        (bundle_dir / "golden_predictions.json").read_text(encoding="utf-8")
    ))
    actual = model.predict_proba(cases)
    tolerance = float(manifest["golden_tolerance"])
    if not all(abs(actual - expected["calibrated_probability"].to_numpy()) <= tolerance):
        raise ValueError("golden probability mismatch")
    policy = json.loads((bundle_dir / "policy.json").read_text(encoding="utf-8"))
    policy_config = load_policy_config(Path("configs/policy.yaml"))
    contract = PolicyContract(**{
        key: policy[key] for key in PolicyContract.__dataclass_fields__
    })
    actions = apply_contract(
        actual, np.asarray(
            cases["requested_amount_paise"].astype("int64").to_numpy(), dtype=np.int64
        ), contract, policy_config
    )
    if [action.value for action in actions] != expected["action"].tolist():
        raise ValueError("golden action mismatch")
    explanations = explain_cases(model, cases, registry)
    reason_codes = [[reason["code"] for reason in reasons] for reasons in explanations]
    if reason_codes != expected["reason_codes"].tolist():
        raise ValueError("golden reason-code mismatch")
    return {"status": "VALID", "files": len(manifest["files"]), "golden_cases": len(cases)}
