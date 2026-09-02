from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from returnguard.config import load_baseline_config, load_model_config, load_policy_config
from returnguard.data.fingerprints import file_sha256, object_sha256
from returnguard.domain.enums import RecommendedAction
from returnguard.evaluation.model_metrics import model_metrics
from returnguard.features.engine import build_point_in_time_features
from returnguard.features.registry import load_feature_registry
from returnguard.models.baselines import rules_score
from returnguard.models.primary import CalibratedRiskModel
from returnguard.policy.selection import PolicyContract, apply_contract
from returnguard.verification.bayesian import bayesian_update


def _opened_data(data_dir: Path, truth: pd.DataFrame, output_dir: Path) -> Path:
    opened = output_dir / "opened_data"
    opened.mkdir(parents=True, exist_ok=False)
    for source in data_dir.glob("*.csv"):
        shutil.copyfile(source, opened / source.name)
    requests = pd.read_csv(opened / "refund_requests.csv")
    truth_requests = truth[[
        "refund_request_id", "is_refund_abuse_simulated", "simulation_scenario",
        "outcome_available_at",
    ]].set_index("refund_request_id")
    requests = requests.set_index("refund_request_id")
    requests.loc[truth_requests.index, truth_requests.columns] = truth_requests
    requests = requests.reset_index()
    requests.to_csv(opened / "refund_requests.csv", index=False, lineterminator="\n")
    verifications = pd.read_csv(opened / "verification_events.csv")
    truth_verifications = truth[[
        "refund_request_id", "completed_at", "result", "result_codes",
    ]].set_index("refund_request_id")
    verifications = verifications.set_index("refund_request_id")
    verifications.loc[truth_verifications.index, truth_verifications.columns] = truth_verifications
    verifications = verifications.reset_index()
    verifications.to_csv(opened / "verification_events.csv", index=False, lineterminator="\n")
    return opened


def evaluate_final_once(
    data_dir: Path, lock_dir: Path, freeze_path: Path, training_dir: Path,
    policy_dir: Path, baseline_dir: Path, feature_config: Path,
    model_config_path: Path, baseline_config_path: Path, policy_config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    results_lock = output_dir / "results.lock.json"
    if results_lock.exists():
        raise FileExistsError("final evaluation is immutable and has already been written")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    expected_freeze_hash = freeze.pop("freeze_manifest_sha256")
    if freeze["state"] != "FROZEN" or object_sha256(freeze) != expected_freeze_hash:
        raise ValueError("valid frozen pre-final manifest required")
    lock_path = lock_dir / "manifest.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    truth_path = lock_dir / "final_truth.csv"
    if lock["state"] != "SEALED" or file_sha256(truth_path) != freeze["final_truth_sha256"]:
        raise ValueError("sealed final truth does not match freeze")
    output_dir.mkdir(parents=True, exist_ok=True)
    truth = pd.read_csv(truth_path)
    opened = _opened_data(data_dir, truth, output_dir)
    registry = load_feature_registry(feature_config)
    all_features = build_point_in_time_features(
        opened, registry,
        frozenset({"train", "calibration", "policy_selection", "final_test"}),
        "FROZEN_FINAL_EVALUATION",
    )
    final = all_features.loc[all_features["partition"] == "final_test"].copy()
    model = joblib.load(training_dir / "model_pipeline.joblib")
    if not isinstance(model, CalibratedRiskModel):
        raise TypeError("frozen model has unexpected type")
    model_config = load_model_config(model_config_path)
    baseline_config = load_baseline_config(baseline_config_path)
    policy_config = load_policy_config(policy_config_path)
    probabilities = model.predict_proba(final)
    raw_scores = model.predict_raw(final)
    labels = final[model_config.label_column].astype(bool).to_numpy()
    amounts = np.asarray(final["requested_amount_paise"].astype("int64"), dtype=np.int64)
    policy_payload = json.loads((policy_dir / "policy.json").read_text(encoding="utf-8"))
    contract = PolicyContract(**{
        key: policy_payload[key] for key in PolicyContract.__dataclass_fields__
    })
    stage_actions = apply_contract(probabilities, amounts, contract, policy_config)
    likelihoods = json.loads(
        (policy_dir / "verifier_likelihoods.json").read_text(encoding="utf-8")
    )["likelihood_ratios"]
    verification_results = pd.read_csv(opened / "verification_events.csv").set_index(
        "refund_request_id"
    )["result"].to_dict()
    posterior: list[float] = []
    final_actions: list[RecommendedAction] = []
    for request_id, prior, amount, stage_action in zip(
        final["refund_request_id"], probabilities, amounts, stage_actions, strict=True,
    ):
        verification_result = str(verification_results[str(request_id)])
        updated = bayesian_update(float(prior), float(likelihoods[verification_result]))
        posterior.append(updated)
        if stage_action != RecommendedAction.VERIFY:
            final_actions.append(stage_action)
            continue
        action = apply_contract(
            np.asarray([updated], dtype=float), np.asarray([amount], dtype=np.int64),
            contract, policy_config,
        )[0]
        if action == RecommendedAction.VERIFY:
            action = (
                RecommendedAction.AUTO_APPROVE
                if verification_result == "consistent" else RecommendedAction.RETURN_FIRST
            )
        final_actions.append(action)
    logistic = joblib.load(baseline_dir / "logistic_pipeline.joblib")
    feature_names = [spec.name for spec in registry]
    logistic_scores = np.asarray(logistic.predict_proba(final[feature_names])[:, 1], dtype=float)
    rule_scores = rules_score(final)
    metrics = {
        "lightgbm_calibrated": model_metrics(
            final, probabilities, 0.5, model_config.calibration_bins
        ),
        "lightgbm_raw": model_metrics(final, raw_scores, 0.5, model_config.calibration_bins),
        "logistic_regression": model_metrics(
            final, logistic_scores, baseline_config.threshold, model_config.calibration_bins
        ),
        "rules": model_metrics(
            final, rule_scores, baseline_config.threshold, model_config.calibration_bins
        ),
    }
    cases = pd.DataFrame({
        "refund_request_id": final["refund_request_id"], "customer_id": final["customer_id"],
        "label_simulated": labels, "requested_amount_paise": amounts,
        "raw_score": raw_scores, "calibrated_probability": probabilities,
        "stage_a_action": [action.value for action in stage_actions],
        "verification_result": [verification_results[str(value)] for value in final["refund_request_id"]],
        "posterior_probability": posterior,
        "final_action": [action.value for action in final_actions],
        "rules_score": rule_scores, "logistic_score": logistic_scores,
    })
    cases_path = output_dir / "final_case_results.parquet"
    cases.to_parquet(cases_path, index=False)
    opened_at = datetime.now(UTC).isoformat()
    result: dict[str, Any] = {
        "schema_version": "1.0", "state": "LOCKED", "opened_at": opened_at,
        "freeze_manifest_sha256": expected_freeze_hash, "support": len(final),
        "prevalence": float(labels.mean()), "metrics": metrics,
        "case_results_sha256": file_sha256(cases_path),
        "simulation_disclosure": "Abuse outcomes and operational fields are simulated.",
        "production_performance_claim": False,
    }
    result["results_sha256"] = object_sha256(result)
    results_lock.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lock.update({"state": "OPENED", "opened_at": opened_at})
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
