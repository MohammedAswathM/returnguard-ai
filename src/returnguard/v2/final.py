from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

from returnguard.data.fingerprints import file_sha256
from returnguard.evaluation.metrics import classification_metrics
from returnguard.v2.config import V2FeatureConfig, V2PolicyConfig
from returnguard.v2.features import FINAL_ACCESS_GRANT, build_features
from returnguard.v2.policy import (
    FrozenPolicy,
    _comparison_cases,
    apply_policy,
    policy_metrics,
)
from returnguard.v2.training import V2RiskModel


def bootstrap_intervals(
    frame: pd.DataFrame, scores: npt.NDArray[np.float64], threshold: float,
    *, resamples: int, seed: int,
) -> dict[str, list[float]]:
    working = frame[["customer_id", "is_refund_abuse_simulated"]].copy()
    working["score"] = scores
    groups = {name: group for name, group in working.groupby("customer_id", sort=False)}
    customer_ids = np.asarray(list(groups), dtype=object)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"average_precision": [], "precision": [], "recall": []}
    for _ in range(resamples):
        selected = rng.choice(customer_ids, len(customer_ids), replace=True)
        sample = pd.concat([groups[value] for value in selected], ignore_index=True)
        labels = sample["is_refund_abuse_simulated"].astype(bool).to_numpy()
        if labels.min() == labels.max():
            continue
        probability = sample["score"].to_numpy(float)
        predicted = probability >= threshold
        tp = int((labels & predicted).sum())
        fp = int((~labels & predicted).sum())
        fn = int((labels & ~predicted).sum())
        values["average_precision"].append(float(average_precision_score(labels, probability)))
        values["precision"].append(tp / (tp + fp) if tp + fp else 0.0)
        values["recall"].append(tp / (tp + fn) if tp + fn else 0.0)
    return {
        name: [float(np.quantile(metric, 0.025)), float(np.quantile(metric, 0.975))]
        for name, metric in values.items()
    }


def evaluate_cases(
    frame: pd.DataFrame, model: V2RiskModel, policy: FrozenPolicy,
    policy_config: V2PolicyConfig, bootstrap_resamples: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = model.raw_score(frame)
    calibrated = model.predict_proba(frame)
    labels = frame["is_refund_abuse_simulated"].astype(bool).to_numpy()
    threshold = policy.manual_review_threshold
    classifier = classification_metrics(labels, calibrated, threshold)
    classifier.update({
        "raw_average_precision": float(average_precision_score(labels, raw)),
        "calibrated_brier_score": float(brier_score_loss(labels, calibrated)),
        "calibrated_log_loss": float(log_loss(labels, calibrated)),
        "bootstrap_95_percent_ci": bootstrap_intervals(
            frame, raw, threshold, resamples=bootstrap_resamples, seed=20260903
        ),
    })
    cases = apply_policy(frame, calibrated, policy)
    cases["raw_score"] = raw
    cases["calibrated_probability"] = calibrated
    comparisons: dict[str, Any] = {}
    for name in ("approve_all", "return_first_all", "fixed_risk_bands", "adaptive_verification"):
        comparison_cases = (
            cases if name == "adaptive_verification"
            else _comparison_cases(frame, calibrated, name, policy)
        )
        comparisons[name] = policy_metrics(comparison_cases, policy_config)

    cold = frame.copy()
    history_groups = {
        "smoothed_customer_history", "time_decayed_refund_velocity",
        "customer_peer_deviation", "supporting_entity_context",
    }
    # This is a counterfactual neutral-prior challenge, not a second fitted model.
    feature_config_path = Path("configs/v2/features.yaml")
    from returnguard.v2.config import load_v2_feature_config
    feature_config = load_v2_feature_config(feature_config_path)
    for spec in feature_config.features:
        if spec.group in history_groups:
            cold[spec.name] = spec.default
    cold["cold_start_indicator"] = 1.0
    cold_scores = model.predict_proba(cold)
    cold_metrics = classification_metrics(labels, cold_scores, threshold)
    cold_metrics["calibrated_brier_score"] = float(brier_score_loss(labels, cold_scores))

    slices: dict[str, Any] = {}
    legitimate = frame.loc[~frame["is_refund_abuse_simulated"].astype(bool)].copy()
    for scenario, subset in legitimate.groupby("simulation_scenario"):
        slice_scores = model.predict_proba(subset)
        slice_cases = apply_policy(subset, slice_scores, policy)
        slices[str(scenario)] = {
            "support": len(subset),
            "mean_calibrated_probability": float(slice_scores.mean()),
            "initial_challenge_rate": float(slice_cases["stage_a_action"].ne("AUTO_APPROVE").mean()),
            "terminal_intervention_rate": float(slice_cases["final_action"].ne("AUTO_APPROVE").mean()),
        }
    return cases, {
        "classifier": classifier, "policy_comparison": comparisons,
        "cold_start_counterfactual": cold_metrics, "hard_legitimate_slices": slices,
    }


def open_and_evaluate_final(
    data_dir: Path, lock_dir: Path, freeze_path: Path,
    feature_config: V2FeatureConfig, policy_config: V2PolicyConfig,
    model_path: Path, policy_path: Path, output_dir: Path,
    *, bootstrap_resamples: int,
) -> dict[str, Any]:
    if (output_dir / "results.lock.json").exists():
        raise FileExistsError("v2 final results already exist; final evaluation cannot be repeated")
    manifest_path = lock_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["state"] != "SEALED" or manifest["opened_at"] is not None:
        raise ValueError("v2 final truth is not sealed for a first opening")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["status"] != "FROZEN_FOR_ONE_TIME_FINAL_EVALUATION":
        raise ValueError("v2 pre-final freeze is incomplete")
    if file_sha256(freeze_path) != Path(f"{freeze_path}.sha256").read_text(encoding="utf-8").split()[0]:
        raise ValueError("v2 freeze manifest hash mismatch")
    truth_path = lock_dir / "final_truth.csv"
    if file_sha256(truth_path) != manifest["truth_sha256"]:
        raise ValueError("v2 final truth hash mismatch")
    model: V2RiskModel = joblib.load(model_path)
    policy_report = json.loads(policy_path.read_text(encoding="utf-8"))
    policy = FrozenPolicy(
        verify_threshold=float(policy_report["verify_threshold"]),
        manual_review_threshold=float(policy_report["manual_review_threshold"]),
        likelihood_ratios=policy_report["likelihoods"]["likelihood_ratios"],
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(manifest_path, lock_dir / "manifest.sealed.json")
    opened_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    manifest.update({"state": "OPENED_ONCE", "opened_at": opened_at})
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    frame = build_features(
        data_dir, feature_config, frozenset({"final_test"}),
        final_access_grant=FINAL_ACCESS_GRANT, final_truth=truth_path,
    )
    cases, metrics = evaluate_cases(
        frame, model, policy, policy_config, bootstrap_resamples
    )
    case_path = output_dir / "case_predictions.parquet"
    cases.to_parquet(case_path, index=False)
    result: dict[str, Any] = {
        "schema_version": "2.0", "benchmark_version": "returnguard-v2.0",
        "status": "LOCKED_FINAL_OPENED_ONCE", "opened_at": opened_at,
        "simulation_disclosure": (
            "UCI transaction-derived foundation with simulated operational fields and abuse labels; "
            "not production performance or realized savings."
        ),
        "support": len(frame), "prevalence": float(frame["is_refund_abuse_simulated"].mean()),
        **metrics,
        "artifacts": {
            "freeze_manifest_sha256": file_sha256(freeze_path),
            "case_predictions_sha256": file_sha256(case_path),
            "final_truth_sha256": file_sha256(truth_path),
            "model_sha256": file_sha256(model_path),
            "policy_sha256": file_sha256(policy_path),
        },
        "integrity_lane": {
            "support": int(len(pd.read_csv(data_dir / "integrity_cases.csv"))),
            "included_in_ml_metrics": False,
            "coverage": "ten deterministic scenarios; executable invariants are covered by payment-integrity tests",
        },
    }
    result_path = output_dir / "results.lock.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(f"{result_path}.sha256").write_text(
        f"{file_sha256(result_path)}  {result_path.name}\n", encoding="utf-8"
    )
    return result
