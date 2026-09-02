from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, brier_score_loss

from returnguard.data.fingerprints import file_sha256, object_sha256
from returnguard.features.engine import build_point_in_time_features
from returnguard.features.registry import load_feature_registry
from returnguard.models.primary import CalibratedRiskModel


def _cluster_intervals(cases: pd.DataFrame, resamples: int = 1000) -> dict[str, Any]:
    cases = cases.reset_index(drop=True)
    groups = [group.index.to_numpy(dtype=int) for _, group in cases.groupby("customer_id", sort=True)]
    labels = cases["label_simulated"].astype(bool).to_numpy()
    raw = cases["raw_score"].to_numpy(dtype=float)
    calibrated = cases["calibrated_probability"].to_numpy(dtype=float)
    rng = np.random.default_rng(20260831)
    values: dict[str, list[float]] = {key: [] for key in ("raw_average_precision", "precision", "recall", "fpr")}
    for _ in range(resamples):
        indices = np.concatenate([groups[index] for index in rng.integers(0, len(groups), len(groups))])
        truth = labels[indices]
        if np.unique(truth).size != 2:
            continue
        predicted = calibrated[indices] >= 0.5
        tp = int((truth & predicted).sum())
        fp = int((~truth & predicted).sum())
        fn = int((truth & ~predicted).sum())
        tn = int((~truth & ~predicted).sum())
        values["raw_average_precision"].append(float(average_precision_score(truth, raw[indices])))
        values["precision"].append(tp / (tp + fp) if tp + fp else 0.0)
        values["recall"].append(tp / (tp + fn) if tp + fn else 0.0)
        values["fpr"].append(fp / (fp + tn) if fp + tn else 0.0)
    return {
        "method": "customer-cluster bootstrap", "seed": 20260831,
        "resamples_requested": resamples, "resamples_valid": len(values["recall"]),
        "metrics": {
            key: {
                "lower_95": float(np.quantile(metric_values, 0.025)),
                "median": float(np.quantile(metric_values, 0.5)),
                "upper_95": float(np.quantile(metric_values, 0.975)),
            }
            for key, metric_values in values.items()
        },
    }


def _subgroup_metrics(cases: pd.DataFrame) -> dict[str, Any]:
    labels = cases["label_simulated"].astype(bool)
    predicted = cases["calibrated_probability"] >= 0.5
    tp = int((labels & predicted).sum())
    fp = int((~labels & predicted).sum())
    fn = int((labels & ~predicted).sum())
    tn = int((~labels & ~predicted).sum())
    abuse_amount = int(cases.loc[labels, "requested_amount_paise"].sum())
    intervened_amount = int(cases.loc[
        labels & cases["final_action"].ne("AUTO_APPROVE"), "requested_amount_paise"
    ].sum())
    result: dict[str, Any] = {
        "support": len(cases), "positive_support": int(labels.sum()),
        "prevalence": float(labels.mean()), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "raw_score_average_precision": (
            float(average_precision_score(labels, cases["raw_score"]))
            if labels.nunique() == 2 else None
        ),
        "calibrated_brier_score": float(brier_score_loss(labels, cases["calibrated_probability"])),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "manual_review_count": int(cases["final_action"].eq("MANUAL_REVIEW").sum()),
        "manual_review_rate": float(cases["final_action"].eq("MANUAL_REVIEW").mean()),
        "simulated_abuse_amount_intervention_rate": (
            intervened_amount / abuse_amount if abuse_amount else None
        ),
        "simulated_abuse_requested_amount_paise": abuse_amount,
        "simulated_intervened_abuse_amount_paise": intervened_amount,
    }
    if labels.nunique() == 2:
        result["bootstrap_confidence_intervals"] = _cluster_intervals(cases)
    return result


def _feature_audit(
    cases: pd.DataFrame, opened_data: Path, training_dir: Path, feature_config: Path,
) -> dict[str, Any]:
    registry = load_feature_registry(feature_config)
    features = build_point_in_time_features(
        opened_data, registry,
        frozenset({"train", "calibration", "policy_selection", "final_test"}),
        "FROZEN_FINAL_EVALUATION",
    ).set_index("refund_request_id").loc[cases["refund_request_id"]]
    model = joblib.load(training_dir / "model_pipeline.joblib")
    if not isinstance(model, CalibratedRiskModel):
        raise TypeError("frozen model has unexpected type")
    transformed = model.transform(features)
    shap_values = shap.TreeExplainer(model.estimator).shap_values(transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    contributions = np.asarray(shap_values, dtype=float)
    split_importance = model.estimator.booster_.feature_importance(importance_type="split")
    gain_importance = model.estimator.booster_.feature_importance(importance_type="gain")
    invalid = cases["exceeds_original_captured_amount"].to_numpy(dtype=bool)
    audited: dict[str, Any] = {}
    for name in ("requested_amount_paise", "amount_paid_ratio"):
        index = model.feature_names.index(name)
        audited[name] = {
            "direct_or_indirect_amount_mismatch_signal": (
                "direct" if name == "amount_paid_ratio" else "indirect"
            ),
            "split_importance": int(split_importance[index]),
            "gain_importance": float(gain_importance[index]),
            "mean_absolute_shap_all": float(np.abs(contributions[:, index]).mean()),
            "mean_shap_exceeds_original_captured": float(contributions[invalid, index].mean()),
            "mean_absolute_shap_exceeds_original_captured": float(
                np.abs(contributions[invalid, index]).mean()
            ),
        }
    ratio_flags = features["amount_paid_ratio"].astype(float).gt(1.0).to_numpy()
    return {
        "features_present": audited,
        "amount_paid_ratio_gt_one_matches_exceeds_original_captured": bool(
            np.array_equal(ratio_flags, invalid)
        ),
        "assessment": (
            "The frozen model directly received requested-to-paid ratio and indirectly received "
            "requested amount. The deterministic over-captured subgroup is therefore a possible "
            "shortcut. This post-lock finding requires a future benchmark contract, not v1 retraining."
        ),
    }


def _policy_statistics(
    cases: pd.DataFrame, initial_actions: pd.Series, final_actions: pd.Series,
) -> dict[str, Any]:
    labels = cases["label_simulated"].astype(bool)
    legitimate = ~labels
    challenged = legitimate & initial_actions.ne("AUTO_APPROVE")
    terminal = legitimate & final_actions.ne("AUTO_APPROVE")
    abuse_intervened = labels & final_actions.ne("AUTO_APPROVE")
    abuse_amount = int(cases.loc[labels, "requested_amount_paise"].sum())
    intervened_amount = int(cases.loc[abuse_intervened, "requested_amount_paise"].sum())
    rescued = challenged & final_actions.eq("AUTO_APPROVE")
    return {
        "support": len(cases),
        "initial_legitimate_challenge_rate": float(challenged.sum() / legitimate.sum()),
        "terminal_legitimate_intervention_rate": float(terminal.sum() / legitimate.sum()),
        "legitimate_return_first_burden": float(
            (legitimate & final_actions.eq("RETURN_FIRST")).sum() / legitimate.sum()
        ),
        "manual_review_count": int(final_actions.eq("MANUAL_REVIEW").sum()),
        "manual_review_rate": float(final_actions.eq("MANUAL_REVIEW").mean()),
        "simulated_abuse_case_intervention_recall": float(abuse_intervened.sum() / labels.sum()),
        "simulated_abuse_amount_intervention_rate": float(intervened_amount / abuse_amount),
        "legitimate_rescue_rate": float(rescued.sum() / legitimate.sum()),
        "monetary_value_claimed": False,
    }


def build_metric_integrity_correction(
    *,
    public_lock: Path,
    preserved_lock: Path,
    cases_path: Path,
    opened_data: Path,
    training_dir: Path,
    feature_config: Path,
    evidence_path: Path,
    transitions_path: Path,
    output_path: Path,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    existing_created_at: str | None = None
    if output_path.is_file():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        existing_created_at = str(existing.get("created_at")) if existing.get("created_at") else None
    lock_hash = file_sha256(public_lock)
    preserved_hash = file_sha256(preserved_lock)
    case_hash = file_sha256(cases_path)
    cases = pd.read_parquet(cases_path)
    locked = json.loads(public_lock.read_text(encoding="utf-8"))
    if case_hash != locked["case_results_sha256"]:
        raise ValueError("case export does not match the frozen result lock")
    requests = pd.read_csv(opened_data / "refund_requests.csv", usecols=["refund_request_id", "order_id"])
    payments = pd.read_csv(opened_data / "payments.csv", usecols=["order_id", "paid_amount_paise", "status"])
    audit = cases.merge(requests, on="refund_request_id", validate="one_to_one").merge(
        payments, on="order_id", validate="many_to_one"
    ).rename(columns={"paid_amount_paise": "original_captured_amount_paise"})
    audit["exceeds_original_captured_amount"] = (
        audit["requested_amount_paise"] > audit["original_captured_amount_paise"]
    )
    audit = audit.sort_values("refund_request_id").reset_index(drop=True)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(evidence_path, index=False, lineterminator="\n")

    policy_path = Path("artifacts/frozen_policy/policy.json")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    transitions = audit[[
        "refund_request_id", "label_simulated", "stage_a_action", "verification_result",
        "posterior_probability", "final_action", "exceeds_original_captured_amount",
    ]].copy()
    transitions["verifier_result_in_flow"] = np.where(
        transitions["stage_a_action"].eq("VERIFY"), transitions["verification_result"], "NOT_REQUESTED"
    )
    transitions["posterior_risk_band"] = np.where(
        transitions["stage_a_action"].ne("VERIFY"), "NOT_APPLICABLE",
        np.where(
            transitions["posterior_probability"] <= float(policy["approve_threshold"]), "LOW",
            np.where(transitions["posterior_probability"] >= float(policy["review_threshold"]), "HIGH", "MIDDLE"),
        ),
    )
    transitions["execution_status"] = "NOT_OBSERVED_V1_BATCH"
    transitions["ground_truth_class"] = np.where(
        transitions["label_simulated"], "SIMULATED_ABUSE", "SIMULATED_LEGITIMATE"
    )
    transitions.to_csv(transitions_path, index=False, lineterminator="\n")

    transition_summary: list[dict[str, Any]] = []
    group_columns = [
        "ground_truth_class", "stage_a_action", "verifier_result_in_flow",
        "posterior_risk_band", "final_action", "execution_status",
    ]
    for keys, group in transitions.groupby(group_columns, sort=True, dropna=False):
        transition_summary.append({
            **dict(zip(group_columns, keys, strict=True)),
            "count": len(group),
            "case_ids": sorted(group["refund_request_id"].astype(str).tolist()),
        })

    legitimate = ~audit["label_simulated"].astype(bool)
    challenged = legitimate & audit["stage_a_action"].ne("AUTO_APPROVE")
    rescued = challenged & audit["verification_result"].eq("consistent") & audit["final_action"].eq("AUTO_APPROVE")
    terminal = challenged & audit["final_action"].ne("AUTO_APPROVE")
    return_first = audit["final_action"].eq("RETURN_FIRST")
    inconsistent = audit["verification_result"].eq("inconsistent")
    inconclusive = audit["verification_result"].eq("inconclusive")
    invalid = audit["exceeds_original_captured_amount"]
    approve_actions = pd.Series("AUTO_APPROVE", index=audit.index)
    return_first_actions = pd.Series("RETURN_FIRST", index=audit.index)
    fixed_actions = audit["stage_a_action"].replace({"VERIFY": "RETURN_FIRST"})
    adaptive_initial = audit["stage_a_action"]
    adaptive_final = audit["final_action"]
    metric_values = {
        "initial_legitimate_challenge_rate": (int(challenged.sum()), int(legitimate.sum())),
        "legitimate_rescue_rate": (int(rescued.sum()), int(legitimate.sum())),
        "challenged_legitimate_rescue_rate": (int(rescued.sum()), int(challenged.sum())),
        "terminal_legitimate_intervention_rate": (int(terminal.sum()), int(legitimate.sum())),
        "manual_reviews_per_1000": (int(audit["final_action"].eq("MANUAL_REVIEW").sum()), len(audit)),
    }
    metric_dictionary: dict[str, Any] = {}
    for name, (numerator, denominator) in metric_values.items():
        per_1000 = name == "manual_reviews_per_1000"
        metric_dictionary[name] = {
            "value": numerator / denominator * (1000 if per_1000 else 1),
            "numerator": numerator, "denominator": denominator,
            "split": "final_test", "unit": "reviews_per_1000" if per_1000 else "proportion",
            "source_artifact": evidence_path.as_posix(),
            "inclusion_criteria": {
                "initial_legitimate_challenge_rate": "legitimate and stage_a_action != AUTO_APPROVE",
                "legitimate_rescue_rate": "legitimate, initially challenged, consistent result, final AUTO_APPROVE",
                "challenged_legitimate_rescue_rate": "rescued legitimate / initially challenged legitimate",
                "terminal_legitimate_intervention_rate": (
                    "legitimate, initially challenged, final_action != AUTO_APPROVE"
                ),
                "manual_reviews_per_1000": "final_action == MANUAL_REVIEW / all requests",
            }[name],
        }
    payload: dict[str, Any] = {
        "schema_version": "1.1", "state": "REPORTING_CORRECTION",
        "created_at": (
            created_at.astimezone(UTC).isoformat()
            if created_at is not None else existing_created_at or datetime.now(UTC).isoformat()
        ),
        "provenance": {
            "public_results_lock": {"path": public_lock.as_posix(), "sha256": lock_hash},
            "preserved_v1_lock": {"path": preserved_lock.as_posix(), "sha256": preserved_hash},
            "frozen_case_export": {"path": cases_path.as_posix(), "sha256": case_hash},
            "public_case_audit": {"path": evidence_path.as_posix(), "sha256": file_sha256(evidence_path)},
            "transition_table": {"path": transitions_path.as_posix(), "sha256": file_sha256(transitions_path)},
        },
        "correction": {
            "original_metric_name": "legitimate_delay_rate",
            "original_value": float(locked["policy"]["legitimate_delay_rate"]),
            "reason": (
                "The original numerator counted only legitimate cases with a non-auto final action. "
                "It excluded legitimate cases delayed by verification and was therefore mislabeled."
            ),
            "authoritative_public_metrics": metric_dictionary,
            "identity": "Reporting-definition correction; not model or policy tuning.",
        },
        "frozen_components_changed": {
            "model": False, "features": False, "calibrator": False, "predictions": False,
            "labels": False, "thresholds": False, "likelihood_ratios": False, "policy_rules": False,
        },
        "payment_state_limitation": {
            "original_captured_amount_available": True,
            "point_in_time_prior_refund_ledger_available": False,
            "remaining_refundable_balance_reconstructable": False,
            "certainly_invalid_exceeds_original_captured_count": int(invalid.sum()),
            "not_exceeding_original_captured_count": int((~invalid).sum()),
            "warning": (
                "Requests not exceeding original captured payment are not proven executable or "
                "refundable-balance verified."
            ),
        },
        "amount_audit": {
            "attempted_simulated_requested_amount_paise": int(audit["requested_amount_paise"].sum()),
            "exceeds_original_captured": {
                "support": int(invalid.sum()),
                "total_requested_amount_paise": int(audit.loc[invalid, "requested_amount_paise"].sum()),
                "total_original_captured_amount_paise": int(audit.loc[invalid, "original_captured_amount_paise"].sum()),
                "total_impossible_excess_paise": int((
                    audit.loc[invalid, "requested_amount_paise"]
                    - audit.loc[invalid, "original_captured_amount_paise"]
                ).sum()),
                "maximum_impossible_excess_paise": int((
                    audit.loc[invalid, "requested_amount_paise"]
                    - audit.loc[invalid, "original_captured_amount_paise"]
                ).max()),
            },
            "request_amount_paise": {
                "median": float(audit["requested_amount_paise"].median()),
                "p95": float(audit["requested_amount_paise"].quantile(0.95)),
                "maximum": int(audit["requested_amount_paise"].max()),
            },
        },
        "frozen_prediction_subgroup_audit": {
            "all_1600_cases": _subgroup_metrics(audit),
            "requests_exceeding_original_captured_payment": _subgroup_metrics(audit.loc[invalid]),
            "requests_not_exceeding_original_captured_payment": _subgroup_metrics(audit.loc[~invalid]),
            "timing": "Post-lock subgroup audit using frozen predictions; no selection or tuning.",
        },
        "policy_flow_reconciliation": {
            "transition_groups": transition_summary,
            "customer_friction_breakdown": {
                "initially_challenged_legitimate": {
                    "count": int(challenged.sum()),
                    "case_ids": sorted(audit.loc[challenged, "refund_request_id"].astype(str).tolist()),
                },
                "automatically_verified_and_rescued": {
                    "count": int(rescued.sum()),
                    "case_ids": sorted(audit.loc[rescued, "refund_request_id"].astype(str).tolist()),
                },
                "manual_reviewed_outcome_unobserved": {
                    "count": int((challenged & audit["final_action"].eq("MANUAL_REVIEW")).sum()),
                    "case_ids": sorted(audit.loc[
                        challenged & audit["final_action"].eq("MANUAL_REVIEW"), "refund_request_id"
                    ].astype(str).tolist()),
                },
                "manually_reviewed_and_approved": {
                    "count": None,
                    "reason": "The frozen batch export contains policy actions, not operator outcomes.",
                },
                "return_first_after_inconsistent_evidence": {
                    "count": int((challenged & return_first & inconsistent).sum()),
                    "case_ids": sorted(audit.loc[
                        challenged & return_first & inconsistent,
                        "refund_request_id",
                    ].astype(str).tolist()),
                },
                "return_first_after_inconclusive_evidence": {
                    "count": int((challenged & return_first & inconclusive).sum()),
                    "case_ids": sorted(audit.loc[
                        challenged & return_first & inconclusive,
                        "refund_request_id",
                    ].astype(str).tolist()),
                },
                "rejected_or_blocked": {"count": 0, "case_ids": []},
                "other_terminal_intervention": {"count": 0, "case_ids": []},
                "verifier_unavailable_or_timeout": {
                    "count": None,
                    "reason": "V1 does not distinguish evidence inconclusive from technical unavailability or timeout.",
                },
                "latency": {
                    "status": "UNAVAILABLE",
                    "reason": "The frozen batch export has no workflow execution timestamps; no latency is inferred.",
                },
            },
            "classifier_false_positives_at_0_5": {
                "count": int((~audit["label_simulated"].astype(bool) & audit["calibrated_probability"].ge(0.5)).sum()),
                "case_ids": sorted(audit.loc[
                    ~audit["label_simulated"].astype(bool) & audit["calibrated_probability"].ge(0.5),
                    "refund_request_id",
                ].astype(str).tolist()),
                "note": "Classifier threshold errors are separate from policy-level customer interventions.",
            },
        },
        "amount_feature_shortcut_audit": _feature_audit(audit, opened_data, training_dir, feature_config),
        "simulation_policy_comparison": {
            "scope": "Frozen v1 cases and actions; submitted-amount policy statistics only.",
            "approve_all": _policy_statistics(audit, approve_actions, approve_actions),
            "return_first_all": _policy_statistics(
                audit, return_first_actions, return_first_actions
            ),
            "fixed_risk_bands": _policy_statistics(audit, adaptive_initial, fixed_actions),
            "adaptive_verification": _policy_statistics(
                audit, adaptive_initial, adaptive_final
            ),
            "interpretation": (
                "Adaptive verification rescues challenged legitimate cases and reduces return-first "
                "burden versus fixed bands. No certified economic superiority is claimed."
            ),
        },
        "economic_reporting": {
            "superseded_estimate_paise_per_1000": float(
                locked["policy"]["estimated_net_value_protected_paise_per_1000"]
            ),
            "status": "WITHDRAWN_UNCERTIFIABLE_FROM_V1",
            "certified_replacement_monetary_result": None,
            "reason": (
                "V1 has no point-in-time prior-refund ledger, and its old calculation used uncapped "
                "requested amounts. Gateway-executable and incremental monetary value are not claimed."
            ),
        },
    }
    payload["correction_sha256"] = object_sha256(payload)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
