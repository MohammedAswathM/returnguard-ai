from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import average_precision_score

from returnguard.config import load_model_config, load_policy_config
from returnguard.domain.enums import RecommendedAction
from returnguard.features.engine import build_point_in_time_features
from returnguard.features.registry import load_feature_registry
from returnguard.models.primary import CalibratedRiskModel, fit_primary_model
from returnguard.policy.engine import action_costs
from returnguard.policy.selection import PolicyContract, apply_contract


def _cluster_bootstrap(
    frame: pd.DataFrame, scores: npt.NDArray[np.float64], resamples: int, seed: int,
) -> dict[str, float]:
    working = frame[["customer_id", "is_refund_abuse_simulated"]].copy()
    working["score"] = scores
    groups = {key: group for key, group in working.groupby("customer_id", sort=True)}
    customers = np.asarray(list(groups))
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(resamples):
        sampled = rng.choice(customers, size=len(customers), replace=True)
        draw = pd.concat([groups[str(customer)] for customer in sampled], ignore_index=True)
        labels = draw["is_refund_abuse_simulated"].astype(bool)
        if labels.nunique() == 2:
            values.append(float(average_precision_score(labels, draw["score"])))
    return {
        "resamples_requested": resamples,
        "resamples_valid": len(values),
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _slice(
    frame: pd.DataFrame, scores: npt.NDArray[np.float64], mask: pd.Series,
) -> dict[str, Any]:
    labels = frame.loc[mask, "is_refund_abuse_simulated"].astype(bool)
    selected = scores[mask.to_numpy()]
    return {
        "support": int(mask.sum()),
        "positive_support": int(labels.sum()),
        "average_precision": (
            float(average_precision_score(labels, selected)) if labels.nunique() == 2 else None
        ),
        "underpowered": bool(mask.sum() < 50 or labels.sum() < 10),
    }


def run_robustness(
    data_dir: Path, training_dir: Path, feature_config: Path,
    model_config_path: Path, output_path: Path, resamples: int = 1000,
    alternate_seed_dirs: tuple[Path, ...] = (),
    policy_config_path: Path = Path("configs/policy.yaml"),
) -> dict[str, Any]:
    registry = load_feature_registry(feature_config)
    config = load_model_config(model_config_path)
    features = build_point_in_time_features(data_dir, registry)
    frame = features.loc[features["partition"] == "policy_selection"].copy()
    model = joblib.load(training_dir / "model_pipeline.joblib")
    if not isinstance(model, CalibratedRiskModel):
        raise TypeError("trusted training artifact has unexpected type")
    scores = model.predict_proba(frame)
    requests = pd.read_csv(data_dir / "refund_requests.csv", usecols=[
        "refund_request_id", "simulation_scenario",
    ])
    frame = frame.merge(requests, on="refund_request_id", how="left", validate="one_to_one")
    missing_all = frame.copy()
    for spec in registry:
        missing_all[spec.name] = spec.default
    unseen = frame.copy()
    unseen["category"] = "UNSEEN_CATEGORY"
    rng = np.random.default_rng(config.random_seed)
    labels = frame[config.label_column].astype(bool).to_numpy()
    noise_results: dict[str, float] = {}
    for rate in (0.05, 0.10):
        noisy = labels.copy()
        indices = rng.choice(len(noisy), size=int(rate * len(noisy)), replace=False)
        noisy[indices] = ~noisy[indices]
        noise_results[str(rate)] = float(average_precision_score(noisy, scores))
    prevalence_results: dict[str, dict[str, float | int]] = {}
    positives = np.flatnonzero(labels)
    negatives = np.flatnonzero(~labels)
    for prevalence in (0.02, 0.05, 0.09, 0.15):
        positive_count = max(1, round(prevalence * len(negatives) / (1 - prevalence)))
        selected = np.concatenate([
            rng.choice(positives, positive_count, replace=positive_count > len(positives)), negatives,
        ])
        prevalence_results[str(prevalence)] = {
            "support": int(len(selected)),
            "realized_prevalence": float(labels[selected].mean()),
            "average_precision": float(average_precision_score(labels[selected], scores[selected])),
        }
    scenarios = {
        name: _slice(frame, scores, frame["simulation_scenario"] == name)
        for name in ("shared_household", "product_defect_burst", "first_order_return")
    }
    policy_config = load_policy_config(policy_config_path)
    policy_payload = json.loads((output_path.parent / "policy.json").read_text(encoding="utf-8"))
    contract = PolicyContract(**{
        key: policy_payload[key] for key in PolicyContract.__dataclass_fields__
    })
    amounts = np.asarray(frame["requested_amount_paise"].astype("int64"), dtype=np.int64)
    friction_sensitivity: dict[str, dict[str, float]] = {}
    for multiplier in (0.5, 1.0, 2.0):
        stressed = policy_config.model_copy(update={
            "legitimate_verification_friction_paise": int(
                policy_config.legitimate_verification_friction_paise * multiplier
            ),
            "legitimate_review_friction_paise": int(
                policy_config.legitimate_review_friction_paise * multiplier
            ),
            "legitimate_delay_cost_paise": int(
                policy_config.legitimate_delay_cost_paise * multiplier
            ),
        })
        actions = apply_contract(scores, amounts, contract, stressed)
        expected_cost = sum(
            action_costs(float(score), int(amount), stressed)[action]
            for score, amount, action in zip(scores, amounts, actions, strict=True)
        )
        friction_sensitivity[str(multiplier)] = {
            "expected_cost_paise_per_1000": float(expected_cost * 1000 / len(frame)),
            "manual_review_rate": float(np.mean([
                action == RecommendedAction.MANUAL_REVIEW for action in actions
            ])),
        }
    feature_groups = {
        "history": [
            spec.name for spec in registry
            if spec.name.startswith("prior_") or spec.name in {"account_tenure_days", "first_order_indicator"}
        ],
        "request_and_order": [
            spec.name for spec in registry if spec.name in {
                "hours_delivery_to_request", "requested_amount_paise", "amount_paid_ratio",
                "requested_quantity_ratio", "returnless_requested", "evidence_provided",
            }
        ],
        "categorical_context": ["reason_code", "category"],
    }
    missing_groups: dict[str, dict[str, float]] = {}
    for group, names in feature_groups.items():
        stressed_frame = frame.copy()
        for spec in registry:
            if spec.name in names:
                stressed_frame[spec.name] = spec.default
        stressed_scores = model.predict_proba(stressed_frame)
        missing_groups[group] = {
            "average_probability": float(stressed_scores.mean()),
            "average_precision": float(average_precision_score(labels, stressed_scores)),
        }
    seed_evidence: list[dict[str, Any]] = []
    for alternate_dir in alternate_seed_dirs:
        alternate_features = build_point_in_time_features(alternate_dir, registry)
        alternate_train = alternate_features.loc[alternate_features["partition"] == "train"]
        alternate_calibration = alternate_features.loc[
            alternate_features["partition"] == "calibration"
        ]
        alternate_policy = alternate_features.loc[
            alternate_features["partition"] == "policy_selection"
        ]
        alternate_model, _ = fit_primary_model(
            alternate_train, alternate_calibration, registry, config
        )
        alternate_scores = alternate_model.predict_proba(alternate_policy)
        alternate_labels = alternate_policy[config.label_column].astype(bool)
        alternate_metadata = json.loads(
            (alternate_dir / "metadata.json").read_text(encoding="utf-8")
        )
        seed_evidence.append({
            "seed": alternate_metadata["generator_seed"],
            "transformed_data_fingerprint": alternate_metadata[
                "transformed_data_fingerprint_sha256"
            ],
            "policy_support": len(alternate_policy),
            "policy_prevalence": float(alternate_labels.mean()),
            "policy_average_precision": float(
                average_precision_score(alternate_labels, alternate_scores)
            ),
            "final_test_accessed": False,
        })
    report: dict[str, Any] = {
        "evaluation_partition": "policy_selection",
        "final_test_accessed": False,
        "support": len(frame),
        "cluster_bootstrap_average_precision": _cluster_bootstrap(
            frame, scores, resamples, config.random_seed
        ),
        "prevalence_stress": prevalence_results,
        "label_noise_stress": noise_results,
        "missing_feature_groups": missing_groups,
        "missing_all_feature_groups": {"average_probability": float(model.predict_proba(missing_all).mean())},
        "unseen_category": {"average_probability": float(model.predict_proba(unseen).mean())},
        "first_order_proxy": _slice(frame, scores, frame["first_order_indicator"] == 1.0),
        "cold_start_challenge": {
            "support": int(len(pd.read_csv(data_dir / "cold_start_mapping.csv"))),
            "evaluation_timing": "Final labels remained sealed during pre-final robustness.",
        },
        "hard_legitimate_scenarios": scenarios,
        "verifier_unavailable": {
            "result": "inconclusive", "policy_requirement": "no autonomous adverse action"
        },
        "friction_cost_sensitivity": friction_sensitivity,
        "generator_seed_evidence": seed_evidence,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
