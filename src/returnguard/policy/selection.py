from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd

from returnguard.config import PolicyConfig, load_model_config, load_policy_config
from returnguard.domain.enums import RecommendedAction
from returnguard.features.engine import build_point_in_time_features
from returnguard.features.registry import load_feature_registry
from returnguard.models.primary import CalibratedRiskModel
from returnguard.policy.engine import action_costs
from returnguard.verification.bayesian import bayesian_update, estimate_likelihoods


@dataclass(frozen=True)
class PolicyContract:
    policy_version: str
    approve_threshold: float
    review_threshold: float
    review_amount_threshold_paise: int
    max_manual_review_rate: float
    max_legitimate_delay_rate: float


def apply_contract(
    probabilities: npt.NDArray[np.float64], amounts: npt.NDArray[np.int64], contract: PolicyContract,
    config: PolicyConfig,
) -> list[RecommendedAction]:
    actions: list[RecommendedAction] = []
    for probability, amount in zip(probabilities, amounts, strict=True):
        if probability <= contract.approve_threshold:
            actions.append(RecommendedAction.AUTO_APPROVE)
        elif (
            probability >= contract.review_threshold
            and amount >= contract.review_amount_threshold_paise
        ):
            actions.append(RecommendedAction.MANUAL_REVIEW)
        else:
            costs = action_costs(float(probability), int(amount), config)
            middle = min(
                (RecommendedAction.VERIFY, RecommendedAction.RETURN_FIRST),
                key=lambda action: costs[action],
            )
            actions.append(middle)
    return actions


def _realized_cost(
    frame: pd.DataFrame, actions: list[RecommendedAction], config: PolicyConfig,
) -> float:
    total = 0.0
    labels = frame["is_refund_abuse_simulated"].astype(bool).to_numpy()
    amounts = frame["requested_amount_paise"].astype("int64").to_numpy()
    for abuse, requested_amount_paise, action in zip(labels, amounts, actions, strict=True):
        exposure = requested_amount_paise * (1 + config.additional_loss_rate)
        if action == RecommendedAction.AUTO_APPROVE:
            total += exposure if abuse else 0
        elif action == RecommendedAction.VERIFY:
            total += config.verification_cost_paise
            total += 0 if abuse else config.legitimate_verification_friction_paise
        elif action == RecommendedAction.RETURN_FIRST:
            total += config.reverse_logistics_cost_paise
            total += exposure * (1 - config.salvage_rate) if abuse else config.legitimate_delay_cost_paise
        else:
            total += config.review_cost_paise
            total += 0 if abuse else config.legitimate_review_friction_paise
    return float(total)


def select_policy(
    data_dir: Path, training_dir: Path, feature_config: Path,
    model_config_path: Path, policy_config_path: Path,
    baseline_predictions_path: Path = Path("artifacts/baselines_uci/case_predictions.csv"),
) -> dict[str, Any]:
    model_config = load_model_config(model_config_path)
    config = load_policy_config(policy_config_path)
    registry = load_feature_registry(feature_config)
    features = build_point_in_time_features(data_dir, registry)
    policy_rows = features.loc[features["partition"] == "policy_selection"].copy()
    model = joblib.load(training_dir / "model_pipeline.joblib")
    if not isinstance(model, CalibratedRiskModel):
        raise TypeError("trusted training artifact has unexpected type")
    probabilities = model.predict_proba(policy_rows)
    amounts = np.asarray(
        policy_rows["requested_amount_paise"].astype("int64").to_numpy(), dtype=np.int64
    )
    labels = policy_rows[model_config.label_column].astype(bool).to_numpy()
    requests = pd.read_csv(data_dir / "refund_requests.csv")
    verifications = pd.read_csv(data_dir / "verification_events.csv")
    likelihoods = estimate_likelihoods(
        requests, verifications, config.laplace_alpha,
        (config.inconclusive_lr_min, config.inconclusive_lr_max),
    )
    candidates: list[tuple[float, PolicyContract, list[RecommendedAction]]] = []
    for delay_rate in (0.10, 0.15, 0.20):
        approve_threshold = float(np.quantile(probabilities, 1 - delay_rate))
        for review_rate in (0.01, 0.03, 0.05):
            review_threshold = float(np.quantile(probabilities, 1 - review_rate))
            eligible_amounts = amounts[probabilities >= review_threshold]
            capacity = max(1, int(np.floor(config.max_manual_review_rate * len(policy_rows))))
            if len(eligible_amounts) > capacity:
                review_amount_threshold = int(np.sort(eligible_amounts)[-capacity])
            else:
                review_amount_threshold = 0
            contract = PolicyContract(
                policy_version=config.policy_version, approve_threshold=approve_threshold,
                review_threshold=review_threshold,
                review_amount_threshold_paise=review_amount_threshold,
                max_manual_review_rate=config.max_manual_review_rate,
                max_legitimate_delay_rate=config.max_legitimate_delay_rate,
            )
            actions = apply_contract(probabilities, amounts, contract, config)
            review = np.mean([action == RecommendedAction.MANUAL_REVIEW for action in actions])
            legitimate_delay = np.mean([
                action != RecommendedAction.AUTO_APPROVE
                for action, label in zip(actions, labels, strict=True) if not label
            ])
            if review <= config.max_manual_review_rate and legitimate_delay <= config.max_legitimate_delay_rate:
                candidates.append((_realized_cost(policy_rows, actions, config), contract, actions))
    if not candidates:
        raise ValueError("no policy satisfies review and legitimate-delay constraints")
    realized_cost, contract, actions = min(candidates, key=lambda item: item[0])
    result_by_request = verifications.set_index("refund_request_id")["result"].to_dict()
    posterior_probabilities: list[float] = []
    adaptive_actions: list[RecommendedAction] = []
    for request_id, prior, amount, stage_action in zip(
        policy_rows["refund_request_id"], probabilities, amounts, actions, strict=True,
    ):
        result = str(result_by_request[str(request_id)])
        posterior = bayesian_update(float(prior), likelihoods["likelihood_ratios"][result])
        posterior_probabilities.append(posterior)
        if stage_action != RecommendedAction.VERIFY:
            adaptive_actions.append(stage_action)
            continue
        post_action = apply_contract(
            np.asarray([posterior], dtype=float), np.asarray([amount], dtype=np.int64),
            contract, config,
        )[0]
        if post_action == RecommendedAction.VERIFY:
            post_action = (
                RecommendedAction.AUTO_APPROVE
                if result == "consistent" else RecommendedAction.RETURN_FIRST
            )
        adaptive_actions.append(post_action)
    rules = pd.read_csv(baseline_predictions_path)
    rules = rules.loc[
        (rules["partition"] == "policy_selection") & (rules["model"] == "rules"),
        ["refund_request_id", "predicted_positive"],
    ]
    aligned_rules = policy_rows[["refund_request_id"]].merge(
        rules, on="refund_request_id", validate="one_to_one"
    )
    rules_actions = [
        RecommendedAction.MANUAL_REVIEW if positive else RecommendedAction.AUTO_APPROVE
        for positive in aligned_rules["predicted_positive"].astype(bool)
    ]
    score_review_actions = [
        RecommendedAction.MANUAL_REVIEW
        if probability >= contract.review_threshold and amount >= contract.review_amount_threshold_paise
        else RecommendedAction.AUTO_APPROVE
        for probability, amount in zip(probabilities, amounts, strict=True)
    ]
    fixed_band_actions: list[RecommendedAction] = []
    for action in actions:
        fixed_band_actions.append(
            RecommendedAction.RETURN_FIRST if action == RecommendedAction.VERIFY else action
        )
    comparison_actions_by_name: dict[str, list[RecommendedAction]] = {
        "approve_all": [RecommendedAction.AUTO_APPROVE] * len(policy_rows),
        "rules_only": rules_actions,
        "score_to_review": score_review_actions,
        "fixed_score_bands": fixed_band_actions,
        "returnguard_adaptive": adaptive_actions,
    }
    comparisons = {
        name: {
            "realized_cost_paise": _realized_cost(policy_rows, comparison_actions, config),
            "manual_review_rate": float(np.mean([
                action == RecommendedAction.MANUAL_REVIEW for action in comparison_actions
            ])),
        }
        for name, comparison_actions in comparison_actions_by_name.items()
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    policy_frame = pd.DataFrame({
        "refund_request_id": policy_rows["refund_request_id"], "label_simulated": labels,
        "requested_amount_paise": amounts, "calibrated_probability": probabilities,
        "stage_a_action": [action.value for action in actions],
        "verification_result": [result_by_request[str(value)] for value in policy_rows["refund_request_id"]],
        "posterior_probability": posterior_probabilities,
        "adaptive_final_action": [action.value for action in adaptive_actions],
    })
    policy_frame.to_parquet(config.output_dir / "policy_selection_cases.parquet", index=False)
    policy_payload = {**asdict(contract), "selection_realized_cost_paise": realized_cost}
    (config.output_dir / "policy.json").write_text(
        json.dumps(policy_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (config.output_dir / "verifier_likelihoods.json").write_text(
        json.dumps(likelihoods, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Exercise every likelihood through the transparent update before freezing.
    posterior_examples = {
        result: bayesian_update(0.5, ratio)
        for result, ratio in likelihoods["likelihood_ratios"].items()
    }
    report = {
        "contract": policy_payload, "review_rate": float(np.mean([
            action == RecommendedAction.MANUAL_REVIEW for action in actions
        ])),
        "legitimate_delay_rate": float(np.mean([
            action != RecommendedAction.AUTO_APPROVE
            for action, label in zip(actions, labels, strict=True) if not label
        ])),
        "posterior_examples_from_0_5": posterior_examples,
        "policy_comparison": comparisons,
        "partitions_used": {"policy_selection": ["policy_selection"], "likelihoods": ["train", "calibration"]},
        "final_test_accessed": False,
    }
    (config.output_dir / "selection_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
