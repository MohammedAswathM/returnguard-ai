from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd

from returnguard.v2.config import V2PolicyConfig
from returnguard.v2.training import V2RiskModel

VERIFIER_RESULTS = (
    "consistent", "inconsistent", "evidence_inconclusive",
    "verifier_unavailable", "verifier_timeout",
)


@dataclass(frozen=True)
class FrozenPolicy:
    verify_threshold: float
    manual_review_threshold: float
    likelihood_ratios: dict[str, float]


def estimate_likelihoods(frame: pd.DataFrame, alpha: float) -> dict[str, Any]:
    labels = frame["is_refund_abuse_simulated"].astype(bool)
    observed = frame["verification_result_simulated"].astype(str)
    estimated_results = ("consistent", "inconsistent", "evidence_inconclusive")
    counts: dict[str, dict[str, int]] = {}
    likelihoods: dict[str, float] = {}
    for result in estimated_results:
        abuse_count = int((labels & observed.eq(result)).sum())
        legitimate_count = int((~labels & observed.eq(result)).sum())
        counts[result] = {"abuse": abuse_count, "legitimate": legitimate_count}
        abuse_probability = (abuse_count + alpha) / (int(labels.sum()) + alpha * len(estimated_results))
        legitimate_probability = (legitimate_count + alpha) / (
            int((~labels).sum()) + alpha * len(estimated_results)
        )
        likelihoods[result] = float(abuse_probability / legitimate_probability)
    likelihoods["verifier_unavailable"] = 1.0
    likelihoods["verifier_timeout"] = 1.0
    if likelihoods["consistent"] >= 1 or likelihoods["inconsistent"] <= 1:
        raise ValueError("v2 verifier likelihood directions are invalid")
    if not 0.5 <= likelihoods["evidence_inconclusive"] <= 2.0:
        raise ValueError("v2 inconclusive evidence likelihood is not neutral enough")
    return {
        "support": len(frame), "laplace_alpha": alpha, "raw_counts": counts,
        "likelihood_ratios": likelihoods,
        "technical_results": {
            "verifier_unavailable": "fixed neutral; not estimated from customer outcomes",
            "verifier_timeout": "fixed neutral; not estimated from customer outcomes",
        },
    }


def posterior_probability(prior: float, result: str, likelihoods: dict[str, float]) -> float:
    if result not in likelihoods:
        result = "evidence_inconclusive"
    clipped = min(max(prior, 1e-6), 1 - 1e-6)
    prior_odds = clipped / (1 - clipped)
    posterior_odds = prior_odds * likelihoods[result]
    return float(posterior_odds / (1 + posterior_odds))


def apply_policy(
    frame: pd.DataFrame, probabilities: npt.NDArray[np.float64], policy: FrozenPolicy,
) -> pd.DataFrame:
    result = frame[[
        "refund_request_id", "customer_id", "is_refund_abuse_simulated",
        "claimed_refund_amount_paise", "verification_result_simulated", "simulation_scenario",
    ]].copy()
    result["initial_probability"] = probabilities
    result["stage_a_action"] = np.where(
        probabilities >= policy.manual_review_threshold, "MANUAL_REVIEW",
        np.where(probabilities >= policy.verify_threshold, "VERIFY", "AUTO_APPROVE"),
    )
    result["posterior_probability"] = probabilities
    verify_mask = result["stage_a_action"].eq("VERIFY")
    result.loc[verify_mask, "posterior_probability"] = [
        posterior_probability(float(prior), str(verifier), policy.likelihood_ratios)
        for prior, verifier in zip(
            result.loc[verify_mask, "initial_probability"],
            result.loc[verify_mask, "verification_result_simulated"], strict=True,
        )
    ]
    result["final_action"] = result["stage_a_action"]
    consistent = verify_mask & result["verification_result_simulated"].eq("consistent")
    inconsistent = verify_mask & result["verification_result_simulated"].eq("inconsistent")
    evidence_unknown = verify_mask & result["verification_result_simulated"].eq("evidence_inconclusive")
    technical_unknown = verify_mask & result["verification_result_simulated"].isin(
        ["verifier_unavailable", "verifier_timeout"]
    )
    result.loc[consistent, "final_action"] = "AUTO_APPROVE"
    result.loc[inconsistent, "final_action"] = "RETURN_FIRST"
    result.loc[evidence_unknown | technical_unknown, "final_action"] = "MANUAL_REVIEW"
    return result


def policy_metrics(cases: pd.DataFrame, config: V2PolicyConfig) -> dict[str, float | int]:
    legitimate = ~cases["is_refund_abuse_simulated"].astype(bool)
    abuse = ~legitimate
    challenge = cases["stage_a_action"].ne("AUTO_APPROVE")
    intervention = cases["final_action"].ne("AUTO_APPROVE")
    review = cases["final_action"].eq("MANUAL_REVIEW")
    returned = cases["final_action"].eq("RETURN_FIRST")
    rescued = legitimate & challenge & ~intervention
    challenged_legitimate = legitimate & challenge
    abuse_amount = float(cases.loc[abuse, "claimed_refund_amount_paise"].sum())
    intervened_amount = float(cases.loc[abuse & intervention, "claimed_refund_amount_paise"].sum())
    cost = 0.0
    cost += float((cases.loc[abuse & ~intervention, "claimed_refund_amount_paise"] * (
        1 + config.additional_loss_rate
    )).sum())
    cost += float(challenge.sum() * config.verification_cost_paise)
    cost += float(review.sum() * config.review_cost_paise)
    cost += float((legitimate & challenge).sum() * config.legitimate_verification_friction_paise)
    cost += float((legitimate & review).sum() * (
        config.legitimate_review_friction_paise + config.legitimate_delay_cost_paise
    ))
    cost += float(returned.sum() * config.reverse_logistics_cost_paise)
    return {
        "support": len(cases), "modeled_cost_paise": cost,
        "initial_legitimate_challenge_rate": float(challenged_legitimate.sum() / legitimate.sum()),
        "legitimate_rescue_rate": float(rescued.sum() / legitimate.sum()),
        "challenged_legitimate_rescue_rate": float(
            rescued.sum() / challenged_legitimate.sum()
        ) if challenged_legitimate.any() else 0.0,
        "terminal_legitimate_intervention_rate": float((legitimate & intervention).sum() / legitimate.sum()),
        "legitimate_return_first_burden": int((legitimate & returned).sum()),
        "manual_reviews": int(review.sum()),
        "manual_reviews_per_1000": float(review.mean() * 1000),
        "abuse_case_intervention_recall": float((abuse & intervention).sum() / abuse.sum()),
        "abuse_amount_intervention_recall": intervened_amount / abuse_amount if abuse_amount else 0.0,
    }


def _comparison_cases(
    frame: pd.DataFrame, probabilities: npt.NDArray[np.float64], name: str,
    policy: FrozenPolicy,
) -> pd.DataFrame:
    cases = apply_policy(frame, probabilities, policy)
    if name == "approve_all":
        cases["stage_a_action"] = "AUTO_APPROVE"
        cases["final_action"] = "AUTO_APPROVE"
    elif name == "return_first_all":
        cases["stage_a_action"] = "RETURN_FIRST"
        cases["final_action"] = "RETURN_FIRST"
    elif name == "fixed_risk_bands":
        cases["stage_a_action"] = np.where(
            probabilities >= policy.manual_review_threshold, "MANUAL_REVIEW", "AUTO_APPROVE"
        )
        cases["final_action"] = cases["stage_a_action"]
    return cases


def select_policy(features: pd.DataFrame, model_path: Path, config: V2PolicyConfig) -> dict[str, Any]:
    output = config.output_dir
    if (output / "policy.json").exists():
        raise FileExistsError("v2 policy artifacts are immutable")
    output.mkdir(parents=True, exist_ok=True)
    model: V2RiskModel = joblib.load(model_path)
    likelihood_frame = features.loc[features["partition"].isin(["train", "calibration"])]
    likelihood_report = estimate_likelihoods(likelihood_frame, config.laplace_alpha)
    policy_frame = features.loc[features["partition"].eq("policy_selection")].copy()
    probabilities = model.predict_proba(policy_frame)
    likelihoods = likelihood_report["likelihood_ratios"]
    candidates: list[tuple[float, FrozenPolicy, dict[str, float | int]]] = []
    for verify_quantile in (0.80, 0.825, 0.85, 0.875, 0.90, 0.925):
        for review_quantile in (0.97, 0.975, 0.98, 0.985, 0.99):
            policy = FrozenPolicy(
                verify_threshold=float(np.quantile(probabilities, verify_quantile)),
                manual_review_threshold=float(np.quantile(probabilities, review_quantile)),
                likelihood_ratios=likelihoods,
            )
            cases = apply_policy(policy_frame, probabilities, policy)
            metrics = policy_metrics(cases, config)
            if (
                float(metrics["manual_reviews_per_1000"]) <= config.max_manual_review_rate * 1000
                and float(metrics["initial_legitimate_challenge_rate"])
                <= config.max_initial_legitimate_challenge_rate
            ):
                candidates.append((float(metrics["modeled_cost_paise"]), policy, metrics))
    if not candidates:
        raise ValueError("no v2 adaptive policy satisfies frozen operating constraints")
    _, selected, selected_metrics = min(candidates, key=lambda item: item[0])
    comparisons: dict[str, Any] = {}
    selected_cases = apply_policy(policy_frame, probabilities, selected)
    selected_cases.to_parquet(output / "policy_selection_cases.parquet", index=False)
    for name in ("approve_all", "return_first_all", "fixed_risk_bands", "adaptive_verification"):
        cases = (
            selected_cases if name == "adaptive_verification"
            else _comparison_cases(policy_frame, probabilities, name, selected)
        )
        comparisons[name] = policy_metrics(cases, config)
    report: dict[str, Any] = {
        "schema_version": "2.0", "policy_version": config.policy_version,
        "selection_partition": "policy_selection", "verify_threshold": selected.verify_threshold,
        "manual_review_threshold": selected.manual_review_threshold,
        "likelihoods": likelihood_report, "selected_metrics": selected_metrics,
        "comparison": comparisons,
        "cost_disclosure": "Simulated policy cost under declared assumptions; not realized savings.",
    }
    (output / "policy.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
