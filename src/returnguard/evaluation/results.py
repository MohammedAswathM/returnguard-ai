from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from returnguard.config import load_policy_config
from returnguard.data.fingerprints import file_sha256, object_sha256
from returnguard.domain.enums import RecommendedAction
from returnguard.policy.selection import _realized_cost


def _bootstrap_ap(cases: pd.DataFrame, resamples: int, seed: int) -> dict[str, Any]:
    labels = cases["label_simulated"].astype(bool).to_numpy()
    scores = cases["calibrated_probability"].to_numpy(dtype=float)
    groups = [group.index.to_numpy(dtype=int) for _, group in cases.groupby("customer_id", sort=True)]
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(resamples):
        sampled = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in sampled])
        if np.unique(labels[indices]).size == 2:
            values.append(float(average_precision_score(labels[indices], scores[indices])))
    return {
        "method": "customer-cluster bootstrap", "resamples": len(values),
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def upgrade_results_lock(
    results_dir: Path, policy_config_path: Path, output_path: Path,
) -> dict[str, Any]:
    source_lock = results_dir / "results.lock.json"
    preserved = results_dir / "results.v1.lock.json"
    if preserved.exists() and output_path.exists():
        raise FileExistsError("results lock upgrade has already been performed")
    if preserved.exists():
        source_lock = preserved
    source = json.loads(source_lock.read_text(encoding="utf-8"))
    if file_sha256(results_dir / "final_case_results.parquet") != source["case_results_sha256"]:
        raise ValueError("case results do not match v1 lock")
    if source_lock != preserved:
        shutil.move(source_lock, preserved)
    cases = pd.read_parquet(results_dir / "final_case_results.parquet")
    config = load_policy_config(policy_config_path)
    labels = cases["label_simulated"].astype(bool)
    final_actions = [RecommendedAction(value) for value in cases["final_action"]]
    stage_actions = [RecommendedAction(value) for value in cases["stage_a_action"]]
    comparison_actions: dict[str, list[RecommendedAction]] = {
        "approve_all": [RecommendedAction.AUTO_APPROVE] * len(cases),
        "score_to_review": [
            RecommendedAction.MANUAL_REVIEW if score >= 0.5 else RecommendedAction.AUTO_APPROVE
            for score in cases["calibrated_probability"]
        ],
        "fixed_score_bands": [
            RecommendedAction.RETURN_FIRST if action == RecommendedAction.VERIFY else action
            for action in stage_actions
        ],
        "returnguard_adaptive": final_actions,
    }
    cost_cases = cases.rename(columns={"label_simulated": "is_refund_abuse_simulated"})
    policy_comparison = {
        name: {
            "realized_cost_paise": _realized_cost(cost_cases, actions, config),
            "manual_review_rate": float(np.mean([
                action == RecommendedAction.MANUAL_REVIEW for action in actions
            ])),
        }
        for name, actions in comparison_actions.items()
    }
    legitimate = ~labels
    rescue = (
        (cases["stage_a_action"] == RecommendedAction.VERIFY.value)
        & (cases["verification_result"] == "consistent")
        & (cases["final_action"] == RecommendedAction.AUTO_APPROVE.value)
        & legitimate
    )
    requested_abuse = cases.loc[labels, "requested_amount_paise"].sum()
    prevented_abuse = cases.loc[
        labels & (cases["final_action"] != RecommendedAction.AUTO_APPROVE.value),
        "requested_amount_paise",
    ].sum()
    adaptive_cost = policy_comparison["returnguard_adaptive"]["realized_cost_paise"]
    approve_cost = policy_comparison["approve_all"]["realized_cost_paise"]
    payload = dict(source)
    payload.update({
        "schema_version": "2.0",
        "supersedes": {"path": preserved.name, "sha256": file_sha256(preserved)},
        "upgrade_reason": (
            "The v1 result schema omitted reconstructable policy and confidence-interval fields. "
            "The model, predictions, operating contract, and case artifact are unchanged."
        ),
        "policy": {
            "comparison": policy_comparison,
            "manual_reviews_per_1000": float(1000 * np.mean([
                action == RecommendedAction.MANUAL_REVIEW for action in final_actions
            ])),
            "legitimate_delay_rate": float(np.mean([
                action != RecommendedAction.AUTO_APPROVE
                for action, is_legitimate in zip(final_actions, legitimate, strict=True)
                if is_legitimate
            ])),
            "legitimate_rescue_count": int(rescue.sum()),
            "abuse_amount_recall": float(prevented_abuse / requested_abuse),
            "estimated_net_value_protected_paise_per_1000": float(
                (approve_cost - adaptive_cost) * 1000 / len(cases)
            ),
            "assumption_label": "Estimated from declared simulated cost assumptions; not realized savings.",
        },
        "confidence_intervals": {
            "calibrated_average_precision": _bootstrap_ap(cases, 1000, 20260831)
        },
    })
    payload.pop("results_sha256", None)
    payload["results_sha256"] = object_sha256(payload)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
