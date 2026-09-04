from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from returnguard.v2.config import V2DataConfig, V2FeatureConfig
from returnguard.v2.training import V2RiskModel, _logistic_pipeline


def _metrics(frame: pd.DataFrame, probabilities: np.ndarray[Any, np.dtype[np.float64]]) -> dict[str, float | int]:
    labels = frame["is_refund_abuse_simulated"].astype(bool).to_numpy()
    return {
        "support": len(frame), "prevalence": float(labels.mean()),
        "raw_average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def _cluster_sample(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    customers = frame["customer_id"].unique()
    sampled = rng.choice(customers, len(customers), replace=True)
    chunks = [frame.loc[frame["customer_id"].eq(customer)].copy() for customer in sampled]
    return pd.concat(chunks, ignore_index=True)


def run_robustness(
    features: pd.DataFrame, feature_config: V2FeatureConfig, data_config: V2DataConfig,
    model_path: Path, output_path: Path,
) -> dict[str, Any]:
    model: V2RiskModel = joblib.load(model_path)
    policy = features.loc[features["partition"].eq("policy_selection")].copy()
    seeds: list[dict[str, Any]] = []
    for seed in data_config.robustness_seeds:
        sample = _cluster_sample(policy, seed)
        seeds.append({"seed": seed, **_metrics(sample, model.predict_proba(sample))})
    aps = [float(item["raw_average_precision"]) for item in seeds]

    scenario_names = (
        "genuine_product_defect_burst", "high_value_loyal_customer", "first_time_refund",
        "cold_start_customer", "unusual_legitimate_purchase", "missing_evidence",
        "verifier_unavailable",
    )
    slices: dict[str, Any] = {}
    for scenario in scenario_names:
        subset = policy.loc[policy["simulation_scenario"].eq(scenario)]
        slices[scenario] = (
            _metrics(subset, model.predict_proba(subset)) if subset["is_refund_abuse_simulated"].nunique() > 1
            else {"support": len(subset), "prevalence": float(subset["is_refund_abuse_simulated"].mean()),
                  "underpowered": True}
        )
    cold = policy.loc[policy["cold_start_indicator"].eq(1.0)]
    slices["cold_start_indicator"] = (
        _metrics(cold, model.predict_proba(cold)) if cold["is_refund_abuse_simulated"].nunique() > 1
        else {"support": len(cold), "underpowered": True}
    )

    stress: dict[str, Any] = {}
    for group, names in {
        "missing_claim_integrity": [
            "reason_history_inconsistency", "evidence_provided", "returnless_requested"
        ],
        "missing_customer_history": [
            "prior_order_count", "smoothed_refund_rate_90d", "prior_matured_adverse_rate"
        ],
    }.items():
        changed = policy.copy()
        for name in names:
            spec = next(item for item in feature_config.features if item.name == name)
            changed[name] = spec.default
        stress[group] = _metrics(policy, model.predict_proba(changed))
    unseen = policy.copy()
    unseen["category"] = "v2_unseen_category"
    stress["unseen_category"] = _metrics(policy, model.predict_proba(unseen))
    duplicated = pd.concat([policy, policy.iloc[:100]], ignore_index=True)
    original = model.predict_proba(policy.iloc[:100])
    replay = model.predict_proba(duplicated.iloc[-100:])
    stress["duplicate_request_prediction_parity"] = {
        "support": 100, "maximum_probability_difference": float(np.max(np.abs(original - replay)))
    }
    amount_cutoff = float(policy["claimed_refund_amount_paise"].quantile(0.99))
    outliers = policy.loc[policy["claimed_refund_amount_paise"].ge(amount_cutoff)]
    stress["amount_outliers_top_1_percent"] = (
        _metrics(outliers, model.predict_proba(outliers))
        if outliers["is_refund_abuse_simulated"].nunique() > 1
        else {"support": len(outliers), "underpowered": True}
    )
    rng = np.random.default_rng(data_config.primary_seed)
    labels = policy["is_refund_abuse_simulated"].astype(bool).to_numpy()
    base_probabilities = model.predict_proba(policy)
    for noise in (0.05, 0.10):
        noisy = labels.copy()
        flip = rng.choice(len(noisy), round(len(noisy) * noise), replace=False)
        noisy[flip] = ~noisy[flip]
        stress[f"label_noise_{int(noise * 100)}pct"] = {
            "support": len(policy), "prevalence": float(noisy.mean()),
            "raw_average_precision": float(average_precision_score(noisy, base_probabilities)),
        }
    for prevalence in (0.02, 0.05, 0.09, 0.15):
        abuse_rows = policy.loc[policy["is_refund_abuse_simulated"].astype(bool)]
        legitimate_rows = policy.loc[~policy["is_refund_abuse_simulated"].astype(bool)]
        target_abuse = max(1, round(len(policy) * prevalence))
        target_legitimate = len(policy) - target_abuse
        shifted = pd.concat([
            abuse_rows.sample(target_abuse, replace=target_abuse > len(abuse_rows), random_state=1),
            legitimate_rows.sample(target_legitimate, replace=target_legitimate > len(legitimate_rows), random_state=2),
        ], ignore_index=True)
        stress[f"prevalence_{int(prevalence * 100)}pct"] = _metrics(
            shifted, model.predict_proba(shifted)
        )

    train = features.loc[features["partition"].eq("train")]
    group_ablation: dict[str, Any] = {}
    all_names = [spec.name for spec in feature_config.features]
    for group in sorted({spec.group for spec in feature_config.features}):
        kept_specs = tuple(spec for spec in feature_config.features if spec.group != group)
        reduced_config = feature_config.model_copy(update={"features": kept_specs})
        kept_names = [name for name in all_names if name in {spec.name for spec in kept_specs}]
        estimator = _logistic_pipeline(reduced_config, data_config.primary_seed)
        estimator.fit(train[kept_names], train["is_refund_abuse_simulated"].astype(bool))
        probability = estimator.predict_proba(policy[kept_names])[:, 1]
        group_ablation[group] = {
            "removed_features": sorted(set(all_names) - set(kept_names)),
            "policy_selection_average_precision": float(average_precision_score(
                policy["is_refund_abuse_simulated"], probability
            )),
        }

    report: dict[str, Any] = {
        "schema_version": "2.0", "scope": "synthetic development stress tests; not held-out evidence",
        "seed_method": (
            "customer-cluster resampling of policy-selection cases using "
            "preregistered non-final robustness seeds"
        ),
        "seed_results": seeds,
        "seed_summary": {"mean_average_precision": float(np.mean(aps)),
                         "standard_deviation_average_precision": float(np.std(aps)),
                         "worst_seed_average_precision": float(np.min(aps))},
        "hard_legitimate_slices": slices, "stress_tests": stress,
        "feature_group_ablations": group_ablation,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
