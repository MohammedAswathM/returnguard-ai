from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from returnguard.v2.policy import FrozenPolicy, apply_policy, posterior_probability
from returnguard.v2.training import V2RiskModel


def test_model_selection_used_development_partitions() -> None:
    report = json.loads(Path("artifacts/v2/training/selection.json").read_text(encoding="utf-8"))
    assert report["scope"] == "development data only"
    assert report["calibration"]["selection_partition"] == "calibration"
    assert report["selected_candidate"] in {"lightgbm_default", "lightgbm_tuned"}


def test_policy_constraints_and_likelihood_directions() -> None:
    report = json.loads(Path("artifacts/v2/policy/policy.json").read_text(encoding="utf-8"))
    lr = report["likelihoods"]["likelihood_ratios"]
    assert lr["consistent"] < 1
    assert lr["inconsistent"] > 1
    assert 0.5 <= lr["evidence_inconclusive"] <= 2
    assert lr["verifier_unavailable"] == 1.0
    assert lr["verifier_timeout"] == 1.0
    assert report["selected_metrics"]["manual_reviews_per_1000"] <= 50
    assert report["selected_metrics"]["initial_legitimate_challenge_rate"] <= 0.20


def test_five_full_generator_seed_replays_kept_final_sealed() -> None:
    report = json.loads(Path(
        "artifacts/v2/development/generator_seed_robustness.json"
    ).read_text(encoding="utf-8"))
    assert len(report["results"]) == 5
    assert len({row["generator_seed"] for row in report["results"]}) == 5
    assert all(row["final_state"] == "SEALED_NOT_EVALUATED" for row in report["results"])
    assert report["not_additional_real_world_evidence"] is True


def test_technical_unavailability_never_changes_probability() -> None:
    likelihoods = {
        "consistent": 0.25, "inconsistent": 4.0, "evidence_inconclusive": 1.1,
        "verifier_unavailable": 1.0, "verifier_timeout": 1.0,
    }
    for prior in (0.01, 0.2, 0.75, 0.99):
        assert posterior_probability(prior, "verifier_unavailable", likelihoods) == prior
        assert posterior_probability(prior, "verifier_timeout", likelihoods) == prior


def test_duplicate_case_has_prediction_parity() -> None:
    model: V2RiskModel = joblib.load("artifacts/v2/training/model.joblib")
    frame = pd.read_parquet("artifacts/v2/development/features.parquet").iloc[:1]
    duplicated = pd.concat([frame, frame], ignore_index=True)
    values = model.predict_proba(duplicated)
    assert values[0] == values[1]


def test_unavailable_verifier_routes_to_review_without_risk_increase() -> None:
    model: V2RiskModel = joblib.load("artifacts/v2/training/model.joblib")
    frame = pd.read_parquet("artifacts/v2/development/features.parquet")
    case = frame.loc[frame["partition"].eq("policy_selection")].iloc[[0]].copy()
    probability = model.predict_proba(case)
    case["verification_result_simulated"] = "verifier_unavailable"
    policy = FrozenPolicy(0.0, 1.0, {
        "consistent": 0.25, "inconsistent": 4.0, "evidence_inconclusive": 1.1,
        "verifier_unavailable": 1.0, "verifier_timeout": 1.0,
    })
    result = apply_policy(case, probability, policy).iloc[0]
    assert result["stage_a_action"] == "VERIFY"
    assert result["final_action"] == "MANUAL_REVIEW"
    assert result["posterior_probability"] == result["initial_probability"]
