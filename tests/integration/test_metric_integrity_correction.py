import json
from pathlib import Path

import pandas as pd

from returnguard.data.fingerprints import file_sha256, object_sha256


def test_correction_reconstructs_customer_and_subgroup_metrics() -> None:
    correction_path = Path("results.metric_integrity.v1.1.json")
    correction = json.loads(correction_path.read_text(encoding="utf-8"))
    declared = correction.pop("correction_sha256")
    assert object_sha256(correction) == declared
    evidence = pd.read_csv("evidence/v1_case_audit.csv")
    assert file_sha256(Path("evidence/v1_case_audit.csv")) == correction["provenance"][
        "public_case_audit"
    ]["sha256"]
    legitimate = ~evidence["label_simulated"].astype(bool)
    challenged = legitimate & evidence["stage_a_action"].ne("AUTO_APPROVE")
    rescued = challenged & evidence["verification_result"].eq("consistent") & evidence[
        "final_action"
    ].eq("AUTO_APPROVE")
    terminal = challenged & evidence["final_action"].ne("AUTO_APPROVE")
    metrics = correction["correction"]["authoritative_public_metrics"]
    expected = {
        "initial_legitimate_challenge_rate": (184, 1451),
        "legitimate_rescue_rate": (168, 1451),
        "challenged_legitimate_rescue_rate": (168, 184),
        "terminal_legitimate_intervention_rate": (16, 1451),
    }
    assert (int(challenged.sum()), int(legitimate.sum())) == expected[
        "initial_legitimate_challenge_rate"
    ]
    assert (int(rescued.sum()), int(legitimate.sum())) == expected["legitimate_rescue_rate"]
    assert (int(rescued.sum()), int(challenged.sum())) == expected[
        "challenged_legitimate_rescue_rate"
    ]
    assert (int(terminal.sum()), int(legitimate.sum())) == expected[
        "terminal_legitimate_intervention_rate"
    ]
    for name, (numerator, denominator) in expected.items():
        assert metrics[name]["numerator"] == numerator
        assert metrics[name]["denominator"] == denominator
        assert metrics[name]["value"] == numerator / denominator
    assert metrics["manual_reviews_per_1000"]["numerator"] == 77
    assert metrics["manual_reviews_per_1000"]["value"] == 48.125


def test_original_locks_and_subgroups_are_bound() -> None:
    correction = json.loads(Path("results.metric_integrity.v1.1.json").read_text())
    assert correction["provenance"]["public_results_lock"]["sha256"] == file_sha256(
        Path("results.lock.json")
    )
    assert correction["provenance"]["public_results_lock"]["sha256"] == (
        "1832c422d79bc965692f89fe910e9dae5b8eca3240116d3c3b9ac842cbaba986"
    )
    evidence = pd.read_csv("evidence/v1_case_audit.csv")
    invalid = evidence["exceeds_original_captured_amount"].astype(bool)
    assert len(evidence) == 1600
    assert int(invalid.sum()) == 55
    assert int((~invalid).sum()) == 1545
    assert bool(evidence.loc[invalid, "label_simulated"].all())
    subgroup = correction["frozen_prediction_subgroup_audit"]
    assert subgroup["all_1600_cases"]["tp"] == 92
    assert subgroup["all_1600_cases"]["fp"] == 27
    assert subgroup["requests_not_exceeding_original_captured_payment"]["tp"] == 37
    assert subgroup["requests_not_exceeding_original_captured_payment"]["fp"] == 27
    policies = correction["simulation_policy_comparison"]
    assert policies["approve_all"]["simulated_abuse_case_intervention_recall"] == 0.0
    assert policies["return_first_all"]["simulated_abuse_case_intervention_recall"] == 1.0
    assert policies["adaptive_verification"]["legitimate_rescue_rate"] == 168 / 1451


def test_public_reporting_uses_correction_terminology() -> None:
    public = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in ("README.md", "MODEL_CARD.md", "DEMO_SCRIPT.md", "POLICY_CARD.md")
    )
    assert "Initial legitimate challenge rate" in public
    assert "Challenged-legitimate rescue rate" in public
    assert "Terminal legitimate intervention rate" in public
    assert "1.10% legitimate delay" not in public
