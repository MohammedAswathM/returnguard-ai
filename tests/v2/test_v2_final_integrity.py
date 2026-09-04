from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def test_v2_final_result_and_case_export_hashes() -> None:
    result_path = Path("artifacts/v2/final_results/results.lock.json")
    expected = Path(f"{result_path}.sha256").read_text(encoding="utf-8").split()[0]
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == expected
    result = json.loads(result_path.read_text(encoding="utf-8"))
    cases = pd.read_parquet("artifacts/v2/final_results/case_predictions.parquet")
    assert len(cases) == result["support"] == 2400
    assert int(cases["is_refund_abuse_simulated"].sum()) == 240
    assert result["classifier"]["tp"] + result["classifier"]["fn"] == 240
    assert result["classifier"]["tn"] + result["classifier"]["fp"] == 2160


def test_v2_correction_preserves_original_and_reconciles_policy_counts() -> None:
    correction = json.loads(Path(
        "results.v2.metric_integrity.v2.0.1.json"
    ).read_text(encoding="utf-8"))
    source = Path(correction["original_result_lock"]["path"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == correction["original_result_lock"]["sha256"]
    assert all(correction["unchanged"].values())
    adaptive = correction["corrections"]["policy_cost"]["comparison"]["adaptive_verification"]
    assert adaptive["manual_reviews"] == 110
    assert adaptive["support"] == 2400
    ci = correction["corrections"]["bootstrap_intervals"]["values"]
    point = json.loads(source.read_text(encoding="utf-8"))["classifier"]
    assert ci["precision"][0] <= point["precision"] <= ci["precision"][1]
    assert ci["recall"][0] <= point["recall"] <= ci["recall"][1]
