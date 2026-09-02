#!/usr/bin/env python3
import json
from pathlib import Path

from returnguard.evaluation.integrity import build_metric_integrity_correction

result = build_metric_integrity_correction(
    public_lock=Path("results.lock.json"),
    preserved_lock=Path("artifacts/final_results/results.v1.lock.json"),
    cases_path=Path("artifacts/final_results/final_case_results.parquet"),
    opened_data=Path("artifacts/final_results/opened_data"),
    training_dir=Path("artifacts/training_run"),
    feature_config=Path("configs/features.yaml"),
    evidence_path=Path("evidence/v1_case_audit.csv"),
    transitions_path=Path("evidence/v1_policy_transitions.csv"),
    output_path=Path("results.metric_integrity.v1.1.json"),
)
print(json.dumps({"status": "PASS", "correction_sha256": result["correction_sha256"]}))
