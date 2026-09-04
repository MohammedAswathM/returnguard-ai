#!/usr/bin/env python3
import json
from pathlib import Path

from returnguard.v2.config import load_v2_feature_config, load_v2_policy_config
from returnguard.v2.final import open_and_evaluate_final

result = open_and_evaluate_final(
    Path("artifacts/v2/data"), Path("artifacts/v2/locked_final"),
    Path("artifacts/v2/freeze_manifest.json"),
    load_v2_feature_config(Path("configs/v2/features.yaml")),
    load_v2_policy_config(Path("configs/v2/policy.yaml")),
    Path("artifacts/v2/training/model.joblib"), Path("artifacts/v2/policy/policy.json"),
    Path("artifacts/v2/final_results"), bootstrap_resamples=1000,
)
print(json.dumps({
    "status": result["status"], "support": result["support"],
    "prevalence": result["prevalence"],
    "raw_average_precision": result["classifier"]["raw_average_precision"],
}))
