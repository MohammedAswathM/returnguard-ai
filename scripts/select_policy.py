#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from returnguard.policy.selection import select_policy

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data_uci"))
parser.add_argument("--training-dir", type=Path, default=Path("artifacts/training_run"))
parser.add_argument("--feature-config", type=Path, default=Path("configs/features.yaml"))
parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
parser.add_argument("--policy-config", type=Path, default=Path("configs/policy.yaml"))
parser.add_argument(
    "--baseline-predictions", type=Path,
    default=Path("artifacts/baselines_uci/case_predictions.csv"),
)
args = parser.parse_args()
print(json.dumps(select_policy(
    args.data_dir, args.training_dir, args.feature_config, args.model_config, args.policy_config,
    args.baseline_predictions,
)))
