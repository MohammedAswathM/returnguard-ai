#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from returnguard.evaluation.final import evaluate_final_once

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data_uci"))
parser.add_argument("--lock-dir", type=Path, default=Path("artifacts/data_uci_locked_final"))
parser.add_argument("--freeze", type=Path, default=Path("artifacts/freeze_manifest.json"))
parser.add_argument("--training-dir", type=Path, default=Path("artifacts/training_run"))
parser.add_argument("--policy-dir", type=Path, default=Path("artifacts/frozen_policy"))
parser.add_argument("--baseline-dir", type=Path, default=Path("artifacts/baselines_uci"))
parser.add_argument("--feature-config", type=Path, default=Path("configs/features.yaml"))
parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
parser.add_argument("--baseline-config", type=Path, default=Path("configs/model_baseline_uci.yaml"))
parser.add_argument("--policy-config", type=Path, default=Path("configs/policy.yaml"))
parser.add_argument("--output", type=Path, default=Path("artifacts/final_results"))
args = parser.parse_args()
print(json.dumps(evaluate_final_once(
    args.data_dir, args.lock_dir, args.freeze, args.training_dir, args.policy_dir,
    args.baseline_dir, args.feature_config, args.model_config, args.baseline_config,
    args.policy_config, args.output,
)))
