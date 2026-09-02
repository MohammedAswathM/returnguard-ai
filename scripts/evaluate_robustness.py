#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from returnguard.evaluation.robustness import run_robustness

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data_uci"))
parser.add_argument("--training-dir", type=Path, default=Path("artifacts/training_run"))
parser.add_argument("--feature-config", type=Path, default=Path("configs/features.yaml"))
parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
parser.add_argument("--output", type=Path, default=Path("artifacts/frozen_policy/robustness.json"))
parser.add_argument("--resamples", type=int, default=1000)
parser.add_argument("--alternate-seed-dir", type=Path, action="append", default=[])
parser.add_argument("--policy-config", type=Path, default=Path("configs/policy.yaml"))
args = parser.parse_args()
print(json.dumps(run_robustness(
    args.data_dir, args.training_dir, args.feature_config, args.model_config,
    args.output, args.resamples, tuple(args.alternate_seed_dir), args.policy_config,
)))
