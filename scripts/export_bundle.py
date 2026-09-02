#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from returnguard.models.bundle import export_bundle

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data_uci"))
parser.add_argument("--training-dir", type=Path, default=Path("artifacts/training_run"))
parser.add_argument("--policy-dir", type=Path, default=Path("artifacts/frozen_policy"))
parser.add_argument("--feature-config", type=Path, default=Path("configs/features.yaml"))
parser.add_argument("--data-card", type=Path, default=Path("docs/DATA_CARD.md"))
parser.add_argument("--output", type=Path, default=Path("artifacts/model_bundle"))
parser.add_argument("--policy-config", type=Path, default=Path("configs/policy.yaml"))
parser.add_argument(
    "--final-results", type=Path, default=Path("artifacts/final_results/results.lock.json")
)
args = parser.parse_args()
print(json.dumps(export_bundle(
    args.data_dir, args.training_dir, args.policy_dir, args.feature_config,
    args.data_card, args.output, args.policy_config, args.final_results,
)))
