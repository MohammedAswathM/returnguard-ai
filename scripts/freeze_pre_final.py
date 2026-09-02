#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from returnguard.evaluation.freeze import freeze_pre_final

parser = argparse.ArgumentParser()
parser.add_argument("--source-commit", required=True)
parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data_uci"))
parser.add_argument("--lock-dir", type=Path, default=Path("artifacts/data_uci_locked_final"))
parser.add_argument("--training-dir", type=Path, default=Path("artifacts/training_run"))
parser.add_argument("--policy-dir", type=Path, default=Path("artifacts/frozen_policy"))
parser.add_argument("--bundle-dir", type=Path, default=Path("artifacts/model_bundle"))
parser.add_argument("--feature-config", type=Path, default=Path("configs/features.yaml"))
parser.add_argument("--policy-config", type=Path, default=Path("configs/policy.yaml"))
parser.add_argument("--output", type=Path, default=Path("artifacts/freeze_manifest.json"))
args = parser.parse_args()
print(json.dumps(freeze_pre_final(
    args.data_dir, args.lock_dir, args.training_dir, args.policy_dir, args.bundle_dir,
    args.feature_config, args.policy_config, args.source_commit,
    (Path("src"), Path("scripts"), Path("configs"), Path("pyproject.toml")), args.output,
)))
