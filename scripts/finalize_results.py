#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from returnguard.evaluation.results import upgrade_results_lock

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", type=Path, default=Path("artifacts/final_results"))
parser.add_argument("--policy-config", type=Path, default=Path("configs/policy.yaml"))
parser.add_argument("--output", type=Path, default=Path("artifacts/final_results/results.lock.json"))
args = parser.parse_args()
print(json.dumps(upgrade_results_lock(args.results_dir, args.policy_config, args.output)))
