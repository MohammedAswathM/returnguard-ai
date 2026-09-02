#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from returnguard.models.training import train_primary

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data_uci"))
parser.add_argument("--feature-config", type=Path, default=Path("configs/features.yaml"))
parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
args = parser.parse_args()
print(json.dumps(train_primary(args.data_dir, args.feature_config, args.model_config)))
