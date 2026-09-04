#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from returnguard.v2.config import load_v2_data_config
from returnguard.v2.generator import generate_v2

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("configs/v2/data.yaml"))
parser.add_argument("--output-dir", type=Path)
parser.add_argument("--lock-dir", type=Path)
args = parser.parse_args()
print(json.dumps(generate_v2(load_v2_data_config(args.config), args.output_dir, args.lock_dir)))
