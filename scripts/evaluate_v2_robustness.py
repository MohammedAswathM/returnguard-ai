#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd

from returnguard.v2.config import load_v2_data_config, load_v2_feature_config
from returnguard.v2.robustness import run_robustness

report = run_robustness(
    pd.read_parquet("artifacts/v2/development/features.parquet"),
    load_v2_feature_config(Path("configs/v2/features.yaml")),
    load_v2_data_config(Path("configs/v2/data.yaml")),
    Path("artifacts/v2/training/model.joblib"),
    Path("artifacts/v2/development/robustness.json"),
)
print(json.dumps(report["seed_summary"]))
