#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd

from returnguard.v2.config import load_v2_feature_config, load_v2_model_config
from returnguard.v2.training import train_v2

features = pd.read_parquet("artifacts/v2/development/features.parquet")
report = train_v2(
    features,
    load_v2_feature_config(Path("configs/v2/features.yaml")),
    load_v2_model_config(Path("configs/v2/model.yaml")),
)
print(json.dumps({
    "selected_candidate": report["selected_candidate"],
    "selected_trial": report["selected_trial"],
    "calibrator": report["calibration"]["selected"],
}))
