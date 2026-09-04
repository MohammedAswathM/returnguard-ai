#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd

from returnguard.v2.config import load_v2_policy_config
from returnguard.v2.policy import select_policy

report = select_policy(
    pd.read_parquet("artifacts/v2/development/features.parquet"),
    Path("artifacts/v2/training/model.joblib"),
    load_v2_policy_config(Path("configs/v2/policy.yaml")),
)
print(json.dumps({
    "review_rate_per_1000": report["selected_metrics"]["manual_reviews_per_1000"],
    "initial_legitimate_challenge_rate": report["selected_metrics"]["initial_legitimate_challenge_rate"],
    "likelihood_ratios": report["likelihoods"]["likelihood_ratios"],
}))
