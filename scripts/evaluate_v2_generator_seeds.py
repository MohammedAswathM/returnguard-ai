#!/usr/bin/env python3
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss

from returnguard.v2.config import load_v2_data_config, load_v2_feature_config
from returnguard.v2.features import build_features
from returnguard.v2.generator import generate_v2
from returnguard.v2.training import V2RiskModel

data_config = load_v2_data_config(Path("configs/v2/data.yaml"))
feature_config = load_v2_feature_config(Path("configs/v2/features.yaml"))
model: V2RiskModel = joblib.load("artifacts/v2/training/model.joblib")
rows = []
for seed in data_config.robustness_seeds:
    data_dir = Path(f"/tmp/returnguard-v2-generator-seed-{seed}")
    lock_dir = Path(f"/tmp/returnguard-v2-generator-seed-{seed}-sealed")
    replay_config = data_config.model_copy(update={"primary_seed": seed})
    metadata = generate_v2(replay_config, data_dir, lock_dir)
    features = build_features(data_dir, feature_config)
    development = features.loc[features["partition"].isin(["calibration", "policy_selection"])]
    labels = development["is_refund_abuse_simulated"].astype(bool).to_numpy()
    raw = model.raw_score(development)
    calibrated = model.predict_proba(development)
    lock = json.loads((lock_dir / "manifest.json").read_text(encoding="utf-8"))
    if lock["state"] != "SEALED" or lock["opened_at"] is not None:
        raise ValueError("robustness generator replay final split was opened")
    rows.append({
        "generator_seed": seed, "development_support": len(development),
        "development_prevalence": float(labels.mean()),
        "raw_average_precision": float(average_precision_score(labels, raw)),
        "calibrated_brier_score": float(brier_score_loss(labels, calibrated)),
        "transformed_data_sha256": metadata["transformed_data_sha256"],
        "final_state": "SEALED_NOT_EVALUATED",
    })
aps = [row["raw_average_precision"] for row in rows]
report = {
    "schema_version": "2.0",
    "scope": "five full synthetic generator replays; development partitions only",
    "not_additional_real_world_evidence": True,
    "results": rows,
    "summary": {
        "mean_raw_average_precision": float(np.mean(aps)),
        "standard_deviation_raw_average_precision": float(np.std(aps)),
        "worst_seed_raw_average_precision": float(np.min(aps)),
    },
}
path = Path("artifacts/v2/development/generator_seed_robustness.json")
path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report["summary"]))
