from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from returnguard.config import load_model_config
from returnguard.data.fingerprints import object_sha256
from returnguard.evaluation.model_metrics import model_metrics
from returnguard.features.engine import build_point_in_time_features
from returnguard.features.registry import load_feature_registry
from returnguard.models.explanations import explain_cases
from returnguard.models.primary import fit_primary_model


def train_primary(
    data_dir: Path, feature_config: Path, model_config_path: Path,
) -> dict[str, Any]:
    config = load_model_config(model_config_path)
    registry = load_feature_registry(feature_config)
    features = build_point_in_time_features(data_dir, registry)
    train = features.loc[features["partition"] == "train"].copy()
    calibration = features.loc[features["partition"] == "calibration"].copy()
    model, calibration_report = fit_primary_model(train, calibration, registry, config)
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "model_pipeline.joblib")
    schema = {
        "schema_version": "1.0",
        "features": [spec.model_dump() for spec in registry],
        "ordered_names": list(model.feature_names),
    }
    schema["schema_sha256"] = object_sha256(schema)
    (output / "feature_schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    predictions: list[pd.DataFrame] = []
    reports: dict[str, Any] = {}
    for partition in ("train", "calibration"):
        rows = features.loc[features["partition"] == partition].copy()
        raw = model.predict_raw(rows)
        calibrated = model.predict_proba(rows)
        reports[partition] = {
            "raw": model_metrics(rows, raw, 0.5, config.calibration_bins),
            "calibrated": model_metrics(rows, calibrated, 0.5, config.calibration_bins),
        }
        predictions.append(pd.DataFrame({
            "refund_request_id": rows["refund_request_id"], "customer_id": rows["customer_id"],
            "partition": partition, "label_simulated": rows[config.label_column].astype(bool),
            "raw_score": raw, "calibrated_probability": calibrated,
            "requested_amount_paise": rows["requested_amount_paise"].astype(int),
        }))
    prediction_frame = pd.concat(predictions, ignore_index=True)
    prediction_frame.to_parquet(output / "development_predictions.parquet", index=False)
    sample = calibration.iloc[: min(50, len(calibration))]
    explanations = explain_cases(model, sample, registry)
    explanation_rows = [
        {"refund_request_id": request_id, "reasons": reasons}
        for request_id, reasons in zip(sample["refund_request_id"], explanations, strict=True)
    ]
    (output / "explanation_samples.json").write_text(
        json.dumps(explanation_rows, indent=2, default=str) + "\n", encoding="utf-8"
    )
    report = {
        "model_version": config.model_version,
        "partitions_used": {"model_fit": ["train"], "calibration": ["calibration"]},
        "calibration": calibration_report, "metrics": reports,
        "final_test_accessed": False,
    }
    (output / "training_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
