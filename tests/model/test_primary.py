from pathlib import Path

import numpy as np

from returnguard.config import ModelConfig
from returnguard.features.engine import build_point_in_time_features, build_serving_feature
from returnguard.features.registry import load_feature_registry
from returnguard.models.explanations import explain_cases
from returnguard.models.primary import fit_primary_model


def model_config(tmp_path: Path) -> ModelConfig:
    return ModelConfig(
        schema_version="1.0", model_version="test-lgbm", label_column="is_refund_abuse_simulated",
        random_seed=7, n_estimators=40, learning_rate=0.05, num_leaves=7,
        min_child_samples=10, subsample=0.9, colsample_bytree=0.9,
        calibration_bins=5, isotonic_min_support=100, output_dir=tmp_path,
    )


def test_primary_probabilities_explanations_and_train_serve_parity(
    generated_data: Path, tmp_path: Path,
) -> None:
    registry = load_feature_registry(Path("configs/features.yaml"))
    features = build_point_in_time_features(generated_data, registry)
    train = features.loc[features["partition"] == "train"]
    calibration = features.loc[features["partition"] == "calibration"]
    model, report = fit_primary_model(train, calibration, registry, model_config(tmp_path))
    scores = model.predict_proba(calibration)
    assert np.isfinite(scores).all()
    assert ((scores > 0) & (scores < 1)).all()
    assert report["selection_partition"] == "calibration"
    sample = calibration.iloc[:3]
    explanations = explain_cases(model, sample, registry)
    feature_names = set(model.feature_names)
    for reasons in explanations:
        assert len(reasons) <= 5
        assert all(reason["feature"] in feature_names for reason in reasons)
        assert all(
            (reason["contribution"] > 0) == (reason["direction"] == "increases_risk")
            for reason in reasons
        )
        assert all("fraudster" not in reason["message"].lower() for reason in reasons)
    request_id = str(calibration.iloc[0]["refund_request_id"])
    serving = build_serving_feature(generated_data, registry, request_id)
    batch = calibration.loc[calibration["refund_request_id"] == request_id].iloc[0]
    assert serving.to_dict() == batch.to_dict()


def test_minor_perturbation_keeps_at_least_one_top_reason(
    generated_data: Path, tmp_path: Path,
) -> None:
    registry = load_feature_registry(Path("configs/features.yaml"))
    features = build_point_in_time_features(generated_data, registry)
    train = features.loc[features["partition"] == "train"]
    calibration = features.loc[features["partition"] == "calibration"]
    model, _ = fit_primary_model(train, calibration, registry, model_config(tmp_path))
    sample = calibration.iloc[[0]].copy()
    perturbed = sample.copy()
    perturbed["requested_amount_paise"] *= 1.0001
    original_codes = {reason["code"] for reason in explain_cases(model, sample, registry)[0]}
    perturbed_codes = {reason["code"] for reason in explain_cases(model, perturbed, registry)[0]}
    assert original_codes & perturbed_codes
