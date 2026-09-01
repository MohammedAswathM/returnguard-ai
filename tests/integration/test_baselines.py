import json
from pathlib import Path

import pandas as pd

from returnguard.config import BaselineConfig
from returnguard.evaluation.metrics import classification_metrics
from returnguard.features.engine import build_point_in_time_features
from returnguard.features.registry import load_feature_registry
from returnguard.models.baselines import train_and_evaluate_baselines


def test_baseline_exports_reconstructable_predictions(generated_data: Path, tmp_path: Path) -> None:
    registry = load_feature_registry(Path("configs/features.yaml"))
    features = build_point_in_time_features(generated_data, registry)
    output = tmp_path / "baselines"
    config = BaselineConfig(
        schema_version="1.0", label_column="is_refund_abuse_simulated", threshold=0.5,
        random_seed=42, max_iter=500, class_weight="balanced", output_dir=output,
    )
    report = train_and_evaluate_baselines(features, registry, config)
    predictions = pd.read_csv(output / "case_predictions.csv")
    policy_logistic = predictions.loc[
        (predictions["partition"] == "policy_selection")
        & (predictions["model"] == "logistic_regression")
    ]
    rebuilt = classification_metrics(
        policy_logistic["label_simulated"].to_numpy(),
        policy_logistic["score"].to_numpy(), 0.5,
    )
    stored = report["metrics"]["logistic_regression:policy_selection"]
    for key in ("support", "tn", "fp", "fn", "tp", "precision", "recall", "f1", "fpr"):
        assert rebuilt[key] == stored[key]
    disk = json.loads((output / "metrics.json").read_text())
    assert "accuracy" not in disk["metrics"]["logistic_regression:policy_selection"]
    assert "final_test" not in set(predictions["partition"])
    assert not any(key.endswith(":final_test") for key in report["metrics"])
    assert report["sanity_checks"]["permutation_average_precision"] < 0.35
