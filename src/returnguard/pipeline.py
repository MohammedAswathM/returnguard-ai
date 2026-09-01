import json
from pathlib import Path
from typing import Any

from returnguard.config import load_baseline_config, load_data_config
from returnguard.data.generator import generate_benchmark
from returnguard.features.engine import build_point_in_time_features
from returnguard.features.registry import load_feature_registry
from returnguard.models.baselines import train_and_evaluate_baselines


def generate(config_path: Path, output_dir: Path | None = None) -> dict[str, Any]:
    return generate_benchmark(load_data_config(config_path), output_dir)


def train_baselines(
    data_dir: Path, model_config_path: Path, features_config_path: Path
) -> dict[str, Any]:
    config = load_baseline_config(model_config_path)
    registry = load_feature_registry(features_config_path)
    feature_frame = build_point_in_time_features(data_dir, registry)
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    development_features = feature_frame.loc[feature_frame["partition"] != "final_test"]
    development_features.to_csv(output / "feature_snapshot.csv", index=False, lineterminator="\n")
    (output / "feature_schema.json").write_text(
        json.dumps([spec.model_dump() for spec in registry], indent=2) + "\n", encoding="utf-8"
    )
    return train_and_evaluate_baselines(feature_frame, registry, config, output)
