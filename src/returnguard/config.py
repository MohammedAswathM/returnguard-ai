from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SplitRatios(StrictModel):
    train: float = Field(gt=0, lt=1)
    calibration: float = Field(gt=0, lt=1)
    policy_selection: float = Field(gt=0, lt=1)
    final_test: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def sums_to_one(self) -> "SplitRatios":
        if abs(sum((self.train, self.calibration, self.policy_selection, self.final_test)) - 1) > 1e-9:
            raise ValueError("split ratios must sum to 1")
        return self


class DataConfig(StrictModel):
    schema_version: str
    seed: int
    source_mode: str = Field(pattern="^(fixture|uci)$")
    uci_archive: Path | None = None
    uci_gbp_to_inr: float = Field(gt=0)
    n_orders: int = Field(ge=100)
    n_refund_requests: int = Field(ge=100)
    abuse_prevalence: float = Field(gt=0, lt=0.5)
    hard_legitimate_fraction: float = Field(ge=0, lt=0.5)
    start_at: str
    duration_days: int = Field(ge=30)
    outcome_delay_days: int = Field(ge=1)
    split_ratios: SplitRatios
    cold_start_fraction: float = Field(gt=0, lt=0.5)
    output_dir: Path

    @model_validator(mode="after")
    def require_uci_archive(self) -> "DataConfig":
        if self.source_mode == "uci" and self.uci_archive is None:
            raise ValueError("uci_archive is required when source_mode is uci")
        return self


class BaselineConfig(StrictModel):
    schema_version: str
    label_column: str
    threshold: float = Field(gt=0, lt=1)
    random_seed: int
    max_iter: int = Field(gt=0)
    class_weight: str | None
    output_dir: Path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"expected mapping in {path}")
    return raw


def load_data_config(path: Path) -> DataConfig:
    return DataConfig.model_validate(load_yaml(path))


def load_baseline_config(path: Path) -> BaselineConfig:
    return BaselineConfig.model_validate(load_yaml(path))
