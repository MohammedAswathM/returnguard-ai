from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class V2SplitCounts(StrictModel):
    train: int = Field(gt=0)
    calibration: int = Field(gt=0)
    policy_selection: int = Field(gt=0)
    final_test: int = Field(gt=0)


class V2DataConfig(StrictModel):
    schema_version: Literal["2.0"]
    benchmark_version: str
    generator_version: str
    source_archive: Path
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary_seed: int
    robustness_seeds: tuple[int, ...]
    n_orders: int = Field(ge=12_000)
    n_ml_claims: int = Field(ge=1_000)
    n_integrity_cases: int = Field(ge=100)
    abuse_prevalence: float = Field(gt=0, lt=0.5)
    hard_legitimate_fraction: float = Field(gt=0, lt=0.5)
    outcome_delay_days: int = Field(ge=1)
    split_counts: V2SplitCounts
    cold_start_fraction: float = Field(gt=0, lt=0.5)
    output_dir: Path
    lock_dir: Path

    @model_validator(mode="after")
    def counts_match(self) -> V2DataConfig:
        support = sum(self.split_counts.model_dump().values())
        if support != self.n_ml_claims:
            raise ValueError("v2 split counts must equal n_ml_claims")
        if len(self.robustness_seeds) < 5:
            raise ValueError("v2 requires at least five robustness seeds")
        return self


class V2FeatureSpec(StrictModel):
    name: str
    type: Literal["float", "category"]
    group: str
    default: float | str
    cutoff_rule: str
    maturity_rule: str


class V2FeatureConfig(StrictModel):
    schema_version: Literal["2.0"]
    features: tuple[V2FeatureSpec, ...]
    forbidden_features: tuple[str, ...]


class V2ModelConfig(StrictModel):
    schema_version: Literal["2.0"]
    model_version: str
    random_seed: int
    model_candidates: tuple[str, ...]
    lightgbm_trials: int = Field(ge=1, le=50)
    rolling_origin_folds: int = Field(ge=2)
    instability_penalty: float = Field(ge=0)
    review_capacity_penalty: float = Field(ge=0)
    max_review_rate: float = Field(gt=0, lt=1)
    calibration_methods: tuple[Literal["sigmoid", "isotonic"], ...]
    calibration_selection: str
    ensemble_selection: str
    bootstrap_resamples: int = Field(ge=1000)
    output_dir: Path


class V2PolicyConfig(StrictModel):
    schema_version: Literal["2.0"]
    policy_version: str
    max_manual_review_rate: float = Field(gt=0, lt=1)
    max_verifications_per_request: Literal[1]
    max_initial_legitimate_challenge_rate: float = Field(gt=0, lt=1)
    additional_loss_rate: float = Field(ge=0)
    salvage_rate: float = Field(ge=0, le=1)
    verification_cost_paise: int = Field(ge=0)
    review_cost_paise: int = Field(ge=0)
    legitimate_verification_friction_paise: int = Field(ge=0)
    legitimate_review_friction_paise: int = Field(ge=0)
    legitimate_delay_cost_paise: int = Field(ge=0)
    reverse_logistics_cost_paise: int = Field(ge=0)
    laplace_alpha: float = Field(gt=0)
    technical_unavailability_likelihood_ratio: float
    output_dir: Path

    @model_validator(mode="after")
    def technical_results_are_neutral(self) -> V2PolicyConfig:
        if self.technical_unavailability_likelihood_ratio != 1.0:
            raise ValueError("technical unavailability likelihood ratio must equal 1.0")
        return self


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def load_v2_data_config(path: Path) -> V2DataConfig:
    return V2DataConfig.model_validate(_load(path))


def load_v2_feature_config(path: Path) -> V2FeatureConfig:
    return V2FeatureConfig.model_validate(_load(path))


def load_v2_model_config(path: Path) -> V2ModelConfig:
    return V2ModelConfig.model_validate(_load(path))


def load_v2_policy_config(path: Path) -> V2PolicyConfig:
    return V2PolicyConfig.model_validate(_load(path))
