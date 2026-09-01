from pathlib import Path

from pydantic import BaseModel, ConfigDict

from returnguard.config import load_yaml


class FeatureSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    type: str
    entity: str
    cutoff_rule: str
    maturity_rule: str
    default: float | str
    explanation: str


def load_feature_registry(path: Path) -> tuple[FeatureSpec, ...]:
    raw = load_yaml(path)
    return tuple(FeatureSpec.model_validate(item) for item in raw["features"])

