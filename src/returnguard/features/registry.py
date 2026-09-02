from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

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
    nullable: bool = False
    owner: str = "risk-model"
    reason_code: str | None = None

    @model_validator(mode="after")
    def populate_reason_code(self) -> "FeatureSpec":
        if self.reason_code is None:
            object.__setattr__(self, "reason_code", self.name.upper())
        return self


def load_feature_registry(path: Path) -> tuple[FeatureSpec, ...]:
    raw = load_yaml(path)
    return tuple(FeatureSpec.model_validate(item) for item in raw["features"])
