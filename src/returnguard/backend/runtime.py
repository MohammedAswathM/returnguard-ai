from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from returnguard.config import load_policy_config
from returnguard.data.fingerprints import object_sha256
from returnguard.features.registry import FeatureSpec, load_feature_registry
from returnguard.models.bundle import validate_bundle
from returnguard.models.explanations import explain_cases
from returnguard.models.primary import CalibratedRiskModel
from returnguard.policy.selection import PolicyContract, apply_contract


@dataclass(frozen=True)
class RuntimeScore:
    raw_score: float
    probability: float
    action: str
    reasons: list[dict[str, Any]]
    snapshot_hash: str


class BundleRuntime:
    def __init__(
        self, bundle_dir: Path, data_dir: Path, feature_config: Path, policy_config_path: Path,
    ) -> None:
        self.bundle_dir = bundle_dir
        self.data_dir = data_dir
        self.feature_config = feature_config
        self.validation_error: str | None = None
        try:
            validate_bundle(bundle_dir, data_dir, feature_config)
            model = joblib.load(bundle_dir / "model_pipeline.joblib")
            if not isinstance(model, CalibratedRiskModel):
                raise TypeError("unexpected model object")
            self.model = model
            self.registry = load_feature_registry(feature_config)
            self.policy_config = load_policy_config(policy_config_path)
            policy = json.loads((bundle_dir / "policy.json").read_text(encoding="utf-8"))
            self.contract = PolicyContract(**{
                key: policy[key] for key in PolicyContract.__dataclass_fields__
            })
            self.likelihoods = json.loads(
                (bundle_dir / "verifier_likelihoods.json").read_text(encoding="utf-8")
            )["likelihood_ratios"]
        except Exception as error:
            self.validation_error = f"{type(error).__name__}: {error}"

    @property
    def healthy(self) -> bool:
        return self.validation_error is None

    def score(self, features: dict[str, Any], amount_paise: int) -> RuntimeScore:
        if not self.healthy:
            raise RuntimeError("trusted bundle is unavailable")
        normalized: dict[str, Any] = {}
        for spec in self.registry:
            value = features.get(spec.name, spec.default)
            if value is None and not spec.nullable:
                value = spec.default
            normalized[spec.name] = value
        frame = pd.DataFrame([normalized], columns=list(self.model.feature_names))
        raw = float(self.model.predict_raw(frame)[0])
        probability = float(self.model.predict_proba(frame)[0])
        action = apply_contract(
            np.asarray([probability]), np.asarray([amount_paise], dtype=np.int64),
            self.contract, self.policy_config,
        )[0]
        reasons = explain_cases(self.model, frame, self.registry)[0]
        return RuntimeScore(
            raw_score=raw, probability=probability, action=action.value, reasons=reasons,
            snapshot_hash=object_sha256(normalized),
        )

    def posterior_action(self, probability: float, result: str, amount_paise: int) -> tuple[float, str]:
        from returnguard.verification.bayesian import bayesian_update

        posterior = bayesian_update(probability, float(self.likelihoods[result]))
        action = apply_contract(
            np.asarray([posterior]), np.asarray([amount_paise], dtype=np.int64),
            self.contract, self.policy_config,
        )[0]
        if action.value == "VERIFY":
            action_value = "AUTO_APPROVE" if result == "consistent" else "RETURN_FIRST"
        else:
            action_value = action.value
        return posterior, action_value


def limited_evidence(registry: tuple[FeatureSpec, ...], features: dict[str, Any]) -> bool:
    return any(spec.name not in features or features[spec.name] is None for spec in registry)
