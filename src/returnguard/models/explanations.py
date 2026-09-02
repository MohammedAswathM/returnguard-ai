from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap

from returnguard.features.registry import FeatureSpec
from returnguard.models.primary import CalibratedRiskModel


@dataclass(frozen=True)
class Reason:
    code: str
    feature: str
    observed: float | str
    reference: float | str
    contribution: float
    direction: str
    message: str


def explain_cases(
    model: CalibratedRiskModel, frame: pd.DataFrame, registry: tuple[FeatureSpec, ...],
) -> list[list[dict[str, Any]]]:
    transformed = model.transform(frame)
    values = shap.TreeExplainer(model.estimator).shap_values(transformed)
    if isinstance(values, list):
        values = values[-1]
    contributions = np.asarray(values, dtype=float)
    explanations: list[list[dict[str, Any]]] = []
    for row_index, (_, row) in enumerate(frame.iterrows()):
        reasons: list[Reason] = []
        for feature_index, spec in enumerate(registry):
            contribution = float(contributions[row_index, feature_index])
            direction = "increases_risk" if contribution > 0 else "mitigates_risk"
            code = spec.reason_code or spec.name.upper()
            message = (
                f"{spec.explanation}: observed value {row[spec.name]!s}; "
                f"development reference {model.reference_values[spec.name]!s}."
            )
            reasons.append(Reason(
                code=code, feature=spec.name, observed=row[spec.name],
                reference=model.reference_values[spec.name], contribution=contribution,
                direction=direction, message=message,
            ))
        increasing = sorted(
            (reason for reason in reasons if reason.contribution > 0),
            key=lambda reason: reason.contribution, reverse=True,
        )[:3]
        mitigating = sorted(
            (reason for reason in reasons if reason.contribution < 0),
            key=lambda reason: reason.contribution,
        )[:2]
        explanations.append([asdict(reason) for reason in [*increasing, *mitigating]])
    return explanations
