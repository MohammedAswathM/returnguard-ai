from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from returnguard.config import ModelConfig
from returnguard.features.leakage import reject_leakage_columns
from returnguard.features.registry import FeatureSpec
from returnguard.models.guard import require_fit_partition


@dataclass
class CalibratedRiskModel:
    model_version: str
    feature_names: tuple[str, ...]
    preprocessor: ColumnTransformer
    estimator: LGBMClassifier
    calibrator_name: str
    calibrator: Any
    reference_values: dict[str, float | str]

    def transform(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        values = self.preprocessor.transform(frame[list(self.feature_names)])
        return np.asarray(values, dtype=float)

    def predict_raw(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        probabilities = np.asarray(self.estimator.predict_proba(self.transform(frame)), dtype=float)
        return probabilities[:, 1]

    def predict_proba(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        raw = self.predict_raw(frame)
        if self.calibrator_name == "sigmoid":
            calibrated = self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        else:
            calibrated = self.calibrator.predict(raw)
        return np.asarray(np.clip(calibrated, 1e-6, 1 - 1e-6), dtype=float)


def _preprocessor(registry: tuple[FeatureSpec, ...]) -> ColumnTransformer:
    numeric = [spec.name for spec in registry if spec.type != "category"]
    categorical = [spec.name for spec in registry if spec.type == "category"]
    if [*numeric, *categorical] != [spec.name for spec in registry]:
        raise ValueError("feature registry must list numeric features before categorical features")
    return ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), categorical),
    ], verbose_feature_names_out=False)


def _fit_calibrator(name: str, scores: npt.NDArray[np.float64], labels: npt.NDArray[np.bool_]) -> Any:
    if name == "sigmoid":
        calibrator = LogisticRegression(random_state=0)
        calibrator.fit(scores.reshape(-1, 1), labels)
        return calibrator
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(scores, labels)
    return calibrator


def _calibrated_scores(name: str, calibrator: Any, scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if name == "sigmoid":
        values = calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
    else:
        values = calibrator.predict(scores)
    return np.asarray(np.clip(values, 1e-6, 1 - 1e-6), dtype=float)


def fit_primary_model(
    train: pd.DataFrame, calibration: pd.DataFrame, registry: tuple[FeatureSpec, ...],
    config: ModelConfig,
) -> tuple[CalibratedRiskModel, dict[str, Any]]:
    require_fit_partition(train)
    if set(calibration["partition"]) != {"calibration"}:
        raise ValueError("calibrator accepts calibration partition only")
    feature_names = tuple(spec.name for spec in registry)
    reject_leakage_columns(feature_names)
    preprocessor = _preprocessor(registry)
    x_train = np.asarray(preprocessor.fit_transform(train[list(feature_names)]), dtype=float)
    estimator = LGBMClassifier(
        objective="binary", n_estimators=config.n_estimators, learning_rate=config.learning_rate,
        num_leaves=config.num_leaves, min_child_samples=config.min_child_samples,
        subsample=config.subsample, colsample_bytree=config.colsample_bytree,
        class_weight="balanced", random_state=config.random_seed, verbosity=-1, n_jobs=1,
    )
    estimator.fit(x_train, train[config.label_column].astype(bool).to_numpy())
    ordered_calibration = calibration.sort_values(["requested_at", "refund_request_id"])
    split = len(ordered_calibration) // 2
    fit_rows, selection_rows = ordered_calibration.iloc[:split], ordered_calibration.iloc[split:]
    fit_probabilities = np.asarray(
        estimator.predict_proba(preprocessor.transform(fit_rows[list(feature_names)])), dtype=float
    )
    selection_probabilities = np.asarray(
        estimator.predict_proba(preprocessor.transform(selection_rows[list(feature_names)])), dtype=float
    )
    fit_raw = fit_probabilities[:, 1]
    selection_raw = selection_probabilities[:, 1]
    candidates = ["sigmoid"]
    if len(calibration) >= config.isotonic_min_support:
        candidates.append("isotonic")
    comparison: dict[str, float] = {}
    for name in candidates:
        candidate = _fit_calibrator(
            name, np.asarray(fit_raw, dtype=float),
            fit_rows[config.label_column].astype(bool).to_numpy(),
        )
        candidate_scores = _calibrated_scores(name, candidate, np.asarray(selection_raw, dtype=float))
        comparison[name] = float(
            brier_score_loss(selection_rows[config.label_column].astype(bool), candidate_scores)
        )
    selected = min(comparison, key=lambda name: comparison[name])
    full_probabilities = np.asarray(
        estimator.predict_proba(preprocessor.transform(ordered_calibration[list(feature_names)])),
        dtype=float,
    )
    full_raw = full_probabilities[:, 1]
    calibrator = _fit_calibrator(
        selected, np.asarray(full_raw, dtype=float),
        ordered_calibration[config.label_column].astype(bool).to_numpy(),
    )
    references: dict[str, float | str] = {}
    for spec in registry:
        series = train[spec.name]
        references[spec.name] = (
            str(series.mode(dropna=True).iloc[0])
            if spec.type == "category" else float(series.median())
        )
    model = CalibratedRiskModel(
        model_version=config.model_version, feature_names=feature_names,
        preprocessor=preprocessor, estimator=estimator, calibrator_name=selected,
        calibrator=calibrator, reference_values=references,
    )
    return model, {
        "selection_partition": "calibration",
        "selection_method": "chronological half-fit/half-evaluate; refit selected on full calibration",
        "candidate_brier": comparison, "selected": selected,
        "support": len(calibration),
    }
