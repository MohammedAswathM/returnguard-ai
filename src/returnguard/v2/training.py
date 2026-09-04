from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from returnguard.data.fingerprints import file_sha256, object_sha256
from returnguard.evaluation.model_metrics import reliability_bins
from returnguard.v2.config import V2FeatureConfig, V2ModelConfig


@dataclass
class V2RiskModel:
    feature_names: tuple[str, ...]
    preprocessor: ColumnTransformer
    estimator: LGBMClassifier
    calibrator_name: str
    calibrator: Any

    def raw_score(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        matrix = np.asarray(self.preprocessor.transform(frame[list(self.feature_names)]), dtype=float)
        probabilities = np.asarray(self.estimator.predict_proba(matrix), dtype=float)
        return probabilities[:, 1]

    def predict_proba(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        raw = self.raw_score(frame)
        if self.calibrator_name == "sigmoid":
            values = self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        else:
            values = self.calibrator.predict(raw)
        return np.asarray(np.clip(values, 1e-6, 1 - 1e-6), dtype=float)


def _tree_preprocessor(config: V2FeatureConfig) -> ColumnTransformer:
    numeric = [spec.name for spec in config.features if spec.type == "float"]
    categorical = [spec.name for spec in config.features if spec.type == "category"]
    return ColumnTransformer([
        ("numeric", "passthrough", numeric),
        ("category", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical),
    ], verbose_feature_names_out=False)


def _logistic_pipeline(config: V2FeatureConfig, seed: int) -> Pipeline:
    numeric = [spec.name for spec in config.features if spec.type == "float"]
    categorical = [spec.name for spec in config.features if spec.type == "category"]
    return Pipeline([
        ("preprocessor", ColumnTransformer([
            ("numeric", StandardScaler(), numeric),
            ("category", OneHotEncoder(handle_unknown="ignore"), categorical),
        ])),
        ("model", LogisticRegression(
            C=0.5, max_iter=1500, class_weight="balanced", random_state=seed
        )),
    ])


def _folds(train: pd.DataFrame, count: int) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    ordered = train.sort_values(["decision_time", "refund_request_id"]).reset_index(drop=True)
    validation_size = len(ordered) // (count + 3)
    initial = len(ordered) - count * validation_size
    return [
        (ordered.iloc[:initial + index * validation_size],
         ordered.iloc[initial + index * validation_size:initial + (index + 1) * validation_size])
        for index in range(count)
    ]


def _sample_trials(config: V2ModelConfig) -> list[dict[str, Any]]:
    rng = np.random.default_rng(config.random_seed)
    trials: list[dict[str, Any]] = [{
        "learning_rate": 0.05, "num_leaves": 15, "max_depth": 5,
        "min_child_samples": 40, "n_estimators": 600, "subsample": 0.9,
        "colsample_bytree": 0.9, "reg_alpha": 0.1, "reg_lambda": 1.0,
        "min_split_gain": 0.0,
    }]
    while len(trials) < config.lightgbm_trials:
        trials.append({
            "learning_rate": float(rng.uniform(0.015, 0.09)),
            "num_leaves": int(rng.integers(7, 32)),
            "max_depth": int(rng.integers(3, 8)),
            "min_child_samples": int(rng.integers(30, 151)),
            "n_estimators": int(rng.integers(300, 1001)),
            "subsample": float(rng.uniform(0.7, 1.0)),
            "colsample_bytree": float(rng.uniform(0.65, 1.0)),
            "reg_alpha": float(rng.uniform(0.0, 2.5)),
            "reg_lambda": float(rng.uniform(0.5, 5.0)),
            "min_split_gain": float(rng.uniform(0.0, 0.15)),
        })
    return trials


def _fit_tree(
    train: pd.DataFrame, validation: pd.DataFrame, features: V2FeatureConfig,
    params: dict[str, Any], seed: int,
) -> tuple[ColumnTransformer, LGBMClassifier, npt.NDArray[np.float64], int]:
    names = [spec.name for spec in features.features]
    preprocessor = _tree_preprocessor(features)
    x_train = np.asarray(preprocessor.fit_transform(train[names]), dtype=float)
    x_validation = np.asarray(preprocessor.transform(validation[names]), dtype=float)
    model = LGBMClassifier(
        objective="binary", random_state=seed, n_jobs=1, verbosity=-1, **params
    )
    model.fit(
        x_train, train["is_refund_abuse_simulated"].astype(bool),
        eval_set=[(x_validation, validation["is_refund_abuse_simulated"].astype(bool))],
        eval_metric="average_precision", callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    probabilities = np.asarray(model.predict_proba(x_validation), dtype=float)
    scores = probabilities[:, 1]
    return preprocessor, model, scores, int(model.best_iteration_ or params["n_estimators"])


def _rules_score(frame: pd.DataFrame) -> npt.NDArray[np.float64]:
    values = (
        0.22 * frame["returnless_requested"].to_numpy(float)
        + 0.22 * frame["missing_evidence_indicator"].to_numpy(float)
        + 0.18 * np.minimum(frame["decayed_refund_velocity_30d"].to_numpy(float), 1.0)
        + 0.18 * frame["reason_history_inconsistency"].to_numpy(float)
        + 0.20 * np.minimum(frame["connected_matured_adverse_rate"].to_numpy(float) * 5, 1.0)
    )
    return np.asarray(np.clip(values, 0, 1), dtype=float)


def _fit_calibrator(name: str, scores: npt.NDArray[np.float64], labels: npt.NDArray[np.bool_]) -> Any:
    if name == "sigmoid":
        model = LogisticRegression(random_state=20260903)
        model.fit(scores.reshape(-1, 1), labels)
        return model
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(scores, labels)
    return model


def _calibrated(name: str, model: Any, scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if name == "sigmoid":
        values = model.predict_proba(scores.reshape(-1, 1))[:, 1]
    else:
        values = model.predict(scores)
    return np.asarray(np.clip(values, 1e-6, 1 - 1e-6), dtype=float)


def train_v2(
    features: pd.DataFrame, feature_config: V2FeatureConfig, model_config: V2ModelConfig,
) -> dict[str, Any]:
    output = model_config.output_dir
    if (output / "selection.json").exists():
        raise FileExistsError("v2 model-selection artifacts are immutable")
    output.mkdir(parents=True, exist_ok=True)
    names = [spec.name for spec in feature_config.features]
    if set(names) & set(feature_config.forbidden_features):
        raise ValueError("forbidden feature in v2 model schema")
    train = features.loc[features["partition"].eq("train")].copy()
    calibration = features.loc[features["partition"].eq("calibration")].copy()
    folds = _folds(train, model_config.rolling_origin_folds)

    trial_rows: list[dict[str, Any]] = []
    for trial_index, params in enumerate(_sample_trials(model_config)):
        fold_aps: list[float] = []
        fold_review_rates: list[float] = []
        iterations: list[int] = []
        for fold_index, (fit, validation) in enumerate(folds):
            _, _, scores, best_iteration = _fit_tree(
                fit, validation, feature_config, params,
                model_config.random_seed + trial_index * 10 + fold_index,
            )
            labels = validation["is_refund_abuse_simulated"].astype(bool)
            fold_aps.append(float(average_precision_score(labels, scores)))
            fold_review_rates.append(float((scores >= 0.5).mean()))
            iterations.append(best_iteration)
        mean_ap = float(np.mean(fold_aps))
        std_ap = float(np.std(fold_aps))
        capacity_excess = max(0.0, float(np.mean(fold_review_rates)) - model_config.max_review_rate)
        objective = (
            mean_ap - model_config.instability_penalty * std_ap
            - model_config.review_capacity_penalty * capacity_excess
        )
        trial_rows.append({
            "trial": trial_index, **params, "fold_average_precision": fold_aps,
            "fold_review_rate_at_0_5": fold_review_rates,
            "best_iterations": iterations, "mean_average_precision": mean_ap,
            "std_average_precision": std_ap, "capacity_excess": capacity_excess,
            "selection_objective": objective,
        })
    best = max(trial_rows, key=lambda row: row["selection_objective"])
    best_params = {key: best[key] for key in (
        "learning_rate", "num_leaves", "max_depth", "min_child_samples", "n_estimators",
        "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "min_split_gain",
    )}

    baseline_rows: list[dict[str, Any]] = []
    for fold_index, (fit, validation) in enumerate(folds):
        logistic = _logistic_pipeline(feature_config, model_config.random_seed + fold_index)
        logistic.fit(fit[names], fit["is_refund_abuse_simulated"].astype(bool))
        baseline_rows.append({
            "fold": fold_index,
            "rules_average_precision": float(average_precision_score(
                validation["is_refund_abuse_simulated"], _rules_score(validation)
            )),
            "logistic_average_precision": float(average_precision_score(
                validation["is_refund_abuse_simulated"], logistic.predict_proba(validation[names])[:, 1]
            )),
            "lightgbm_default_average_precision": trial_rows[0]["fold_average_precision"][fold_index],
            "lightgbm_tuned_average_precision": best["fold_average_precision"][fold_index],
        })

    final_preprocessor = _tree_preprocessor(feature_config)
    x_train = np.asarray(final_preprocessor.fit_transform(train[names]), dtype=float)
    final_estimator = LGBMClassifier(
        objective="binary", random_state=model_config.random_seed, n_jobs=1, verbosity=-1,
        **{**best_params, "n_estimators": max(1, int(np.mean(best["best_iterations"])))},
    )
    final_estimator.fit(x_train, train["is_refund_abuse_simulated"].astype(bool))
    ordered_calibration = calibration.sort_values(["decision_time", "refund_request_id"])
    calibration_probabilities = np.asarray(final_estimator.predict_proba(
        final_preprocessor.transform(ordered_calibration[names])
    ), dtype=float)
    raw_calibration = calibration_probabilities[:, 1]
    midpoint = len(calibration) // 2
    calibration_comparison: dict[str, dict[str, float]] = {}
    calibration_labels = np.asarray(
        ordered_calibration["is_refund_abuse_simulated"].astype(bool).to_numpy(), dtype=bool
    )
    for method in model_config.calibration_methods:
        candidate = _fit_calibrator(
            method, raw_calibration[:midpoint], calibration_labels[:midpoint]
        )
        probabilities = _calibrated(method, candidate, raw_calibration[midpoint:])
        calibration_comparison[method] = {
            "brier_score": float(brier_score_loss(calibration_labels[midpoint:], probabilities)),
            "log_loss": float(log_loss(calibration_labels[midpoint:], probabilities)),
        }
    selected_calibrator = min(
        calibration_comparison,
        key=lambda method: (
            calibration_comparison[method]["brier_score"],
            calibration_comparison[method]["log_loss"],
        ),
    )
    calibrator = _fit_calibrator(selected_calibrator, raw_calibration, calibration_labels)
    calibrated = _calibrated(selected_calibrator, calibrator, raw_calibration)
    model = V2RiskModel(
        feature_names=tuple(names), preprocessor=final_preprocessor, estimator=final_estimator,
        calibrator_name=selected_calibrator, calibrator=calibrator,
    )
    joblib.dump(model, output / "model.joblib")

    sample = ordered_calibration.iloc[: min(500, len(ordered_calibration))]
    matrix = np.asarray(final_preprocessor.transform(sample[names]), dtype=float)
    shap_values = np.asarray(shap.TreeExplainer(final_estimator).shap_values(matrix), dtype=float)
    if shap_values.ndim == 3:
        shap_values = shap_values[-1]
    mean_absolute_shap = np.mean(np.abs(shap_values), axis=0)
    shap_total = float(mean_absolute_shap.sum())
    shap_rows: list[dict[str, float | str]] = [
        {"feature": name, "mean_absolute_shap": float(value),
         "share": float(value / shap_total) if shap_total else 0.0}
        for name, value in zip(names, mean_absolute_shap, strict=True)
    ]
    shap_rows.sort(key=lambda row: float(row["mean_absolute_shap"]), reverse=True)

    trials_path = output / "lightgbm_trials.json"
    trials_path.write_text(json.dumps(trial_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report: dict[str, Any] = {
        "schema_version": "2.0", "scope": "development data only",
        "selected_candidate": "lightgbm_tuned",
        "selection_reason": "highest preregistered penalized rolling-origin objective",
        "selected_trial": int(best["trial"]), "selected_parameters": best_params,
        "selected_best_iterations": best["best_iterations"],
        "rolling_origin_comparison": baseline_rows,
        "calibration": {
            "selection_partition": "calibration", "candidate_metrics": calibration_comparison,
            "selected": selected_calibrator,
            "raw_average_precision": float(average_precision_score(
                calibration_labels, raw_calibration
            )),
            "calibrated_brier_score": float(brier_score_loss(calibration_labels, calibrated)),
            "calibrated_log_loss": float(log_loss(calibration_labels, calibrated)),
            "reliability_bins": reliability_bins(calibration_labels, calibrated, 10),
        },
        "shap_concentration": {
            "sample_support": len(sample), "features": shap_rows,
            "top_feature_share": shap_rows[0]["share"],
            "top_three_share": sum(float(row["share"]) for row in shap_rows[:3]),
        },
        "feature_schema_sha256": object_sha256([
            spec.model_dump(mode="json") for spec in feature_config.features
        ]),
        "training_feature_artifact_sha256": file_sha256(Path("artifacts/v2/development/features.parquet")),
    }
    (output / "selection.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
