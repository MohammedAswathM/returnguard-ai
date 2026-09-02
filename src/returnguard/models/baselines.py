from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from returnguard.config import BaselineConfig
from returnguard.evaluation.metrics import classification_metrics
from returnguard.features.leakage import reject_leakage_columns
from returnguard.features.registry import FeatureSpec
from returnguard.models.guard import require_fit_partition

IDENTITY_COLUMNS = {"refund_request_id", "customer_id", "requested_at", "partition"}


def rules_score(frame: pd.DataFrame) -> npt.NDArray[np.float64]:
    score = (
        0.20 * (frame["prior_refund_count_30d"].to_numpy() >= 2)
        + 0.20 * (frame["prior_refund_count_90d"].to_numpy() >= 4)
        + 0.20 * (frame["amount_paid_ratio"].to_numpy() > 1.0)
        + 0.15 * (frame["hours_delivery_to_request"].to_numpy() < 24)
        + 0.15 * frame["returnless_requested"].to_numpy()
        + 0.10 * (frame["prior_matured_adverse_outcome_count"].to_numpy() > 0)
    )
    return np.asarray(np.clip(score, 0.0, 1.0), dtype=float)


def build_logistic_pipeline(registry: tuple[FeatureSpec, ...], config: BaselineConfig) -> Pipeline:
    numeric = [spec.name for spec in registry if spec.type != "category"]
    categorical = [spec.name for spec in registry if spec.type == "category"]
    reject_leakage_columns([*numeric, *categorical])
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    model = LogisticRegression(
        max_iter=config.max_iter, class_weight=config.class_weight, random_state=config.random_seed,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def fit_logistic(
    frame: pd.DataFrame, registry: tuple[FeatureSpec, ...], config: BaselineConfig,
    labels: npt.NDArray[np.bool_] | None = None,
) -> Pipeline:
    require_fit_partition(frame)
    pipeline = build_logistic_pipeline(registry, config)
    feature_names = [spec.name for spec in registry]
    target = frame[config.label_column].astype(bool).to_numpy() if labels is None else labels
    pipeline.fit(frame[feature_names], target)
    return pipeline


def _feature_order_hash(registry: tuple[FeatureSpec, ...]) -> str:
    payload = json.dumps([spec.name for spec in registry], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _anti_shortcut_check(
    train: pd.DataFrame, validation: pd.DataFrame, registry: tuple[FeatureSpec, ...], seed: int,
) -> dict[str, float]:
    numeric = [spec.name for spec in registry if spec.type != "category"]
    stump = DecisionTreeClassifier(max_depth=1, random_state=seed, class_weight="balanced")
    stump.fit(train[numeric].fillna(0), train["is_refund_abuse_simulated"])
    scores = stump.predict_proba(validation[numeric].fillna(0))[:, 1]
    ap = float(average_precision_score(validation["is_refund_abuse_simulated"], scores))
    if ap >= 0.95:
        raise ValueError(f"generator shortcut detected: decision-stump AP={ap:.3f}")
    return {"decision_stump_average_precision": ap}


def train_and_evaluate_baselines(
    features: pd.DataFrame, registry: tuple[FeatureSpec, ...], config: BaselineConfig,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    output = output_dir or config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    feature_names = [spec.name for spec in registry]
    reject_leakage_columns(feature_names)
    train = features.loc[features["partition"] == "train"].copy()
    policy = features.loc[features["partition"] == "policy_selection"].copy()
    model = fit_logistic(train, registry, config)
    predictions: list[pd.DataFrame] = []
    metrics: dict[str, Any] = {}
    for partition in ("train", "calibration", "policy_selection"):
        subset = features.loc[features["partition"] == partition].copy()
        labels = subset[config.label_column].astype(bool).to_numpy()
        logistic_scores = model.predict_proba(subset[feature_names])[:, 1]
        rule_scores = rules_score(subset)
        for model_name, scores in (("rules", rule_scores), ("logistic_regression", logistic_scores)):
            metrics[f"{model_name}:{partition}"] = classification_metrics(
                labels, scores, config.threshold
            )
            predictions.append(pd.DataFrame({
                "refund_request_id": subset["refund_request_id"].to_numpy(),
                "customer_id": subset["customer_id"].to_numpy(), "partition": partition,
                "model": model_name, "label_simulated": labels,
                "score": scores, "threshold": config.threshold,
                "predicted_positive": scores >= config.threshold,
            }))
    rng = np.random.default_rng(config.random_seed)
    permutation_aps: list[float] = []
    for _ in range(5):
        permuted = rng.permutation(train[config.label_column].astype(bool).to_numpy())
        permuted_model = fit_logistic(train, registry, config, labels=permuted)
        permutation_scores = permuted_model.predict_proba(policy[feature_names])[:, 1]
        permutation_aps.append(
            float(average_precision_score(policy[config.label_column], permutation_scores))
        )
    permutation_ap = float(np.mean(permutation_aps))
    prevalence = float(policy[config.label_column].mean())
    if permutation_ap > prevalence + 0.15:
        raise ValueError("label permutation retained suspicious predictive performance")
    checks: dict[str, Any] = _anti_shortcut_check(train, policy, registry, config.random_seed)
    checks.update({
        "permutation_average_precision": permutation_ap,
        "permutation_average_precision_runs": permutation_aps,
        "policy_selection_prevalence": prevalence,
        "feature_order_sha256": _feature_order_hash(registry),
    })
    prediction_frame = pd.concat(predictions, ignore_index=True)
    prediction_frame.to_csv(output / "case_predictions.csv", index=False, lineterminator="\n")
    joblib.dump(model, output / "logistic_pipeline.joblib")
    report = {
        "benchmark_disclosure": "Held-out simulation benchmark; not production performance.",
        "threshold_source": "fixed baseline configuration; not selected on final test",
        "metrics": metrics, "sanity_checks": checks,
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
