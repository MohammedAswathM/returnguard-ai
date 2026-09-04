from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from returnguard.v2.config import V2FeatureConfig


def _pipeline(config: V2FeatureConfig) -> Pipeline:
    numeric = [spec.name for spec in config.features if spec.type == "float"]
    categorical = [spec.name for spec in config.features if spec.type == "category"]
    return Pipeline([
        ("preprocessor", ColumnTransformer([
            ("numeric", StandardScaler(), numeric),
            ("category", OneHotEncoder(handle_unknown="ignore"), categorical),
        ])),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=20260903)),
    ])


def run_shortcut_audit(
    features: pd.DataFrame, config: V2FeatureConfig, output_path: Path | None = None,
) -> dict[str, Any]:
    train = features.loc[features["partition"].eq("train")].copy()
    policy = features.loc[features["partition"].eq("policy_selection")].copy()
    label = "is_refund_abuse_simulated"
    y_train = train[label].astype(bool).to_numpy()
    y_policy = policy[label].astype(bool).to_numpy()
    diagnostics: list[dict[str, Any]] = []
    for spec in config.features:
        if spec.type == "float":
            train_value = train[[spec.name]].astype(float)
            policy_value = policy[[spec.name]].astype(float)
            direct = policy_value[spec.name].to_numpy()
            direct_scaled = (direct - direct.min()) / max(float(np.ptp(direct)), 1e-12)
            ap = max(
                average_precision_score(y_policy, direct_scaled),
                average_precision_score(y_policy, 1.0 - direct_scaled),
            )
            auc = max(roc_auc_score(y_policy, direct), 1.0 - roc_auc_score(y_policy, direct))
            discrete = False
            train_encoded = train_value.to_numpy()
        else:
            rates = train.groupby(spec.name, observed=True)[label].mean()
            score = policy[spec.name].map(rates).fillna(float(y_train.mean())).to_numpy()
            ap = average_precision_score(y_policy, score)
            auc = roc_auc_score(y_policy, score)
            discrete = True
            combined = pd.concat([train[spec.name], policy[spec.name]], ignore_index=True)
            codes, _ = pd.factorize(combined)
            train_encoded = codes[:len(train)].reshape(-1, 1)
        stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=25, random_state=20260903)
        stump.fit(train_encoded, y_train)
        if spec.type == "float":
            stump_input = policy_value.to_numpy()
        else:
            stump_input = codes[len(train):].reshape(-1, 1)
        stump_score = stump.predict_proba(stump_input)[:, 1]
        mi = mutual_info_classif(
            train_encoded, y_train, discrete_features=discrete, random_state=20260903
        )[0]
        diagnostics.append({
            "feature": spec.name,
            "single_feature_average_precision": float(ap),
            "single_feature_roc_auc": float(auc),
            "decision_stump_average_precision": float(average_precision_score(y_policy, stump_score)),
            "mutual_information": float(mi),
        })

    rng = np.random.default_rng(20260903)
    feature_names = [spec.name for spec in config.features]
    permutation_aps: list[float] = []
    for _ in range(5):
        model = _pipeline(config)
        model.fit(train[feature_names], rng.permutation(y_train))
        permutation_aps.append(float(average_precision_score(
            y_policy, model.predict_proba(policy[feature_names])[:, 1]
        )))
    max_single_ap = max(item["single_feature_average_precision"] for item in diagnostics)
    max_stump_ap = max(item["decision_stump_average_precision"] for item in diagnostics)
    prevalence = float(y_policy.mean())
    passed = bool(
        max_single_ap < 0.80
        and max_stump_ap < 0.80
        and float(np.mean(permutation_aps)) <= prevalence + 0.05
    )
    report: dict[str, Any] = {
        "schema_version": "2.0",
        "scope": "development partitions only",
        "policy_selection_prevalence": prevalence,
        "feature_diagnostics": diagnostics,
        "max_single_feature_average_precision": max_single_ap,
        "max_decision_stump_average_precision": max_stump_ap,
        "permuted_label_average_precision_runs": permutation_aps,
        "permuted_label_average_precision_mean": float(np.mean(permutation_aps)),
        "thresholds": {
            "max_single_feature_average_precision": 0.80,
            "max_decision_stump_average_precision": 0.80,
            "permutation_excess_over_prevalence": 0.05,
        },
        "passed": passed,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        raise ValueError("v2 development shortcut audit failed; final must remain sealed")
    return report
