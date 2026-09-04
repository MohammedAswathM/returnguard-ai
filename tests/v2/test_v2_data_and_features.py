from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from returnguard.v2.config import load_v2_feature_config
from returnguard.v2.features import build_features, validate_ml_lane

DATA = Path("artifacts/v2/data")
LOCK = Path("artifacts/v2/locked_final")


def test_preregistration_hash_and_one_time_final_open() -> None:
    prereg = Path("evidence/v2/preregistration.json")
    expected = Path("evidence/v2/preregistration.sha256").read_text(encoding="utf-8").split()[0]
    assert hashlib.sha256(prereg.read_bytes()).hexdigest() == expected
    manifest = json.loads((LOCK / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "OPENED_ONCE"
    assert manifest["opened_at"] is not None
    sealed = json.loads((LOCK / "manifest.sealed.json").read_text(encoding="utf-8"))
    assert sealed["state"] == "SEALED"
    assert sealed["opened_at"] is None
    assert sealed["truth_sha256"] == manifest["truth_sha256"]
    public = pd.read_csv(DATA / "refund_requests.csv")
    final = public.loc[public["partition"].eq("final_test")]
    assert len(final) == 2400
    assert final["is_refund_abuse_simulated"].isna().all()


def test_two_lanes_are_disjoint_and_ml_lane_is_valid() -> None:
    checks = validate_ml_lane(DATA)
    assert all(checks.values())
    requests = pd.read_csv(DATA / "refund_requests.csv")
    integrity = pd.read_csv(DATA / "integrity_cases.csv")
    assert set(requests["lane"]) == {"ML_ELIGIBLE_AMBIGUOUS_CLAIM"}
    assert set(integrity["lane"]) == {"DETERMINISTIC_PAYMENT_INTEGRITY"}
    assert not set(requests["refund_request_id"]) & set(integrity["integrity_case_id"])


def test_chronological_split_boundaries_do_not_overlap() -> None:
    requests = pd.read_csv(DATA / "refund_requests.csv", parse_dates=["decision_time"])
    partitions = ["train", "calibration", "policy_selection", "final_test"]
    supports = [7200, 1200, 1200, 2400]
    for name, support in zip(partitions, supports, strict=True):
        assert int(requests["partition"].eq(name).sum()) == support
    for earlier, later in zip(partitions[:-1], partitions[1:], strict=True):
        assert requests.loc[requests["partition"].eq(earlier), "decision_time"].max() < requests.loc[
            requests["partition"].eq(later), "decision_time"
        ].min()


def test_feature_matrix_is_allowlisted_and_development_only() -> None:
    config = load_v2_feature_config(Path("configs/v2/features.yaml"))
    frame = build_features(DATA, config)
    assert set(frame["partition"]) == {"train", "calibration", "policy_selection"}
    assert len(frame) == 9600
    feature_names = {spec.name for spec in config.features}
    assert not feature_names & set(config.forbidden_features)
    assert feature_names <= set(frame.columns)
    assert not frame[list(feature_names)].isna().any().any()
    saved = pd.read_parquet("artifacts/v2/development/features.parquet")
    pd.testing.assert_frame_equal(
        frame.reset_index(drop=True), saved.reset_index(drop=True), check_exact=True
    )


def test_current_request_is_not_self_counted() -> None:
    config = load_v2_feature_config(Path("configs/v2/features.yaml"))
    frame = build_features(DATA, config).sort_values(["decision_time", "refund_request_id"])
    first = frame.groupby("customer_id", sort=False).head(1)
    assert first["cold_start_indicator"].eq(1.0).all()
    assert first["decayed_refund_velocity_30d"].eq(0.0).all()


def test_final_access_requires_explicit_grant_and_truth() -> None:
    config = load_v2_feature_config(Path("configs/v2/features.yaml"))
    with pytest.raises(ValueError, match="explicit frozen-evaluation grant"):
        build_features(DATA, config, frozenset({"final_test"}))
