import shutil
from pathlib import Path

import pandas as pd

from returnguard.features.engine import build_point_in_time_features
from returnguard.features.registry import load_feature_registry

REGISTRY = Path("configs/features.yaml")


def test_compute_before_update_and_strict_past(generated_data: Path) -> None:
    features = build_point_in_time_features(generated_data, load_feature_registry(REGISTRY))
    customer = features.groupby("customer_id").size().sort_values(ascending=False).index[0]
    rows = features.loc[features["customer_id"] == customer].sort_values("requested_at")
    assert rows.iloc[0]["prior_refund_count_90d"] == 0
    assert (rows["prior_refund_count_90d"] <= range(len(rows))).all()


def test_immature_outcomes_never_count(generated_data: Path) -> None:
    features = build_point_in_time_features(generated_data, load_feature_registry(REGISTRY))
    requests = pd.read_csv(
        generated_data / "refund_requests.csv", parse_dates=["requested_at", "outcome_available_at"]
    )
    merged = features.merge(
        requests[["refund_request_id", "customer_id", "requested_at", "outcome_available_at",
                  "is_refund_abuse_simulated"]],
        on=["refund_request_id", "customer_id"], suffixes=("", "_raw"),
    )
    for row in merged.itertuples():
        prior = requests.loc[
            (requests["customer_id"] == row.customer_id)
            & (requests["requested_at"] < row.requested_at)
            & (requests["outcome_available_at"] < row.requested_at)
        ]
        assert row.prior_matured_adverse_outcome_count == prior["is_refund_abuse_simulated"].sum()


def test_feature_schema_excludes_simulation_shortcuts() -> None:
    names = {item.name for item in load_feature_registry(REGISTRY)}
    assert not names & {
        "simulation_scenario", "generator_seed", "is_refund_abuse_simulated",
        "outcome_available_at", "verification_result", "recommended_action",
    }


def test_future_request_does_not_change_past_features(
    generated_data: Path, tmp_path: Path
) -> None:
    registry = load_feature_registry(REGISTRY)
    baseline = build_point_in_time_features(generated_data, registry)
    modified_dir = tmp_path / "future_event"
    shutil.copytree(generated_data, modified_dir)
    requests = pd.read_csv(modified_dir / "refund_requests.csv")
    future = requests.loc[requests["partition"] != "final_test"].iloc[-1].copy()
    future["refund_request_id"] = "rr_future_only"
    future["requested_at"] = "2099-01-01T00:00:00Z"
    future["outcome_available_at"] = "2099-01-15T00:00:00Z"
    requests = pd.concat([requests, future.to_frame().T], ignore_index=True)
    requests.to_csv(modified_dir / "refund_requests.csv", index=False)
    replay = build_point_in_time_features(modified_dir, registry)
    original_replay = replay.loc[replay["refund_request_id"] != "rr_future_only"].reset_index(drop=True)
    pd.testing.assert_frame_equal(baseline.reset_index(drop=True), original_replay)
