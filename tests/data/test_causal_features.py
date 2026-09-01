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
