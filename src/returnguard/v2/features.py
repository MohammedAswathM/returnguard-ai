from __future__ import annotations

import heapq
import math
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd

from returnguard.v2.config import V2FeatureConfig

DEVELOPMENT_PARTITIONS = frozenset({"train", "calibration", "policy_selection"})
FINAL_ACCESS_GRANT = "V2_FROZEN_FINAL_EVALUATION"


@dataclass(frozen=True)
class MaturedOutcome:
    customer_id: str
    category: str
    address_id: str
    device_id: str
    adverse: bool
    defect: bool


def _parse_inputs(data_dir: Path, final_truth: Path | None) -> tuple[pd.DataFrame, ...]:
    customers = pd.read_csv(data_dir / "customers.csv", parse_dates=["created_at"])
    orders = pd.read_csv(
        data_dir / "orders.csv", parse_dates=["ordered_at", "delivered_at"]
    )
    requests = pd.read_csv(data_dir / "refund_requests.csv")
    requests["decision_time"] = pd.to_datetime(requests["decision_time"], utc=True)
    requests["outcome_available_at"] = pd.to_datetime(
        requests["outcome_available_at"], utc=True
    )
    if final_truth is not None:
        truth = pd.read_csv(final_truth)
        truth["outcome_available_at"] = pd.to_datetime(
            truth["outcome_available_at"], utc=True
        )
        replacement = truth.set_index("refund_request_id")
        final_mask = requests["partition"].eq("final_test")
        for column in (
            "is_refund_abuse_simulated", "simulation_scenario",
            "outcome_available_at", "verification_result_simulated",
        ):
            requests.loc[final_mask, column] = requests.loc[
                final_mask, "refund_request_id"
            ].map(replacement[column])
    return customers, orders, requests


def validate_ml_lane(data_dir: Path) -> dict[str, bool]:
    requests = pd.read_csv(data_dir / "refund_requests.csv")
    snapshots = pd.read_csv(data_dir / "payment_snapshots.csv")
    joined = requests.merge(
        snapshots, on=["refund_request_id", "payment_id", "order_id"],
        validate="one_to_one", suffixes=("_request", "_payment"),
    )
    checks = {
        "payment_exists": bool(joined["payment_exists"].all()),
        "merchant_matches": bool(joined["merchant_matches"].all()),
        "payment_captured": bool(joined["payment_status"].eq("captured").all()),
        "currency_matches": bool(joined["currency_matches"].all()),
        "positive_amount": bool(joined["claimed_refund_amount_paise"].gt(0).all()),
        "within_point_in_time_balance": bool(
            joined["claimed_refund_amount_paise"].le(joined["refundable_balance_paise"]).all()
        ),
        "authorized_amount_matches": bool(
            joined["claimed_refund_amount_paise"].eq(
                joined["gateway_authorized_execution_amount_paise"]
            ).all()
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"v2 ML lane contains deterministic payment-invalid cases: {failed}")
    return checks


def build_features(
    data_dir: Path,
    config: V2FeatureConfig,
    partitions: frozenset[str] = DEVELOPMENT_PARTITIONS,
    *,
    final_access_grant: str | None = None,
    final_truth: Path | None = None,
) -> pd.DataFrame:
    if "final_test" in partitions and (
        final_access_grant != FINAL_ACCESS_GRANT or final_truth is None
    ):
        raise ValueError("v2 final features require the explicit frozen-evaluation grant and truth")
    validate_ml_lane(data_dir)
    customers, orders, all_requests = _parse_inputs(data_dir, final_truth)
    if all_requests.loc[all_requests["partition"].isin(partitions), "is_refund_abuse_simulated"].isna().any():
        raise ValueError("labels are unavailable for a requested feature partition")

    requested = all_requests.loc[all_requests["partition"].isin(partitions)].copy()
    history_partitions = set(partitions)
    if "final_test" in partitions:
        history_partitions.update(DEVELOPMENT_PARTITIONS)
    replay = all_requests.loc[all_requests["partition"].isin(history_partitions)].sort_values(
        ["decision_time", "refund_request_id"]
    )
    selected_ids = set(requested["refund_request_id"].astype(str))
    order_by_id = orders.set_index("order_id").to_dict(orient="index")
    created_by_customer = customers.set_index("customer_id")["created_at"].to_dict()
    order_times: dict[str, list[pd.Timestamp]] = defaultdict(list)
    for raw in orders.sort_values("ordered_at").to_dict(orient="records"):
        order = cast(dict[str, Any], raw)
        order_times[str(order["customer_id"])].append(order["ordered_at"])

    request_history: dict[str, list[pd.Timestamp]] = defaultdict(list)
    reason_history: dict[str, Counter[str]] = defaultdict(Counter)
    entity_claims: dict[str, int] = defaultdict(int)
    matured_customer: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    matured_category: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    matured_entity: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    category_claims: Counter[str] = Counter()
    all_claims = 0
    maturity_heap: list[tuple[pd.Timestamp, str, MaturedOutcome]] = []
    rows: list[dict[str, Any]] = []

    for raw in replay.to_dict(orient="records"):
        request = cast(dict[str, Any], raw)
        now = cast(pd.Timestamp, request["decision_time"])
        while maturity_heap and maturity_heap[0][0] < now:
            _, _, outcome = heapq.heappop(maturity_heap)
            matured_customer[outcome.customer_id][0] += int(outcome.adverse)
            matured_customer[outcome.customer_id][1] += 1
            matured_category[outcome.category][0] += int(outcome.defect)
            matured_category[outcome.category][1] += 1
            for entity in {outcome.address_id, outcome.device_id}:
                matured_entity[entity][0] += int(outcome.adverse)
                matured_entity[entity][1] += 1

        customer_id = str(request["customer_id"])
        order = cast(dict[str, Any], order_by_id[str(request["order_id"])])
        category = str(order["category"])
        address_id = str(order["shipping_address_id"])
        device_id = str(order["device_id"])
        prior_times = request_history[customer_id]
        prior_90 = len(prior_times) - bisect_left(prior_times, now - timedelta(days=90))
        velocity = sum(
            math.exp(-((now - event).total_seconds() / 86400) / 15.0)
            for event in prior_times
            if event >= now - timedelta(days=30)
        )
        prior_orders = bisect_left(order_times[customer_id], now)
        customer_rate = (prior_90 + 1.0) / (prior_orders + 20.0)
        peer_rate = (all_claims + 5.0) / (sum(map(len, order_times.values())) + 100.0)
        customer_adverse, customer_matured = matured_customer[customer_id]
        category_defects, category_matured = matured_category[category]
        connected_entities = {address_id, device_id}
        connected_claims = sum(entity_claims[value] for value in connected_entities)
        connected_adverse = sum(matured_entity[value][0] for value in connected_entities)
        connected_matured = sum(matured_entity[value][1] for value in connected_entities)
        reasons = reason_history[customer_id]
        reason_total = sum(reasons.values())
        reason = str(request["reason_code"])
        current_reason_share = reasons[reason] / reason_total if reason_total else 1.0
        label = bool(request["is_refund_abuse_simulated"])

        feature_values: dict[str, Any] = {
            "account_tenure_days": max(
                0.0, (now - created_by_customer[customer_id]).total_seconds() / 86400
            ),
            "requested_amount_paise": float(request["claimed_refund_amount_paise"]),
            "requested_quantity_ratio": float(request["requested_quantity"]) / max(
                1.0, float(order["item_count"])
            ),
            "hours_delivery_to_request": max(
                0.0, (now - order["delivered_at"]).total_seconds() / 3600
            ),
            "reason_code": reason,
            "category": category,
            "prior_order_count": float(prior_orders),
            "smoothed_refund_rate_90d": customer_rate,
            "prior_matured_adverse_rate": (customer_adverse + 1.0) / (customer_matured + 20.0),
            "decayed_refund_velocity_30d": velocity,
            "customer_peer_refund_deviation": customer_rate - peer_rate,
            "category_matured_defect_rate": (category_defects + 1.0) / (category_matured + 20.0),
            "reason_history_inconsistency": 1.0 - current_reason_share,
            "evidence_provided": float(bool(request["evidence_provided"])),
            "returnless_requested": float(bool(request["returnless_requested"])),
            "prior_connected_claim_count": float(connected_claims),
            "connected_matured_adverse_rate": (connected_adverse + 1.0) / (connected_matured + 20.0),
            "missing_evidence_indicator": float(not bool(request["evidence_provided"])),
            "cold_start_indicator": float(len(prior_times) == 0),
        }
        expected = [spec.name for spec in config.features]
        if set(feature_values) != set(expected):
            raise ValueError("v2 feature implementation and frozen allowlist differ")
        forbidden = set(config.forbidden_features) & set(feature_values)
        if forbidden:
            raise ValueError(f"forbidden v2 features entered the matrix: {sorted(forbidden)}")
        if str(request["refund_request_id"]) in selected_ids:
            rows.append({
                "refund_request_id": request["refund_request_id"],
                "customer_id": customer_id,
                "decision_time": now,
                "partition": request["partition"],
                "is_refund_abuse_simulated": label,
                "verification_result_simulated": request["verification_result_simulated"],
                "simulation_scenario": request["simulation_scenario"],
                "claimed_refund_amount_paise": request["claimed_refund_amount_paise"],
                **{name: feature_values[name] for name in expected},
            })

        # Compute-before-update: the current request enters every history only here.
        prior_times.append(now)
        reasons[reason] += 1
        category_claims[category] += 1
        all_claims += 1
        for entity in connected_entities:
            entity_claims[entity] += 1
        outcome = MaturedOutcome(
            customer_id=customer_id, category=category,
            address_id=address_id, device_id=device_id,
            adverse=label,
            defect=(reason == "damaged" and not label),
        )
        heapq.heappush(
            maturity_heap,
            (cast(pd.Timestamp, request["outcome_available_at"]), str(request["refund_request_id"]), outcome),
        )
    return pd.DataFrame(rows)


def assert_ordered_schema(frame: pd.DataFrame, config: V2FeatureConfig) -> None:
    feature_names = [spec.name for spec in config.features]
    positions = [cast(int, frame.columns.get_loc(name)) for name in feature_names]
    if positions != sorted(positions) or len(set(positions)) != len(feature_names):
        raise ValueError("v2 feature order does not match the frozen schema")
