from __future__ import annotations

import heapq
from bisect import bisect_left
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd

from returnguard.features.registry import FeatureSpec


def build_point_in_time_features(
    data_dir: Path, registry: tuple[FeatureSpec, ...],
    partitions: frozenset[str] = frozenset({"train", "calibration", "policy_selection"}),
    final_access_grant: str | None = None,
) -> pd.DataFrame:
    customers = pd.read_csv(data_dir / "customers.csv", parse_dates=["created_at"])
    orders = pd.read_csv(data_dir / "orders.csv", parse_dates=["ordered_at", "delivered_at"])
    requests = pd.read_csv(data_dir / "refund_requests.csv")
    for column in ("requested_at", "outcome_available_at"):
        requests[column] = pd.to_datetime(requests[column], utc=True, format="mixed")
    allowed = {"train", "calibration", "policy_selection"}
    if final_access_grant == "FROZEN_FINAL_EVALUATION":
        allowed.add("final_test")
    forbidden = partitions - allowed
    if forbidden:
        raise ValueError(f"sealed partitions cannot enter development feature replay: {sorted(forbidden)}")
    requests = requests.loc[requests["partition"].isin(partitions)].sort_values(
        ["requested_at", "refund_request_id"]
    )
    customer_created = customers.set_index("customer_id")["created_at"].to_dict()
    order_by_id = orders.set_index("order_id").to_dict(orient="index")
    order_times_by_customer: dict[str, list[pd.Timestamp]] = defaultdict(list)
    for row in orders.sort_values("ordered_at").to_dict(orient="records"):
        typed_row = cast(dict[str, Any], row)
        order_times_by_customer[str(typed_row["customer_id"])].append(typed_row["ordered_at"])
    cancellation_times: dict[str, list[pd.Timestamp]] = defaultdict(list)
    transaction_path = data_dir / "transaction_events.csv"
    if transaction_path.exists():
        transaction_events = pd.read_csv(transaction_path)
        if not transaction_events.empty:
            transaction_events["event_time"] = pd.to_datetime(
                transaction_events["event_time"], utc=True
            )
            for raw_event in transaction_events.sort_values("event_time").to_dict(orient="records"):
                event = cast(dict[str, Any], raw_event)
                if event["event_type"] == "cancellation":
                    cancellation_times[str(event["customer_id"])].append(event["event_time"])
    request_times: dict[str, list[pd.Timestamp]] = defaultdict(list)
    prior_returnless: dict[str, int] = defaultdict(int)
    outcome_heaps: dict[str, list[tuple[pd.Timestamp, bool]]] = defaultdict(list)
    matured_adverse: dict[str, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []
    for raw_request in requests.to_dict(orient="records"):
        request = cast(dict[str, Any], raw_request)
        now = request["requested_at"]
        customer_id = str(request["customer_id"])
        order = order_by_id[str(request["order_id"])]
        prior_order_count = bisect_left(order_times_by_customer[customer_id], now)
        history_times = request_times[customer_id]
        prior_30_count = len(history_times) - bisect_left(history_times, now - timedelta(days=30))
        prior_90_count = len(history_times) - bisect_left(history_times, now - timedelta(days=90))
        maturity_heap = outcome_heaps[customer_id]
        while maturity_heap and maturity_heap[0][0] < now:
            _, adverse = heapq.heappop(maturity_heap)
            matured_adverse[customer_id] += int(adverse)
        paid = float(order["gross_amount_paise"])
        features: dict[str, Any] = {
            "refund_request_id": request["refund_request_id"], "customer_id": customer_id,
            "requested_at": now, "partition": request["partition"],
            "is_refund_abuse_simulated": bool(request["is_refund_abuse_simulated"]),
            "account_tenure_days": max(0.0, (now - customer_created[customer_id]).total_seconds() / 86400),
            "prior_order_count": float(prior_order_count),
            "prior_refund_count_30d": float(prior_30_count),
            "prior_refund_count_90d": float(prior_90_count),
            "prior_matured_adverse_outcome_count": float(matured_adverse[customer_id]),
            "prior_returnless_count": float(prior_returnless[customer_id]),
            "prior_cancellation_count": float(
                bisect_left(cancellation_times[customer_id], now)
            ),
            "hours_delivery_to_request": max(0.0, (now - order["delivered_at"]).total_seconds() / 3600),
            "requested_amount_paise": float(request["requested_amount_paise"]),
            "amount_paid_ratio": float(request["requested_amount_paise"]) / paid,
            "requested_quantity_ratio": float(request["requested_quantity"]) / float(order["item_count"]),
            "returnless_requested": float(bool(request["returnless_requested"])),
            "evidence_provided": float(bool(request["evidence_provided"])),
            "first_order_indicator": float(prior_order_count <= 1),
            "reason_code": request["reason_code"], "category": order["category"],
        }
        expected = [spec.name for spec in registry]
        missing = set(expected) - features.keys()
        if missing:
            raise ValueError(f"feature engine missing registry entries: {sorted(missing)}")
        rows.append({key: features[key] for key in [
            "refund_request_id", "customer_id", "requested_at", "partition",
            "is_refund_abuse_simulated", *expected,
        ]})
        # The current request becomes history only after its feature vector is complete.
        history_times.append(now)
        prior_returnless[customer_id] += int(bool(request["returnless_requested"]))
        heapq.heappush(
            maturity_heap,
            (request["outcome_available_at"], bool(request["is_refund_abuse_simulated"])),
        )
    return pd.DataFrame(rows)


def build_serving_feature(
    data_dir: Path, registry: tuple[FeatureSpec, ...], refund_request_id: str,
) -> pd.Series:
    replay = build_point_in_time_features(data_dir, registry)
    matches = replay.loc[replay["refund_request_id"] == refund_request_id]
    if len(matches) != 1:
        raise KeyError(f"development request not found exactly once: {refund_request_id}")
    return matches.iloc[0]
