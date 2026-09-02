#!/usr/bin/env python3
import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from returnguard.features.engine import build_point_in_time_features
from returnguard.features.registry import load_feature_registry

parser = argparse.ArgumentParser()
parser.add_argument("--api-url", default="http://127.0.0.1:8000")
parser.add_argument("--suffix", required=True)
parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data_uci"))
parser.add_argument("--policy-cases", type=Path, default=Path(
    "artifacts/frozen_policy/policy_selection_cases.parquet"
))
args = parser.parse_args()

registry = load_feature_registry(Path("configs/features.yaml"))
features = build_point_in_time_features(args.data_dir, registry).set_index("refund_request_id")
policy = pd.read_parquet(args.policy_cases)
candidates = policy.loc[
    (policy["stage_a_action"] == "VERIFY")
    & (policy["verification_result"] == "consistent")
    & (policy["adaptive_final_action"] == "AUTO_APPROVE")
]
if candidates.empty:
    raise RuntimeError("frozen policy has no legitimate-rescue demo candidate")
candidate = candidates.iloc[0]
source_id = str(candidate["refund_request_id"])
feature_row = features.loc[source_id]
request_id = f"demo-rescue-{args.suffix}"
feature_payload: dict[str, Any] = {
    spec.name: feature_row[spec.name].item()
    if hasattr(feature_row[spec.name], "item") else feature_row[spec.name]
    for spec in registry
}
with httpx.Client(base_url=args.api_url, timeout=30.0) as client:
    created = client.post(f"/api/v1/refund-requests/{request_id}", json={
        "merchant_id": "demo-merchant",
        "customer_id": str(feature_row["customer_id"]),
        "payment_id": f"demo-payment-{request_id}", "order_id": f"demo-order-{request_id}",
        "currency": "INR",
        "requested_amount_paise": int(feature_row["requested_amount_paise"]),
        "payment_snapshot": {
            "merchant_id": "demo-merchant", "payment_id": f"demo-payment-{request_id}",
            "razorpay_payment_id": None, "order_id": f"demo-order-{request_id}",
            "currency": "INR", "captured_amount_paise": int(feature_row["requested_amount_paise"]),
            "amount_refunded_paise": 0,
            "refundable_balance_paise": int(feature_row["requested_amount_paise"]),
            "payment_status": "captured", "captured_at": None,
            "snapshot_as_of": datetime.now(UTC).isoformat(), "source": "LABELLED_DEMO_FIXTURE",
        },
        "features": feature_payload,
    })
    created.raise_for_status()
    score = client.post(f"/api/v1/refund-requests/{request_id}/score")
    score.raise_for_status()
    if score.json()["action"] != "VERIFY":
        raise RuntimeError(f"expected VERIFY, received {score.json()['action']}")
    verification = client.post(f"/api/v1/refund-requests/{request_id}/verifications")
    verification.raise_for_status()
    completed = client.post(
        f"/api/v1/verifications/{verification.json()['verification_id']}/complete",
        json={
            "requested_amount_paise": int(feature_row["requested_amount_paise"]),
            "refundable_balance_paise": int(feature_row["requested_amount_paise"]),
            "requested_quantity": 1, "eligible_quantity": 1, "delivered": True,
            "reason_code": "damaged", "evidence_token": "demo-consistent",
        },
    )
    completed.raise_for_status()
    if completed.json()["action"] != "AUTO_APPROVE":
        raise RuntimeError("consistent verification did not rescue the case")
    execution = client.post(f"/api/v1/refund-requests/{request_id}/execute")
    execution.raise_for_status()
    response_metadata = json.loads(execution.json()["response_json"])
    print({
        "request_id": request_id, "prior": score.json()["probability"],
        "posterior": completed.json()["probability"], "action": completed.json()["action"],
        "execution_status": execution.json()["status"],
        "adapter": response_metadata["adapter"],
    })
