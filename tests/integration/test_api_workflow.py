import asyncio
import hashlib
import hmac
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
from fastapi import FastAPI

from returnguard.backend.api import create_app
from returnguard.backend.gateway import MockRazorpayGateway
from returnguard.backend.runtime import RuntimeScore


class LocalClient:
    def __init__(self, app: FastAPI) -> None:
        self.app = app

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        async def send() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(send())

    def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", path, **kwargs)

    def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", path, **kwargs)


class DeterministicRuntime:
    def __init__(self, action: str = "AUTO_APPROVE", healthy: bool = True) -> None:
        self.action = action
        self.healthy = healthy
        self.validation_error = None if healthy else "test hash mismatch"
        self.model = SimpleNamespace(model_version="test-model")
        self.registry = ()

    def score(self, features: dict[str, Any], amount_paise: int) -> RuntimeScore:
        probability = float(features.get("test_probability", 0.2))
        return RuntimeScore(
            raw_score=probability, probability=probability, action=self.action,
            reasons=[], snapshot_hash="a" * 64,
        )

    def posterior_action(self, probability: float, result: str, amount_paise: int) -> tuple[float, str]:
        if result == "consistent":
            return probability / 5, "AUTO_APPROVE"
        if result == "inconsistent":
            return min(0.99, probability * 2), "MANUAL_REVIEW"
        return probability, "RETURN_FIRST"


class WebhookConfirmedGateway(MockRazorpayGateway):
    provider_name = "RAZORPAY_TEST_MODE"
    requires_webhook_confirmation = True


def case_body(probability: float = 0.2) -> dict[str, Any]:
    return {
        "merchant_id": "merchant-1", "customer_id": "customer-1",
        "payment_id": "payment-1", "order_id": "order-1", "currency": "INR",
        "requested_amount_paise": 5000,
        "payment_snapshot": {
            "merchant_id": "merchant-1", "payment_id": "payment-1",
            "razorpay_payment_id": None, "order_id": "order-1", "currency": "INR",
            "captured_amount_paise": 5000, "amount_refunded_paise": 0,
            "refundable_balance_paise": 5000, "payment_status": "captured",
            "captured_at": None, "snapshot_as_of": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
            "source": "TEST_FIXTURE",
        },
        "features": {"test_probability": probability},
    }


def verification_body(token: str | None = "demo-consistent") -> dict[str, Any]:
    return {
        "requested_amount_paise": 5000, "refundable_balance_paise": 5000,
        "requested_quantity": 1, "eligible_quantity": 1, "delivered": True,
        "reason_code": "damaged", "evidence_token": token,
    }


def test_low_risk_refund_is_idempotent_and_audited(tmp_path: Path) -> None:
    gateway = MockRazorpayGateway()
    app = create_app(
        database_path=tmp_path / "api.db", gateway=gateway,
        runtime_override=DeterministicRuntime(),  # type: ignore[arg-type]
    )
    client = LocalClient(app)
    assert client.post("/api/v1/refund-requests/rr-low", json=case_body()).status_code == 200
    scored = client.post("/api/v1/refund-requests/rr-low/score")
    assert scored.status_code == 200
    first = client.post("/api/v1/refund-requests/rr-low/execute")
    second = client.post("/api/v1/refund-requests/rr-low/execute")
    assert first.status_code == second.status_code == 200
    assert first.json()["gateway_refund_id"] == second.json()["gateway_refund_id"]
    assert len(gateway.effects) == 1
    audit = client.get("/api/v1/refund-requests/rr-low/audit").json()
    assert [event["after_state"] for event in audit] == [
        "RECEIVED", "AUTO_APPROVE", "REFUND_PROCESSING", "REFUNDED",
    ]
    repository = app.state.repository
    with sqlite3.connect(tmp_path / "api.db") as connection:
        try:
            connection.execute("DELETE FROM audit_events")
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("append-only trigger did not reject audit deletion")
    repository.close()


def test_legitimate_verification_rescue_keeps_decisions_separate(tmp_path: Path) -> None:
    app = create_app(
        database_path=tmp_path / "rescue.db",
        runtime_override=DeterministicRuntime("VERIFY"),  # type: ignore[arg-type]
    )
    client = LocalClient(app)
    client.post("/api/v1/refund-requests/rr-rescue", json=case_body(0.6))
    assert client.post("/api/v1/refund-requests/rr-rescue/score").json()["action"] == "VERIFY"
    verification = client.post("/api/v1/refund-requests/rr-rescue/verifications").json()
    duplicate = client.post("/api/v1/refund-requests/rr-rescue/verifications")
    assert duplicate.status_code == 409
    completed = client.post(
        f"/api/v1/verifications/{verification['verification_id']}/complete",
        json=verification_body(),
    )
    assert completed.status_code == 200
    assert completed.json()["probability"] < 0.6
    assert completed.json()["action"] == "AUTO_APPROVE"
    decisions = app.state.repository.decisions("rr-rescue")
    assert [decision["stage"] for decision in decisions] == [
        "pre_verification", "post_verification",
    ]


def test_inconsistent_case_requires_named_operator_and_moves_no_money(tmp_path: Path) -> None:
    gateway = MockRazorpayGateway()
    app = create_app(
        database_path=tmp_path / "inconsistent.db", gateway=gateway,
        runtime_override=DeterministicRuntime("VERIFY"),  # type: ignore[arg-type]
    )
    client = LocalClient(app)
    client.post("/api/v1/refund-requests/rr-review", json=case_body(0.7))
    client.post("/api/v1/refund-requests/rr-review/score")
    verification = client.post("/api/v1/refund-requests/rr-review/verifications").json()
    body = verification_body("not-allowlisted")
    completed = client.post(
        f"/api/v1/verifications/{verification['verification_id']}/complete", json=body
    )
    assert completed.json()["action"] == "MANUAL_REVIEW"
    assert client.post("/api/v1/refund-requests/rr-review/execute").status_code == 409
    rejected = client.post("/api/v1/refund-requests/rr-review/decisions", json={
        "operator_id": "reviewer@example.test", "approved": False,
        "reason": "Structured order facts require reconsideration.",
    })
    assert rejected.json()["appeal_state"] == "RECONSIDERATION_AVAILABLE"
    assert gateway.effects == {}


def test_invalid_bundle_degrades_health_and_fails_closed(tmp_path: Path) -> None:
    app = create_app(
        database_path=tmp_path / "degraded.db",
        runtime_override=DeterministicRuntime(healthy=False),  # type: ignore[arg-type]
    )
    client = LocalClient(app)
    health = client.get("/health").json()
    assert health["status"] == "degraded"
    assert health["refund_adapter"] == "MOCK_RAZORPAY_TEST_ADAPTER"
    assert health["refund_completion"] == "IMMEDIATE_ADAPTER_CONFIRMATION"
    client.post("/api/v1/refund-requests/rr-safe", json=case_body())
    assert client.post("/api/v1/refund-requests/rr-safe/score").status_code == 503
    audit = client.get("/api/v1/refund-requests/rr-safe/audit").json()
    assert audit[-1]["after_state"] == "FAILED_SAFE"


def test_payment_integrity_failure_prevents_model_decision(tmp_path: Path) -> None:
    app = create_app(
        database_path=tmp_path / "integrity.db",
        runtime_override=DeterministicRuntime(),  # type: ignore[arg-type]
    )
    client = LocalClient(app)
    body = case_body()
    body["requested_amount_paise"] = 5_001
    assert client.post("/api/v1/refund-requests/rr-over", json=body).status_code == 200
    scored = client.post("/api/v1/refund-requests/rr-over/score")
    assert scored.status_code == 200
    assert scored.json()["reason_code"] == "REFUND_BALANCE_EXHAUSTED"
    assert scored.json()["decision_type"] == "deterministic_payment_integrity"
    assert app.state.repository.decisions("rr-over") == []


def test_unavailable_balance_fails_closed_before_model(tmp_path: Path) -> None:
    app = create_app(
        database_path=tmp_path / "unavailable.db",
        runtime_override=DeterministicRuntime(),  # type: ignore[arg-type]
    )
    client = LocalClient(app)
    body = case_body()
    body["payment_snapshot"] = None
    client.post("/api/v1/refund-requests/rr-unavailable", json=body)
    scored = client.post("/api/v1/refund-requests/rr-unavailable/score")
    assert scored.json()["reason_code"] == "REFUND_BALANCE_UNAVAILABLE"
    assert scored.json()["action"] == "MANUAL_REVIEW"
    assert app.state.repository.decisions("rr-unavailable") == []


def test_webhook_replay_is_idempotent(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", "test-webhook-secret")
    app = create_app(
        database_path=tmp_path / "webhook.db",
        runtime_override=DeterministicRuntime(),  # type: ignore[arg-type]
    )
    client = LocalClient(app)
    raw = json.dumps({"id": "evt-test-1", "event": "refund.processed"}).encode()
    signature = hmac.new(b"test-webhook-secret", raw, hashlib.sha256).hexdigest()
    headers = {"X-Razorpay-Signature": signature, "Content-Type": "application/json"}
    first = client.post("/api/v1/webhooks/razorpay", content=raw, headers=headers)
    second = client.post("/api/v1/webhooks/razorpay", content=raw, headers=headers)
    assert first.json()["duplicate"] is False
    assert second.json()["duplicate"] is True


def test_signed_webhook_finalizes_reservation_once(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", "test-webhook-secret")
    gateway = WebhookConfirmedGateway()
    app = create_app(
        database_path=tmp_path / "webhook-finalize.db", gateway=gateway,
        runtime_override=DeterministicRuntime(),  # type: ignore[arg-type]
    )
    client = LocalClient(app)
    client.post("/api/v1/refund-requests/rr-webhook", json=case_body())
    client.post("/api/v1/refund-requests/rr-webhook/score")
    dispatched = client.post("/api/v1/refund-requests/rr-webhook/execute")
    assert dispatched.json()["status"] == "REFUND_PROCESSING"
    assert app.state.repository.refund_ledger("payment-1")[0]["status"] == "processing"

    raw = json.dumps({
        "event": "refund.processed",
        "payload": {"refund": {"entity": {"id": dispatched.json()["gateway_refund_id"]}}},
    }).encode()
    signature = hmac.new(b"test-webhook-secret", raw, hashlib.sha256).hexdigest()
    headers = {"X-Razorpay-Signature": signature, "Content-Type": "application/json"}
    first = client.post("/api/v1/webhooks/razorpay", content=raw, headers=headers)
    second = client.post("/api/v1/webhooks/razorpay", content=raw, headers=headers)
    assert first.json() == {
        "accepted": True, "duplicate": False, "event_type": "refund.processed",
        "matched": True, "refund_status": "REFUNDED",
    }
    assert second.json()["duplicate"] is True
    assert app.state.repository.execution("rr-webhook")["status"] == "REFUNDED"
    assert app.state.repository.refund_ledger("payment-1")[0]["status"] == "succeeded"
    assert len(gateway.effects) == 1
