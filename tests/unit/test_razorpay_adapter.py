import hashlib
import hmac
from datetime import UTC, datetime

import httpx
import pytest

from returnguard.backend.razorpay import RazorpayTestGateway, verify_webhook_signature


def test_razorpay_test_gateway_checks_balance_and_idempotency_header() -> None:
    seen_header: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={
                "id": "pay_test", "status": "captured", "amount": 10_000,
                "amount_refunded": 2_000, "currency": "INR", "created_at": 1_700_000_000,
            })
        seen_header.append(request.headers["X-Refund-Idempotency"])
        return httpx.Response(200, json={
            "id": "rfnd_test", "status": "processed", "amount": 5_000,
        })

    client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.razorpay.com"
    )
    gateway = RazorpayTestGateway("rzp_test_key", "secret", "pay_test", client)
    payment = gateway.payment_snapshot(
        "configured-test-merchant", "configured-test-payment", "order-1", "INR",
        datetime(2026, 9, 2, tzinfo=UTC),
    )
    assert payment.refundable_balance_paise == 8_000
    assert payment.payment_id == "configured-test-payment"
    assert payment.razorpay_payment_id is None
    assert payment.source == "RAZORPAY_TEST_API"
    result = gateway.refund("rr-1", 5_000, "deterministic-key")
    assert result.refund_id == "rfnd_test"
    assert result.metadata["adapter"] == "RAZORPAY_TEST_MODE"
    assert "payment_id" not in result.metadata
    assert seen_header == ["deterministic-key"]


def test_razorpay_gateway_rejects_live_key_and_excess_refund() -> None:
    with pytest.raises(ValueError, match="test keys"):
        RazorpayTestGateway("rzp_live_key", "secret", "pay_test")

    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
            "status": "captured", "amount": 1_000, "amount_refunded": 900,
        })),
        base_url="https://api.razorpay.com",
    )
    gateway = RazorpayTestGateway("rzp_test_key", "secret", "pay_test", client)
    with pytest.raises(RuntimeError, match="refundable balance"):
        gateway.refund("rr-1", 500, "deterministic-key")


def test_webhook_signature_uses_raw_body_and_constant_time_comparison() -> None:
    body = b'{"event":"refund.processed"}'
    signature = hmac.new(b"secret", body, hashlib.sha256).hexdigest()
    assert verify_webhook_signature(body, signature, "secret")
    assert not verify_webhook_signature(body + b" ", signature, "secret")
