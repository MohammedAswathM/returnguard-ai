from __future__ import annotations

import hashlib
import hmac
from datetime import UTC, datetime
from typing import Any

import httpx

from returnguard.backend.gateway import RefundResult
from returnguard.domain.enums import PaymentStatus
from returnguard.domain.schemas import PaymentSnapshot


class RazorpayTestGateway:
    provider_name = "RAZORPAY_TEST_MODE"
    authoritative_balance = True
    requires_webhook_confirmation = True

    def __init__(
        self, key_id: str, key_secret: str, payment_id: str,
        client: httpx.Client | None = None,
        merchant_id: str = "configured-test-merchant",
        payment_alias: str = "configured-test-payment",
    ) -> None:
        if not key_id.startswith("rzp_test_"):
            raise ValueError("only Razorpay test keys are accepted")
        if not key_secret or not payment_id.startswith("pay_"):
            raise ValueError("test key secret and captured test payment are required")
        self.payment_id = payment_id
        self.merchant_id = merchant_id
        self.payment_alias = payment_alias
        self.client = client or httpx.Client(
            base_url="https://api.razorpay.com", auth=(key_id, key_secret), timeout=15.0
        )

    def payment_snapshot(
        self, merchant_id: str, payment_id: str, order_id: str,
        currency: str, as_of: datetime,
    ) -> PaymentSnapshot:
        if merchant_id != self.merchant_id or payment_id != self.payment_alias:
            raise RuntimeError("configured test payment does not belong to request scope")
        response = self.client.get(f"/v1/payments/{self.payment_id}")
        response.raise_for_status()
        payment: dict[str, Any] = response.json()
        paid = int(payment.get("amount", 0))
        refunded = int(payment.get("amount_refunded", 0))
        payment_currency = str(payment.get("currency", currency)).upper()
        captured_at_raw = payment.get("created_at")
        captured_at = (
            datetime.fromtimestamp(int(captured_at_raw), tz=UTC)
            if captured_at_raw is not None else None
        )
        return PaymentSnapshot(
            merchant_id=merchant_id,
            payment_id=payment_id,
            razorpay_payment_id=None,
            order_id=order_id,
            currency=payment_currency,
            captured_amount_paise=paid,
            amount_refunded_paise=refunded,
            refundable_balance_paise=max(0, paid - refunded),
            payment_status=PaymentStatus(str(payment.get("status", "created"))),
            captured_at=captured_at,
            snapshot_as_of=as_of.astimezone(UTC),
            source="RAZORPAY_TEST_API",
        )

    def refund(self, request_id: str, amount_paise: int, idempotency_key: str) -> RefundResult:
        payment_response = self.client.get(f"/v1/payments/{self.payment_id}")
        payment_response.raise_for_status()
        payment = payment_response.json()
        if payment.get("status") != "captured":
            raise RuntimeError("Razorpay test payment is not captured")
        paid = int(payment.get("amount", 0))
        refunded = int(payment.get("amount_refunded", 0))
        if amount_paise <= 0 or amount_paise > paid - refunded:
            raise RuntimeError("requested amount exceeds Razorpay refundable balance")
        response = self.client.post(
            f"/v1/payments/{self.payment_id}/refund",
            headers={"X-Refund-Idempotency": idempotency_key},
            json={
                "amount": amount_paise, "speed": "normal",
                "receipt": f"returnguard-{request_id}"[:40],
                "notes": {"refund_request_id": request_id, "environment": "test"},
            },
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        refund_id = str(payload.get("id", ""))
        if not refund_id.startswith("rfnd_"):
            raise RuntimeError("Razorpay response did not contain a refund id")
        return RefundResult(
            refund_id=refund_id, status=str(payload.get("status", "unknown")),
            metadata={
                "adapter": self.provider_name, "payment_reference": self.payment_alias,
                "amount_paise": int(payload.get("amount", amount_paise)),
                "status": str(payload.get("status", "unknown")),
            },
        )


def verify_webhook_signature(raw_body: bytes, signature: str, secret: str) -> bool:
    if not signature or not secret:
        return False
    expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
