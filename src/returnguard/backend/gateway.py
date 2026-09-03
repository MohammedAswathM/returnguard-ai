from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from returnguard.domain.schemas import PaymentSnapshot


@dataclass(frozen=True)
class RefundResult:
    refund_id: str
    status: str
    metadata: dict[str, Any]


class RefundGateway(Protocol):
    provider_name: str
    authoritative_balance: bool
    requires_webhook_confirmation: bool

    def refund(self, request_id: str, amount_paise: int, idempotency_key: str) -> RefundResult: ...

    def payment_snapshot(
        self, merchant_id: str, payment_id: str, order_id: str,
        currency: str, as_of: datetime,
    ) -> PaymentSnapshot: ...


class MockRazorpayGateway:
    """Labelled in-memory test adapter; it never contacts Razorpay."""

    provider_name = "MOCK_RAZORPAY_TEST_ADAPTER"
    authoritative_balance = False
    requires_webhook_confirmation = False

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.effects: dict[str, RefundResult] = {}

    def refund(self, request_id: str, amount_paise: int, idempotency_key: str) -> RefundResult:
        if idempotency_key in self.effects:
            return self.effects[idempotency_key]
        if self.fail:
            raise RuntimeError("mock gateway unavailable")
        result = RefundResult(
            refund_id=f"rfnd_mock_{request_id}", status="processed",
            metadata={"adapter": self.provider_name, "amount_paise": amount_paise},
        )
        self.effects[idempotency_key] = result
        return result

    def payment_snapshot(
        self, merchant_id: str, payment_id: str, order_id: str,
        currency: str, as_of: datetime,
    ) -> PaymentSnapshot:
        raise RuntimeError("mock adapter is not an authoritative balance source")
