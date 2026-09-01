from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from returnguard.domain.enums import PaymentStatus, RefundReason, RequestChannel
from returnguard.domain.schemas import Order, Payment, RefundRequest, validate_refund_eligibility


def make_contracts() -> tuple[Order, Payment, RefundRequest]:
    ordered = datetime(2026, 1, 1, tzinfo=UTC)
    order = Order(
        order_id="o1", customer_id="c1", ordered_at=ordered,
        delivered_at=ordered + timedelta(days=2), currency="INR", gross_amount_paise=10_000,
        item_count=2, category="apparel", shipping_address_id="a1", device_id="d1",
    )
    payment = Payment(
        payment_id="p1", order_id="o1", paid_amount_paise=10_000,
        status=PaymentStatus.CAPTURED, paid_at=ordered + timedelta(minutes=1),
    )
    request = RefundRequest(
        refund_request_id="r1", order_id="o1", customer_id="c1",
        requested_at=ordered + timedelta(days=3), reason_code=RefundReason.SIZE,
        requested_amount_paise=5_000, requested_quantity=1, returnless_requested=False,
        evidence_provided=True, channel=RequestChannel.APP,
        outcome_available_at=ordered + timedelta(days=10),
    )
    return order, payment, request


def test_contracts_accept_utc_and_integer_paise() -> None:
    order, payment, request = make_contracts()
    validate_refund_eligibility(request, order, payment)
    assert isinstance(request.requested_amount_paise, int)


def test_naive_timestamp_and_nonpositive_amount_fail() -> None:
    order, _, _ = make_contracts()
    with pytest.raises(ValidationError, match="timezone-aware"):
        Order(**{**order.model_dump(), "ordered_at": datetime(2026, 1, 1)})
    with pytest.raises(ValidationError):
        Payment(payment_id="p", order_id="o", paid_amount_paise=0, status="captured")


def test_refund_cannot_exceed_balance_or_quantity() -> None:
    order, payment, request = make_contracts()
    with pytest.raises(ValueError, match="refundable balance"):
        validate_refund_eligibility(request.model_copy(update={"requested_amount_paise": 10_001}), order, payment)
    with pytest.raises(ValueError, match="order quantity"):
        validate_refund_eligibility(request.model_copy(update={"requested_quantity": 3}), order, payment)

