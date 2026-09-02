from __future__ import annotations

from returnguard.domain.enums import PaymentIntegrityCode, PaymentStatus
from returnguard.domain.schemas import PaymentIntegrityDecision, PaymentSnapshot


def evaluate_payment_integrity(
    *,
    snapshot: PaymentSnapshot | None,
    merchant_id: str,
    payment_id: str,
    order_id: str,
    currency: str,
    requested_amount_paise: int,
) -> PaymentIntegrityDecision:
    if snapshot is None:
        return PaymentIntegrityDecision(
            eligible_for_risk_scoring=False,
            reason_code=PaymentIntegrityCode.REFUND_BALANCE_UNAVAILABLE,
        )
    if snapshot.payment_id != payment_id or snapshot.order_id != order_id:
        return PaymentIntegrityDecision(
            eligible_for_risk_scoring=False,
            reason_code=PaymentIntegrityCode.PAYMENT_NOT_FOUND,
            snapshot_as_of=snapshot.snapshot_as_of,
        )
    if snapshot.merchant_id != merchant_id:
        code = PaymentIntegrityCode.MERCHANT_MISMATCH
    elif snapshot.payment_status != PaymentStatus.CAPTURED:
        code = PaymentIntegrityCode.PAYMENT_NOT_CAPTURED
    elif snapshot.currency != currency:
        code = PaymentIntegrityCode.CURRENCY_MISMATCH
    elif requested_amount_paise <= 0:
        code = PaymentIntegrityCode.INVALID_REFUND_AMOUNT
    elif snapshot.refundable_balance_paise <= 0:
        code = PaymentIntegrityCode.REFUND_BALANCE_EXHAUSTED
    elif requested_amount_paise > snapshot.refundable_balance_paise:
        code = PaymentIntegrityCode.REFUND_BALANCE_EXHAUSTED
    else:
        return PaymentIntegrityDecision(
            eligible_for_risk_scoring=True,
            refundable_balance_paise=snapshot.refundable_balance_paise,
            snapshot_as_of=snapshot.snapshot_as_of,
        )
    return PaymentIntegrityDecision(
        eligible_for_risk_scoring=False,
        reason_code=code,
        refundable_balance_paise=snapshot.refundable_balance_paise,
        snapshot_as_of=snapshot.snapshot_as_of,
    )
