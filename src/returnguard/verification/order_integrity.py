from dataclasses import dataclass

from returnguard.domain.enums import PaymentStatus, RefundReason, VerificationResult


@dataclass(frozen=True)
class IntegrityCheck:
    order_exists: bool
    payment_exists: bool
    payment_matches_order: bool
    payment_status: PaymentStatus | None
    requested_amount_paise: int
    refundable_balance_paise: int
    requested_quantity: int
    eligible_quantity: int
    delivered: bool | None
    reason_code: RefundReason
    evidence_token: str | None
    allowed_evidence_tokens: frozenset[str]
    evidence_expired: bool = False
    verifier_available: bool = True


@dataclass(frozen=True)
class IntegrityFinding:
    result: VerificationResult
    reason_codes: tuple[str, ...]


def verify_order_integrity(check: IntegrityCheck) -> IntegrityFinding:
    if not check.verifier_available:
        return IntegrityFinding(VerificationResult.INCONCLUSIVE, ("VERIFIER_UNAVAILABLE",))
    if check.evidence_expired:
        return IntegrityFinding(VerificationResult.EXPIRED, ("EVIDENCE_EXPIRED",))
    if not check.order_exists or not check.payment_exists:
        return IntegrityFinding(VerificationResult.INCONCLUSIVE, ("ORDER_OR_PAYMENT_UNAVAILABLE",))
    inconsistencies: list[str] = []
    if not check.payment_matches_order:
        inconsistencies.append("PAYMENT_ORDER_MISMATCH")
    if check.payment_status != PaymentStatus.CAPTURED:
        inconsistencies.append("PAYMENT_NOT_CAPTURED")
    if check.requested_amount_paise > check.refundable_balance_paise:
        inconsistencies.append("AMOUNT_EXCEEDS_REFUNDABLE_BALANCE")
    if check.requested_quantity > check.eligible_quantity:
        inconsistencies.append("QUANTITY_EXCEEDS_ELIGIBLE")
    if check.reason_code == RefundReason.NOT_RECEIVED and check.delivered is None:
        return IntegrityFinding(VerificationResult.INCONCLUSIVE, ("DELIVERY_STATUS_UNAVAILABLE",))
    if check.evidence_token is None:
        return IntegrityFinding(VerificationResult.INCONCLUSIVE, ("EVIDENCE_MISSING",))
    if check.evidence_token not in check.allowed_evidence_tokens:
        inconsistencies.append("EVIDENCE_TOKEN_NOT_ALLOWLISTED")
    if inconsistencies:
        return IntegrityFinding(VerificationResult.INCONSISTENT, tuple(inconsistencies))
    return IntegrityFinding(VerificationResult.CONSISTENT, ("ORDER_INTEGRITY_CONFIRMED",))
