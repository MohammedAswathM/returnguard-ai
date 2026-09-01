from enum import StrEnum


class AccountStatus(StrEnum):
    ACTIVE = "active"
    RESTRICTED = "restricted"
    CLOSED = "closed"


class PaymentStatus(StrEnum):
    CREATED = "created"
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    REFUNDED = "refunded"


class RefundReason(StrEnum):
    MISSING = "missing"
    DAMAGED = "damaged"
    NOT_RECEIVED = "not_received"
    SIZE = "size"
    OTHER = "other"


class RequestChannel(StrEnum):
    WEB = "web"
    APP = "app"
    SUPPORT = "support"


class VerificationType(StrEnum):
    ORDER_INTEGRITY = "order_integrity"
    DELIVERY_PROOF = "delivery_proof"
    ITEM_MATCH = "item_match"
    RETURN_FIRST = "return_first"
    MANUAL = "manual"


class VerificationResult(StrEnum):
    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    INCONCLUSIVE = "inconclusive"
    EXPIRED = "expired"


class DecisionStage(StrEnum):
    PRE_VERIFICATION = "pre_verification"
    POST_VERIFICATION = "post_verification"
    OPERATOR = "operator"


class RecommendedAction(StrEnum):
    AUTO_APPROVE = "AUTO_APPROVE"
    VERIFY = "VERIFY"
    RETURN_FIRST = "RETURN_FIRST"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class ActorType(StrEnum):
    SYSTEM = "system"
    OPERATOR = "operator"
    WEBHOOK = "webhook"


class WorkflowState(StrEnum):
    RECEIVED = "RECEIVED"
    SCORED = "SCORED"
    AUTO_APPROVE = "AUTO_APPROVE"
    VERIFY = "VERIFY"
    VERIFYING = "VERIFYING"
    VERIFIED_CLEAR = "VERIFIED_CLEAR"
    VERIFIED_INCONSISTENT = "VERIFIED_INCONSISTENT"
    INCONCLUSIVE = "INCONCLUSIVE"
    RETURN_FIRST = "RETURN_FIRST"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    APPROVED = "APPROVED"
    HELD = "HELD"
    REJECTED_BY_OPERATOR = "REJECTED_BY_OPERATOR"
    REFUND_PROCESSING = "REFUND_PROCESSING"
    REFUNDED = "REFUNDED"
    FAILED_SAFE = "FAILED_SAFE"

