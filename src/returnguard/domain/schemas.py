from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from returnguard.domain.enums import (
    AccountStatus,
    ActorType,
    DecisionStage,
    PaymentIntegrityCode,
    PaymentStatus,
    RecommendedAction,
    RefundLedgerStatus,
    RefundReason,
    RequestChannel,
    VerificationResult,
    VerificationType,
    WorkflowState,
)
from returnguard.domain.time import require_utc


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Customer(Contract):
    customer_id: str = Field(min_length=1)
    created_at: datetime
    country: str = Field(min_length=2, max_length=64)
    account_status: AccountStatus = AccountStatus.ACTIVE

    @model_validator(mode="after")
    def validate_time(self) -> "Customer":
        require_utc(self.created_at)
        return self


class Order(Contract):
    order_id: str = Field(min_length=1)
    customer_id: str = Field(min_length=1)
    razorpay_order_id: str | None = None
    ordered_at: datetime
    delivered_at: datetime | None = None
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    gross_amount_paise: int = Field(gt=0)
    item_count: int = Field(gt=0)
    category: str = Field(min_length=1)
    shipping_address_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_times(self) -> "Order":
        require_utc(self.ordered_at)
        if self.delivered_at is not None:
            require_utc(self.delivered_at)
            if self.delivered_at < self.ordered_at:
                raise ValueError("delivered_at cannot precede ordered_at")
        return self


class Payment(Contract):
    payment_id: str = Field(min_length=1)
    razorpay_payment_id: str | None = None
    order_id: str = Field(min_length=1)
    paid_amount_paise: int = Field(gt=0)
    status: PaymentStatus
    paid_at: datetime | None = None

    @model_validator(mode="after")
    def validate_time(self) -> "Payment":
        if self.paid_at is not None:
            require_utc(self.paid_at)
        return self


class PaymentSnapshot(Contract):
    merchant_id: str = Field(min_length=1)
    payment_id: str = Field(min_length=1)
    razorpay_payment_id: str | None = None
    order_id: str = Field(min_length=1)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    captured_amount_paise: int = Field(ge=0)
    amount_refunded_paise: int = Field(ge=0)
    refundable_balance_paise: int = Field(ge=0)
    payment_status: PaymentStatus
    captured_at: datetime | None = None
    snapshot_as_of: datetime
    source: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_balance_and_times(self) -> "PaymentSnapshot":
        require_utc(self.snapshot_as_of)
        if self.captured_at is not None:
            require_utc(self.captured_at)
            if self.captured_at > self.snapshot_as_of:
                raise ValueError("captured_at cannot follow snapshot_as_of")
        expected = self.captured_amount_paise - self.amount_refunded_paise
        if expected < 0 or self.refundable_balance_paise != expected:
            raise ValueError("refundable balance must equal captured amount minus refunded amount")
        return self


class RefundLedgerEntry(Contract):
    refund_id: str = Field(min_length=1)
    payment_id: str = Field(min_length=1)
    merchant_id: str = Field(min_length=1)
    amount_paise: int = Field(gt=0)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    status: RefundLedgerStatus
    created_at: datetime
    processed_at: datetime | None = None
    idempotency_key: str = Field(min_length=1)
    razorpay_refund_id: str | None = None

    @model_validator(mode="after")
    def validate_ledger_times(self) -> "RefundLedgerEntry":
        require_utc(self.created_at)
        if self.processed_at is not None:
            require_utc(self.processed_at)
            if self.processed_at < self.created_at:
                raise ValueError("processed_at cannot precede created_at")
        return self


class PaymentIntegrityDecision(Contract):
    eligible_for_risk_scoring: bool
    reason_code: PaymentIntegrityCode | None = None
    refundable_balance_paise: int | None = Field(default=None, ge=0)
    snapshot_as_of: datetime | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> "PaymentIntegrityDecision":
        if self.snapshot_as_of is not None:
            require_utc(self.snapshot_as_of)
        if self.eligible_for_risk_scoring == (self.reason_code is not None):
            raise ValueError("eligible decisions have no failure reason; ineligible decisions require one")
        return self


class RefundRequest(Contract):
    refund_request_id: str = Field(min_length=1)
    order_id: str = Field(min_length=1)
    customer_id: str = Field(min_length=1)
    requested_at: datetime
    reason_code: RefundReason
    requested_amount_paise: int = Field(gt=0)
    requested_quantity: int = Field(gt=0)
    returnless_requested: bool
    evidence_provided: bool
    channel: RequestChannel
    is_refund_abuse_simulated: bool | None = None
    simulation_scenario: str | None = None
    outcome_available_at: datetime | None = None

    @model_validator(mode="after")
    def validate_times(self) -> "RefundRequest":
        require_utc(self.requested_at)
        if self.outcome_available_at is not None:
            require_utc(self.outcome_available_at)
            if self.outcome_available_at <= self.requested_at:
                raise ValueError("outcome must mature after request")
        return self


class VerificationEvent(Contract):
    verification_id: str = Field(min_length=1)
    refund_request_id: str = Field(min_length=1)
    verification_type: VerificationType
    requested_at: datetime
    completed_at: datetime | None = None
    result: VerificationResult | None = None
    result_codes: tuple[str, ...] = ()
    estimated_friction_cost_paise: int = Field(ge=0)
    evidence_hash: str | None = None

    @model_validator(mode="after")
    def validate_times(self) -> "VerificationEvent":
        require_utc(self.requested_at)
        if self.completed_at is not None:
            require_utc(self.completed_at)
            if self.completed_at < self.requested_at:
                raise ValueError("completed_at cannot precede requested_at")
        if (self.completed_at is None) != (self.result is None):
            raise ValueError("completed_at and result must be set together")
        return self


class RiskDecision(Contract):
    decision_id: str = Field(min_length=1)
    refund_request_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    raw_score: float = Field(ge=0.0, le=1.0)
    calibrated_probability: float = Field(ge=0.0, le=1.0)
    decision_stage: DecisionStage
    recommended_action: RecommendedAction
    reason_codes: tuple[str, ...] = ()
    feature_snapshot_hash: str = Field(min_length=64, max_length=64)
    created_at: datetime

    @model_validator(mode="after")
    def validate_time(self) -> "RiskDecision":
        require_utc(self.created_at)
        return self


class AuditEvent(Contract):
    event_id: str = Field(min_length=1)
    refund_request_id: str = Field(min_length=1)
    actor_type: ActorType
    actor_id: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    before_state: WorkflowState | None = None
    after_state: WorkflowState
    reason: str = Field(min_length=1)
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_time(self) -> "AuditEvent":
        require_utc(self.created_at)
        return self


def validate_refund_eligibility(
    request: RefundRequest, order: Order, payment: Payment, already_refunded_paise: int = 0
) -> None:
    if request.order_id != order.order_id or payment.order_id != order.order_id:
        raise ValueError("request, order, and payment must refer to the same order")
    if request.customer_id != order.customer_id:
        raise ValueError("request customer does not own order")
    refundable = payment.paid_amount_paise - already_refunded_paise
    if already_refunded_paise < 0 or request.requested_amount_paise > refundable:
        raise ValueError("requested amount exceeds refundable balance")
    if request.requested_quantity > order.item_count:
        raise ValueError("requested quantity exceeds order quantity")
    if request.requested_at <= order.ordered_at:
        raise ValueError("refund request must follow order")
