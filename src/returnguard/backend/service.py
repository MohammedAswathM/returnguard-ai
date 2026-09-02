from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from returnguard.backend.gateway import RefundGateway
from returnguard.backend.payment_integrity import evaluate_payment_integrity
from returnguard.backend.repository import SQLiteRepository
from returnguard.backend.runtime import BundleRuntime
from returnguard.domain.enums import VerificationResult, WorkflowState
from returnguard.domain.schemas import PaymentSnapshot
from returnguard.verification.order_integrity import IntegrityCheck, verify_order_integrity

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


class WorkflowError(ValueError):
    pass


class ReturnGuardService:
    def __init__(
        self, repository: SQLiteRepository, runtime: BundleRuntime,
        gateway: RefundGateway, clock: Clock = utc_now,
    ) -> None:
        self.repository = repository
        self.runtime = runtime
        self.gateway = gateway
        self.clock = clock

    def _now(self) -> str:
        value = self.clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("clock must return UTC-aware timestamps")
        return value.astimezone(UTC).isoformat()

    def _audit(
        self, request_id: str, event_type: str, before: str | None, after: str,
        reason: str, actor_id: str = "returnguard-system", actor_type: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.repository.add_audit({
            "event_id": str(uuid.uuid4()), "refund_request_id": request_id,
            "actor_type": actor_type, "actor_id": actor_id, "event_type": event_type,
            "before_state": before, "after_state": after, "reason": reason,
            "metadata_json": json.dumps(metadata or {}, sort_keys=True), "created_at": self._now(),
        })

    def register_case(
        self, request_id: str, merchant_id: str, customer_id: str, payment_id: str,
        order_id: str, currency: str, amount_paise: int, features: dict[str, Any],
        payment_snapshot: PaymentSnapshot | None, limited_evidence: bool = False,
    ) -> dict[str, Any]:
        now = self._now()
        if payment_snapshot is not None:
            snapshot_values = payment_snapshot.model_dump(mode="json")
            snapshot_values["payment_status"] = payment_snapshot.payment_status.value
            self.repository.upsert_payment_snapshot(snapshot_values)
        self.repository.create_case(
            request_id, merchant_id, customer_id, payment_id, order_id, currency,
            amount_paise, features, now, limited_evidence
        )
        self._audit(request_id, "REQUEST_RECEIVED", None, "RECEIVED", "Refund request received.")
        return self._required_case(request_id)

    def _required_case(self, request_id: str) -> dict[str, Any]:
        case = self.repository.case(request_id)
        if case is None:
            raise KeyError(request_id)
        return case

    def score(self, request_id: str) -> dict[str, Any]:
        case = self._required_case(request_id)
        if case["state"] != WorkflowState.RECEIVED.value:
            stored_integrity = self.repository.integrity_decision(request_id)
            if stored_integrity is not None and not bool(stored_integrity["eligible_for_risk_scoring"]):
                return {**stored_integrity, "decision_type": "deterministic_payment_integrity"}
            existing = self.repository.decisions(request_id)
            if existing:
                return existing[0]
            raise WorkflowError("request is not scoreable")
        snapshot_row = self.repository.payment_snapshot(str(case["payment_id"]))
        snapshot = PaymentSnapshot.model_validate(snapshot_row) if snapshot_row else None
        integrity = evaluate_payment_integrity(
            snapshot=snapshot, merchant_id=str(case["merchant_id"]),
            payment_id=str(case["payment_id"]), order_id=str(case["order_id"]),
            currency=str(case["currency"]),
            requested_amount_paise=int(case["requested_amount_paise"]),
        )
        integrity_values = {
            "integrity_decision_id": str(uuid.uuid4()), "refund_request_id": request_id,
            "eligible_for_risk_scoring": int(integrity.eligible_for_risk_scoring),
            "reason_code": integrity.reason_code.value if integrity.reason_code else None,
            "refundable_balance_paise": integrity.refundable_balance_paise,
            "snapshot_as_of": integrity.snapshot_as_of.isoformat() if integrity.snapshot_as_of else None,
            "created_at": self._now(),
        }
        self.repository.add_integrity_decision(integrity_values)
        if not integrity.eligible_for_risk_scoring:
            self.repository.set_state(request_id, WorkflowState.MANUAL_REVIEW.value, self._now())
            self._audit(
                request_id, "PAYMENT_INTEGRITY_FAILED", case["state"], "MANUAL_REVIEW",
                "Deterministic payment eligibility failed before model scoring.",
                metadata={"reason_code": integrity_values["reason_code"]},
            )
            return {
                **integrity_values, "decision_type": "deterministic_payment_integrity",
                "action": "MANUAL_REVIEW",
            }
        if not self.runtime.healthy:
            self.repository.set_state(request_id, WorkflowState.FAILED_SAFE.value, self._now())
            self._audit(
                request_id, "MODEL_UNAVAILABLE", case["state"], "FAILED_SAFE",
                "Trusted model bundle validation failed.",
            )
            raise RuntimeError("trusted model unavailable; request failed safe")
        features = json.loads(case["features_json"])
        score = self.runtime.score(features, int(case["requested_amount_paise"]))
        now = self._now()
        decision = {
            "decision_id": str(uuid.uuid4()), "refund_request_id": request_id,
            "stage": "pre_verification", "probability": score.probability,
            "raw_score": score.raw_score, "action": score.action,
            "reason_codes_json": json.dumps([reason["code"] for reason in score.reasons]),
            "feature_snapshot_hash": score.snapshot_hash,
            "model_version": self.runtime.model.model_version, "created_at": now,
        }
        self.repository.add_decision(decision)
        state = score.action
        self.repository.set_state(request_id, state, now)
        self._audit(
            request_id, "STAGE_A_SCORED", case["state"], state,
            "Frozen policy applied to calibrated stage-A probability.",
            metadata={"decision_id": decision["decision_id"], "limited_evidence": bool(case["limited_evidence"])},
        )
        return {**decision, "reasons": score.reasons}

    def request_verification(self, request_id: str) -> dict[str, Any]:
        case = self._required_case(request_id)
        if case["state"] != WorkflowState.VERIFY.value:
            raise WorkflowError("verification is allowed only from VERIFY")
        verification_id = str(uuid.uuid4())
        try:
            self.repository.add_verification({
                "verification_id": verification_id, "refund_request_id": request_id,
                "requested_at": self._now(),
            })
        except sqlite3.IntegrityError as error:
            raise WorkflowError("one active verification is already present") from error
        self.repository.set_state(request_id, WorkflowState.VERIFYING.value, self._now())
        self._audit(
            request_id, "VERIFICATION_REQUESTED", case["state"], "VERIFYING",
            "One order-integrity verification requested.",
            metadata={"verification_id": verification_id},
        )
        return {"verification_id": verification_id, "status": "ACTIVE"}

    def complete_verification(self, verification_id: str, check: IntegrityCheck) -> dict[str, Any]:
        verification = self.repository.verification(verification_id)
        if verification is None or verification["status"] != "ACTIVE":
            raise WorkflowError("active verification not found")
        request_id = str(verification["refund_request_id"])
        case = self._required_case(request_id)
        outcome = verify_order_integrity(check)
        now = self._now()
        self.repository.complete_verification(
            verification_id, outcome.result.value, list(outcome.reason_codes), now
        )
        initial = self.repository.decisions(request_id)[0]
        posterior, action = self.runtime.posterior_action(
            float(initial["probability"]), outcome.result.value,
            int(case["requested_amount_paise"]),
        )
        if outcome.result == VerificationResult.INCONSISTENT:
            action = "MANUAL_REVIEW"
        elif outcome.result in {VerificationResult.INCONCLUSIVE, VerificationResult.EXPIRED}:
            action = "RETURN_FIRST"
        state_by_result = {
            VerificationResult.CONSISTENT: "VERIFIED_CLEAR",
            VerificationResult.INCONSISTENT: "VERIFIED_INCONSISTENT",
            VerificationResult.INCONCLUSIVE: "INCONCLUSIVE",
            VerificationResult.EXPIRED: "INCONCLUSIVE",
        }
        intermediate = state_by_result[outcome.result]
        self.repository.set_state(request_id, intermediate, now)
        self._audit(
            request_id, "VERIFICATION_COMPLETED", case["state"], intermediate,
            "Structured order-integrity verification completed.",
            metadata={"result": outcome.result.value, "result_codes": list(outcome.reason_codes)},
        )
        decision = {
            "decision_id": str(uuid.uuid4()), "refund_request_id": request_id,
            "stage": "post_verification", "probability": posterior,
            "raw_score": float(initial["raw_score"]), "action": action,
            "reason_codes_json": json.dumps(list(outcome.reason_codes)),
            "feature_snapshot_hash": str(initial["feature_snapshot_hash"]),
            "model_version": str(initial["model_version"]), "created_at": now,
        }
        self.repository.add_decision(decision)
        self.repository.set_state(request_id, action, now)
        self._audit(
            request_id, "POSTERIOR_DECISION", intermediate, action,
            "Bayesian posterior and frozen policy applied as a separate decision.",
            metadata={"decision_id": decision["decision_id"], "posterior": posterior},
        )
        return {**decision, "verification_result": outcome.result.value}

    def operator_decision(
        self, request_id: str, operator_id: str, approved: bool, reason: str,
    ) -> dict[str, Any]:
        case = self._required_case(request_id)
        if not operator_id.strip() or not reason.strip():
            raise WorkflowError("named operator and reason are required")
        if case["state"] not in {"MANUAL_REVIEW", "VERIFIED_INCONSISTENT", "HELD"}:
            raise WorkflowError("operator decision is not allowed from current state")
        state = "APPROVED" if approved else "REJECTED_BY_OPERATOR"
        now = self._now()
        values = {
            "id": str(uuid.uuid4()), "refund_request_id": request_id,
            "operator_id": operator_id, "approved": int(approved), "reason": reason,
            "appeal_state": "NOT_REQUIRED" if approved else "RECONSIDERATION_AVAILABLE",
            "created_at": now,
        }
        self.repository.add_operator_decision(values)
        self.repository.set_state(request_id, state, now)
        self._audit(
            request_id, "OPERATOR_DECISION", case["state"], state, reason,
            actor_id=operator_id, actor_type="operator",
            metadata={"appeal_state": values["appeal_state"]},
        )
        return {**values, "state": state}

    def execute(self, request_id: str) -> dict[str, Any]:
        existing = self.repository.execution(request_id)
        if existing is not None:
            return existing
        case = self._required_case(request_id)
        if case["state"] not in {"AUTO_APPROVE", "APPROVED"}:
            raise WorkflowError("refund execution requires approval")
        key = hashlib.sha256(f"returnguard-refund:{request_id}".encode()).hexdigest()
        if self.gateway.authoritative_balance:
            try:
                refreshed = self.gateway.payment_snapshot(
                    str(case["merchant_id"]), str(case["payment_id"]), str(case["order_id"]),
                    str(case["currency"]), self.clock(),
                )
            except Exception as error:
                self.repository.set_state(request_id, "FAILED_SAFE", self._now())
                self._audit(
                    request_id, "BALANCE_SOURCE_UNAVAILABLE", case["state"], "FAILED_SAFE",
                    "Authoritative refundable balance could not be refreshed.",
                    metadata={"reason_code": "REFUND_BALANCE_UNAVAILABLE", "error_type": type(error).__name__},
                )
                raise RuntimeError("refund balance unavailable; request failed safe") from error
            refreshed_values = refreshed.model_dump(mode="json")
            refreshed_values["payment_status"] = refreshed.payment_status.value
            self.repository.upsert_payment_snapshot(refreshed_values)
        current_row = self.repository.payment_snapshot(str(case["payment_id"]))
        current = PaymentSnapshot.model_validate(current_row) if current_row else None
        integrity = evaluate_payment_integrity(
            snapshot=current, merchant_id=str(case["merchant_id"]),
            payment_id=str(case["payment_id"]), order_id=str(case["order_id"]),
            currency=str(case["currency"]),
            requested_amount_paise=int(case["requested_amount_paise"]),
        )
        if not integrity.eligible_for_risk_scoring:
            raise WorkflowError(integrity.reason_code.value if integrity.reason_code else "REFUND_BALANCE_UNAVAILABLE")
        refund_id = str(uuid.uuid4())
        reserved, reason = self.repository.reserve_refund({
            "refund_id": refund_id, "refund_request_id": request_id,
            "payment_id": str(case["payment_id"]), "merchant_id": str(case["merchant_id"]),
            "amount_paise": int(case["requested_amount_paise"]), "currency": str(case["currency"]),
            "created_at": self._now(), "idempotency_key": key,
        })
        if not reserved:
            existing = self.repository.execution(request_id)
            if existing is not None:
                return existing
            raise WorkflowError(reason or "CONCURRENT_BALANCE_CONFLICT")
        now = self._now()
        self.repository.mark_refund_processing(refund_id)
        self.repository.set_state(request_id, "REFUND_PROCESSING", now)
        self._audit(
            request_id, "REFUND_EXECUTION_STARTED", case["state"], "REFUND_PROCESSING",
            "Authorized refund execution started.", metadata={"idempotency_key": key},
        )
        try:
            result = self.gateway.refund(request_id, int(case["requested_amount_paise"]), key)
        except Exception as error:
            self.repository.release_refund(refund_id, self._now())
            self.repository.set_state(request_id, "FAILED_SAFE", self._now())
            self._audit(
                request_id, "REFUND_EXECUTION_FAILED", "REFUND_PROCESSING", "FAILED_SAFE",
                "Refund gateway failed; no success was recorded.", metadata={"error_type": type(error).__name__},
            )
            raise RuntimeError("refund failed safe") from error
        self.repository.finalize_refund(refund_id, result.refund_id, self._now())
        values = {
            "refund_request_id": request_id, "idempotency_key": key,
            "gateway_refund_id": result.refund_id, "status": "REFUNDED",
            "response_json": json.dumps(result.metadata, sort_keys=True),
            "created_at": now, "updated_at": self._now(),
        }
        self.repository.add_execution(values)
        self.repository.set_state(request_id, "REFUNDED", values["updated_at"])
        self._audit(
            request_id, "REFUND_EXECUTED", "REFUND_PROCESSING", "REFUNDED",
            "Gateway confirmed refund in the configured adapter.",
            metadata={"gateway_refund_id": result.refund_id, "provider": self.gateway.provider_name},
        )
        return values
