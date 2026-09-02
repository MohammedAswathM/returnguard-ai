from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from returnguard.backend.gateway import MockRazorpayGateway, RefundGateway
from returnguard.backend.razorpay import RazorpayTestGateway, verify_webhook_signature
from returnguard.backend.repository import SQLiteRepository
from returnguard.backend.runtime import BundleRuntime, limited_evidence
from returnguard.backend.service import ReturnGuardService, WorkflowError
from returnguard.domain.enums import PaymentStatus, RefundReason
from returnguard.domain.schemas import PaymentSnapshot
from returnguard.verification.order_integrity import IntegrityCheck


class Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateCaseBody(Body):
    merchant_id: str = Field(min_length=1)
    customer_id: str = Field(min_length=1)
    payment_id: str = Field(min_length=1)
    order_id: str = Field(min_length=1)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    requested_amount_paise: int
    features: dict[str, Any]
    payment_snapshot: PaymentSnapshot | None


class CompleteVerificationBody(Body):
    order_exists: bool = True
    payment_exists: bool = True
    payment_matches_order: bool = True
    payment_status: PaymentStatus = PaymentStatus.CAPTURED
    requested_amount_paise: int = Field(gt=0)
    refundable_balance_paise: int = Field(ge=0)
    requested_quantity: int = Field(gt=0)
    eligible_quantity: int = Field(ge=0)
    delivered: bool
    reason_code: RefundReason
    evidence_token: str | None = None
    allowed_evidence_tokens: frozenset[str] = frozenset({"demo-consistent"})
    verifier_available: bool = True
    evidence_expired: bool = False


class OperatorDecisionBody(Body):
    operator_id: str = Field(min_length=1)
    approved: bool
    reason: str = Field(min_length=1)


def create_app(
    database_path: Path = Path("artifacts/returnguard.db"),
    bundle_dir: Path = Path("artifacts/model_bundle"),
    data_dir: Path = Path("artifacts/data_uci"),
    feature_config: Path = Path("configs/features.yaml"),
    policy_config: Path = Path("configs/policy.yaml"),
    gateway: RefundGateway | None = None,
    runtime_override: BundleRuntime | None = None,
) -> FastAPI:
    repository = SQLiteRepository(database_path)
    runtime = runtime_override or BundleRuntime(bundle_dir, data_dir, feature_config, policy_config)
    configured_gateway = gateway
    if configured_gateway is None and os.getenv("RAZORPAY_KEY_ID"):
        configured_gateway = RazorpayTestGateway(
            os.environ["RAZORPAY_KEY_ID"], os.environ.get("RAZORPAY_KEY_SECRET", ""),
            os.environ.get("RAZORPAY_TEST_PAYMENT_ID", ""),
            merchant_id=os.environ.get("RAZORPAY_TEST_MERCHANT_ID", "configured-test-merchant"),
        )
    service = ReturnGuardService(repository, runtime, configured_gateway or MockRazorpayGateway())
    app = FastAPI(title="ReturnGuard API", version="1.0.0")
    app.state.repository = repository
    app.state.runtime = runtime
    app.state.service = service

    def call(operation: Any, *args: Any) -> Any:
        try:
            return operation(*args)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="refund request not found") from error
        except WorkflowError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.post("/api/v1/refund-requests/{request_id}")
    async def create_case(request_id: str, body: CreateCaseBody) -> dict[str, Any]:
        if repository.case(request_id) is not None:
            raise HTTPException(status_code=409, detail="DUPLICATE_REQUEST")
        return service.register_case(
            request_id, body.merchant_id, body.customer_id, body.payment_id, body.order_id,
            body.currency, body.requested_amount_paise, body.features, body.payment_snapshot,
            limited_evidence(runtime.registry, body.features) if runtime.healthy else True,
        )

    @app.post("/api/v1/refund-requests/{request_id}/score")
    async def score(request_id: str) -> dict[str, Any]:
        return cast(dict[str, Any], call(service.score, request_id))

    @app.post("/api/v1/refund-requests/{request_id}/verifications")
    async def request_verification(request_id: str) -> dict[str, Any]:
        return cast(dict[str, Any], call(service.request_verification, request_id))

    @app.post("/api/v1/verifications/{verification_id}/complete")
    async def complete_verification(
        verification_id: str, body: CompleteVerificationBody,
    ) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            call(service.complete_verification, verification_id, IntegrityCheck(**body.model_dump())),
        )

    @app.post("/api/v1/refund-requests/{request_id}/decisions")
    async def operator_decision(request_id: str, body: OperatorDecisionBody) -> dict[str, Any]:
        return cast(dict[str, Any], call(
            service.operator_decision, request_id, body.operator_id, body.approved, body.reason
        ))

    @app.post("/api/v1/refund-requests/{request_id}/execute")
    async def execute(request_id: str) -> dict[str, Any]:
        return cast(dict[str, Any], call(service.execute, request_id))

    @app.post("/api/v1/evaluations/run")
    async def run_evaluation() -> dict[str, Any]:
        run_id = str(uuid.uuid4())
        results_path = Path("results.lock.json")
        result = (
            json.loads(results_path.read_text(encoding="utf-8"))
            if results_path.is_file() else {"status": "UNAVAILABLE"}
        )
        repository.add_evaluation(run_id, result, service._now())
        return {"run_id": run_id, "status": "COMPLETE"}

    @app.get("/api/v1/evaluations/{run_id}")
    async def evaluation(run_id: str) -> dict[str, Any]:
        row = repository.evaluation(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="evaluation not found")
        return {**row, "result": json.loads(row["result_json"])}

    @app.get("/api/v1/refund-requests/{request_id}/audit")
    async def audit(request_id: str) -> list[dict[str, Any]]:
        if repository.case(request_id) is None:
            raise HTTPException(status_code=404, detail="refund request not found")
        return repository.audit(request_id)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "healthy" if runtime.healthy else "degraded",
            "bundle_valid": runtime.healthy,
            "error": runtime.validation_error,
        }

    @app.post("/api/v1/webhooks/razorpay")
    async def razorpay_webhook(request: Request) -> dict[str, Any]:
        raw_body = await request.body()
        signature = request.headers.get("X-Razorpay-Signature", "")
        secret = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")
        if not verify_webhook_signature(raw_body, signature, secret):
            raise HTTPException(status_code=401, detail="invalid webhook signature")
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError as error:
            raise HTTPException(status_code=400, detail="invalid webhook JSON") from error
        event_type = str(payload.get("event", "unknown"))
        event_id = str(payload.get("id") or hashlib.sha256(raw_body).hexdigest())
        inserted = repository.record_webhook(
            event_id, event_type, hashlib.sha256(raw_body).hexdigest(), service._now()
        )
        return {"accepted": True, "duplicate": not inserted, "event_type": event_type}

    return app
