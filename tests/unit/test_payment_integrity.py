from __future__ import annotations

import concurrent.futures
from datetime import UTC, datetime
from pathlib import Path

from returnguard.backend.payment_integrity import evaluate_payment_integrity
from returnguard.backend.repository import SQLiteRepository
from returnguard.domain.enums import PaymentIntegrityCode, PaymentStatus
from returnguard.domain.schemas import PaymentSnapshot

NOW = datetime(2026, 9, 2, tzinfo=UTC)


def snapshot(**changes: object) -> PaymentSnapshot:
    values = {
        "merchant_id": "merchant-1", "payment_id": "payment-1",
        "razorpay_payment_id": "pay_test_1", "order_id": "order-1", "currency": "INR",
        "captured_amount_paise": 10_000, "amount_refunded_paise": 2_000,
        "refundable_balance_paise": 8_000, "payment_status": PaymentStatus.CAPTURED,
        "captured_at": NOW, "snapshot_as_of": NOW, "source": "TEST_FIXTURE",
    }
    values.update(changes)
    return PaymentSnapshot(**values)  # type: ignore[arg-type]


def decision(payment: PaymentSnapshot | None, amount: int = 8_000, **changes: object):  # type: ignore[no-untyped-def]
    values = {
        "snapshot": payment, "merchant_id": "merchant-1", "payment_id": "payment-1",
        "order_id": "order-1", "currency": "INR", "requested_amount_paise": amount,
    }
    values.update(changes)
    return evaluate_payment_integrity(**values)  # type: ignore[arg-type]


def test_payment_integrity_boundary_codes() -> None:
    assert decision(snapshot()).eligible_for_risk_scoring
    assert decision(snapshot(), 8_001).reason_code == PaymentIntegrityCode.REFUND_BALANCE_EXHAUSTED
    original_balance = snapshot(amount_refunded_paise=0, refundable_balance_paise=10_000)
    assert decision(original_balance, 10_001).reason_code == PaymentIntegrityCode.REFUND_BALANCE_EXHAUSTED
    assert decision(snapshot(), 0).reason_code == PaymentIntegrityCode.INVALID_REFUND_AMOUNT
    assert decision(snapshot(), -1).reason_code == PaymentIntegrityCode.INVALID_REFUND_AMOUNT
    assert decision(snapshot(), currency="USD").reason_code == PaymentIntegrityCode.CURRENCY_MISMATCH
    assert decision(snapshot(), merchant_id="merchant-2").reason_code == PaymentIntegrityCode.MERCHANT_MISMATCH
    assert decision(snapshot(), payment_id="payment-2").reason_code == PaymentIntegrityCode.PAYMENT_NOT_FOUND
    uncaptured = snapshot(payment_status=PaymentStatus.AUTHORIZED)
    assert decision(uncaptured).reason_code == PaymentIntegrityCode.PAYMENT_NOT_CAPTURED
    assert decision(None).reason_code == PaymentIntegrityCode.REFUND_BALANCE_UNAVAILABLE


def setup_repository(path: Path) -> SQLiteRepository:
    repository = SQLiteRepository(path)
    repository.upsert_payment_snapshot(snapshot().model_dump(mode="json"))
    for index in (1, 2):
        repository.create_case(
            f"request-{index}", "merchant-1", f"customer-{index}", "payment-1",
            "order-1", "INR", 5_000, {}, NOW.isoformat(),
        )
    return repository


def reserve(repository: SQLiteRepository, request: int, amount: int, key: str) -> tuple[bool, str | None]:
    return repository.reserve_refund({
        "refund_id": f"refund-{request}", "refund_request_id": f"request-{request}",
        "payment_id": "payment-1", "merchant_id": "merchant-1", "amount_paise": amount,
        "currency": "INR", "created_at": NOW.isoformat(), "idempotency_key": key,
    })


def test_cumulative_partial_refunds_and_duplicate_key(tmp_path: Path) -> None:
    repository = setup_repository(tmp_path / "ledger.db")
    assert reserve(repository, 1, 3_000, "key-1") == (True, None)
    repository.finalize_refund("refund-1", "rfnd_test_1", NOW.isoformat())
    assert repository.payment_snapshot("payment-1")["refundable_balance_paise"] == 5_000  # type: ignore[index]
    assert reserve(repository, 1, 3_000, "key-1") == (False, "DUPLICATE_REQUEST")
    assert reserve(repository, 2, 5_001, "key-2") == (False, "REFUND_BALANCE_EXHAUSTED")


def test_failed_refund_releases_reservation(tmp_path: Path) -> None:
    repository = setup_repository(tmp_path / "failure.db")
    assert reserve(repository, 1, 5_000, "key-1") == (True, None)
    repository.release_refund("refund-1", NOW.isoformat())
    assert reserve(repository, 2, 5_000, "key-2") == (True, None)


def test_only_one_concurrent_operation_reserves_final_balance(tmp_path: Path) -> None:
    path = tmp_path / "concurrent.db"
    first = setup_repository(path)
    second = SQLiteRepository(path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda args: reserve(args[0], args[1], 5_000, f"key-{args[1]}"),
            ((first, 1), (second, 2)),
        ))
    assert sum(result[0] for result in results) == 1
    assert {result[1] for result in results if not result[0]} == {"CONCURRENT_BALANCE_CONFLICT"}
