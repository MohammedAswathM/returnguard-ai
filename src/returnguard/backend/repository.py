from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS refund_cases (
    refund_request_id TEXT PRIMARY KEY,
    merchant_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    payment_id TEXT NOT NULL,
    order_id TEXT NOT NULL,
    currency TEXT NOT NULL CHECK(length(currency) = 3),
    requested_amount_paise INTEGER NOT NULL,
    state TEXT NOT NULL,
    features_json TEXT NOT NULL,
    limited_evidence INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS payment_snapshots (
    payment_id TEXT PRIMARY KEY,
    merchant_id TEXT NOT NULL,
    razorpay_payment_id TEXT,
    order_id TEXT NOT NULL,
    currency TEXT NOT NULL CHECK(length(currency) = 3),
    captured_amount_paise INTEGER NOT NULL CHECK(captured_amount_paise >= 0),
    amount_refunded_paise INTEGER NOT NULL CHECK(amount_refunded_paise >= 0),
    refundable_balance_paise INTEGER NOT NULL CHECK(refundable_balance_paise >= 0),
    payment_status TEXT NOT NULL,
    captured_at TEXT,
    snapshot_as_of TEXT NOT NULL,
    source TEXT NOT NULL,
    CHECK(refundable_balance_paise = captured_amount_paise - amount_refunded_paise)
);
CREATE TABLE IF NOT EXISTS refund_ledger (
    refund_id TEXT PRIMARY KEY,
    refund_request_id TEXT NOT NULL UNIQUE REFERENCES refund_cases(refund_request_id),
    payment_id TEXT NOT NULL REFERENCES payment_snapshots(payment_id),
    merchant_id TEXT NOT NULL,
    amount_paise INTEGER NOT NULL CHECK(amount_paise > 0),
    currency TEXT NOT NULL CHECK(length(currency) = 3),
    status TEXT NOT NULL CHECK(status IN ('reserved','processing','succeeded','failed','released')),
    created_at TEXT NOT NULL,
    processed_at TEXT,
    idempotency_key TEXT NOT NULL UNIQUE,
    razorpay_refund_id TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY,
    refund_request_id TEXT NOT NULL REFERENCES refund_cases(refund_request_id),
    stage TEXT NOT NULL,
    probability REAL NOT NULL CHECK(probability >= 0 AND probability <= 1),
    raw_score REAL NOT NULL CHECK(raw_score >= 0 AND raw_score <= 1),
    action TEXT NOT NULL,
    reason_codes_json TEXT NOT NULL,
    feature_snapshot_hash TEXT NOT NULL,
    model_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(refund_request_id, stage)
);
CREATE TABLE IF NOT EXISTS integrity_decisions (
    integrity_decision_id TEXT PRIMARY KEY,
    refund_request_id TEXT NOT NULL REFERENCES refund_cases(refund_request_id),
    eligible_for_risk_scoring INTEGER NOT NULL,
    reason_code TEXT,
    refundable_balance_paise INTEGER,
    snapshot_as_of TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(refund_request_id)
);
CREATE TRIGGER IF NOT EXISTS integrity_decisions_no_update BEFORE UPDATE ON integrity_decisions
BEGIN SELECT RAISE(ABORT, 'integrity decisions are immutable'); END;
CREATE TRIGGER IF NOT EXISTS integrity_decisions_no_delete BEFORE DELETE ON integrity_decisions
BEGIN SELECT RAISE(ABORT, 'integrity decisions are immutable'); END;
CREATE TRIGGER IF NOT EXISTS decisions_no_update BEFORE UPDATE ON decisions
BEGIN SELECT RAISE(ABORT, 'decisions are immutable'); END;
CREATE TRIGGER IF NOT EXISTS decisions_no_delete BEFORE DELETE ON decisions
BEGIN SELECT RAISE(ABORT, 'decisions are immutable'); END;
CREATE TABLE IF NOT EXISTS verifications (
    verification_id TEXT PRIMARY KEY,
    refund_request_id TEXT NOT NULL REFERENCES refund_cases(refund_request_id),
    status TEXT NOT NULL,
    result TEXT,
    result_codes_json TEXT NOT NULL DEFAULT '[]',
    requested_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_verification
ON verifications(refund_request_id) WHERE status = 'ACTIVE';
CREATE TABLE IF NOT EXISTS operator_decisions (
    id TEXT PRIMARY KEY,
    refund_request_id TEXT NOT NULL REFERENCES refund_cases(refund_request_id),
    operator_id TEXT NOT NULL,
    approved INTEGER NOT NULL,
    reason TEXT NOT NULL,
    appeal_state TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS executions (
    refund_request_id TEXT PRIMARY KEY REFERENCES refund_cases(refund_request_id),
    idempotency_key TEXT NOT NULL UNIQUE,
    gateway_refund_id TEXT,
    status TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    refund_request_id TEXT NOT NULL REFERENCES refund_cases(refund_request_id),
    actor_type TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    before_state TEXT,
    after_state TEXT NOT NULL,
    reason TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_events
BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_events
BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
CREATE TABLE IF NOT EXISTS evaluation_runs (
    run_id TEXT PRIMARY KEY, status TEXT NOT NULL, result_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS webhook_events (
    event_id TEXT PRIMARY KEY, event_type TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
    received_at TEXT NOT NULL
);
"""


class SQLiteRepository:
    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)
        self._migrate_legacy_cases()
        self._transaction_lock = threading.RLock()

    def _migrate_legacy_cases(self) -> None:
        columns = {
            str(row["name"])
            for row in self.connection.execute("PRAGMA table_info(refund_cases)").fetchall()
        }
        additions = {
            "merchant_id": "TEXT NOT NULL DEFAULT ''",
            "payment_id": "TEXT NOT NULL DEFAULT ''",
            "order_id": "TEXT NOT NULL DEFAULT ''",
            "currency": "TEXT NOT NULL DEFAULT 'INR'",
        }
        with self.connection:
            for name, declaration in additions.items():
                if name not in columns:
                    self.connection.execute(
                        f"ALTER TABLE refund_cases ADD COLUMN {name} {declaration}"
                    )

    def close(self) -> None:
        self.connection.close()

    def create_case(
        self, request_id: str, merchant_id: str, customer_id: str, payment_id: str,
        order_id: str, currency: str, amount_paise: int,
        features: dict[str, Any], created_at: str, limited_evidence: bool = False,
    ) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT INTO refund_cases
                (refund_request_id, merchant_id, customer_id, payment_id, order_id, currency,
                 requested_amount_paise, state, features_json, limited_evidence, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'RECEIVED', ?, ?, ?, ?)""",
                (request_id, merchant_id, customer_id, payment_id, order_id, currency,
                 amount_paise, json.dumps(features, sort_keys=True), int(limited_evidence),
                 created_at, created_at),
            )

    def upsert_payment_snapshot(self, values: dict[str, Any]) -> None:
        with self._transaction_lock, self.connection:
            self.connection.execute(
                """INSERT INTO payment_snapshots
                (payment_id, merchant_id, razorpay_payment_id, order_id, currency,
                 captured_amount_paise, amount_refunded_paise, refundable_balance_paise,
                 payment_status, captured_at, snapshot_as_of, source)
                VALUES (:payment_id, :merchant_id, :razorpay_payment_id, :order_id, :currency,
                 :captured_amount_paise, :amount_refunded_paise, :refundable_balance_paise,
                 :payment_status, :captured_at, :snapshot_as_of, :source)
                ON CONFLICT(payment_id) DO UPDATE SET
                 merchant_id=excluded.merchant_id, razorpay_payment_id=excluded.razorpay_payment_id,
                 order_id=excluded.order_id, currency=excluded.currency,
                 captured_amount_paise=excluded.captured_amount_paise,
                 amount_refunded_paise=excluded.amount_refunded_paise,
                 refundable_balance_paise=excluded.refundable_balance_paise,
                 payment_status=excluded.payment_status, captured_at=excluded.captured_at,
                 snapshot_as_of=excluded.snapshot_as_of, source=excluded.source""",
                values,
            )

    def payment_snapshot(self, payment_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM payment_snapshots WHERE payment_id = ?", (payment_id,)
        ).fetchone()
        return dict(row) if row else None

    def reserve_refund(self, values: dict[str, Any]) -> tuple[bool, str | None]:
        """Atomically reserve balance; return a stable integrity code on rejection."""
        with self._transaction_lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                duplicate = self.connection.execute(
                    "SELECT * FROM refund_ledger WHERE idempotency_key = ? OR refund_request_id = ?",
                    (values["idempotency_key"], values["refund_request_id"]),
                ).fetchone()
                if duplicate is not None:
                    self.connection.rollback()
                    return False, "DUPLICATE_REQUEST"
                snapshot = self.connection.execute(
                    "SELECT * FROM payment_snapshots WHERE payment_id = ?", (values["payment_id"],)
                ).fetchone()
                if snapshot is None:
                    self.connection.rollback()
                    return False, "REFUND_BALANCE_UNAVAILABLE"
                active = self.connection.execute(
                    """SELECT COALESCE(SUM(amount_paise), 0) FROM refund_ledger
                    WHERE payment_id = ? AND status IN ('reserved', 'processing')""",
                    (values["payment_id"],),
                ).fetchone()[0]
                available = int(snapshot["refundable_balance_paise"]) - int(active)
                amount = int(values["amount_paise"])
                if amount > available:
                    self.connection.rollback()
                    code = (
                        "CONCURRENT_BALANCE_CONFLICT"
                        if amount <= int(snapshot["refundable_balance_paise"]) else
                        "REFUND_BALANCE_EXHAUSTED"
                    )
                    return False, code
                self.connection.execute(
                    """INSERT INTO refund_ledger
                    (refund_id, refund_request_id, payment_id, merchant_id, amount_paise,
                     currency, status, created_at, processed_at, idempotency_key, razorpay_refund_id)
                    VALUES (:refund_id, :refund_request_id, :payment_id, :merchant_id,
                     :amount_paise, :currency, 'reserved', :created_at, NULL,
                     :idempotency_key, NULL)""",
                    values,
                )
                self.connection.commit()
                return True, None
            except Exception:
                self.connection.rollback()
                raise

    def mark_refund_processing(self, refund_id: str) -> None:
        with self._transaction_lock, self.connection:
            self.connection.execute(
                "UPDATE refund_ledger SET status = 'processing' WHERE refund_id = ? AND status = 'reserved'",
                (refund_id,),
            )

    def finalize_refund(
        self, refund_id: str, razorpay_refund_id: str, processed_at: str,
    ) -> None:
        with self._transaction_lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                entry = self.connection.execute(
                    "SELECT * FROM refund_ledger WHERE refund_id = ?", (refund_id,)
                ).fetchone()
                if entry is None or entry["status"] not in {"reserved", "processing"}:
                    raise sqlite3.IntegrityError("active refund reservation not found")
                updated = self.connection.execute(
                    """UPDATE payment_snapshots SET
                    amount_refunded_paise = amount_refunded_paise + ?,
                    refundable_balance_paise = refundable_balance_paise - ?,
                    snapshot_as_of = ?
                    WHERE payment_id = ? AND refundable_balance_paise >= ?""",
                    (entry["amount_paise"], entry["amount_paise"], processed_at,
                     entry["payment_id"], entry["amount_paise"]),
                )
                if updated.rowcount != 1:
                    raise sqlite3.IntegrityError("reserved balance cannot be finalized")
                self.connection.execute(
                    """UPDATE refund_ledger SET status = 'succeeded', processed_at = ?,
                    razorpay_refund_id = ? WHERE refund_id = ?""",
                    (processed_at, razorpay_refund_id, refund_id),
                )
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise

    def attach_gateway_refund(self, refund_id: str, gateway_refund_id: str) -> None:
        with self._transaction_lock, self.connection:
            updated = self.connection.execute(
                """UPDATE refund_ledger SET razorpay_refund_id = ?
                WHERE refund_id = ? AND status = 'processing' AND razorpay_refund_id IS NULL""",
                (gateway_refund_id, refund_id),
            )
            if updated.rowcount != 1:
                raise sqlite3.IntegrityError("processing refund cannot be linked to gateway")

    def refund_by_gateway_id(self, gateway_refund_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM refund_ledger WHERE razorpay_refund_id = ?", (gateway_refund_id,)
        ).fetchone()
        return dict(row) if row else None

    def release_refund(self, refund_id: str, processed_at: str) -> None:
        with self._transaction_lock, self.connection:
            self.connection.execute(
                """UPDATE refund_ledger SET status = 'failed', processed_at = ?
                WHERE refund_id = ? AND status IN ('reserved', 'processing')""",
                (processed_at, refund_id),
            )

    def refund_ledger(self, payment_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM refund_ledger WHERE payment_id = ? ORDER BY created_at, refund_id",
            (payment_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def case(self, request_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM refund_cases WHERE refund_request_id = ?", (request_id,)
        ).fetchone()
        return dict(row) if row else None

    def set_state(self, request_id: str, state: str, updated_at: str) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE refund_cases SET state = ?, updated_at = ? WHERE refund_request_id = ?",
                (state, updated_at, request_id),
            )

    def add_decision(self, values: dict[str, Any]) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT INTO decisions
                (decision_id, refund_request_id, stage, probability, raw_score, action,
                 reason_codes_json, feature_snapshot_hash, model_version, created_at)
                VALUES (:decision_id, :refund_request_id, :stage, :probability, :raw_score,
                 :action, :reason_codes_json, :feature_snapshot_hash, :model_version, :created_at)""",
                values,
            )

    def add_integrity_decision(self, values: dict[str, Any]) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT INTO integrity_decisions
                (integrity_decision_id, refund_request_id, eligible_for_risk_scoring,
                 reason_code, refundable_balance_paise, snapshot_as_of, created_at)
                VALUES (:integrity_decision_id, :refund_request_id,
                 :eligible_for_risk_scoring, :reason_code, :refundable_balance_paise,
                 :snapshot_as_of, :created_at)""",
                values,
            )

    def integrity_decision(self, request_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM integrity_decisions WHERE refund_request_id = ?", (request_id,)
        ).fetchone()
        return dict(row) if row else None

    def decisions(self, request_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM decisions WHERE refund_request_id = ? ORDER BY created_at", (request_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def add_verification(self, values: dict[str, Any]) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT INTO verifications
                (verification_id, refund_request_id, status, requested_at)
                VALUES (:verification_id, :refund_request_id, 'ACTIVE', :requested_at)""", values,
            )

    def verification(self, verification_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM verifications WHERE verification_id = ?", (verification_id,)
        ).fetchone()
        return dict(row) if row else None

    def complete_verification(
        self, verification_id: str, result: str, codes: list[str], completed_at: str,
    ) -> None:
        with self.connection:
            self.connection.execute(
                """UPDATE verifications SET status = 'COMPLETE', result = ?,
                result_codes_json = ?, completed_at = ? WHERE verification_id = ? AND status = 'ACTIVE'""",
                (result, json.dumps(codes), completed_at, verification_id),
            )

    def add_operator_decision(self, values: dict[str, Any]) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT INTO operator_decisions VALUES
                (:id, :refund_request_id, :operator_id, :approved, :reason, :appeal_state, :created_at)""",
                values,
            )

    def execution(self, request_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM executions WHERE refund_request_id = ?", (request_id,)
        ).fetchone()
        return dict(row) if row else None

    def add_execution(self, values: dict[str, Any]) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT INTO executions VALUES
                (:refund_request_id, :idempotency_key, :gateway_refund_id, :status,
                 :response_json, :created_at, :updated_at)""", values,
            )

    def update_execution_status(self, request_id: str, status: str, updated_at: str) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE executions SET status = ?, updated_at = ? WHERE refund_request_id = ?",
                (status, updated_at, request_id),
            )

    def add_audit(self, values: dict[str, Any]) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT INTO audit_events
                (event_id, refund_request_id, actor_type, actor_id, event_type, before_state,
                 after_state, reason, metadata_json, created_at)
                VALUES (:event_id, :refund_request_id, :actor_type, :actor_id, :event_type,
                 :before_state, :after_state, :reason, :metadata_json, :created_at)""", values,
            )

    def audit(self, request_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM audit_events WHERE refund_request_id = ? ORDER BY sequence", (request_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def add_evaluation(self, run_id: str, result: dict[str, Any], created_at: str) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO evaluation_runs VALUES (?, 'COMPLETE', ?, ?)",
                (run_id, json.dumps(result, sort_keys=True), created_at),
            )

    def evaluation(self, run_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM evaluation_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None

    def record_webhook(self, event_id: str, event_type: str, digest: str, received_at: str) -> bool:
        try:
            with self.connection:
                self.connection.execute(
                    "INSERT INTO webhook_events VALUES (?, ?, ?, ?)",
                    (event_id, event_type, digest, received_at),
                )
        except sqlite3.IntegrityError:
            return False
        return True
