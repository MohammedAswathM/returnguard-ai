from __future__ import annotations

import json
from datetime import UTC, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from returnguard.data.fingerprints import file_sha256, object_sha256
from returnguard.data.uci import build_uci_foundation
from returnguard.v2.config import V2DataConfig

PARTITIONS = ("train", "calibration", "policy_selection", "final_test")
HARD_LEGITIMATE = (
    "genuine_product_defect_burst", "high_value_loyal_customer", "first_time_refund",
    "cold_start_customer", "unusual_legitimate_purchase", "missing_evidence",
    "verifier_unavailable",
)
ABUSE_SCENARIOS = (
    "serial_false_claim", "repeated_claim_pattern", "peer_deviation", "claim_timing",
    "claim_reason_inconsistency", "supporting_entity_pattern", "cross_category_escalation",
    "evidence_inconsistency",
)
INTEGRITY_SCENARIOS = (
    "requested_amount_above_refundable_balance", "cumulative_partial_refund_exhaustion",
    "payment_not_captured", "merchant_mismatch", "currency_mismatch", "payment_not_found",
    "duplicate_idempotency_key", "concurrent_balance_conflict",
    "gateway_failure_reservation_release", "webhook_replay",
)


def _iso(value: pd.Timestamp) -> str:
    return value.tz_convert(UTC).isoformat().replace("+00:00", "Z")


def _partitions(config: V2DataConfig) -> list[str]:
    return [
        partition
        for partition in PARTITIONS
        for _ in range(int(getattr(config.split_counts, partition)))
    ]


def _integrity_cases(config: V2DataConfig) -> pd.DataFrame:
    per_scenario = config.n_integrity_cases // len(INTEGRITY_SCENARIOS)
    rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(INTEGRITY_SCENARIOS):
        for index in range(per_scenario):
            captured = 10_000
            refunded = 2_000
            reserved = 0
            requested = 3_000
            status = "captured"
            currency = "INR"
            request_currency = "INR"
            merchant_matches = True
            payment_exists = True
            expected = "ELIGIBLE"
            if scenario == "requested_amount_above_refundable_balance":
                requested, expected = 8_001, "REFUND_BALANCE_EXHAUSTED"
            elif scenario == "cumulative_partial_refund_exhaustion":
                refunded, requested, expected = 8_000, 2_001, "REFUND_BALANCE_EXHAUSTED"
            elif scenario == "payment_not_captured":
                status, expected = "authorized", "PAYMENT_NOT_CAPTURED"
            elif scenario == "merchant_mismatch":
                merchant_matches, expected = False, "MERCHANT_MISMATCH"
            elif scenario == "currency_mismatch":
                request_currency, expected = "USD", "CURRENCY_MISMATCH"
            elif scenario == "payment_not_found":
                payment_exists, expected = False, "PAYMENT_NOT_FOUND"
            elif scenario == "duplicate_idempotency_key":
                expected = "DUPLICATE_REQUEST"
            elif scenario == "concurrent_balance_conflict":
                reserved, requested, expected = 5_500, 3_000, "CONCURRENT_BALANCE_CONFLICT"
            elif scenario == "gateway_failure_reservation_release":
                expected = "RESERVATION_RELEASED"
            elif scenario == "webhook_replay":
                expected = "WEBHOOK_DEDUPLICATED"
            rows.append({
                "integrity_case_id": f"v2_int_{scenario_index:02d}_{index:04d}",
                "scenario": scenario, "payment_exists": payment_exists,
                "merchant_matches": merchant_matches, "payment_status": status,
                "payment_currency": currency, "request_currency": request_currency,
                "captured_amount_paise": captured, "successful_refunds_paise": refunded,
                "active_reservations_paise": reserved,
                "refundable_balance_paise": captured - refunded - reserved,
                "requested_amount_paise": requested, "expected_result": expected,
                "lane": "DETERMINISTIC_PAYMENT_INTEGRITY",
            })
    return pd.DataFrame(rows)


def generate_v2(config: V2DataConfig, output_dir: Path | None = None, lock_dir: Path | None = None) -> dict[str, Any]:
    output = output_dir or config.output_dir
    lock = lock_dir or config.lock_dir
    if (output / "metadata.json").exists() or (lock / "manifest.json").exists():
        raise FileExistsError("v2 benchmark paths are immutable once generated")
    if file_sha256(config.source_archive) != config.source_sha256:
        raise ValueError("v2 UCI source hash does not match preregistration")
    output.mkdir(parents=True, exist_ok=False)
    lock.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(config.primary_seed)
    customers, orders, _payments, _order_items, _transaction_events, foundation = build_uci_foundation(
        config.source_archive, config.n_orders, config.primary_seed, 100.0
    )
    orders = orders.sort_values(["ordered_at", "order_id"]).reset_index(drop=True)
    selected_indices = np.sort(rng.choice(len(orders), config.n_ml_claims, replace=False))
    selected = orders.iloc[selected_indices].copy().reset_index(drop=True)
    selected["delivered_at"] = pd.to_datetime(selected["delivered_at"], utc=True)
    delays = rng.uniform(6, 24 * 21, len(selected))
    selected["decision_time"] = selected["delivered_at"] + pd.to_timedelta(delays, unit="h")
    selected = selected.sort_values(["decision_time", "order_id"]).reset_index(drop=True)
    selected["partition"] = _partitions(config)

    customer_frequency = selected["customer_id"].map(selected["customer_id"].value_counts()).astype(float)
    fast_signal = rng.beta(2.0, 5.0, len(selected))
    returnless = rng.random(len(selected)) < (0.12 + 0.18 * fast_signal)
    evidence = rng.random(len(selected)) > (0.22 + 0.18 * fast_signal)
    entity_signal = rng.beta(1.5, 5.0, len(selected))
    repeated_signal = rng.beta(1.8, 4.5, len(selected))
    peer_signal = np.log1p(customer_frequency) / np.log1p(max(2.0, customer_frequency.max()))
    latent = (
        0.9 * fast_signal + 0.8 * returnless.astype(float) + 0.7 * (~evidence).astype(float)
        + 0.8 * entity_signal + 0.7 * repeated_signal + 0.6 * peer_signal
        + rng.normal(0, 0.75, len(selected))
    )
    reasons = np.array(["missing", "damaged", "not_received", "size", "other"])
    categories = selected["category"].astype(str).to_numpy(copy=True)
    reason_values = rng.choice(reasons, len(selected), p=[0.25, 0.28, 0.20, 0.15, 0.12])
    hard = np.zeros(len(selected), dtype=bool)
    for partition in PARTITIONS:
        partition_index = np.flatnonzero(selected["partition"].to_numpy() == partition)
        count = round(len(partition_index) * config.hard_legitimate_fraction)
        hard[rng.choice(partition_index, count, replace=False)] = True
    labels = np.zeros(len(selected), dtype=bool)
    for partition in PARTITIONS:
        candidates = np.flatnonzero((selected["partition"].to_numpy() == partition) & ~hard)
        count = round(int((selected["partition"] == partition).sum()) * config.abuse_prevalence)
        weights = np.exp(np.clip(latent[candidates] - latent[candidates].mean(), -2.5, 2.5))
        weights /= weights.sum()
        labels[rng.choice(candidates, count, replace=False, p=weights)] = True
    scenario = np.full(len(selected), "routine_legitimate", dtype=object)
    abuse_positions = np.flatnonzero(labels)
    for position, index in enumerate(abuse_positions):
        scenario[index] = ABUSE_SCENARIOS[position % len(ABUSE_SCENARIOS)]
    hard_positions = np.flatnonzero(hard)
    for position, index in enumerate(hard_positions):
        scenario[index] = HARD_LEGITIMATE[position % len(HARD_LEGITIMATE)]
    reason_values[scenario == "genuine_product_defect_burst"] = "damaged"
    reason_values[scenario == "missing_evidence"] = "other"
    evidence[scenario == "missing_evidence"] = False
    evidence[scenario == "verifier_unavailable"] = False
    returnless[scenario == "high_value_loyal_customer"] = False
    categories[scenario == "genuine_product_defect_burst"] = "simulated_defect_batch"

    captured = selected["gross_amount_paise"].astype("int64").clip(lower=100)
    prior_fraction = np.where(rng.random(len(selected)) < 0.18, rng.uniform(0.05, 0.35, len(selected)), 0)
    previously_refunded = np.floor(captured.to_numpy() * prior_fraction).astype("int64")
    remaining = captured.to_numpy() - previously_refunded
    claimed_fraction = rng.uniform(0.08, 0.92, len(selected))
    claimed = np.maximum(1, np.floor(remaining * claimed_fraction).astype("int64"))
    quantity = np.maximum(
        1,
        np.floor(
            selected["item_count"].to_numpy() * rng.uniform(0.2, 1.0, len(selected))
        ).astype(int),
    )
    quantity = np.minimum(quantity, selected["item_count"].astype(int).to_numpy())
    decision_times = pd.to_datetime(selected["decision_time"], utc=True)
    outcome_times = decision_times + pd.to_timedelta(
        config.outcome_delay_days + rng.integers(0, 10, len(selected)), unit="D"
    )
    verification_results = np.empty(len(selected), dtype=object)
    for index in range(len(selected)):
        if scenario[index] == "verifier_unavailable":
            verification_results[index] = "verifier_unavailable"
        elif scenario[index] == "missing_evidence":
            verification_results[index] = "evidence_inconclusive"
        elif labels[index]:
            verification_results[index] = rng.choice(
                ["consistent", "inconsistent", "evidence_inconclusive"], p=[0.18, 0.68, 0.14]
            )
        else:
            verification_results[index] = rng.choice(
                ["consistent", "inconsistent", "evidence_inconclusive"], p=[0.88, 0.04, 0.08]
            )

    requests = pd.DataFrame({
        "refund_request_id": [f"v2_rr_{i:07d}" for i in range(len(selected))],
        "order_id": selected["order_id"].astype(str),
        "payment_id": [f"v2_pay_{i:07d}" for i in range(len(selected))],
        "customer_id": selected["customer_id"].astype(str),
        "decision_time": [_iso(value) for value in decision_times],
        "reason_code": reason_values,
        "claimed_refund_amount_paise": claimed,
        "gateway_authorized_execution_amount_paise": claimed,
        "requested_quantity": quantity,
        "returnless_requested": returnless,
        "evidence_provided": evidence,
        "is_refund_abuse_simulated": labels,
        "simulation_scenario": scenario,
        "outcome_available_at": [_iso(value) for value in outcome_times],
        "verification_result_simulated": verification_results,
        "partition": selected["partition"],
        "lane": "ML_ELIGIBLE_AMBIGUOUS_CLAIM",
    })
    snapshots = pd.DataFrame({
        "refund_request_id": requests["refund_request_id"],
        "payment_id": requests["payment_id"], "merchant_id": "v2_benchmark_merchant",
        "order_id": requests["order_id"], "currency": "INR",
        "captured_amount_paise": captured.to_numpy(),
        "amount_refunded_paise": previously_refunded,
        "active_reservation_amount_paise": 0,
        "refundable_balance_paise": remaining,
        "payment_status": "captured", "payment_exists": True,
        "merchant_matches": True, "currency_matches": True,
        "snapshot_as_of": requests["decision_time"], "source": "SIMULATED_PAYMENT_LEDGER",
    })
    prior_entries = snapshots.loc[snapshots["amount_refunded_paise"] > 0].copy()
    ledger = pd.DataFrame({
        "refund_ledger_entry_id": [f"v2_prior_{i:07d}" for i in range(len(prior_entries))],
        "payment_id": prior_entries["payment_id"].to_numpy(),
        "merchant_id": "v2_benchmark_merchant",
        "amount_paise": prior_entries["amount_refunded_paise"].to_numpy(),
        "currency": "INR", "status": "succeeded",
        "created_at": [
            _iso(pd.Timestamp(value) - timedelta(days=7)) for value in prior_entries["snapshot_as_of"]
        ],
        "processed_at": [
            _iso(pd.Timestamp(value) - timedelta(days=6)) for value in prior_entries["snapshot_as_of"]
        ],
        "idempotency_key": [f"v2_prior_key_{i:07d}" for i in range(len(prior_entries))],
    })
    selected_orders = selected.copy()
    selected_orders["category"] = categories
    shared_device = rng.random(len(selected_orders)) < 0.12
    shared_address = rng.random(len(selected_orders)) < 0.08
    selected_orders.loc[shared_device, "device_id"] = [
        f"v2_shared_device_{value % 240:04d}"
        for value in np.flatnonzero(shared_device)
    ]
    selected_orders.loc[shared_address, "shipping_address_id"] = [
        f"v2_shared_address_{value % 160:04d}"
        for value in np.flatnonzero(shared_address)
    ]
    order_columns = [
        "order_id", "customer_id", "ordered_at", "delivered_at", "currency",
        "gross_amount_paise", "item_count", "category", "shipping_address_id", "device_id",
    ]
    selected_orders = selected_orders[order_columns]
    selected_customer_ids = set(requests["customer_id"])
    selected_customers = customers.loc[customers["customer_id"].isin(selected_customer_ids)].copy()
    integrity = _integrity_cases(config)

    final_mask = requests["partition"] == "final_test"
    truth_columns = [
        "refund_request_id", "is_refund_abuse_simulated", "simulation_scenario",
        "outcome_available_at", "verification_result_simulated",
    ]
    final_truth = requests.loc[final_mask, truth_columns].copy()
    truth_path = lock / "final_truth.csv"
    final_truth.to_csv(truth_path, index=False, lineterminator="\n")
    lock_manifest = {
        "schema_version": "2.0", "state": "SEALED", "support": int(final_mask.sum()),
        "truth_sha256": file_sha256(truth_path), "opened_at": None,
        "preregistration_sha256": file_sha256(Path("evidence/v2/preregistration.json")),
    }
    (lock / "manifest.json").write_text(
        json.dumps(lock_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for column in truth_columns[1:]:
        requests[column] = requests[column].astype("object")
        requests.loc[final_mask, column] = pd.NA

    challenge_count = round(int(final_mask.sum()) * config.cold_start_fraction)
    final_ids = requests.loc[final_mask, "refund_request_id"].to_numpy()
    challenge_ids = rng.choice(final_ids, challenge_count, replace=False)
    cold_start = pd.DataFrame({
        "refund_request_id": challenge_ids,
        "challenge_customer_id": [f"v2_cold_{i:06d}" for i in range(challenge_count)],
        "mapping_purpose": "separate_cold_start_challenge",
    })
    tables = {
        "customers": selected_customers, "orders": selected_orders,
        "refund_requests": requests, "payment_snapshots": snapshots,
        "refund_ledger_entries": ledger, "integrity_cases": integrity,
        "cold_start_mapping": cold_start,
    }
    fingerprints: dict[str, str] = {}
    for name, frame in tables.items():
        path = output / f"{name}.csv"
        frame.to_csv(path, index=False, lineterminator="\n")
        fingerprints[name] = file_sha256(path)
    split_summary = []
    for partition in PARTITIONS:
        subset = requests.loc[requests["partition"] == partition]
        split_summary.append({
            "partition": partition, "support": len(subset),
            "prevalence": None if partition == "final_test" else float(
                subset["is_refund_abuse_simulated"].astype(bool).mean()
            ),
            "start_at": str(subset["decision_time"].min()),
            "end_at": str(subset["decision_time"].max()),
        })
    metadata = {
        "schema_version": "2.0", "benchmark_version": config.benchmark_version,
        "generator_version": config.generator_version, "generator_seed": config.primary_seed,
        "source_archive_sha256": config.source_sha256,
        "preregistration_sha256": file_sha256(Path("evidence/v2/preregistration.json")),
        "split_summary": split_summary, "fingerprints": fingerprints,
        "transformed_data_sha256": object_sha256(fingerprints),
        "lane_support": {
            "deterministic_payment_integrity": len(integrity),
            "ml_eligible_ambiguous_claims": len(requests),
        },
        "ml_eligibility": {
            "all_payment_exists": bool(snapshots["payment_exists"].all()),
            "all_merchant_matches": bool(snapshots["merchant_matches"].all()),
            "all_captured": bool((snapshots["payment_status"] == "captured").all()),
            "all_currency_matches": bool(snapshots["currency_matches"].all()),
            "all_positive_claims": bool((requests["claimed_refund_amount_paise"] > 0).all()),
            "all_within_refundable_balance": bool((
                requests["claimed_refund_amount_paise"].to_numpy()
                <= snapshots["refundable_balance_paise"].to_numpy()
            ).all()),
            "complete_prior_refund_history": True,
        },
        "field_provenance": {
            "source_transaction_derived": [
                "customer_id", "ordered_at", "gross_amount_paise", "item_count", "country"
            ],
            "deterministically_derived": [
                "order_id", "category", "delivered_at", "account_status", "created_at"
            ],
            "simulated_operational": [
                "refund_request_id", "payment_id", "merchant_id", "decision_time",
                "reason_code", "claimed_refund_amount_paise",
                "gateway_authorized_execution_amount_paise", "requested_quantity",
                "returnless_requested", "evidence_provided", "outcome_available_at",
                "verification_result_simulated", "captured_amount_paise",
                "amount_refunded_paise", "active_reservation_amount_paise",
                "refundable_balance_paise", "payment_status", "payment_exists",
                "merchant_matches", "currency_matches", "snapshot_as_of", "source",
                "shipping_address_id", "device_id"
            ],
            "simulated_labels_or_scenario_metadata": [
                "is_refund_abuse_simulated", "simulation_scenario"
            ],
        },
        "foundation": foundation,
        "final_lock": lock_manifest,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata
