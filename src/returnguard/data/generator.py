from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from returnguard.config import DataConfig
from returnguard.data.final_lock import seal_final_truth
from returnguard.data.fingerprints import file_sha256, object_sha256
from returnguard.data.provenance import field_provenance
from returnguard.data.splitter import assign_chronological_partitions, summarize_splits
from returnguard.data.uci import build_uci_foundation
from returnguard.data.validation import validate_uci_archive

TABLES = (
    "customers", "orders", "order_items", "payments", "transaction_events",
    "refund_requests", "verification_events", "risk_decisions", "audit_events",
)
HARD_LEGITIMATE = (
    "shared_household", "loyal_high_volume", "product_defect_burst", "apparel_size_fit",
    "carrier_incident", "legitimate_high_value_damage", "first_order_return",
    "consistent_evidence_rescue", "missing_evidence_inconclusive",
)
ABUSE_SCENARIOS = (
    "serial_claims", "high_value_concentration", "reason_repetition", "fast_after_delivery",
    "returnless_exploitation", "linked_accounts", "amount_mismatch", "verification_evasion",
)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _empty_table(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _fixture_foundation(
    config: DataConfig, rng: np.random.Generator, start: datetime
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    customer_count = max(250, config.n_orders // 8)
    customer_ids = np.array([f"cus_{i:06d}" for i in range(customer_count)])
    countries = np.array(["GB", "GB", "GB", "IE", "FR", "DE"])
    created_offsets = rng.integers(1, 365, customer_count)
    customers = pd.DataFrame({
        "customer_id": customer_ids,
        "created_at": [_iso(start - timedelta(days=int(x))) for x in created_offsets],
        "country": rng.choice(countries, customer_count),
        "account_status": "active",
    })
    popularity = rng.zipf(1.8, customer_count).astype(float)
    popularity /= popularity.sum()
    order_customer = rng.choice(customer_ids, config.n_orders, p=popularity)
    order_seconds = np.sort(rng.integers(0, config.duration_days * 86400, config.n_orders))
    categories = np.array(["apparel", "electronics", "home", "beauty", "grocery", "other"])
    order_rows: list[dict[str, Any]] = []
    payment_rows: list[dict[str, Any]] = []
    for i, seconds in enumerate(order_seconds):
        ordered = start + timedelta(seconds=int(seconds))
        delivered = ordered + timedelta(days=int(rng.integers(2, 9)))
        item_count = int(rng.integers(1, 7))
        gross = int(max(10_000, rng.lognormal(11.2, 0.75)))
        customer_id = str(order_customer[i])
        order_id = f"ord_{i:07d}"
        order_rows.append({
            "order_id": order_id, "customer_id": customer_id, "razorpay_order_id": "",
            "ordered_at": _iso(ordered), "delivered_at": _iso(delivered), "currency": "INR",
            "gross_amount_paise": gross, "item_count": item_count,
            "category": str(rng.choice(categories)),
            "shipping_address_id": f"addr_{int(customer_id[-6:]) // 2:06d}",
            "device_id": f"dev_{int(customer_id[-6:]) // 3:06d}",
        })
        payment_rows.append({
            "payment_id": f"pay_{i:07d}", "razorpay_payment_id": "", "order_id": order_id,
            "paid_amount_paise": gross, "status": "captured",
            "paid_at": _iso(ordered + timedelta(minutes=2)),
        })
    orders = pd.DataFrame(order_rows)
    order_items = pd.DataFrame({
        "order_item_id": [f"item_{index:07d}_0000" for index in range(len(orders))],
        "order_id": orders["order_id"], "product_id": "fixture_product",
        "description_present": True, "quantity": orders["item_count"],
        "unit_price_source_gbp": 0.0,
    })
    transaction_events = _empty_table([
        "transaction_event_id", "customer_id", "event_time", "event_type", "source_invoice_id",
    ])
    return customers, orders, pd.DataFrame(payment_rows), order_items, transaction_events, {
        "fixture_disclosure": "Transaction foundation is synthetic."
    }


def generate_benchmark(config: DataConfig, output_dir: Path | None = None) -> dict[str, Any]:
    output = output_dir or config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    start = datetime.fromisoformat(config.start_at.replace("Z", "+00:00")).astimezone(UTC)
    if config.source_mode == "uci":
        if config.uci_archive is None:  # Defensive narrowing beyond config validation.
            raise ValueError("uci_archive is required")
        source_validation = validate_uci_archive(
            config.uci_archive, output / "source_validation.json"
        )
        if source_validation["status"] != "VALIDATED_OFFICIAL_SOURCE":
            raise ValueError("UCI source failed official identity validation")
        customers, orders, payments, order_items, transaction_events, foundation_notes = build_uci_foundation(
            config.uci_archive, config.n_orders, config.seed, config.uci_gbp_to_inr
        )
    else:
        source_validation = None
        customers, orders, payments, order_items, transaction_events, foundation_notes = (
            _fixture_foundation(config, rng, start)
        )

    eligible = np.arange(config.n_orders)
    replace = config.n_refund_requests > len(eligible)
    selected = rng.choice(eligible, config.n_refund_requests, replace=replace)
    selected.sort()
    exact_abuse = np.zeros(config.n_refund_requests, dtype=bool)
    exact_abuse[: round(config.n_refund_requests * config.abuse_prevalence)] = True
    rng.shuffle(exact_abuse)
    legitimate_indices = np.flatnonzero(~exact_abuse)
    hard_count = min(len(legitimate_indices), round(config.n_refund_requests * config.hard_legitimate_fraction))
    hard_indices = set(int(x) for x in rng.choice(legitimate_indices, hard_count, replace=False))
    hard_scenario_by_index = {
        index: HARD_LEGITIMATE[position % len(HARD_LEGITIMATE)]
        for position, index in enumerate(sorted(hard_indices))
    }
    first_order_ids = set(
        orders.sort_values("ordered_at").groupby("customer_id", sort=False).first()["order_id"]
    )
    first_order_candidate = next(
        (
            index for index in sorted(hard_indices)
            if str(orders.iloc[int(selected[index])]["order_id"]) in first_order_ids
        ),
        None,
    )
    if first_order_candidate is not None:
        hard_scenario_by_index[first_order_candidate] = "first_order_return"
    customer_order_counts = cast(pd.Series, orders["customer_id"].value_counts())
    loyal_cutoff = float(customer_order_counts.quantile(0.9))
    loyal_customers = set(customer_order_counts.loc[customer_order_counts >= loyal_cutoff].index)
    loyal_candidate = next(
        (
            index for index in sorted(hard_indices)
            if str(orders.iloc[int(selected[index])]["customer_id"]) in loyal_customers
        ),
        None,
    )
    if loyal_candidate is not None:
        hard_scenario_by_index[loyal_candidate] = "loyal_high_volume"
    reasons = np.array(["missing", "damaged", "not_received", "size", "other"])
    request_rows: list[dict[str, Any]] = []
    verification_rows: list[dict[str, Any]] = []
    shared_household_count = 0
    for request_i, order_i in enumerate(selected):
        order = orders.iloc[int(order_i)]
        delivered = datetime.fromisoformat(str(order["delivered_at"]).replace("Z", "+00:00"))
        abuse = bool(exact_abuse[request_i])
        scenario = str(rng.choice(ABUSE_SCENARIOS)) if abuse else "routine_legitimate"
        if not abuse and request_i in hard_scenario_by_index:
            scenario = hard_scenario_by_index[request_i]
        if scenario == "shared_household":
            household = shared_household_count // 2
            orders.at[int(order_i), "shipping_address_id"] = f"sim_household_addr_{household:04d}"
            orders.at[int(order_i), "device_id"] = f"sim_household_dev_{household:04d}"
            shared_household_count += 1
        if scenario == "product_defect_burst":
            orders.at[int(order_i), "category"] = "simulated_defect_batch"
        fast = abuse and scenario in {"fast_after_delivery", "serial_claims", "returnless_exploitation"}
        delay_hours = float(rng.uniform(1, 30) if fast else rng.uniform(12, 24 * 21))
        requested = delivered + timedelta(hours=delay_hours)
        amount_ratio = float(rng.uniform(0.7, 1.12) if abuse else rng.uniform(0.2, 1.0))
        if scenario == "amount_mismatch":
            amount_ratio = float(rng.uniform(1.02, 1.25))
        if scenario in {"legitimate_high_value_damage", "loyal_high_volume"}:
            amount_ratio = float(rng.uniform(0.75, 1.0))
        amount = max(1, round(int(order["gross_amount_paise"]) * amount_ratio))
        returnless = bool(rng.random() < (0.72 if abuse else 0.20))
        if scenario in {"apparel_size_fit", "legitimate_high_value_damage"}:
            returnless = False
        evidence = bool(rng.random() < (0.45 if abuse else 0.72))
        if scenario in {"consistent_evidence_rescue", "legitimate_high_value_damage"}:
            evidence = True
        reason = str(rng.choice(reasons, p=[0.28, 0.28, 0.22, 0.10, 0.12]))
        if scenario == "apparel_size_fit":
            reason = "size"
        elif scenario in {"product_defect_burst", "legitimate_high_value_damage"}:
            reason = "damaged"
        elif scenario == "carrier_incident":
            reason = "not_received"
        request_id = f"rr_{request_i:07d}"
        outcome_at = requested + timedelta(days=config.outcome_delay_days + int(rng.integers(0, 10)))
        request_rows.append({
            "refund_request_id": request_id, "order_id": str(order["order_id"]),
            "customer_id": str(order["customer_id"]), "requested_at": _iso(requested),
            "reason_code": reason, "requested_amount_paise": amount,
            "requested_quantity": int(rng.integers(1, int(order["item_count"]) + 1)),
            "returnless_requested": returnless, "evidence_provided": evidence,
            "channel": str(rng.choice(["web", "app", "support"])),
            "is_refund_abuse_simulated": abuse, "simulation_scenario": scenario,
            "outcome_available_at": _iso(outcome_at),
        })
        if scenario == "missing_evidence_inconclusive":
            result = "inconclusive"
        elif scenario in {
            "shared_household", "loyal_high_volume", "product_defect_burst",
            "carrier_incident", "legitimate_high_value_damage", "first_order_return",
            "consistent_evidence_rescue",
        }:
            result = "consistent"
        elif abuse and rng.random() < 0.70:
            result = "inconsistent"
        elif not abuse and rng.random() < 0.86:
            result = "consistent"
        else:
            result = str(rng.choice(["consistent", "inconsistent", "inconclusive"] ))
        verification_rows.append({
            "verification_id": f"ver_{request_i:07d}", "refund_request_id": request_id,
            "verification_type": "order_integrity", "requested_at": _iso(requested + timedelta(hours=1)),
            "completed_at": _iso(requested + timedelta(hours=6)), "result": result,
            "result_codes": json.dumps([f"SIMULATED_{result.upper()}"]),
            "estimated_friction_cost_paise": 2500, "evidence_hash": "",
        })
    requests = pd.DataFrame(request_rows)
    ratios = config.split_ratios
    requests = assign_chronological_partitions(
        requests, (ratios.train, ratios.calibration, ratios.policy_selection, ratios.final_test)
    )
    split_map = requests[["refund_request_id", "partition"]]
    verifications = pd.DataFrame(verification_rows).merge(split_map, on="refund_request_id", how="left")
    requests, verifications, final_lock = seal_final_truth(
        requests, verifications, output.parent / f"{output.name}_locked_final"
    )

    train_customers = set(requests.loc[requests["partition"] == "train", "customer_id"])
    final_rows = requests.loc[requests["partition"] == "final_test"]
    challenge_count = max(1, round(len(final_rows) * config.cold_start_fraction))
    challenge_sample = final_rows.sample(n=challenge_count, random_state=config.seed)
    cold_start = pd.DataFrame({
        "refund_request_id": challenge_sample["refund_request_id"],
        "primary_customer_id": challenge_sample["customer_id"],
        "challenge_customer_id": [f"cold_{i:06d}" for i in range(challenge_count)],
        "mapping_purpose": "separate_cold_start_challenge",
    })
    assert not set(cold_start["challenge_customer_id"]) & train_customers

    tables = {
        "customers": customers, "orders": orders, "order_items": order_items,
        "payments": payments, "transaction_events": transaction_events,
        "refund_requests": requests, "verification_events": verifications,
        "risk_decisions": _empty_table([
            "decision_id", "refund_request_id", "model_version", "raw_score",
            "calibrated_probability", "decision_stage", "recommended_action", "reason_codes",
            "feature_snapshot_hash", "created_at",
        ]),
        "audit_events": _empty_table([
            "event_id", "refund_request_id", "actor_type", "actor_id", "event_type",
            "before_state", "after_state", "reason", "created_at",
        ]),
    }
    fingerprints: dict[str, str] = {}
    for name, table in tables.items():
        path = output / f"{name}.csv"
        table.to_csv(path, index=False, lineterminator="\n")
        fingerprints[name] = file_sha256(path)
    cold_start.to_csv(output / "cold_start_mapping.csv", index=False, lineterminator="\n")
    fingerprints["cold_start_mapping"] = file_sha256(output / "cold_start_mapping.csv")
    summaries = [asdict(item) for item in summarize_splits(requests, "is_refund_abuse_simulated")]
    development_scenario_support = {
        str(name): int(count)
        for name, count in requests.loc[
            requests["partition"] != "final_test", "simulation_scenario"
        ].value_counts().sort_index().items()
    }
    metadata = {
        "schema_version": config.schema_version,
        "generator_seed": config.seed,
        "source_mode": config.source_mode,
        "source_disclosure": (
            "Deterministic synthetic fixture; not UCI and not production data."
            if config.source_mode == "fixture" else "UCI-derived transaction base with simulated labels."
        ),
        "config": config.model_dump(mode="json"),
        "generator_config_fingerprint_sha256": object_sha256(config.model_dump(mode="json")),
        "split_summary": summaries,
        "split_fingerprint_sha256": object_sha256(summaries),
        "development_scenario_support": development_scenario_support,
        "fingerprints": fingerprints,
        "transformed_data_fingerprint_sha256": object_sha256(fingerprints),
        "label_disclosure": "is_refund_abuse_simulated is generated and is not a UCI cancellation label.",
        "cold_start_disclosure": "Mapping is separate and does not alter the primary final-test rows.",
        "final_lock": final_lock,
        "foundation_notes": foundation_notes,
        "source_validation": source_validation,
        "field_provenance": field_provenance({
            name: [str(column) for column in table.columns] for name, table in tables.items()
        }),
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata
