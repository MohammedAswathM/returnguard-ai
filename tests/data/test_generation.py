import json
from pathlib import Path

import pandas as pd

from returnguard.config import DataConfig
from returnguard.data.generator import TABLES, generate_benchmark
from returnguard.data.splitter import PARTITIONS, assert_non_overlapping


def test_generation_is_reproducible(small_config: DataConfig, tmp_path: Path) -> None:
    first = generate_benchmark(small_config, tmp_path / "first")
    second = generate_benchmark(small_config, tmp_path / "second")
    assert first["fingerprints"] == second["fingerprints"]


def test_canonical_tables_and_metadata_exist(generated_data: Path) -> None:
    assert all((generated_data / f"{table}.csv").exists() for table in TABLES)
    metadata = json.loads((generated_data / "metadata.json").read_text())
    assert metadata["source_mode"] == "fixture"
    assert "not a UCI cancellation label" in metadata["label_disclosure"]
    final_summary = next(row for row in metadata["split_summary"] if row["partition"] == "final_test")
    assert final_summary["prevalence"] is None
    for table in TABLES:
        columns = pd.read_csv(generated_data / f"{table}.csv", nrows=0).columns
        assert set(columns) == set(metadata["field_provenance"][table])


def test_splits_are_ordered_and_non_overlapping(generated_data: Path) -> None:
    requests = pd.read_csv(generated_data / "refund_requests.csv")
    assert tuple(requests["partition"].drop_duplicates()) == PARTITIONS
    assert_non_overlapping(requests)
    assert set(requests.groupby("partition").size().index) == set(PARTITIONS)


def test_cold_start_mapping_does_not_modify_primary_rows(generated_data: Path) -> None:
    requests = pd.read_csv(generated_data / "refund_requests.csv")
    mapping = pd.read_csv(generated_data / "cold_start_mapping.csv")
    final_ids = set(requests.loc[requests["partition"] == "final_test", "refund_request_id"])
    assert set(mapping["refund_request_id"]) <= final_ids
    assert not set(mapping["challenge_customer_id"]) & set(requests["customer_id"])


def test_difficult_legitimate_and_simulated_labels_are_disclosed(generated_data: Path) -> None:
    requests = pd.read_csv(generated_data / "refund_requests.csv")
    development = requests.loc[requests["partition"] != "final_test"].copy()
    development["is_refund_abuse_simulated"] = development[
        "is_refund_abuse_simulated"
    ].astype(bool)
    legitimate = development.loc[~development["is_refund_abuse_simulated"]]
    assert (legitimate["simulation_scenario"] != "routine_legitimate").any()
    assert not development.loc[
        development["is_refund_abuse_simulated"], "simulation_scenario"
    ].isna().any()


def test_final_truth_is_redacted_from_development_tables(generated_data: Path) -> None:
    requests = pd.read_csv(generated_data / "refund_requests.csv")
    final_rows = requests.loc[requests["partition"] == "final_test"]
    assert final_rows["is_refund_abuse_simulated"].isna().all()
    assert final_rows["simulation_scenario"].isna().all()
    assert final_rows["outcome_available_at"].isna().all()
    manifest_path = generated_data.parent / f"{generated_data.name}_locked_final" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["state"] == "SEALED"
    assert manifest["opened_at"] is None
    assert manifest["support"] == len(final_rows)


def test_hard_legitimate_cases_have_observable_conditions(generated_data: Path) -> None:
    requests = pd.read_csv(generated_data / "refund_requests.csv")
    requests = requests.loc[requests["partition"] != "final_test"]
    orders = pd.read_csv(generated_data / "orders.csv")
    verifications = pd.read_csv(generated_data / "verification_events.csv")
    cases = requests.merge(orders, on=["order_id", "customer_id"]).merge(
        verifications[["refund_request_id", "result"]], on="refund_request_id"
    )
    required = {
        "shared_household", "loyal_high_volume", "product_defect_burst", "carrier_incident",
        "legitimate_high_value_damage", "first_order_return", "missing_evidence_inconclusive",
    }
    assert required <= set(cases["simulation_scenario"])
    shared = cases.loc[cases["simulation_scenario"] == "shared_household"]
    assert shared.groupby("device_id")["customer_id"].nunique().max() >= 2
    defect = cases.loc[cases["simulation_scenario"] == "product_defect_burst"]
    assert set(defect["category"]) == {"simulated_defect_batch"}
    assert set(defect["reason_code"]) == {"damaged"}
    carrier = cases.loc[cases["simulation_scenario"] == "carrier_incident"]
    assert set(carrier["reason_code"]) == {"not_received"}
    missing = cases.loc[cases["simulation_scenario"] == "missing_evidence_inconclusive"]
    assert set(missing["result"]) == {"inconclusive"}
