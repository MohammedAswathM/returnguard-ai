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
    legitimate = requests.loc[~requests["is_refund_abuse_simulated"]]
    assert (legitimate["simulation_scenario"] != "routine_legitimate").any()
    assert not requests.loc[requests["is_refund_abuse_simulated"], "simulation_scenario"].isna().any()
