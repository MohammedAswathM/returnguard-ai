from pathlib import Path

import pytest

from returnguard.config import DataConfig, SplitRatios
from returnguard.data.generator import generate_benchmark


@pytest.fixture
def small_config(tmp_path: Path) -> DataConfig:
    return DataConfig(
        schema_version="1.0", seed=12345, source_mode="fixture", uci_gbp_to_inr=100.0,
        n_orders=1200,
        n_refund_requests=400, abuse_prevalence=0.09, hard_legitimate_fraction=0.18,
        start_at="2024-01-01T00:00:00Z", duration_days=365, outcome_delay_days=14,
        split_ratios=SplitRatios(
            train=0.6, calibration=0.1, policy_selection=0.1, final_test=0.2,
        ),
        cold_start_fraction=0.1, output_dir=tmp_path / "data",
    )


@pytest.fixture
def generated_data(small_config: DataConfig) -> Path:
    generate_benchmark(small_config)
    return small_config.output_dir
