from pathlib import Path

import pandas as pd
import pytest

from returnguard.features.engine import build_point_in_time_features
from returnguard.features.leakage import reject_leakage_columns
from returnguard.features.registry import load_feature_registry
from returnguard.models.guard import require_fit_partition


@pytest.mark.parametrize("name", [
    "is_refund_abuse_simulated", "simulation_scenario", "generator_seed",
    "future_action", "verification_result", "inspection_outcome",
])
def test_leakage_columns_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="leakage"):
        reject_leakage_columns(["requested_amount_paise", name])


def test_final_rows_cannot_enter_fit() -> None:
    with pytest.raises(ValueError, match="final_test"):
        require_fit_partition(pd.DataFrame({"partition": ["train", "final_test"]}))


def test_final_partition_cannot_enter_development_feature_replay(generated_data: Path) -> None:
    with pytest.raises(ValueError, match="sealed"):
        build_point_in_time_features(
            generated_data,
            load_feature_registry(Path("configs/features.yaml")),
            partitions=frozenset({"final_test"}),
        )
