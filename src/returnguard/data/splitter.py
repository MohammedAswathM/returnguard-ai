from dataclasses import dataclass

import pandas as pd

PARTITIONS = ("train", "calibration", "policy_selection", "final_test")
FIT_PARTITIONS = frozenset({"train"})


@dataclass(frozen=True)
class SplitSummary:
    partition: str
    support: int
    prevalence: float | None
    start_at: str
    end_at: str


def assign_chronological_partitions(
    frame: pd.DataFrame, ratios: tuple[float, float, float, float]
) -> pd.DataFrame:
    ordered = frame.sort_values(["requested_at", "refund_request_id"]).reset_index(drop=True).copy()
    n = len(ordered)
    boundaries = [int(n * ratios[0]), int(n * sum(ratios[:2])), int(n * sum(ratios[:3]))]
    labels: list[str] = []
    for index in range(n):
        if index < boundaries[0]:
            labels.append("train")
        elif index < boundaries[1]:
            labels.append("calibration")
        elif index < boundaries[2]:
            labels.append("policy_selection")
        else:
            labels.append("final_test")
    ordered["partition"] = labels
    return ordered


def summarize_splits(frame: pd.DataFrame, label: str) -> list[SplitSummary]:
    result: list[SplitSummary] = []
    for partition in PARTITIONS:
        rows = frame.loc[frame["partition"] == partition]
        result.append(SplitSummary(
            partition=partition,
            support=len(rows),
            prevalence=None if partition == "final_test" else float(rows[label].mean()),
            start_at=str(rows["requested_at"].min()),
            end_at=str(rows["requested_at"].max()),
        ))
    return result


def assert_non_overlapping(frame: pd.DataFrame) -> None:
    previous_end: pd.Timestamp | None = None
    for partition in PARTITIONS:
        times = pd.to_datetime(frame.loc[frame["partition"] == partition, "requested_at"], utc=True)
        if times.empty:
            raise ValueError(f"empty partition: {partition}")
        if previous_end is not None and times.min() < previous_end:
            raise ValueError("chronological partitions overlap")
        previous_end = times.max()
