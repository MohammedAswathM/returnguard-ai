import pandas as pd


def require_fit_partition(frame: pd.DataFrame, allowed: frozenset[str] = frozenset({"train"})) -> None:
    if "partition" not in frame:
        raise ValueError("fit data must carry its partition provenance")
    observed = frozenset(str(value) for value in frame["partition"].unique())
    forbidden = observed - allowed
    if forbidden:
        raise ValueError(f"fit rejected non-training partitions: {sorted(forbidden)}")

