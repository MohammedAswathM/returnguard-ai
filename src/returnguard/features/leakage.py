import re
from collections.abc import Iterable

LEAKAGE_PATTERN = re.compile(
    r"(^|_)(target|label|abuse|scenario|seed|outcome|verification_result|recommended_action|"
    r"future|inspection)(_|$)", re.IGNORECASE,
)
ALLOWED_MATURED_HISTORY = frozenset({"prior_matured_adverse_outcome_count"})


def reject_leakage_columns(columns: Iterable[str]) -> None:
    rejected = sorted(
        column
        for column in columns
        if LEAKAGE_PATTERN.search(column) and column not in ALLOWED_MATURED_HISTORY
    )
    if rejected:
        raise ValueError(f"leakage columns are forbidden: {', '.join(rejected)}")
