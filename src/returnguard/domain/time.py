from datetime import UTC, datetime


def require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    if value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("timestamp must use UTC")
    return value


def utc_now() -> datetime:
    return datetime.now(UTC)

