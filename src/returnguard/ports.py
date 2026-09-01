from datetime import datetime
from typing import Protocol, TypeVar

from returnguard.domain.schemas import AuditEvent

T = TypeVar("T")


class Clock(Protocol):
    def now(self) -> datetime: ...


class Repository(Protocol[T]):
    def add(self, item: T) -> None: ...
    def get(self, item_id: str) -> T | None: ...


class AppendOnlyAuditRepository(Protocol):
    def append(self, event: AuditEvent) -> None: ...
    def list_for_request(self, refund_request_id: str) -> tuple[AuditEvent, ...]: ...

