"""Serializable simulator event records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class EventRecord:
    sequence: int
    simulated_time_s: float
    event_type: str
    request_id: str | None = None
    worker_id: str | None = None
    stage_id: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
