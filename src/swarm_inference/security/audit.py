"""Durable audit evidence records."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AuditEvidence:
    audit_id: str
    request_id: str
    stage_id: int
    token_position: int
    primary_worker_id: str
    audit_worker_id: str
    input_checksum: str
    primary_output_checksum: str
    audit_output_checksum: str
    agreed: bool
    exact: bool
    maximum_absolute_error: float
    timestamp: str

    @classmethod
    def create(cls, **values: object) -> AuditEvidence:
        return cls(timestamp=datetime.now(UTC).isoformat(), **values)  # type: ignore[arg-type]


class AuditLog:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser().resolve() if path else None
        self.entries: list[AuditEvidence] = []

    def append(self, evidence: AuditEvidence) -> None:
        self.entries.append(evidence)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(asdict(evidence), sort_keys=True) + "\n")
