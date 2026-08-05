"""Atomic, auditable coordinator trust store for worker fingerprints."""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from swarm_inference.config.models import StrictModel
from swarm_inference.exceptions import IntegrityError
from swarm_inference.filesystem import replace_atomically

TRUST_STORE_DOCUMENT_TYPE: Literal["swarm-trusted-workers"] = "swarm-trusted-workers"
TRUST_STORE_FORMAT_VERSION: Literal[1] = 1
_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")


def normalize_fingerprint(value: str) -> str:
    normalized = value.strip().lower()
    if normalized.startswith("sha256:"):
        normalized = normalized.removeprefix("sha256:")
    if not _FINGERPRINT.fullmatch(normalized):
        raise ValueError("worker fingerprint must be 64 lowercase SHA-256 hex characters")
    return normalized


class TrustedWorker(StrictModel):
    fingerprint: str
    label: str | None = None
    notes: str | None = None
    trusted_at: str
    updated_at: str


class TrustedWorkersDocument(StrictModel):
    document_type: Literal["swarm-trusted-workers"] = TRUST_STORE_DOCUMENT_TYPE
    format_version: Literal[1] = TRUST_STORE_FORMAT_VERSION
    workers: list[TrustedWorker] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_fingerprints(self) -> TrustedWorkersDocument:
        normalized = [normalize_fingerprint(item.fingerprint) for item in self.workers]
        if len(normalized) != len(set(normalized)):
            raise ValueError("trusted worker fingerprints must be unique")
        if normalized != sorted(normalized):
            raise ValueError("trusted worker fingerprints must be deterministically sorted")
        return self


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        replace_atomically(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class WorkerTrustStore:
    """Reload a valid immutable snapshot for every security decision."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.audit_path = self.path.with_name(f"{self.path.stem}.audit.jsonl")
        self._lock = threading.RLock()

    def load(self) -> TrustedWorkersDocument:
        with self._lock:
            if not self.path.exists():
                return TrustedWorkersDocument()
            if not self.path.is_file():
                raise IntegrityError(f"worker trust store is not a file: {self.path}")
            try:
                return TrustedWorkersDocument.model_validate_json(
                    self.path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as exc:
                raise IntegrityError(f"invalid worker trust store {self.path}: {exc}") from exc

    def fingerprints(self) -> tuple[str, ...]:
        return tuple(item.fingerprint for item in self.load().workers)

    def contains(self, fingerprint: str) -> bool:
        normalized = normalize_fingerprint(fingerprint)
        return normalized in self.fingerprints()

    def trust(
        self,
        fingerprint: str,
        *,
        label: str | None = None,
        notes: str | None = None,
    ) -> tuple[TrustedWorker, bool]:
        normalized = normalize_fingerprint(fingerprint)
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        with self._lock:
            document = self.load()
            existing = next(
                (item for item in document.workers if item.fingerprint == normalized),
                None,
            )
            added = existing is None
            record = TrustedWorker(
                fingerprint=normalized,
                label=label if label is not None else (existing.label if existing else None),
                notes=notes if notes is not None else (existing.notes if existing else None),
                trusted_at=existing.trusted_at if existing else now,
                updated_at=now,
            )
            workers = [item for item in document.workers if item.fingerprint != normalized]
            workers.append(record)
            updated = document.model_copy(
                update={"workers": sorted(workers, key=lambda item: item.fingerprint)}
            )
            _atomic_write(self.path, updated.model_dump_json(indent=2) + "\n")
            self._audit("trust" if added else "trust_update", record)
            return record, added

    def untrust(self, fingerprint: str) -> bool:
        normalized = normalize_fingerprint(fingerprint)
        with self._lock:
            document = self.load()
            existing = next(
                (item for item in document.workers if item.fingerprint == normalized),
                None,
            )
            if existing is None:
                return False
            updated = document.model_copy(
                update={
                    "workers": [item for item in document.workers if item.fingerprint != normalized]
                }
            )
            _atomic_write(self.path, updated.model_dump_json(indent=2) + "\n")
            self._audit("untrust", existing)
            return True

    def _audit(self, event: str, record: TrustedWorker) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "event": event,
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "fingerprint": record.fingerprint,
            "label": record.label,
            "notes": record.notes,
            "trust_store": str(self.path),
        }
        with self.audit_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


__all__ = [
    "TRUST_STORE_DOCUMENT_TYPE",
    "TRUST_STORE_FORMAT_VERSION",
    "TrustedWorker",
    "TrustedWorkersDocument",
    "WorkerTrustStore",
    "normalize_fingerprint",
]
