"""Exact stage-input replay log for cache reconstruction."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from swarm_inference.config.models import OperationKind
from swarm_inference.exceptions import IntegrityError, ReplayUnavailableError
from swarm_inference.protocol.checksums import sha256_bytes


@dataclass(frozen=True, slots=True)
class ReplayEntry:
    request_id: str
    model_revision: str
    stage_id: int
    cache_generation: int
    token_position: int
    operation: OperationKind
    payload: bytes
    checksum: str
    recorded_monotonic_ns: int


class ReplayLog:
    """In-memory replay log with bounded total bytes and deterministic ordering."""

    def __init__(self, *, maximum_bytes: int = 2 * 1024 * 1024 * 1024) -> None:
        if maximum_bytes <= 0:
            raise ValueError("maximum_bytes must be positive")
        self._maximum_bytes = maximum_bytes
        self._total_bytes = 0
        self._entries: dict[tuple[str, str, int, int], list[ReplayEntry]] = {}
        self._lock = threading.RLock()

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    def append(
        self,
        *,
        request_id: str,
        model_revision: str,
        stage_id: int,
        cache_generation: int,
        token_position: int,
        operation: OperationKind,
        payload: bytes,
        recorded_monotonic_ns: int,
    ) -> ReplayEntry:
        checksum = sha256_bytes(payload)
        entry = ReplayEntry(
            request_id=request_id,
            model_revision=model_revision,
            stage_id=stage_id,
            cache_generation=cache_generation,
            token_position=token_position,
            operation=operation,
            payload=bytes(payload),
            checksum=checksum,
            recorded_monotonic_ns=recorded_monotonic_ns,
        )
        key = (request_id, model_revision, stage_id, cache_generation)
        with self._lock:
            if self._total_bytes + len(payload) > self._maximum_bytes:
                raise ReplayUnavailableError(
                    f"replay log capacity {self._maximum_bytes} bytes exceeded"
                )
            entries = self._entries.setdefault(key, [])
            if any(item.token_position == token_position for item in entries):
                raise IntegrityError(
                    f"duplicate replay input for request={request_id} stage={stage_id} "
                    f"generation={cache_generation} position={token_position}"
                )
            entries.append(entry)
            entries.sort(key=lambda item: (item.token_position, item.recorded_monotonic_ns))
            self._total_bytes += len(payload)
        return entry

    def entries_for(
        self,
        *,
        request_id: str,
        model_revision: str,
        stage_id: int,
        cache_generation: int,
        through_token_position: int | None = None,
    ) -> list[ReplayEntry]:
        key = (request_id, model_revision, stage_id, cache_generation)
        with self._lock:
            entries = list(self._entries.get(key, []))
        if not entries:
            raise ReplayUnavailableError(
                f"no replay inputs for request={request_id} stage={stage_id} "
                f"generation={cache_generation}"
            )
        if through_token_position is not None:
            entries = [entry for entry in entries if entry.token_position <= through_token_position]
        expected = list(range(entries[0].token_position, entries[-1].token_position + 1))
        actual = [entry.token_position for entry in entries]
        if actual != expected:
            raise ReplayUnavailableError(
                f"replay log has a position gap for request={request_id} stage={stage_id}: "
                f"{actual!r}"
            )
        for entry in entries:
            if sha256_bytes(entry.payload) != entry.checksum:
                raise IntegrityError(
                    f"replay checksum mismatch at stage={stage_id} position={entry.token_position}"
                )
        return entries

    def delete_request(self, request_id: str) -> int:
        removed = 0
        with self._lock:
            for key in [key for key in self._entries if key[0] == request_id]:
                entries = self._entries.pop(key)
                removed += sum(len(entry.payload) for entry in entries)
            self._total_bytes -= removed
        return removed
