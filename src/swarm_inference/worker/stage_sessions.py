"""Persistent, topology-scoped stage session registry."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from swarm_inference.execution.interfaces import StageExecutor
from swarm_inference.protocol.stage_worker import StageSessionStatus


@dataclass(slots=True)
class StageSessionRecord:
    topology_id: str
    session_id: str
    model_revision: str
    route_generation: int
    stage_id: int
    cache_position: int
    opened_monotonic_ns: int
    last_operation_monotonic_ns: int
    cancelled: bool = False


class StageSessionRegistry:
    """Own session identity and cache-position state for one resident stage."""

    def __init__(self, *, maximum_sessions: int) -> None:
        if maximum_sessions <= 0:
            raise ValueError("maximum stage session count must be positive")
        self.maximum_sessions = maximum_sessions
        self._sessions: dict[tuple[str, str], StageSessionRecord] = {}
        self._lock = threading.RLock()

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def open(
        self,
        executor: StageExecutor,
        *,
        topology_id: str,
        session_id: str,
        model_revision: str,
        route_generation: int,
        stage_id: int,
    ) -> StageSessionRecord:
        key = (topology_id, session_id)
        with self._lock:
            if key in self._sessions:
                raise ValueError(
                    f"stage session {session_id!r} is already open in topology {topology_id!r}"
                )
            if len(self._sessions) >= self.maximum_sessions:
                raise RuntimeError(
                    f"maximum active stage session count {self.maximum_sessions} reached"
                )
            executor.open_session(session_id)
            now = time.monotonic_ns()
            record = StageSessionRecord(
                topology_id=topology_id,
                session_id=session_id,
                model_revision=model_revision,
                route_generation=route_generation,
                stage_id=stage_id,
                cache_position=0,
                opened_monotonic_ns=now,
                last_operation_monotonic_ns=now,
            )
            self._sessions[key] = record
            return record

    def require(
        self,
        *,
        topology_id: str,
        session_id: str,
        model_revision: str,
        route_generation: int,
        stage_id: int,
        cache_position_start: int | None = None,
    ) -> StageSessionRecord:
        key = (topology_id, session_id)
        with self._lock:
            try:
                record = self._sessions[key]
            except KeyError as exc:
                raise ValueError(
                    f"stage session {session_id!r} is not open in topology {topology_id!r}"
                ) from exc
            if record.model_revision != model_revision:
                raise ValueError("stage session model revision mismatch")
            if record.route_generation != route_generation:
                raise ValueError("stage session route generation mismatch")
            if record.stage_id != stage_id:
                raise ValueError("stage session is bound to another stage")
            if cache_position_start is not None and cache_position_start != record.cache_position:
                raise ValueError(
                    f"stage cache position {cache_position_start} does not match session "
                    f"position {record.cache_position}"
                )
            return record

    def update_cache_position(
        self,
        *,
        topology_id: str,
        session_id: str,
        new_position: int,
    ) -> None:
        if new_position < 0:
            raise ValueError("stage cache position cannot be negative")
        with self._lock:
            record = self._sessions[(topology_id, session_id)]
            if new_position < record.cache_position:
                raise ValueError("stage execution moved the cache position backwards")
            record.cache_position = new_position
            record.last_operation_monotonic_ns = time.monotonic_ns()

    def close(
        self,
        executor: StageExecutor,
        *,
        topology_id: str,
        session_id: str,
    ) -> int:
        key = (topology_id, session_id)
        with self._lock:
            if key not in self._sessions:
                raise ValueError("stage session is not open")
            released = executor.close_session(session_id)
            del self._sessions[key]
            return released

    def cancel(
        self,
        executor: StageExecutor,
        *,
        topology_id: str,
        session_id: str,
    ) -> int:
        key = (topology_id, session_id)
        with self._lock:
            try:
                record = self._sessions[key]
            except KeyError as exc:
                raise ValueError("stage session is not open") from exc
            record.cancelled = True
            released = executor.cancel_session(session_id)
            del self._sessions[key]
            return released

    def cancel_all(self, executor: StageExecutor) -> int:
        released = 0
        with self._lock:
            for key, record in list(self._sessions.items()):
                record.cancelled = True
                released += executor.cancel_session(record.session_id)
                del self._sessions[key]
        return released

    def statuses(self, executor: StageExecutor) -> list[StageSessionStatus]:
        with self._lock:
            return [
                StageSessionStatus(
                    topology_id=record.topology_id,
                    session_id=record.session_id,
                    model_revision=record.model_revision,
                    route_generation=record.route_generation,
                    stage_id=record.stage_id,
                    cache_position=record.cache_position,
                    kv_cache_bytes=executor.kv_cache_bytes(record.session_id),
                    opened_monotonic_ns=record.opened_monotonic_ns,
                    last_operation_monotonic_ns=record.last_operation_monotonic_ns,
                    cancelled=record.cancelled,
                )
                for record in sorted(
                    self._sessions.values(), key=lambda item: (item.topology_id, item.session_id)
                )
            ]


__all__ = ["StageSessionRecord", "StageSessionRegistry"]
