"""Coordinator-owned worker and stage-replica registry."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from swarm_inference.config.models import (
    HealthStatus,
    StageReplica,
    WorkerCapability,
)
from swarm_inference.exceptions import ConfigurationError


@dataclass(slots=True)
class RegistryRecord:
    capability: WorkerCapability
    registered_monotonic_s: float
    last_heartbeat_monotonic_s: float
    measured_registration: bool


class WorkerRegistry:
    """Thread-safe registry with heartbeat expiry and explicit replicas."""

    def __init__(self, *, heartbeat_timeout_s: float = 15.0) -> None:
        if heartbeat_timeout_s <= 0:
            raise ValueError("heartbeat_timeout_s must be positive")
        self._heartbeat_timeout_s = heartbeat_timeout_s
        self._workers: dict[str, RegistryRecord] = {}
        self._replicas: dict[tuple[int, str], StageReplica] = {}
        self._lock = threading.RLock()

    def register(
        self,
        capability: WorkerCapability,
        *,
        benchmark_verified: bool,
        now: float | None = None,
    ) -> None:
        if not benchmark_verified:
            raise ConfigurationError(
                f"worker {capability.worker_id} registration lacks a measured benchmark"
            )
        timestamp = time.monotonic() if now is None else now
        with self._lock:
            self._workers[capability.worker_id] = RegistryRecord(
                capability=capability,
                registered_monotonic_s=timestamp,
                last_heartbeat_monotonic_s=timestamp,
                measured_registration=True,
            )

    def heartbeat(
        self,
        worker_id: str,
        *,
        queue_depth: int,
        assignments: list[int],
        now: float | None = None,
    ) -> None:
        timestamp = time.monotonic() if now is None else now
        with self._lock:
            try:
                record = self._workers[worker_id]
            except KeyError as exc:
                raise ConfigurationError(f"heartbeat from unknown worker {worker_id}") from exc
            record.last_heartbeat_monotonic_s = timestamp
            record.capability.current_queue_depth = queue_depth
            record.capability.current_shard_assignments = list(assignments)
            for key, replica in self._replicas.items():
                if key[1] == worker_id:
                    replica.queue_depth = queue_depth
                    if replica.health != HealthStatus.QUARANTINED:
                        replica.health = HealthStatus.HEALTHY

    def add_replica(self, replica: StageReplica) -> None:
        with self._lock:
            if replica.worker_id not in self._workers:
                raise ConfigurationError(
                    f"cannot assign stage {replica.stage_id} to unknown worker {replica.worker_id}"
                )
            self._replicas[(replica.stage_id, replica.worker_id)] = replica

    def remove_worker(self, worker_id: str) -> None:
        with self._lock:
            self._workers.pop(worker_id, None)
            for key in [key for key in self._replicas if key[1] == worker_id]:
                self._replicas.pop(key, None)

    def expire(self, *, now: float | None = None) -> list[str]:
        timestamp = time.monotonic() if now is None else now
        expired: list[str] = []
        with self._lock:
            for worker_id, record in self._workers.items():
                age = timestamp - record.last_heartbeat_monotonic_s
                if age > self._heartbeat_timeout_s:
                    expired.append(worker_id)
                    for key, replica in self._replicas.items():
                        if key[1] == worker_id and replica.health != HealthStatus.QUARANTINED:
                            replica.health = HealthStatus.UNHEALTHY
        return sorted(expired)

    def quarantine(self, worker_id: str) -> None:
        with self._lock:
            for key, replica in self._replicas.items():
                if key[1] == worker_id:
                    replica.health = HealthStatus.QUARANTINED

    def mark_unhealthy(self, worker_id: str) -> None:
        """Exclude a failed worker immediately, without waiting for heartbeat expiry."""

        with self._lock:
            for key, replica in self._replicas.items():
                if key[1] == worker_id and replica.health != HealthStatus.QUARANTINED:
                    replica.health = HealthStatus.UNHEALTHY

    def workers(self) -> list[WorkerCapability]:
        with self._lock:
            return [record.capability.model_copy(deep=True) for record in self._workers.values()]

    def capability(self, worker_id: str) -> WorkerCapability:
        with self._lock:
            try:
                return self._workers[worker_id].capability.model_copy(deep=True)
            except KeyError as exc:
                raise ConfigurationError(f"unknown worker {worker_id}") from exc

    def replicas(self, stage_id: int | None = None) -> list[StageReplica]:
        with self._lock:
            replicas = list(self._replicas.values())
            if stage_id is not None:
                replicas = [replica for replica in replicas if replica.stage_id == stage_id]
            return [replica.model_copy(deep=True) for replica in replicas]
