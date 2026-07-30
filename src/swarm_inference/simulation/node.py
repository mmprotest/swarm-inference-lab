"""Queueing and service model for one simulated worker."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from swarm_inference.config.models import OperationKind
from swarm_inference.exceptions import BackpressureError


@dataclass(frozen=True, slots=True)
class PlannedOperation:
    worker_id: str
    stage_id: int
    queued_at_s: float
    started_at_s: float
    completed_at_s: float
    queueing_s: float
    execution_s: float
    operation: OperationKind


@dataclass(slots=True)
class SimWorker:
    worker_id: str
    profile_name: str
    memory_bytes: int
    compute_rate_layers_s: float
    reliability: float
    max_concurrent_operations: int
    queue_capacity: int
    corrupt: bool = False
    healthy: bool = True
    quarantined: bool = False
    reputation: float = 1.0
    assigned_stage_id: int | None = None
    assigned_stage_bytes: int = 0
    busy_time_s: float = 0.0
    bytes_sent: int = 0
    bytes_received: int = 0
    operations: int = 0
    failures: int = 0
    audit_count: int = 0
    audit_disagreements: int = 0
    tokens_contributed: int = 0
    service_times_s: list[float] = field(default_factory=list)
    _slots: list[float] = field(default_factory=list, repr=False)
    _completion_times: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self._slots:
            self._slots = [0.0] * self.max_concurrent_operations
            heapq.heapify(self._slots)

    def assign(self, *, stage_id: int, required_bytes: int) -> None:
        if required_bytes > self.memory_bytes:
            raise ValueError(
                f"stage {stage_id} requires {required_bytes} bytes but worker "
                f"{self.worker_id} has {self.memory_bytes}"
            )
        self.assigned_stage_id = stage_id
        self.assigned_stage_bytes = required_bytes

    def queue_depth(self, now_s: float) -> int:
        self._completion_times = [
            timestamp for timestamp in self._completion_times if timestamp > now_s
        ]
        return len(self._completion_times)

    def predicted_completion(
        self,
        *,
        now_s: float,
        stage_layers: int,
        work_multiplier: float,
    ) -> float:
        available = min(self._slots)
        execution = stage_layers * work_multiplier / self.compute_rate_layers_s
        return max(now_s, available) + execution

    def plan(
        self,
        *,
        now_s: float,
        stage_id: int,
        stage_layers: int,
        operation: OperationKind,
        work_multiplier: float,
        coordinator_overhead_s: float,
    ) -> PlannedOperation:
        if not self.healthy or self.quarantined:
            raise BackpressureError(f"worker {self.worker_id} is unavailable")
        if self.queue_depth(now_s) >= self.queue_capacity:
            raise BackpressureError(
                f"worker {self.worker_id} queue reached capacity {self.queue_capacity}"
            )
        available = heapq.heappop(self._slots)
        started = max(now_s, available)
        execution = (
            stage_layers * work_multiplier / self.compute_rate_layers_s + coordinator_overhead_s
        )
        completed = started + execution
        heapq.heappush(self._slots, completed)
        self._completion_times.append(completed)
        self.busy_time_s += execution
        self.operations += 1
        self.service_times_s.append(execution)
        return PlannedOperation(
            worker_id=self.worker_id,
            stage_id=stage_id,
            queued_at_s=now_s,
            started_at_s=started,
            completed_at_s=completed,
            queueing_s=started - now_s,
            execution_s=execution,
            operation=operation,
        )
