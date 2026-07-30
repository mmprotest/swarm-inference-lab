"""Coordinator-authoritative atomic route reservations."""

from __future__ import annotations

import math
import statistics
import threading
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from swarm_inference.config.models import (
    HealthStatus,
    StageDefinition,
    StageReplica,
    WorkerCapability,
    WorkloadClass,
)
from swarm_inference.exceptions import NoValidRouteError

ReplicaKey = tuple[int, str]


@dataclass(slots=True)
class ReplicaReservationState:
    stage_id: int
    worker_id: str
    healthy: bool = True
    warm: bool = True
    reserved_requests: int = 0
    in_flight_stage_operations: int = 0
    reserved_token_steps: int = 0
    measured_service_time_ewma_ms: float = 0.0
    measured_service_time_variance_ms2: float = 0.0
    data_plane_queue_depth: int = 0
    data_plane_bytes_in_flight: int = 0
    recent_failures: int = 0
    route_lease_count: int = 0
    last_assignment_sequence: int = -1
    assignment_count: int = 0
    completed_operations: int = 0
    busy_time_ms: float = 0.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "worker_id": self.worker_id,
            "healthy": self.healthy,
            "warm": self.warm,
            "reserved_requests": self.reserved_requests,
            "in_flight_stage_operations": self.in_flight_stage_operations,
            "reserved_token_steps": self.reserved_token_steps,
            "measured_service_time_ewma_ms": self.measured_service_time_ewma_ms,
            "measured_service_time_variance_ms2": (self.measured_service_time_variance_ms2),
            "data_plane_queue_depth": self.data_plane_queue_depth,
            "data_plane_bytes_in_flight": self.data_plane_bytes_in_flight,
            "recent_failures": self.recent_failures,
            "route_lease_count": self.route_lease_count,
            "last_assignment_sequence": self.last_assignment_sequence,
            "assignment_count": self.assignment_count,
            "completed_operations": self.completed_operations,
            "busy_time_ms": self.busy_time_ms,
        }


@dataclass(slots=True)
class RouteLease:
    route_id: str
    request_id: str
    generation: int
    assignments: dict[int, StageReplica]
    reserved_token_steps: int
    created_monotonic_s: float
    expires_monotonic_s: float
    released: bool = False
    release_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ReservationDecision:
    route_id: str
    generation: int
    assignments: dict[int, StageReplica]
    candidate_costs_ms: dict[str, float]
    reservation_time_ms: float
    lease_expiry_monotonic_s: float


class AtomicRouteAllocator:
    """Select every stage and reserve it under one coordinator lock."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._replicas: dict[ReplicaKey, ReplicaReservationState] = {}
        self._leases: dict[str, RouteLease] = {}
        self._request_routes: dict[str, str] = {}
        self._sequence = 0
        self._reservation_leaks = 0

    @staticmethod
    def _replica_key(replica: StageReplica) -> ReplicaKey:
        return (replica.stage_id, replica.worker_id)

    def _state_for(self, replica: StageReplica) -> ReplicaReservationState:
        key = self._replica_key(replica)
        state = self._replicas.get(key)
        initial_service_ms = (
            1000.0 / replica.measured_service_rate
            if replica.measured_service_rate > 0
            else math.inf
        )
        if state is None:
            state = ReplicaReservationState(
                stage_id=replica.stage_id,
                worker_id=replica.worker_id,
                healthy=replica.health == HealthStatus.HEALTHY,
                warm=replica.warm,
                measured_service_time_ewma_ms=initial_service_ms,
                data_plane_queue_depth=replica.queue_depth,
                recent_failures=replica.failure_count,
            )
            self._replicas[key] = state
        else:
            state.healthy = replica.health == HealthStatus.HEALTHY
            state.warm = replica.warm
            state.data_plane_queue_depth = replica.queue_depth
            state.recent_failures = replica.failure_count
            if not math.isfinite(state.measured_service_time_ewma_ms) and math.isfinite(
                initial_service_ms
            ):
                state.measured_service_time_ewma_ms = initial_service_ms
        return state

    def allocate(
        self,
        *,
        request_id: str,
        stages: list[StageDefinition],
        replicas: list[StageReplica],
        workers: list[WorkerCapability],
        token_steps: int,
        activation_bytes: int,
        workload_class: WorkloadClass,
        lease_seconds: float,
        now: float | None = None,
    ) -> ReservationDecision:
        if token_steps <= 0:
            raise ValueError("token_steps must be positive")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        started = time.perf_counter_ns()
        timestamp = time.monotonic() if now is None else now
        worker_by_id = {worker.worker_id: worker for worker in workers}
        with self._lock:
            existing_id = self._request_routes.get(request_id)
            if existing_id is not None:
                existing = self._leases[existing_id]
                if not existing.released:
                    raise ValueError(f"request {request_id} already owns route {existing_id}")
            selected: dict[int, StageReplica] = {}
            candidate_costs: dict[str, float] = {}
            previous_worker: WorkerCapability | None = None
            for stage in sorted(stages, key=lambda item: item.stage_id):
                stage_rates = [
                    replica.measured_service_rate
                    for replica in replicas
                    if replica.stage_id == stage.stage_id
                    and replica.health == HealthStatus.HEALTHY
                    and replica.measured_service_rate > 0
                ]
                homogeneous_stage = bool(stage_rates) and (
                    max(stage_rates) / min(stage_rates) <= 1.5
                )
                homogeneous_service_ms = (
                    statistics.median(1000 / rate for rate in stage_rates)
                    if homogeneous_stage
                    else 0.0
                )
                candidates: list[tuple[float, int, str, StageReplica]] = []
                for replica in replicas:
                    if replica.stage_id != stage.stage_id:
                        continue
                    worker = worker_by_id.get(replica.worker_id)
                    if worker is None or replica.endpoint is None:
                        continue
                    state = self._state_for(replica)
                    if (
                        not state.healthy
                        or replica.load_status != "loaded"
                        or replica.measured_service_rate <= 0
                    ):
                        continue
                    service_ms = (
                        homogeneous_service_ms
                        if homogeneous_stage
                        else state.measured_service_time_ewma_ms
                    )
                    if not math.isfinite(service_ms) or service_ms <= 0:
                        continue
                    authoritative_work = (
                        state.reserved_token_steps
                        + state.in_flight_stage_operations
                        + state.data_plane_queue_depth
                    )
                    queue_cost = authoritative_work * service_ms
                    if previous_worker is None:
                        transfer_ms = worker.coordinator_latency_ms + (
                            activation_bytes / max(worker.download_bandwidth_bytes_s, 1.0) * 1000
                        )
                    else:
                        transfer_ms = (
                            activation_bytes
                            / max(
                                min(
                                    previous_worker.upload_bandwidth_bytes_s,
                                    worker.download_bandwidth_bytes_s,
                                ),
                                1.0,
                            )
                            * 1000
                        )
                    reliability_weight = {
                        WorkloadClass.INTERACTIVE: 3.0,
                        WorkloadClass.STANDARD: 1.5,
                        WorkloadClass.BACKGROUND: 0.5,
                    }[workload_class]
                    reliability_penalty = (
                        1.0 - worker.reliability_score
                    ) * service_ms * reliability_weight + state.recent_failures * service_ms
                    cold_penalty = 0.0 if state.warm else service_ms * 4
                    bytes_penalty = (
                        state.data_plane_bytes_in_flight
                        / max(worker.upload_bandwidth_bytes_s, 1.0)
                        * 1000
                    )
                    cost = (
                        queue_cost
                        + service_ms
                        + transfer_ms
                        + reliability_penalty
                        + cold_penalty
                        + bytes_penalty
                    )
                    candidate_costs[f"{stage.stage_id}:{replica.worker_id}"] = cost
                    candidates.append(
                        (
                            cost,
                            state.last_assignment_sequence,
                            replica.worker_id,
                            replica,
                        )
                    )
                if not candidates:
                    raise NoValidRouteError(
                        f"no positive-capacity healthy replica for stage {stage.stage_id}"
                    )
                _, _, _, winner = min(candidates, key=lambda item: item[:3])
                selected[stage.stage_id] = winner.model_copy(deep=True)
                previous_worker = worker_by_id[winner.worker_id]
            route_id = uuid4().hex
            self._sequence += 1
            for replica in selected.values():
                state = self._state_for(replica)
                state.reserved_requests += 1
                state.reserved_token_steps += token_steps
                state.route_lease_count += 1
                state.assignment_count += 1
                state.last_assignment_sequence = self._sequence
            lease = RouteLease(
                route_id=route_id,
                request_id=request_id,
                generation=1,
                assignments=selected,
                reserved_token_steps=token_steps,
                created_monotonic_s=timestamp,
                expires_monotonic_s=timestamp + lease_seconds,
            )
            self._leases[route_id] = lease
            self._request_routes[request_id] = route_id
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        return ReservationDecision(
            route_id=route_id,
            generation=1,
            assignments={key: value.model_copy(deep=True) for key, value in selected.items()},
            candidate_costs_ms=candidate_costs,
            reservation_time_ms=elapsed_ms,
            lease_expiry_monotonic_s=lease.expires_monotonic_s,
        )

    def release(self, route_id: str, *, reason: str) -> bool:
        """Release exactly once; return whether this call performed the release."""

        with self._lock:
            lease = self._leases.get(route_id)
            if lease is None or lease.released:
                return False
            for replica in lease.assignments.values():
                state = self._replicas[self._replica_key(replica)]
                state.reserved_requests = max(0, state.reserved_requests - 1)
                state.reserved_token_steps = max(
                    0,
                    state.reserved_token_steps - lease.reserved_token_steps,
                )
                state.route_lease_count = max(0, state.route_lease_count - 1)
            lease.released = True
            lease.release_reason = reason
            if self._request_routes.get(lease.request_id) == route_id:
                self._request_routes.pop(lease.request_id, None)
            return True

    def cancel_request(self, request_id: str) -> bool:
        with self._lock:
            route_id = self._request_routes.get(request_id)
        return False if route_id is None else self.release(route_id, reason="cancelled")

    def dispatch_failed(self, route_id: str) -> bool:
        return self.release(route_id, reason="dispatch-failed")

    def reconcile_expired(self, *, now: float | None = None) -> list[str]:
        timestamp = time.monotonic() if now is None else now
        with self._lock:
            expired = [
                route_id
                for route_id, lease in self._leases.items()
                if not lease.released and lease.expires_monotonic_s <= timestamp
            ]
        for route_id in expired:
            if self.release(route_id, reason="lease-expired"):
                self._reservation_leaks += 1
        return sorted(expired)

    def record_operation_start(self, stage_id: int, worker_id: str, payload_bytes: int) -> None:
        with self._lock:
            state = self._replicas[(stage_id, worker_id)]
            state.in_flight_stage_operations += 1
            state.data_plane_bytes_in_flight += max(0, payload_bytes)

    def record_operation_complete(
        self,
        stage_id: int,
        worker_id: str,
        *,
        payload_bytes: int,
        effective_service_ms: float,
        execution_ms: float,
    ) -> None:
        with self._lock:
            state = self._replicas[(stage_id, worker_id)]
            state.in_flight_stage_operations = max(0, state.in_flight_stage_operations - 1)
            state.data_plane_bytes_in_flight = max(
                0, state.data_plane_bytes_in_flight - max(0, payload_bytes)
            )
            state.completed_operations += 1
            state.busy_time_ms += max(0.0, execution_ms)
            sample = max(0.001, effective_service_ms)
            previous = state.measured_service_time_ewma_ms
            if not math.isfinite(previous) or previous <= 0:
                state.measured_service_time_ewma_ms = sample
                state.measured_service_time_variance_ms2 = 0.0
            else:
                alpha = 0.15
                delta = sample - previous
                state.measured_service_time_ewma_ms = previous + alpha * delta
                state.measured_service_time_variance_ms2 = (1 - alpha) * (
                    state.measured_service_time_variance_ms2 + alpha * delta**2
                )

    def record_operation_aborted(
        self,
        stage_id: int,
        worker_id: str,
        payload_bytes: int,
    ) -> None:
        with self._lock:
            state = self._replicas.get((stage_id, worker_id))
            if state is None:
                return
            state.in_flight_stage_operations = max(
                0,
                state.in_flight_stage_operations - 1,
            )
            state.data_plane_bytes_in_flight = max(
                0,
                state.data_plane_bytes_in_flight - max(0, payload_bytes),
            )

    def mark_failed(self, stage_id: int, worker_id: str) -> None:
        with self._lock:
            state = self._replicas.get((stage_id, worker_id))
            if state is not None:
                state.healthy = False
                state.recent_failures += 1

    def replace(
        self,
        route_id: str,
        *,
        stage_id: int,
        replacement: StageReplica,
    ) -> int:
        """Atomically move one reservation and increment the route generation."""

        with self._lock:
            lease = self._leases[route_id]
            if lease.released:
                raise ValueError(f"route {route_id} is already released")
            existing = lease.assignments[stage_id]
            if existing.worker_id == replacement.worker_id:
                return lease.generation
            old_state = self._replicas[self._replica_key(existing)]
            new_state = self._state_for(replacement)
            old_state.reserved_requests = max(0, old_state.reserved_requests - 1)
            old_state.reserved_token_steps = max(
                0, old_state.reserved_token_steps - lease.reserved_token_steps
            )
            old_state.route_lease_count = max(0, old_state.route_lease_count - 1)
            new_state.reserved_requests += 1
            new_state.reserved_token_steps += lease.reserved_token_steps
            new_state.route_lease_count += 1
            new_state.assignment_count += 1
            self._sequence += 1
            new_state.last_assignment_sequence = self._sequence
            lease.assignments[stage_id] = replacement.model_copy(deep=True)
            lease.generation += 1
            return lease.generation

    def replace_failed(
        self,
        route_id: str,
        *,
        stage_id: int,
        candidates: list[StageReplica],
    ) -> tuple[StageReplica, int]:
        """Select and reserve a compatible replacement under the scheduler lock."""

        with self._lock:
            lease = self._leases[route_id]
            if lease.released:
                raise ValueError(f"route {route_id} is already released")
            failed = lease.assignments[stage_id]
            eligible: list[tuple[float, int, str, StageReplica]] = []
            for candidate in candidates:
                if (
                    candidate.stage_id != stage_id
                    or candidate.worker_id == failed.worker_id
                    or candidate.endpoint is None
                    or candidate.health != HealthStatus.HEALTHY
                    or candidate.load_status != "loaded"
                    or candidate.measured_service_rate <= 0
                ):
                    continue
                state = self._state_for(candidate)
                if (
                    not state.healthy
                    or not math.isfinite(state.measured_service_time_ewma_ms)
                    or state.measured_service_time_ewma_ms <= 0
                ):
                    continue
                outstanding = (
                    state.reserved_token_steps
                    + state.in_flight_stage_operations
                    + state.data_plane_queue_depth
                )
                cost = (outstanding + 1) * state.measured_service_time_ewma_ms
                eligible.append(
                    (
                        cost,
                        state.last_assignment_sequence,
                        candidate.worker_id,
                        candidate,
                    )
                )
            if not eligible:
                raise NoValidRouteError(
                    f"no compatible replacement for failed stage {stage_id} "
                    f"worker {failed.worker_id}"
                )
            replacement = min(eligible, key=lambda item: item[:3])[3]
            generation = self.replace(
                route_id,
                stage_id=stage_id,
                replacement=replacement,
            )
            return replacement.model_copy(deep=True), generation

    def lease(self, route_id: str) -> RouteLease:
        with self._lock:
            lease = self._leases[route_id]
            return RouteLease(
                route_id=lease.route_id,
                request_id=lease.request_id,
                generation=lease.generation,
                assignments={
                    key: value.model_copy(deep=True) for key, value in lease.assignments.items()
                },
                reserved_token_steps=lease.reserved_token_steps,
                created_monotonic_s=lease.created_monotonic_s,
                expires_monotonic_s=lease.expires_monotonic_s,
                released=lease.released,
                release_reason=lease.release_reason,
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "replicas": {
                    f"{stage_id}:{worker_id}": state.snapshot()
                    for (stage_id, worker_id), state in sorted(self._replicas.items())
                },
                "active_route_leases": sum(
                    1 for lease in self._leases.values() if not lease.released
                ),
                "released_route_leases": sum(
                    1 for lease in self._leases.values() if lease.released
                ),
                "reservation_leaks": self._reservation_leaks,
            }

    def release_all(self, *, reason: str) -> list[str]:
        with self._lock:
            active = [route_id for route_id, lease in self._leases.items() if not lease.released]
        for route_id in active:
            self.release(route_id, reason=reason)
        return sorted(active)
