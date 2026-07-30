"""Route-cost and workload-tier schedulers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from swarm_inference.config.models import (
    HealthStatus,
    SchedulerMode,
    StageDefinition,
    StageReplica,
    WorkerCapability,
    WorkloadClass,
)
from swarm_inference.exceptions import NoValidRouteError


@dataclass(frozen=True, slots=True)
class RouteCandidate:
    stage_id: int
    worker_id: str
    predicted_execution_ms: float
    predicted_queue_ms: float
    predicted_network_ms: float
    reliability_penalty_ms: float
    total_cost_ms: float
    accepted: bool
    rejection_reason: str | None = None


@dataclass(slots=True)
class RouteDecision:
    scheduler: SchedulerMode
    workload_class: WorkloadClass
    candidates: list[RouteCandidate] = field(default_factory=list)
    selected_route: list[StageReplica] = field(default_factory=list)
    actual_elapsed_ms: float | None = None

    @property
    def predicted_total_ms(self) -> float:
        selected = {replica.worker_id for replica in self.selected_route}
        return sum(
            candidate.total_cost_ms
            for candidate in self.candidates
            if candidate.accepted and candidate.worker_id in selected
        )

    @property
    def prediction_error_ms(self) -> float | None:
        if self.actual_elapsed_ms is None:
            return None
        return self.actual_elapsed_ms - self.predicted_total_ms


def predicted_network_capacity(stage_capacities: Iterable[float]) -> float:
    values = list(stage_capacities)
    if not values:
        return 0.0
    if any(value < 0 for value in values):
        raise ValueError("stage capacities cannot be negative")
    return min(values)


def route_cost(
    *,
    stage: StageDefinition,
    replica: StageReplica,
    worker: WorkerCapability,
    payload_bytes: int,
    previous_worker: WorkerCapability | None,
    workload_class: WorkloadClass,
) -> RouteCandidate:
    backend_key = worker.backend.value
    execution_ms = stage.estimated_execution_ms.get(backend_key)
    if execution_ms is None:
        execution_ms = (
            1000.0 / replica.measured_service_rate
            if replica.measured_service_rate > 0
            else float("inf")
        )
    queue_ms = (
        worker.current_queue_depth * execution_ms / max(worker.max_concurrent_stage_operations, 1)
    )
    if previous_worker is None:
        network_ms = worker.coordinator_latency_ms + (
            payload_bytes / max(worker.download_bandwidth_bytes_s, 1.0) * 1000
        )
    else:
        bandwidth = min(
            previous_worker.upload_bandwidth_bytes_s,
            worker.download_bandwidth_bytes_s,
        )
        network_ms = (
            previous_worker.coordinator_latency_ms
            + worker.coordinator_latency_ms
            + payload_bytes / max(bandwidth, 1.0) * 1000
        )
    reliability_weight = {
        WorkloadClass.INTERACTIVE: 3.0,
        WorkloadClass.STANDARD: 1.5,
        WorkloadClass.BACKGROUND: 0.5,
    }[workload_class]
    reliability_penalty = (1.0 - worker.reliability_score) * execution_ms * reliability_weight
    rejection_reason: str | None = None
    accepted = True
    if replica.health != HealthStatus.HEALTHY:
        accepted = False
        rejection_reason = f"replica health is {replica.health.value}"
    elif replica.load_status != "loaded":
        accepted = False
        rejection_reason = f"replica load status is {replica.load_status}"
    elif stage.required_memory_bytes > worker.effective_memory_bytes:
        accepted = False
        rejection_reason = "stage exceeds worker memory limit"
    elif workload_class == WorkloadClass.INTERACTIVE and worker.reliability_score < 0.9:
        accepted = False
        rejection_reason = "reliability below interactive threshold"
    total = execution_ms + queue_ms + network_ms + reliability_penalty
    return RouteCandidate(
        stage_id=stage.stage_id,
        worker_id=worker.worker_id,
        predicted_execution_ms=execution_ms,
        predicted_queue_ms=queue_ms,
        predicted_network_ms=network_ms,
        reliability_penalty_ms=reliability_penalty,
        total_cost_ms=total,
        accepted=accepted,
        rejection_reason=rejection_reason,
    )


class Scheduler:
    """Choose an ordered stage route without forcing slow nodes onto fast traffic."""

    def __init__(self, mode: SchedulerMode) -> None:
        self.mode = mode

    def choose_route(
        self,
        *,
        stages: list[StageDefinition],
        replicas: list[StageReplica],
        workers: list[WorkerCapability],
        payload_bytes: int,
        workload_class: WorkloadClass,
    ) -> RouteDecision:
        worker_by_id = {worker.worker_id: worker for worker in workers}
        decision = RouteDecision(scheduler=self.mode, workload_class=workload_class)
        previous: WorkerCapability | None = None
        for stage in sorted(stages, key=lambda item: item.stage_id):
            stage_replicas = sorted(
                [replica for replica in replicas if replica.stage_id == stage.stage_id],
                key=lambda replica: replica.worker_id,
            )
            stage_candidates: list[tuple[StageReplica, RouteCandidate]] = []
            for replica in stage_replicas:
                worker = worker_by_id.get(replica.worker_id)
                if worker is None:
                    continue
                candidate = route_cost(
                    stage=stage,
                    replica=replica,
                    worker=worker,
                    payload_bytes=payload_bytes,
                    previous_worker=previous,
                    workload_class=workload_class,
                )
                decision.candidates.append(candidate)
                if candidate.accepted:
                    stage_candidates.append((replica, candidate))
            if not stage_candidates:
                reasons = [
                    f"{candidate.worker_id}: {candidate.rejection_reason}"
                    for candidate in decision.candidates
                    if candidate.stage_id == stage.stage_id
                ]
                raise NoValidRouteError(
                    f"no valid replica for stage {stage.stage_id}; " + "; ".join(reasons)
                )
            if self.mode == SchedulerMode.STATIC:
                selected, _ = stage_candidates[0]
            else:
                selected, _ = min(stage_candidates, key=lambda item: item[1].total_cost_ms)
            decision.selected_route.append(selected.model_copy(deep=True))
            previous = worker_by_id[selected.worker_id]
        return decision
