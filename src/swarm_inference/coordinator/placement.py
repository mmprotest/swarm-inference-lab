"""Memory-safe bottleneck-aware stage placement."""

from __future__ import annotations

from dataclasses import dataclass, field

from swarm_inference.config.models import (
    HealthStatus,
    StageDefinition,
    StageReplica,
    WorkerCapability,
)
from swarm_inference.coordinator.scheduler import predicted_network_capacity
from swarm_inference.exceptions import InsufficientStageCoverageError


@dataclass(frozen=True, slots=True)
class PlacementDecision:
    worker_id: str
    stage_id: int | None
    service_rate: float
    marginal_throughput: float
    accepted: bool
    reason: str


@dataclass(slots=True)
class PlacementPlan:
    replicas: list[StageReplica] = field(default_factory=list)
    decisions: list[PlacementDecision] = field(default_factory=list)

    def stage_capacities(self) -> dict[int, float]:
        capacities: dict[int, float] = {}
        for replica in self.replicas:
            if replica.health == HealthStatus.HEALTHY:
                capacities[replica.stage_id] = (
                    capacities.get(replica.stage_id, 0.0) + replica.measured_service_rate
                )
        return capacities


def estimate_worker_stage_rate(
    worker: WorkerCapability,
    stage: StageDefinition,
) -> float:
    matching = [
        benchmark
        for benchmark in worker.stage_benchmarks
        if benchmark.stage_id in {None, stage.stage_id}
        and benchmark.operation.value == "decode"
        and benchmark.mean_ms > 0
    ]
    if matching:
        best = min(matching, key=lambda item: item.mean_ms)
        return 1000.0 / best.mean_ms
    estimated = stage.estimated_execution_ms.get(worker.backend.value)
    if estimated and estimated > 0:
        return 1000.0 / estimated
    return 0.0


def marginal_capacity_gain(
    current_capacities: dict[int, float],
    *,
    stage_id: int,
    added_service_rate: float,
) -> float:
    if added_service_rate <= 0:
        return 0.0
    before = predicted_network_capacity(current_capacities.values())
    updated = dict(current_capacities)
    updated[stage_id] = updated.get(stage_id, 0.0) + added_service_rate
    after = predicted_network_capacity(updated.values())
    return max(0.0, after - before)


def place_replicas(
    *,
    stages: list[StageDefinition],
    workers: list[WorkerCapability],
    shard_hashes: dict[int, str] | None = None,
) -> PlacementPlan:
    """Assign each worker to at most one stage.

    Initial coverage is selected by scarcity: stages with fewer eligible
    workers are placed first. Remaining workers are admitted only for a
    strictly positive predicted bottleneck-capacity gain.
    """

    if not stages:
        raise ValueError("at least one stage is required")
    plan = PlacementPlan()
    remaining = {worker.worker_id: worker for worker in workers}
    capacities = {stage.stage_id: 0.0 for stage in stages}
    eligibility: dict[int, list[WorkerCapability]] = {
        stage.stage_id: [
            worker
            for worker in workers
            if stage.required_memory_bytes <= worker.effective_memory_bytes
            and estimate_worker_stage_rate(worker, stage) > 0
        ]
        for stage in stages
    }
    for stage in sorted(stages, key=lambda item: (len(eligibility[item.stage_id]), item.stage_id)):
        candidates = [
            worker for worker in eligibility[stage.stage_id] if worker.worker_id in remaining
        ]
        if not candidates:
            raise InsufficientStageCoverageError(
                f"no unassigned compatible worker can host stage {stage.stage_id}"
            )
        selected = max(
            candidates,
            key=lambda worker: (
                estimate_worker_stage_rate(worker, stage),
                worker.reliability_score,
                worker.worker_id,
            ),
        )
        rate = estimate_worker_stage_rate(selected, stage)
        plan.replicas.append(
            StageReplica(
                stage_id=stage.stage_id,
                worker_id=selected.worker_id,
                shard_hash=(shard_hashes or {}).get(stage.stage_id, "synthetic"),
                load_status="loaded",
                warm=True,
                measured_service_rate=rate,
                health=HealthStatus.HEALTHY,
                endpoint=selected.endpoint,
            )
        )
        capacities[stage.stage_id] += rate
        plan.decisions.append(
            PlacementDecision(
                worker_id=selected.worker_id,
                stage_id=stage.stage_id,
                service_rate=rate,
                marginal_throughput=rate,
                accepted=True,
                reason="required initial stage coverage",
            )
        )
        remaining.pop(selected.worker_id)

    # Plan spare workers as a batch. This matters when several stages are tied
    # bottlenecks: the first replica alone has zero min-capacity gain, while a
    # balanced set of replicas has a real gain. Leave-one-out pruning then
    # removes workers whose contribution to the final pipeline is non-positive.
    tentative: list[tuple[WorkerCapability, StageDefinition, float]] = []
    rejected_no_fit: list[WorkerCapability] = []
    projected = dict(capacities)
    for worker in sorted(
        remaining.values(),
        key=lambda item: (
            -max(
                (
                    estimate_worker_stage_rate(item, stage)
                    for stage in stages
                    if stage.required_memory_bytes <= item.effective_memory_bytes
                ),
                default=0.0,
            ),
            item.worker_id,
        ),
    ):
        choices = [
            (stage, estimate_worker_stage_rate(worker, stage))
            for stage in stages
            if stage.required_memory_bytes <= worker.effective_memory_bytes
            and estimate_worker_stage_rate(worker, stage) > 0
        ]
        if not choices:
            rejected_no_fit.append(worker)
            continue
        selected_stage, rate = min(
            choices,
            key=lambda item: (
                projected[item[0].stage_id] + item[1],
                projected[item[0].stage_id],
                item[0].stage_id,
            ),
        )
        tentative.append((worker, selected_stage, rate))
        projected[selected_stage.stage_id] += rate

    base_capacity = predicted_network_capacity(capacities.values())
    changed = True
    while changed and tentative:
        changed = False
        full_capacity = predicted_network_capacity(projected.values())
        for worker, stage, rate in list(tentative):
            without = dict(projected)
            without[stage.stage_id] -= rate
            leave_one_out_gain = full_capacity - predicted_network_capacity(without.values())
            if leave_one_out_gain <= 0:
                tentative.remove((worker, stage, rate))
                projected[stage.stage_id] -= rate
                plan.decisions.append(
                    PlacementDecision(
                        worker_id=worker.worker_id,
                        stage_id=None,
                        service_rate=rate,
                        marginal_throughput=0.0,
                        accepted=False,
                        reason="non-positive marginal predicted throughput",
                    )
                )
                changed = True
                break

    final_capacity = predicted_network_capacity(projected.values())
    group_gain = max(0.0, final_capacity - base_capacity)
    for worker, selected_stage, rate in tentative:
        without = dict(projected)
        without[selected_stage.stage_id] -= rate
        contribution = final_capacity - predicted_network_capacity(without.values())
        if contribution <= 0:
            contribution = group_gain / max(len(tentative), 1)
        plan.replicas.append(
            StageReplica(
                stage_id=selected_stage.stage_id,
                worker_id=worker.worker_id,
                shard_hash=(shard_hashes or {}).get(selected_stage.stage_id, "synthetic"),
                load_status="loaded",
                warm=True,
                measured_service_rate=rate,
                health=HealthStatus.HEALTHY,
                endpoint=worker.endpoint,
            )
        )
        plan.decisions.append(
            PlacementDecision(
                worker_id=worker.worker_id,
                stage_id=selected_stage.stage_id,
                service_rate=rate,
                marginal_throughput=max(0.0, contribution),
                accepted=True,
                reason="positive leave-one-out bottleneck-capacity contribution",
            )
        )
    for worker in rejected_no_fit:
        plan.decisions.append(
            PlacementDecision(
                worker_id=worker.worker_id,
                stage_id=None,
                service_rate=0.0,
                marginal_throughput=0.0,
                accepted=False,
                reason="no memory-compatible stage with a measured service rate",
            )
        )
    return plan
