from __future__ import annotations

from datetime import UTC, datetime

import pytest

from swarm_inference.config.models import (
    Backend,
    CacheSpec,
    HealthStatus,
    OperationKind,
    StageBenchmark,
    StageDefinition,
    StageReplica,
    TensorSpec,
    WorkerCapability,
    WorkloadClass,
)
from swarm_inference.config.validation import (
    validate_assignment,
    validate_route,
    validate_stage_coverage,
)
from swarm_inference.coordinator.placement import (
    marginal_capacity_gain,
    place_replicas,
)
from swarm_inference.coordinator.scheduler import route_cost
from swarm_inference.exceptions import (
    InsufficientStageCoverageError,
    MemoryLimitExceededError,
    NoValidRouteError,
)
from swarm_inference.simulation.model import partition_layer_ranges


def stage(stage_id: int, required: int = 100) -> StageDefinition:
    return StageDefinition(
        stage_id=stage_id,
        layer_start=stage_id * 2,
        layer_end=stage_id * 2 + 2,
        required_memory_bytes=required,
        input_spec=TensorSpec(dtype="float32", shape=[1, 4]),
        output_spec=TensorSpec(dtype="float32", shape=[1, 4]),
        cache_spec=CacheSpec(bytes_per_token=8),
        estimated_execution_ms={"synthetic": 2.0},
    )


def worker(worker_id: str, *, memory: int = 1000, mean_ms: float = 2.0) -> WorkerCapability:
    return WorkerCapability(
        worker_id=worker_id,
        public_key="test",
        hostname=worker_id,
        operating_system="test",
        architecture="test",
        backend=Backend.SYNTHETIC,
        cpu_model="test",
        logical_cpu_count=1,
        total_ram_bytes=memory,
        available_ram_bytes=memory,
        supported_dtypes=["float32"],
        upload_bandwidth_bytes_s=1_000_000,
        download_bandwidth_bytes_s=1_000_000,
        coordinator_latency_ms=1,
        reliability_score=0.99,
        memory_limit_bytes=memory,
        stage_benchmarks=[
            StageBenchmark(
                worker_class="test",
                operation=OperationKind.DECODE,
                sequence_length=1,
                batch_size=1,
                mean_ms=mean_ms,
                p95_ms=mean_ms,
                samples=3,
                measured=True,
            )
        ],
        last_heartbeat=datetime.now(UTC),
    )


def test_partition_assigns_every_layer_once() -> None:
    ranges = partition_layer_ranges(17, 5)
    assert [layer for start, end in ranges for layer in range(start, end)] == list(range(17))
    assert (
        max(end - start for start, end in ranges) - min(end - start for start, end in ranges) <= 1
    )


def test_memory_limit_enforced() -> None:
    with pytest.raises(MemoryLimitExceededError):
        validate_assignment(stage(0, required=1001), worker("small", memory=1000))


def test_route_cost_includes_queue_network_and_reliability() -> None:
    candidate_worker = worker("w")
    candidate_worker.current_queue_depth = 3
    replica = StageReplica(
        stage_id=0,
        worker_id="w",
        shard_hash="x",
        load_status="loaded",
        warm=True,
        measured_service_rate=500,
        health=HealthStatus.HEALTHY,
    )
    cost = route_cost(
        stage=stage(0),
        replica=replica,
        worker=candidate_worker,
        payload_bytes=1000,
        previous_worker=None,
        workload_class=WorkloadClass.STANDARD,
    )
    assert cost.accepted
    assert cost.predicted_queue_ms > 0
    assert cost.predicted_network_ms > 0
    assert cost.total_cost_ms > cost.predicted_execution_ms


def test_balanced_worker_batch_is_assigned() -> None:
    plan = place_replicas(
        stages=[stage(0), stage(1)],
        workers=[worker(f"w{i}") for i in range(4)],
    )
    assert len(plan.replicas) == 4
    counts = {
        stage_id: sum(replica.stage_id == stage_id for replica in plan.replicas)
        for stage_id in (0, 1)
    }
    assert counts == {0: 2, 1: 2}
    assert all(decision.marginal_throughput > 0 for decision in plan.decisions if decision.accepted)


def test_nonpositive_rate_worker_is_not_selected() -> None:
    slow = worker("slow")
    slow.stage_benchmarks = []
    stages = [stage(0), stage(1)]
    for item in stages:
        item.estimated_execution_ms = {}
    plan = place_replicas(
        stages=stages,
        workers=[worker("w0"), worker("w1"), slow],
    )
    assert all(replica.worker_id != "slow" for replica in plan.replicas)
    rejected = next(item for item in plan.decisions if item.worker_id == "slow")
    assert not rejected.accepted


def test_capacity_gain_never_negative() -> None:
    before = {0: 10.0, 1: 12.0}
    assert marginal_capacity_gain(before, stage_id=0, added_service_rate=3) == 2
    assert marginal_capacity_gain(before, stage_id=1, added_service_rate=3) == 0


def test_stage_coverage_and_route_validation() -> None:
    stages = [stage(0), stage(1)]
    replicas = [
        StageReplica(
            stage_id=index,
            worker_id=f"w{index}",
            shard_hash="x",
            load_status="loaded",
            warm=True,
            health=HealthStatus.HEALTHY,
        )
        for index in range(2)
    ]
    validate_stage_coverage(stages, replicas)
    validate_route(replicas, stages)
    with pytest.raises(NoValidRouteError):
        validate_route(list(reversed(replicas)), stages)
    with pytest.raises(InsufficientStageCoverageError):
        validate_stage_coverage(stages, replicas[:1])
