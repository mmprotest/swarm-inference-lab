from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from swarm_inference.config.models import (
    Backend,
    HealthStatus,
    OperationKind,
    StageBenchmark,
    StageReplica,
    SyntheticModelConfig,
    WorkerCapability,
    WorkloadClass,
)
from swarm_inference.coordinator.reservations import AtomicRouteAllocator
from swarm_inference.exceptions import NoValidRouteError
from swarm_inference.simulation.model import build_synthetic_stages


def _worker(worker_id: str) -> WorkerCapability:
    return WorkerCapability(
        worker_id=worker_id,
        public_key="test",
        hostname="localhost",
        operating_system="test",
        architecture="test",
        backend=Backend.SYNTHETIC,
        cpu_model="test",
        logical_cpu_count=1,
        total_ram_bytes=2**30,
        available_ram_bytes=2**30,
        supported_dtypes=["float32"],
        upload_bandwidth_bytes_s=1e9,
        download_bandwidth_bytes_s=1e9,
        coordinator_latency_ms=0.1,
        reliability_score=1,
        endpoint=f"127.0.0.1:{worker_id.removeprefix('w') or '1'}",
        stage_benchmarks=[
            StageBenchmark(
                worker_class="test",
                operation=OperationKind.DECODE,
                sequence_length=1,
                batch_size=1,
                mean_ms=10,
                p95_ms=10,
                samples=10,
            )
        ],
    )


def _replica(stage_id: int, worker_id: str, rate: float = 100) -> StageReplica:
    return StageReplica(
        stage_id=stage_id,
        worker_id=worker_id,
        shard_hash=f"stage-{stage_id}",
        load_status="loaded",
        warm=True,
        measured_service_rate=rate,
        health=HealthStatus.HEALTHY,
        endpoint=f"127.0.0.1:{worker_id.removeprefix('w') or '1'}",
    )


def _fixture(
    *, rates: tuple[float, float] = (100, 100)
) -> tuple[AtomicRouteAllocator, list, list[StageReplica], list[WorkerCapability]]:
    allocator = AtomicRouteAllocator()
    stages = build_synthetic_stages(
        SyntheticModelConfig(layer_count=2, stage_count=2, bytes_per_layer=1024)
    )
    replicas = [
        _replica(stage, f"w{stage * 2 + replica + 1}", rates[replica])
        for stage in range(2)
        for replica in range(2)
    ]
    workers = [_worker(replica.worker_id) for replica in replicas]
    return allocator, stages, replicas, workers


def _allocate(
    allocator: AtomicRouteAllocator,
    stages: list,
    replicas: list[StageReplica],
    workers: list[WorkerCapability],
    request_id: str,
    *,
    now: float | None = None,
):
    return allocator.allocate(
        request_id=request_id,
        stages=stages,
        replicas=replicas,
        workers=workers,
        token_steps=8,
        activation_bytes=16 * 1024,
        workload_class=WorkloadClass.STANDARD,
        lease_seconds=60,
        now=now,
    )


def test_sixty_four_concurrent_admissions_are_atomically_balanced() -> None:
    allocator, stages, replicas, workers = _fixture()
    with ThreadPoolExecutor(max_workers=16) as pool:
        decisions = list(
            pool.map(
                lambda index: _allocate(
                    allocator,
                    stages,
                    replicas,
                    workers,
                    f"request-{index}",
                ),
                range(64),
            )
        )

    for stage_id in range(2):
        counts = {
            worker_id: sum(
                decision.assignments[stage_id].worker_id == worker_id for decision in decisions
            )
            for worker_id in [f"w{stage_id * 2 + 1}", f"w{stage_id * 2 + 2}"]
        }
        assert max(counts.values()) - min(counts.values()) <= 1
    snapshot = allocator.snapshot()
    assert snapshot["active_route_leases"] == 64
    assert all(state["reserved_requests"] == 32 for state in snapshot["replicas"].values())


def test_reservations_are_immediate_and_release_is_idempotent() -> None:
    allocator, stages, replicas, workers = _fixture()
    first = _allocate(allocator, stages, replicas, workers, "first")
    second = _allocate(allocator, stages, replicas, workers, "second")
    assert first.assignments[0].worker_id != second.assignments[0].worker_id
    assert allocator.release(first.route_id, reason="finished")
    assert not allocator.release(first.route_id, reason="again")
    assert allocator.snapshot()["active_route_leases"] == 1


def test_cancellation_dispatch_failure_and_expiry_release_once() -> None:
    allocator, stages, replicas, workers = _fixture()
    _allocate(allocator, stages, replicas, workers, "cancelled")
    failed = _allocate(allocator, stages, replicas, workers, "failed")
    expired = _allocate(
        allocator,
        stages,
        replicas,
        workers,
        "expired",
        now=10,
    )
    assert allocator.cancel_request("cancelled")
    assert not allocator.cancel_request("cancelled")
    assert allocator.dispatch_failed(failed.route_id)
    assert not allocator.dispatch_failed(failed.route_id)
    assert allocator.reconcile_expired(now=71) == [expired.route_id]
    snapshot = allocator.snapshot()
    assert snapshot["active_route_leases"] == 0
    assert snapshot["reservation_leaks"] == 1


def test_failed_and_non_positive_capacity_replicas_are_excluded() -> None:
    allocator, stages, replicas, workers = _fixture()
    replicas[0] = replicas[0].model_copy(update={"health": HealthStatus.UNHEALTHY})
    replicas[2] = replicas[2].model_copy(update={"measured_service_rate": 0})
    decision = _allocate(allocator, stages, replicas, workers, "healthy-only")
    assert decision.assignments[0].worker_id == replicas[1].worker_id
    assert decision.assignments[1].worker_id == replicas[3].worker_id

    for replica in replicas:
        replica.measured_service_rate = 0
    with pytest.raises(NoValidRouteError):
        _allocate(allocator, stages, replicas, workers, "no-capacity")


def test_unequal_service_rates_receive_proportionate_placement() -> None:
    allocator, stages, replicas, workers = _fixture(rates=(200, 100))
    decisions = [
        _allocate(allocator, stages, replicas, workers, f"weighted-{index}") for index in range(90)
    ]
    for stage_id in range(2):
        fast_id = f"w{stage_id * 2 + 1}"
        slow_id = f"w{stage_id * 2 + 2}"
        fast = sum(decision.assignments[stage_id].worker_id == fast_id for decision in decisions)
        slow = sum(decision.assignments[stage_id].worker_id == slow_id for decision in decisions)
        assert 1.7 <= fast / slow <= 2.3
