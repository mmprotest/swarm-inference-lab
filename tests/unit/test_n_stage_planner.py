from __future__ import annotations

import inspect
import time

import pytest

from swarm_inference.cluster.models import NetworkLinkMeasurement
from swarm_inference.config.models import (
    Backend,
    OperationKind,
    StageBenchmark,
    WorkerCapability,
)
from swarm_inference.coordinator.model_catalog import InspectedProductModel
from swarm_inference.coordinator.stage_planner import ProductStagePlanner
from swarm_inference.model.product import (
    ModelResolutionPolicy,
    ProductLayerCost,
    ProductModelMetadata,
    ProductModelReference,
    ProductModelSpec,
)
from swarm_inference.protocol.product import ModelPlanRequest, WorkerEligibilityReport
from swarm_inference.protocol.stage_ring import STAGE_RING_PROTOCOL_VERSION


def _reference() -> ProductModelReference:
    return ProductModelReference(
        model_id="test/olmoe",
        model_revision="immutable-model-revision",
        tokenizer_revision="immutable-tokenizer-revision",
        dtype="float32",
        resolution_policy=ModelResolutionPolicy.LOCAL_ONLY,
    )


def _metadata(layer_count: int, *, weight_bytes: int = 100) -> ProductModelMetadata:
    return ProductModelMetadata(
        layer_costs=tuple(
            ProductLayerCost(
                layer_id=index,
                execution_ns=100,
                weight_bytes=weight_bytes,
                kv_bytes_per_token=1,
                peak_temporary_bytes=20,
                activation_bytes=4096,
                measured=True,
            )
            for index in range(layer_count)
        ),
        embedding_weight_bytes=100,
        final_weight_bytes=100,
        dtype_bytes=4,
        hidden_size=32,
        metadata_hash=f"metadata-{layer_count}-{weight_bytes}",
    )


def _worker(index: int, *, memory: int = 10_000, mean_ms: float = 1.0) -> WorkerCapability:
    node_id = f"node-{index:08x}"
    worker_id = f"{node_id}/cpu-0"
    return WorkerCapability(
        worker_id=worker_id,
        node_id=node_id,
        public_key=f"key-{index}",
        hostname=node_id,
        operating_system="test",
        architecture="x86_64",
        backend=Backend.TORCH_CPU,
        cpu_model="test",
        logical_cpu_count=1,
        total_ram_bytes=memory,
        available_ram_bytes=memory,
        supported_dtypes=["float32"],
        stage_benchmarks=[
            StageBenchmark(
                worker_class="test",
                operation=OperationKind.DECODE,
                sequence_length=1,
                batch_size=1,
                mean_ms=mean_ms,
                median_ms=mean_ms,
                p95_ms=mean_ms,
                samples=5,
                measured=True,
                device="cpu",
                dtype="float32",
                measured_at_unix_ns=time.time_ns(),
                measurement_source="selected-device-torch",
            )
        ],
        upload_bandwidth_bytes_s=0,
        download_bandwidth_bytes_s=0,
        coordinator_latency_ms=999,
        memory_limit_bytes=memory,
        endpoint=f"127.0.0.1:{50_000 + index}",
        control_endpoint=f"127.0.0.1:{50_000 + index}",
        data_plane_endpoint=f"127.0.0.1:{51_000 + index}",
        device_identifier="cpu",
        stage_ring_protocol_version=STAGE_RING_PROTOCOL_VERSION,
        supported_model_adapters=["olmoe"],
        supported_stage_execution_backends=["canonical-contiguous-olmoe"],
        supported_activation_dtypes=["float32"],
        configured_memory_limit_bytes=memory,
        stage_runtime_enabled=True,
    )


def _inspected(
    metadata: ProductModelMetadata, workers: list[WorkerCapability]
) -> InspectedProductModel:
    return InspectedProductModel(
        spec=ProductModelSpec.resolved(_reference(), metadata),
        metadata=metadata,
        capabilities={worker.worker_id: worker for worker in workers},
        eligibility=tuple(
            WorkerEligibilityReport(
                worker_id=worker.worker_id,
                eligible=True,
                effective_memory_bytes=worker.effective_memory_bytes,
                active_session_count=worker.active_session_count,
                exact_model_identity=True,
                measured_profile=True,
            )
            for worker in workers
        ),
        all_capabilities={worker.worker_id: worker for worker in workers},
    )


def _links(
    workers: list[WorkerCapability],
    *,
    measured_at: int | None = None,
    bandwidth: float = 100_000_000,
) -> list[NetworkLinkMeasurement]:
    timestamp = measured_at or time.time_ns()
    return [
        NetworkLinkMeasurement(
            source_worker_id=source.worker_id,
            destination_worker_id=destination.worker_id,
            source_node_id=source.node_id,
            destination_node_id=destination.node_id,
            measured_at_unix_ns=timestamp,
            round_trip_latency_ms=1,
            one_way_estimate_ms=0.5,
            upload_bytes_per_s=bandwidth,
            download_bytes_per_s=bandwidth,
            payload_sizes=[4096],
            sample_count=3,
            p95_transfer_ms=1,
            source_endpoint=source.data_plane_endpoint,
            destination_endpoint=destination.data_plane_endpoint,
            measured=True,
            probe_ticket_id=f"ticket-{source.worker_id}-{destination.worker_id}",
            authentication_verified=True,
            payload_checksums_verified=True,
        )
        for source in workers
        for destination in workers
        if source.worker_id != destination.worker_id
    ]


@pytest.mark.parametrize("stage_count", [3, 4, 8])
def test_plans_three_four_and_eight_stage_directed_rings(stage_count: int) -> None:
    workers = [_worker(index) for index in range(stage_count)]
    measurements = _links(workers)
    planner = ProductStagePlanner(
        beam_width=64,
        network_measurement_provider=lambda: measurements,
    )

    plan = planner.build_plan(
        ModelPlanRequest(
            reference=_reference(),
            stage_count=stage_count,
            partition_method="equal",
            require_distributed=True,
            max_sequence_tokens=8,
        ),
        _inspected(_metadata(stage_count), workers),
    )

    assert plan.stage_count == stage_count
    assert [item.stage_id for item in plan.assignments] == list(range(stage_count))
    assert [layer for item in plan.assignments for layer in item.assignment.layer_ids] == list(
        range(stage_count)
    )
    assert len(plan.report.directed_links_selected) == stage_count - 1
    assert all(link.measured and link.fresh for link in plan.report.directed_links_selected)
    assert plan.report.confidence == "measured"
    assert plan.report.search_method == "bounded-deterministic-beam-search"


def test_speed_mode_excludes_slow_node_when_it_reduces_utility() -> None:
    workers = [
        _worker(0, mean_ms=0.2),
        _worker(1, mean_ms=0.4),
        _worker(2, mean_ms=20),
    ]
    measurements = _links(workers)
    planner = ProductStagePlanner(
        beam_width=64,
        network_measurement_provider=lambda: measurements,
    )

    plan = planner.build_plan(
        ModelPlanRequest(reference=_reference(), mode="speed", max_sequence_tokens=8),
        _inspected(_metadata(4), workers),
    )

    assert plan.stage_count == 1
    assert plan.assignments[0].worker_id == workers[0].worker_id
    slow_utility = next(
        item for item in plan.report.node_utility if item.worker_id == workers[2].worker_id
    )
    assert not slow_utility.included
    assert "utility" in slow_utility.reason or "objective" in slow_utility.reason


def test_capacity_mode_includes_slow_node_when_collective_fit_requires_it() -> None:
    workers = [
        _worker(0, memory=250, mean_ms=1),
        _worker(1, memory=250, mean_ms=1),
        _worker(2, memory=250, mean_ms=1),
        _worker(3, memory=250, mean_ms=20),
    ]
    measurements = _links(workers)
    planner = ProductStagePlanner(
        beam_width=128,
        network_measurement_provider=lambda: measurements,
    )

    plan = planner.build_plan(
        ModelPlanRequest(reference=_reference(), mode="capacity", max_sequence_tokens=8),
        _inspected(_metadata(4), workers),
    )

    assert plan.stage_count == 4
    assert workers[3].worker_id in {item.worker_id for item in plan.assignments}
    assert all(
        item.required_memory_bytes <= item.effective_memory_bytes for item in plan.assignments
    )


def test_required_node_is_included_and_ties_are_deterministic() -> None:
    workers = [_worker(index) for index in range(4)]
    measurements = _links(workers)
    planner = ProductStagePlanner(
        beam_width=64,
        network_measurement_provider=lambda: measurements,
    )
    request = ModelPlanRequest(
        reference=_reference(),
        stage_count=3,
        partition_method="equal",
        mode="balanced",
        required_node_ids=[workers[3].node_id or ""],
        max_sequence_tokens=8,
    )
    inspected = _inspected(_metadata(6), workers)

    first = planner.build_plan(request, inspected)
    second = planner.build_plan(request, inspected)

    assert workers[3].worker_id in {item.worker_id for item in first.assignments}
    assert [item.worker_id for item in first.assignments] == [
        item.worker_id for item in second.assignments
    ]
    assert first.plan_id == second.plan_id
    assert set(first.report.objective_components) == {
        "throughput_penalty",
        "memory_headroom_penalty",
        "reliability_penalty",
        "participation_penalty",
    }


def test_stale_directed_links_are_rejected_for_automatic_distributed_plans() -> None:
    now = time.time_ns()
    workers = [_worker(0, memory=250), _worker(1, memory=250)]
    stale = _links(workers, measured_at=now - 901_000_000_000)
    planner = ProductStagePlanner(
        network_measurement_ttl_seconds=900,
        network_measurement_provider=lambda: stale,
        clock_ns=lambda: now,
    )

    with pytest.raises(RuntimeError, match="no fresh directed measurement"):
        planner.build_plan(
            ModelPlanRequest(
                reference=_reference(),
                require_distributed=True,
                mode="capacity",
                max_sequence_tokens=8,
            ),
            _inspected(_metadata(2), workers),
        )


def test_planner_source_has_no_factorial_permutation_path() -> None:
    source = inspect.getsource(ProductStagePlanner)
    assert "permutations" not in source
    assert "itertools" not in source
    assert "beam_width" in source
