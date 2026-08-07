from __future__ import annotations

from dataclasses import replace

import pytest

from swarm_inference.config.models import (
    Backend,
    OperationKind,
    StageBenchmark,
    WorkerCapability,
)
from swarm_inference.coordinator.model_catalog import (
    InspectedProductModel,
    ProductModelCatalog,
)
from swarm_inference.coordinator.registry import WorkerRegistry
from swarm_inference.coordinator.stage_planner import ProductStagePlanner
from swarm_inference.model.product import (
    ModelResolutionPolicy,
    ProductLayerCost,
    ProductModelMetadata,
    ProductModelReference,
    ProductModelSpec,
)
from swarm_inference.protocol.product import (
    ModelPlanRequest,
    WorkerEligibilityReport,
    WorkerModelProbeRequest,
    WorkerModelProbeResponse,
)
from swarm_inference.protocol.stage_ring import STAGE_RING_PROTOCOL_VERSION
from swarm_inference.protocol.stage_worker import (
    GetStageCapabilitiesRequest,
    GetStageCapabilitiesResponse,
)


def _capability(
    worker_id: str,
    *,
    memory_bytes: int = 5_000,
    mean_ms: float = 1.0,
    adapter: bool = True,
    active_sessions: int = 0,
) -> WorkerCapability:
    suffix = int(worker_id.rsplit("-", 1)[-1]) + 1
    return WorkerCapability(
        worker_id=worker_id,
        public_key=f"key-{worker_id}",
        hostname="localhost",
        operating_system="test",
        architecture="x86_64",
        backend=Backend.TORCH_CPU,
        cpu_model="test",
        logical_cpu_count=1,
        total_ram_bytes=max(memory_bytes, 1),
        available_ram_bytes=memory_bytes,
        supported_dtypes=["float32"],
        stage_benchmarks=[
            StageBenchmark(
                worker_class="test",
                operation=OperationKind.DECODE,
                sequence_length=1,
                batch_size=1,
                mean_ms=mean_ms,
                p95_ms=mean_ms,
                samples=5,
                measured=True,
            )
        ],
        upload_bandwidth_bytes_s=1_000_000,
        download_bandwidth_bytes_s=1_000_000,
        coordinator_latency_ms=0.1,
        memory_limit_bytes=max(memory_bytes, 1),
        max_concurrent_stage_operations=4,
        endpoint=f"127.0.0.1:{50_000 + suffix}",
        control_endpoint=f"127.0.0.1:{50_000 + suffix}",
        data_plane_endpoint=f"127.0.0.1:{51_000 + suffix}",
        device_identifier="cpu",
        stage_ring_protocol_version=STAGE_RING_PROTOCOL_VERSION,
        supported_model_adapters=["qwen3_dense"] if adapter else [],
        supported_stage_execution_backends=["qwen3-transformers-eager"],
        supported_activation_dtypes=["float32"],
        configured_memory_limit_bytes=max(memory_bytes, 1),
        active_session_count=active_sessions,
        stage_runtime_enabled=True,
    )


def _metadata(
    *,
    metadata_hash: str = "metadata-a",
    execution: tuple[int, ...] = (4, 4, 1, 1),
) -> ProductModelMetadata:
    return ProductModelMetadata(
        layer_costs=tuple(
            ProductLayerCost(
                layer_id=index,
                execution_ns=value,
                weight_bytes=100,
                kv_bytes_per_token=1,
                peak_temporary_bytes=20,
                activation_bytes=16,
                measured=True,
            )
            for index, value in enumerate(execution)
        ),
        embedding_weight_bytes=800,
        final_weight_bytes=50,
        dtype_bytes=4,
        hidden_size=4,
        metadata_hash=metadata_hash,
        adapter_id="qwen3_dense",
    )


def _reference() -> ProductModelReference:
    return ProductModelReference(
        model_id="test/qwen3",
        model_revision="model-commit",
        tokenizer_revision="tokenizer-commit",
        dtype="float32",
        adapter_id="qwen3_dense",
        resolution_policy=ModelResolutionPolicy.LOCAL_ONLY,
    )


class _ProbeTransport:
    def __init__(
        self,
        capabilities: dict[str, WorkerCapability],
        metadata: dict[str, ProductModelMetadata],
    ) -> None:
        self.capabilities = capabilities
        self.metadata = metadata

    async def get_stage_capabilities(
        self,
        _endpoint: str,
        request: GetStageCapabilitiesRequest,
    ) -> GetStageCapabilitiesResponse:
        return GetStageCapabilitiesResponse(
            worker_id=request.worker_id,
            request_id=request.request_id,
            capability=self.capabilities[request.worker_id],
        )

    async def inspect_stage_model(
        self,
        _endpoint: str,
        request: WorkerModelProbeRequest,
    ) -> WorkerModelProbeResponse:
        metadata = self.metadata[request.worker_id]
        return WorkerModelProbeResponse(
            worker_id=request.worker_id,
            request_id=request.request_id,
            available=True,
            spec=ProductModelSpec.resolved(request.reference, metadata),
            metadata=metadata,
        )


@pytest.mark.asyncio
async def test_worker_eligibility_requires_health_identity_adapter_load_and_memory() -> None:
    registry = WorkerRegistry(heartbeat_timeout_s=1.0)
    capabilities = {
        "worker-0": _capability("worker-0"),
        "worker-1": _capability("worker-1"),
        "worker-2": _capability("worker-2"),
        "worker-3": _capability("worker-3", adapter=False),
        "worker-4": _capability("worker-4", memory_bytes=10),
        "worker-5": _capability("worker-5", active_sessions=8),
        "worker-6": _capability("worker-6"),
    }
    for worker_id, capability in capabilities.items():
        registry.register(
            capability,
            benchmark_verified=True,
            now=0.0 if worker_id == "worker-2" else None,
        )
    matching = _metadata()
    probes = {worker_id: matching for worker_id in capabilities}
    probes["worker-1"] = _metadata()
    probes["worker-6"] = _metadata(metadata_hash="metadata-other")
    catalog = ProductModelCatalog(
        registry=registry,
        transport=_ProbeTransport(capabilities, probes),
        maximum_active_sessions_per_worker=8,
    )

    inspected = await catalog.inspect(_reference())

    assert set(inspected.capabilities) == {"worker-0", "worker-1"}
    reports = {item.worker_id: item for item in inspected.eligibility}
    assert reports["worker-0"].eligible
    assert any("heartbeat" in value for value in reports["worker-2"].rejection_reasons)
    assert any("adapter" in value for value in reports["worker-3"].rejection_reasons)
    assert any("smallest valid stage" in value for value in reports["worker-4"].rejection_reasons)
    assert any("active load" in value for value in reports["worker-5"].rejection_reasons)
    assert any("differs" in value for value in reports["worker-6"].rejection_reasons)


def _inspected(*workers: WorkerCapability) -> InspectedProductModel:
    reference = _reference()
    metadata = _metadata()
    return InspectedProductModel(
        spec=ProductModelSpec.resolved(reference, metadata),
        metadata=metadata,
        capabilities={item.worker_id: item for item in workers},
        eligibility=tuple(
            WorkerEligibilityReport(
                worker_id=item.worker_id,
                eligible=True,
                effective_memory_bytes=item.effective_memory_bytes,
                active_session_count=item.active_session_count,
                exact_model_identity=True,
                measured_profile=True,
            )
            for item in workers
        ),
    )


def test_exact_two_stage_equal_plan_is_contiguous_and_memory_aware() -> None:
    low_memory = _capability("worker-0", memory_bytes=500)
    high_memory = _capability("worker-1", memory_bytes=5_000)
    plan = ProductStagePlanner().build_plan(
        ModelPlanRequest(
            reference=_reference(),
            stage_count=2,
            partition_method="equal",
            require_distributed=True,
            max_sequence_tokens=8,
        ),
        _inspected(low_memory, high_memory),
    )

    assert plan.stage_count == 2
    assert plan.partition_method == "equal"
    assert [item.assignment.layer_ids for item in plan.assignments] == [(0, 1), (2, 3)]
    assert plan.assignments[0].worker_id == "worker-1"
    assert plan.assignments[1].worker_id == "worker-0"
    assert all(
        item.required_memory_bytes <= item.effective_memory_bytes for item in plan.assignments
    )
    assert plan.report.selected_topology == "two-stage-equal-ring"


def test_balanced_partition_and_automatic_choice_are_utility_driven() -> None:
    first = _capability("worker-0", memory_bytes=20_000, mean_ms=1.0)
    second = _capability("worker-1", memory_bytes=20_000, mean_ms=1.1)
    inspected = _inspected(first, second)
    inspected = replace(inspected, metadata=_metadata(execution=(8, 1, 1, 1)))
    inspected = replace(
        inspected,
        spec=ProductModelSpec.resolved(_reference(), inspected.metadata),
    )
    planner = ProductStagePlanner()

    balanced = planner.build_plan(
        ModelPlanRequest(
            reference=_reference(),
            stage_count=2,
            partition_method="balanced",
            require_distributed=True,
            max_sequence_tokens=8,
        ),
        inspected,
    )
    automatic = planner.build_plan(ModelPlanRequest(reference=_reference()), inspected)

    assert [item.assignment.layer_ids for item in balanced.assignments] == [(0,), (1, 2, 3)]
    reports = {item.name: item for item in automatic.report.candidates}
    assert set(reports) == {
        "local-monolithic",
        "two-stage-equal-ring",
        "two-stage-balanced-ring",
    }
    feasible = [item for item in reports.values() if item.feasible]
    selected = next(item for item in feasible if item.selected)
    assert selected.expected_utility_tokens_s == max(
        item.expected_utility_tokens_s for item in feasible
    )
    assert "highest measured expected token utility" in automatic.report.reason_for_selection
