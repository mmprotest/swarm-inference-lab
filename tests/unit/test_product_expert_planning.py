from __future__ import annotations

import time

import pytest

from swarm_inference.config.models import (
    Backend,
    OperationKind,
    StageBenchmark,
    WorkerCapability,
    WorkerRole,
)
from swarm_inference.coordinator.model_catalog import InspectedProductModel
from swarm_inference.coordinator.session_controller import ProductSessionController
from swarm_inference.coordinator.stage_planner import ProductStagePlanner
from swarm_inference.exceptions import IntegrityError
from swarm_inference.model.product import (
    ModelResolutionPolicy,
    ProductLayerCost,
    ProductModelMetadata,
    ProductModelReference,
    ProductModelSpec,
)
from swarm_inference.protocol.product import (
    ModelPlanRequest,
    ProductTokenPublication,
    WorkerEligibilityReport,
)
from swarm_inference.protocol.stage_ring import STAGE_RING_PROTOCOL_VERSION


def _reference() -> ProductModelReference:
    return ProductModelReference(
        model_id="test/qwen3-moe",
        model_revision="model-revision",
        tokenizer_revision="tokenizer-revision",
        dtype="float32",
        adapter_id="qwen3_moe",
        resolution_policy=ModelResolutionPolicy.LOCAL_ONLY,
    )


def _metadata() -> ProductModelMetadata:
    return ProductModelMetadata(
        layer_costs=(
            ProductLayerCost(
                layer_id=0,
                execution_ns=30_000_000,
                weight_bytes=1_000,
                kv_bytes_per_token=1,
                peak_temporary_bytes=20,
                activation_bytes=16,
                measured=True,
                expert_weight_bytes=800,
                expert_execution_ns=20_000_000,
            ),
        ),
        embedding_weight_bytes=50,
        final_weight_bytes=20,
        dtype_bytes=4,
        hidden_size=4,
        metadata_hash="metadata-hash",
        expert_count=2,
        experts_per_token=1,
        expert_intermediate_size=8,
        model_fingerprint="model-fingerprint",
        quantization_fingerprint="quantization-fingerprint",
        adapter_id="qwen3_moe",
    )


def _worker(worker_id: str, *, memory_bytes: int = 2_000) -> WorkerCapability:
    suffix = sum(ord(value) for value in worker_id) % 1_000
    return WorkerCapability(
        worker_id=worker_id,
        public_key=f"test-key-{worker_id}",
        hostname="localhost",
        operating_system="test",
        architecture="x86_64",
        backend=Backend.TORCH_CPU,
        cpu_model="test",
        logical_cpu_count=1,
        total_ram_bytes=memory_bytes,
        available_ram_bytes=memory_bytes,
        supported_dtypes=["float32"],
        stage_benchmarks=[
            StageBenchmark(
                worker_class="test",
                operation=OperationKind.DECODE,
                sequence_length=1,
                batch_size=1,
                mean_ms=1,
                p95_ms=1,
                samples=3,
                measured=True,
            )
        ],
        upload_bandwidth_bytes_s=1_000_000_000,
        download_bandwidth_bytes_s=1_000_000_000,
        coordinator_latency_ms=0,
        memory_limit_bytes=memory_bytes,
        endpoint=f"127.0.0.1:{50_000 + suffix}",
        control_endpoint=f"127.0.0.1:{50_000 + suffix}",
        data_plane_endpoint=f"127.0.0.1:{51_000 + suffix}",
        device_identifier="cpu",
        stage_ring_protocol_version=STAGE_RING_PROTOCOL_VERSION,
        supported_model_adapters=["qwen3_moe"],
        supported_stage_execution_backends=["qwen3-transformers-eager"],
        supported_activation_dtypes=["float32"],
        configured_memory_limit_bytes=memory_bytes,
        stage_runtime_enabled=True,
        roles=[WorkerRole.CONTIGUOUS_STAGE],
    )


def _expert_worker(
    worker_id: str,
    *,
    role: WorkerRole,
    whole: bool = False,
    slice_start: int | None = None,
    slice_end: int | None = None,
    service_ms: float = 1,
) -> WorkerCapability:
    worker = _worker(worker_id, memory_bytes=2_000)
    microshards = []
    if slice_start is not None and slice_end is not None:
        microshards = [
            {
                "layer_id": 0,
                "expert_id": expert_id,
                "hidden_start": slice_start,
                "hidden_end": slice_end,
                "logical_intermediate_dimension": 8,
                "quantization_group_size": 2,
                "content_hash": f"sha256:{worker_id}:{expert_id}",
            }
            for expert_id in range(2)
        ]
    return worker.model_copy(
        update={
            "stage_runtime_enabled": False,
            "roles": [role],
            "data_plane_endpoint": None,
            "expert_data_plane_endpoint": f"127.0.0.1:{52_000 + len(worker_id)}",
            "owned_experts": {"0": [0, 1]} if whole else {},
            "owned_microshards": microshards,
            "expert_content_hashes": {
                f"0:{expert_id}": f"sha256:{worker_id}:{expert_id}" for expert_id in range(2)
            },
            "expert_memory_budget_bytes": 2_000,
            "expert_cache_budget_bytes": 1_000,
            "model_fingerprint": "model-fingerprint",
            "quantisation_fingerprint": "quantization-fingerprint",
            "supported_expert_codecs": ["raw_fp32"],
            "supported_reduction_modes": ["fixed_order_fp32"],
            "measured_expert_service_rates": {
                "whole_expert_ms": service_ms,
                "microshard_ms": service_ms,
                "reduction_ms": 0.01,
            },
        }
    )


def _inspected(
    stage: WorkerCapability,
    *expert_workers: WorkerCapability,
) -> InspectedProductModel:
    metadata = _metadata()
    return InspectedProductModel(
        spec=ProductModelSpec.resolved(_reference(), metadata),
        metadata=metadata,
        capabilities={stage.worker_id: stage},
        eligibility=(
            WorkerEligibilityReport(
                worker_id=stage.worker_id,
                eligible=True,
                effective_memory_bytes=stage.effective_memory_bytes,
                active_session_count=0,
                exact_model_identity=True,
                measured_profile=True,
            ),
        ),
        all_capabilities={
            stage.worker_id: stage,
            **{worker.worker_id: worker for worker in expert_workers},
        },
    )


def _request(
    *,
    policy: str = "auto",
    require_remote: bool = False,
    allow_fallback: bool = False,
) -> ModelPlanRequest:
    return ModelPlanRequest(
        reference=_reference(),
        stage_count=1,
        partition_method="equal",
        max_sequence_tokens=8,
        expert_policy=policy,
        require_remote_experts=require_remote,
        allow_expert_local_fallback=allow_fallback,
    )


def test_hierarchical_planner_uses_whole_experts_for_capacity() -> None:
    stage = _worker("stage-0", memory_bytes=500)
    whole = _expert_worker("whole-0", role=WorkerRole.WHOLE_EXPERT, whole=True)
    plan = ProductStagePlanner().build_plan(_request(), _inspected(stage, whole))
    placements = plan.expert_plans[0].placements
    assert plan.stage_count == 1
    assert plan.assignments[0].assignment.layer_ids == (0,)
    assert all(item.strategy == "whole-remote" for item in placements)
    assert all(item.capacity_required for item in placements)
    assert all(item.worker_ids == ["whole-0"] for item in placements)
    assert all(
        item.rejected and all(row["reasons"] for row in item.rejected) for item in placements
    )


def test_hierarchical_planner_selects_gap_free_physical_microshards() -> None:
    stage = _worker("stage-0", memory_bytes=500)
    first = _expert_worker(
        "micro-0",
        role=WorkerRole.EXPERT_MICROSHARD,
        slice_start=0,
        slice_end=4,
    )
    second = _expert_worker(
        "micro-1",
        role=WorkerRole.EXPERT_MICROSHARD,
        slice_start=4,
        slice_end=8,
    )
    plan = ProductStagePlanner().build_plan(
        _request(policy="microshard-remote", require_remote=True),
        _inspected(stage, first, second),
    )
    for placement in plan.expert_plans[0].placements:
        assert placement.strategy == "microshard-remote"
        assert placement.worker_ids == ["micro-0", "micro-1"]
        assert [(item["hidden_start"], item["hidden_end"]) for item in placement.microshards] == [
            (0, 4),
            (4, 8),
        ]
        assert all(
            item["hidden_end"] - item["hidden_start"] < item["logical_intermediate_dimension"]
            for item in placement.microshards
        )
        assert placement.forced_remote
        assert not placement.local_fallback_permitted


def test_hierarchical_planner_rejects_advertised_microshard_logical_width_mismatch() -> None:
    stage = _worker("stage-0", memory_bytes=500)
    first = _expert_worker(
        "micro-0",
        role=WorkerRole.EXPERT_MICROSHARD,
        slice_start=0,
        slice_end=4,
    )
    second = _expert_worker(
        "micro-1",
        role=WorkerRole.EXPERT_MICROSHARD,
        slice_start=4,
        slice_end=8,
    )
    first.owned_microshards[0]["logical_intermediate_dimension"] = 10
    with pytest.raises(RuntimeError, match="no exact expert strategy"):
        ProductStagePlanner().build_plan(
            _request(policy="microshard-remote", require_remote=True),
            _inspected(stage, first, second),
        )


@pytest.mark.parametrize("field", ["supported_expert_codecs", "supported_reduction_modes"])
def test_hierarchical_planner_rejects_whole_expert_without_exact_protocol(
    field: str,
) -> None:
    stage = _worker("stage-0", memory_bytes=500)
    whole = _expert_worker("whole-0", role=WorkerRole.WHOLE_EXPERT, whole=True)
    setattr(whole, field, [])
    with pytest.raises(RuntimeError, match="no exact expert strategy"):
        ProductStagePlanner().build_plan(
            _request(policy="whole-remote", require_remote=True),
            _inspected(stage, whole),
        )


def test_hierarchical_planner_rejects_microshard_without_exact_reduction() -> None:
    stage = _worker("stage-0", memory_bytes=500)
    first = _expert_worker(
        "micro-0",
        role=WorkerRole.EXPERT_MICROSHARD,
        slice_start=0,
        slice_end=4,
    )
    second = _expert_worker(
        "micro-1",
        role=WorkerRole.EXPERT_MICROSHARD,
        slice_start=4,
        slice_end=8,
    )
    second.supported_reduction_modes = []
    with pytest.raises(RuntimeError, match="no exact expert strategy"):
        ProductStagePlanner().build_plan(
            _request(policy="microshard-remote", require_remote=True),
            _inspected(stage, first, second),
        )


def test_hierarchical_planner_prefers_local_when_remote_utility_is_negative() -> None:
    stage = _worker("stage-0", memory_bytes=2_000)
    slow = _expert_worker(
        "whole-0",
        role=WorkerRole.WHOLE_EXPERT,
        whole=True,
        service_ms=25,
    )
    plan = ProductStagePlanner().build_plan(_request(), _inspected(stage, slow))
    for placement in plan.expert_plans[0].placements:
        assert placement.strategy == "local"
        rejected = next(item for item in placement.rejected if item["strategy"] == "whole-remote")
        assert any("non-positive" in reason for reason in rejected["reasons"])


def test_planner_permits_fallback_only_for_remote_experts_that_fit_locally() -> None:
    stage = _worker("stage-0", memory_bytes=2_000).model_copy(
        update={"measured_memory_bandwidth_bytes_s": 1_000_000_000}
    )
    fast = _expert_worker(
        "whole-0",
        role=WorkerRole.WHOLE_EXPERT,
        whole=True,
        service_ms=0.01,
    )
    plan = ProductStagePlanner().build_plan(
        _request(allow_fallback=True),
        _inspected(stage, fast),
    )
    assert all(item.strategy == "whole-remote" for item in plan.expert_plans[0].placements)
    assert all(item.local_fallback_permitted for item in plan.expert_plans[0].placements)

    capacity_stage = _worker("capacity-stage", memory_bytes=500)
    capacity_plan = ProductStagePlanner().build_plan(
        _request(allow_fallback=True),
        _inspected(capacity_stage, fast),
    )
    assert all(item.strategy == "whole-remote" for item in capacity_plan.expert_plans[0].placements)
    assert all(item.capacity_required for item in capacity_plan.expert_plans[0].placements)
    assert not any(
        item.local_fallback_permitted for item in capacity_plan.expert_plans[0].placements
    )


def test_planner_does_not_attach_fallback_permission_to_local_selection() -> None:
    stage = _worker("stage-0", memory_bytes=2_000)
    slow = _expert_worker(
        "whole-0",
        role=WorkerRole.WHOLE_EXPERT,
        whole=True,
        service_ms=25,
    )
    plan = ProductStagePlanner().build_plan(
        _request(allow_fallback=True),
        _inspected(stage, slow),
    )
    assert all(item.strategy == "local" for item in plan.expert_plans[0].placements)
    assert not any(item.local_fallback_permitted for item in plan.expert_plans[0].placements)


def test_microshard_role_is_never_considered_a_whole_expert_owner() -> None:
    stage = _worker("stage-0", memory_bytes=500)
    mislabeled = _expert_worker(
        "micro-0",
        role=WorkerRole.EXPERT_MICROSHARD,
        whole=True,
        slice_start=0,
        slice_end=4,
    )
    try:
        ProductStagePlanner().build_plan(
            _request(policy="whole-remote", require_remote=True),
            _inspected(stage, mislabeled),
        )
    except RuntimeError as exc:
        assert "no exact expert strategy" in str(exc)
    else:  # pragma: no cover - a role-boundary regression must fail loudly
        raise AssertionError("microshard-only worker was selected for whole-expert execution")


def _publication(
    plan: object,
    *,
    event: dict[str, object],
    metrics: dict[str, int],
) -> ProductTokenPublication:
    from swarm_inference.protocol.product import ProductStagePlan

    selected = ProductStagePlan.model_validate(plan)
    return ProductTokenPublication(
        worker_id=selected.assignments[0].worker_id,
        request_id="request-1",
        session_id="session-1",
        topology_id=selected.topology_id,
        route_generation=selected.generation,
        model_revision=selected.model.model_revision,
        token_position=0,
        token_id=1,
        published_monotonic_ns=time.monotonic_ns(),
        expert_trace=[event],
        expert_metrics=metrics,
    )


def test_forced_remote_trace_requires_exact_workers_and_matching_metrics() -> None:
    stage = _worker("stage-0", memory_bytes=500)
    whole = _expert_worker("whole-0", role=WorkerRole.WHOLE_EXPERT, whole=True)
    plan = ProductStagePlanner().build_plan(
        _request(policy="whole-remote", require_remote=True),
        _inspected(stage, whole),
    )
    event: dict[str, object] = {
        "event": "remote_whole_expert_result_consumed",
        "session_id": "session-1",
        "request_id": "request-1:layer-0:expert-0",
        "token_position": 0,
        "layer_id": 0,
        "expert_id": 0,
        "worker_ids": ["whole-0"],
        "request_bytes": 100,
        "response_bytes": 80,
        "result_hash": "sha256:result",
    }
    publication = _publication(
        plan,
        event=event,
        metrics={
            "remote_expert_calls": 1,
            "remote_whole_expert_calls": 1,
            "remote_microshard_calls": 0,
            "fallbacks": 0,
            "bytes_transferred": 180,
        },
    )
    ProductSessionController._validate_expert_trace(publication, plan)
    with pytest.raises(IntegrityError, match="byte metric"):
        ProductSessionController._validate_expert_trace(
            publication.model_copy(update={"expert_metrics": {"remote_expert_calls": 1}}),
            plan,
        )
    local_event = {**event, "event": "local_expert_result_consumed", "worker_ids": []}
    with pytest.raises(IntegrityError, match="local expert contribution"):
        ProductSessionController._validate_expert_trace(
            publication.model_copy(update={"expert_trace": [local_event]}),
            plan,
        )


def test_microshard_trace_cannot_omit_a_planned_physical_owner() -> None:
    stage = _worker("stage-0", memory_bytes=500)
    first = _expert_worker(
        "micro-0",
        role=WorkerRole.EXPERT_MICROSHARD,
        slice_start=0,
        slice_end=4,
    )
    second = _expert_worker(
        "micro-1",
        role=WorkerRole.EXPERT_MICROSHARD,
        slice_start=4,
        slice_end=8,
    )
    plan = ProductStagePlanner().build_plan(
        _request(policy="microshard-remote", require_remote=True),
        _inspected(stage, first, second),
    )
    publication = _publication(
        plan,
        event={
            "event": "remote_microshard_result_consumed",
            "session_id": "session-1",
            "request_id": "request-1:layer-0:expert-0",
            "token_position": 0,
            "layer_id": 0,
            "expert_id": 0,
            "worker_ids": ["micro-0"],
            "request_bytes": 100,
            "response_bytes": 80,
            "result_hash": "sha256:result",
        },
        metrics={"remote_expert_calls": 1, "bytes_transferred": 180},
    )
    with pytest.raises(IntegrityError, match="exactly match the plan"):
        ProductSessionController._validate_expert_trace(publication, plan)
