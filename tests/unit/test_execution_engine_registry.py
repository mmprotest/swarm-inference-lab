from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from swarm_inference.coordinator.canonical_planner import (
    CanonicalPlanner,
    MechanismEvidence,
)
from swarm_inference.engines.colibri import ColibriExecutionEngine
from swarm_inference.engines.interfaces import (
    ClusterCapabilities,
    Deployment,
    EngineSupportReport,
    EngineSupportStatus,
    ExecutionDevice,
    ExecutionEngineCapability,
    ExecutionPlan,
    ExecutionProfileCapability,
    ExecutionRequest,
    InferenceEvent,
    InferenceRequest,
    PhasePlan,
    WorkerExecutionCapability,
)
from swarm_inference.engines.native_stage import NativeStageEngine
from swarm_inference.engines.registry import ExecutionEngineRegistry
from swarm_inference.model.descriptor import ModelFileDescriptor, ResolvedModelDescriptor


def _model() -> ResolvedModelDescriptor:
    return ResolvedModelDescriptor(
        model_id="org/model",
        revision="a" * 40,
        content_fingerprint="sha256:" + "b" * 64,
        source_type="huggingface",
        format="gguf",
        architecture="test",
        files=(ModelFileDescriptor(relative_path="model.gguf", size_bytes=1),),
        weight_bytes=1,
        layer_count=1,
    )


def _plan(mechanisms: dict[str, bool]) -> ExecutionPlan:
    roles: dict[str, str] = {}
    return ExecutionPlan(
        plan_id="plan",
        engine_id="test-engine",
        model_fingerprint="sha256:" + "b" * 64,
        execution_identity="runtime-identity",
        objective="speed",
        topology="local",
        worker_roles=roles,
        optional_mechanisms=mechanisms,
        prefill_plan=PhasePlan(phase="prefill", worker_roles=roles),
        decode_plan=PhasePlan(phase="decode", worker_roles=roles),
        predicted_ttft_ms=1,
        predicted_decode_tokens_s=1,
        predicted_aggregate_tokens_s=1,
        score=1,
    )


class _Engine:
    engine_id = "test-engine"

    def __init__(self, *, status: EngineSupportStatus, plan: ExecutionPlan) -> None:
        self.status = status
        self.plan = plan
        self.candidate_calls = 0

    def probe(self, model, cluster) -> EngineSupportReport:
        del model, cluster
        return EngineSupportReport(
            engine_id=self.engine_id,
            status=self.status,
            reason="test probe",
        )

    async def candidate_plans(self, model, cluster, request) -> list[ExecutionPlan]:
        del model, cluster, request
        self.candidate_calls += 1
        return [self.plan]

    async def prepare(self, plan: ExecutionPlan) -> Deployment:
        raise AssertionError(plan)

    async def submit(
        self, deployment: Deployment, request: InferenceRequest
    ) -> AsyncIterator[InferenceEvent]:
        del deployment, request
        if False:
            yield InferenceEvent(event_type="completed", request_id="never", sequence_number=0)

    async def unload(self, deployment: Deployment) -> None:
        raise AssertionError(deployment)


@pytest.mark.asyncio
async def test_forced_engine_fails_closed_without_calling_another_candidate() -> None:
    engine = _Engine(status=EngineSupportStatus.MISSING_RUNTIME, plan=_plan({}))
    registry = ExecutionEngineRegistry((engine,))

    with pytest.raises(RuntimeError, match=r"forced engine.*MISSING_RUNTIME"):
        await registry.compete(
            _model(),
            ClusterCapabilities(workers=()),
            ExecutionRequest(requested_engine="test-engine"),
        )

    assert engine.candidate_calls == 0


@pytest.mark.asyncio
async def test_rejected_default_is_not_eligible_without_explicit_force() -> None:
    engine = _Engine(
        status=EngineSupportStatus.SUPPORTED,
        plan=_plan({"aggressive_paging": True}),
    )
    planner = CanonicalPlanner(ExecutionEngineRegistry((engine,)))

    with pytest.raises(RuntimeError, match="REJECTED_DEFAULT"):
        await planner.plan(_model(), ClusterCapabilities(workers=()), ExecutionRequest())


@pytest.mark.asyncio
async def test_conditional_mechanism_requires_matching_runtime_positive_evidence() -> None:
    engine = _Engine(
        status=EngineSupportStatus.SUPPORTED,
        plan=_plan({"routing_aware_placement": True}),
    )
    planner = CanonicalPlanner(ExecutionEngineRegistry((engine,)))
    mismatched = MechanismEvidence(
        mechanism="routing_aware_placement",
        exactness_passed=True,
        measured_utility=1,
        evidence_fingerprint="evidence",
        runtime_fingerprint="different-runtime",
    )
    with pytest.raises(RuntimeError, match="matched-runtime"):
        await planner.plan(
            _model(),
            ClusterCapabilities(workers=()),
            ExecutionRequest(),
            mechanism_evidence=(mismatched,),
        )

    matching = mismatched.model_copy(update={"runtime_fingerprint": "runtime-identity"})
    decision = await planner.plan(
        _model(),
        ClusterCapabilities(workers=()),
        ExecutionRequest(),
        mechanism_evidence=(matching,),
    )
    assert decision.selected.plan_id == "plan"


@pytest.mark.asyncio
async def test_forced_distributed_candidates_cannot_be_readmitted_by_composition() -> None:
    local = _plan({}).model_copy(
        update={
            "plan_id": "local",
            "topology": "local",
            "worker_roles": {"node-a/worker": "critical_path_stage"},
            "prefill_plan": PhasePlan(
                phase="prefill", worker_roles={"node-a/worker": "critical_path_stage"}
            ),
            "decode_plan": PhasePlan(
                phase="decode", worker_roles={"node-a/worker": "critical_path_stage"}
            ),
            "score": 100,
        }
    )
    distributed = local.model_copy(
        update={
            "plan_id": "distributed",
            "topology": "two-host",
            "worker_roles": {
                "node-a/worker": "critical_path_stage",
                "node-b/worker": "tensor_rpc_compute",
            },
            "prefill_plan": PhasePlan(
                phase="prefill",
                worker_roles={
                    "node-a/worker": "critical_path_stage",
                    "node-b/worker": "tensor_rpc_compute",
                },
            ),
            "decode_plan": PhasePlan(
                phase="decode",
                worker_roles={
                    "node-a/worker": "critical_path_stage",
                    "node-b/worker": "tensor_rpc_compute",
                },
            ),
            "score": 1,
        }
    )

    class MultiplePlanEngine(_Engine):
        async def candidate_plans(self, model, cluster, request) -> list[ExecutionPlan]:
            del model, cluster, request
            return [local, distributed]

    engine = MultiplePlanEngine(status=EngineSupportStatus.SUPPORTED, plan=local)
    cluster = ClusterCapabilities(
        workers=(
            WorkerExecutionCapability(worker_id="node-a/worker", node_id="node-a", engines=()),
            WorkerExecutionCapability(worker_id="node-b/worker", node_id="node-b", engines=()),
        )
    )
    decision = await CanonicalPlanner(ExecutionEngineRegistry((engine,))).plan(
        _model(), cluster, ExecutionRequest(require_distributed=True)
    )

    assert decision.selected.plan_id == "distributed"
    assert [item.plan_id for item in decision.candidates] == ["distributed"]


@pytest.mark.asyncio
async def test_colibri_routing_policy_is_a_separate_evidence_gated_candidate() -> None:
    model = _model().model_copy(
        update={"format": "safetensors", "architecture": "OlmoeForCausalLM"}
    )
    capability = ExecutionEngineCapability(
        engine_id="colibri",
        enabled=True,
        runtime_revision="pinned-runtime",
        binary_hashes={"colibri": "sha256:binary"},
        formats=("safetensors",),
        adapters=("olmoe",),
        fast_paths=("routing-aware-placement",),
        execution_profiles=(
            ExecutionProfileCapability(
                profile_id="olmoe-hot-v1",
                mechanism="routing_aware_placement",
                adapter_id="olmoe",
                model_fingerprint=model.content_fingerprint,
                content_fingerprint="sha256:" + "4" * 64,
                exactness_passed=True,
                measured_utility=1,
                evidence_fingerprint="frozen-colibri-evidence",
            ),
        ),
        devices=(
            ExecutionDevice(
                device_id="cpu",
                device_type="cpu",
                name="test-cpu",
                usable_memory_bytes=1024,
                measured_decode_tokens_s=10,
            ),
        ),
    )
    cluster = ClusterCapabilities(
        workers=(
            WorkerExecutionCapability(
                worker_id="node-a/worker", node_id="node-a", engines=(capability,)
            ),
        )
    )
    engine = ColibriExecutionEngine()
    first = await engine.candidate_plans(model, cluster, ExecutionRequest())
    second = await engine.candidate_plans(model, cluster, ExecutionRequest())
    assert [item.plan_id for item in first] == [item.plan_id for item in second]
    assert {item.fast_paths["node-a/worker"] for item in first} == {
        "backend-native",
        "routing-aware-placement",
    }

    planner = CanonicalPlanner(ExecutionEngineRegistry((engine,)))
    automatic = await planner.plan(model, cluster, ExecutionRequest())
    assert automatic.selected.fast_paths["node-a/worker"] == "routing-aware-placement"
    assert automatic.selected.engine_parameters["routing_profile_id"] == "olmoe-hot-v1"
    assert not automatic.rejected_plans

    routing = next(item for item in first if item.optional_mechanisms["routing_aware_placement"])
    rejected = await planner.plan(
        model,
        cluster,
        ExecutionRequest(),
        mechanism_evidence=(
            MechanismEvidence(
                mechanism="routing_aware_placement",
                exactness_passed=False,
                measured_utility=-1,
                evidence_fingerprint="newer-negative-evidence",
                runtime_fingerprint=routing.execution_identity,
            ),
        ),
    )
    assert rejected.selected.fast_paths["node-a/worker"] == "backend-native"
    assert routing.plan_id in rejected.rejected_plans


def _native_worker(
    worker_id: str,
    *,
    memory: int,
    rate: float,
    role: str = "critical_path_stage",
    device_uuid: str | None = None,
) -> WorkerExecutionCapability:
    return WorkerExecutionCapability(
        worker_id=worker_id,
        node_id=worker_id.split("/", 1)[0],
        engines=(
            ExecutionEngineCapability(
                engine_id="native-stage",
                enabled=True,
                runtime_revision="native-v3",
                formats=("safetensors",),
                adapters=("qwen3_dense",),
                roles=(role,),
                devices=(
                    ExecutionDevice(
                        device_id="cuda:0",
                        device_type="cuda",
                        uuid=device_uuid,
                        name="test-gpu",
                        usable_memory_bytes=memory,
                        measured_decode_tokens_s=rate,
                        runtime_version="torch-test",
                        driver_version="driver-test",
                    ),
                ),
            ),
        ),
    )


def _qwen_model(*, weight_bytes: int = 100) -> ResolvedModelDescriptor:
    return _model().model_copy(
        update={
            "format": "safetensors",
            "architecture": "Qwen3ForCausalLM",
            "weight_bytes": weight_bytes,
            "layer_count": 4,
            "tokenizer_identity": "a" * 40,
        }
    )


@pytest.mark.asyncio
async def test_native_identity_uses_only_selected_stage_runtime_facts() -> None:
    engine = NativeStageEngine()
    fast = _native_worker(
        "node-a/worker",
        memory=200,
        rate=100,
        device_uuid="gpu-a",
    )
    slow = _native_worker(
        "node-b/worker",
        memory=200,
        rate=1,
        device_uuid="gpu-b",
    )
    first = await engine.candidate_plans(
        _qwen_model(),
        ClusterCapabilities(workers=(fast, slow)),
        ExecutionRequest(objective="speed"),
    )
    fast_plan = next(
        item for item in first if item.worker_roles == {fast.worker_id: "critical_path_stage"}
    )
    assert fast_plan.fast_paths[fast.worker_id] == "eager"

    unused = _native_worker(
        "node-c/storage",
        memory=1000,
        rate=1000,
        role="storage_cache",
        device_uuid="gpu-c",
    )
    second = await engine.candidate_plans(
        _qwen_model(),
        ClusterCapabilities(workers=(fast, slow, unused)),
        ExecutionRequest(objective="speed"),
    )
    repeated = next(
        item for item in second if item.worker_roles == {fast.worker_id: "critical_path_stage"}
    )
    assert repeated.execution_identity == fast_plan.execution_identity
    assert unused.worker_id not in repeated.worker_roles


@pytest.mark.asyncio
async def test_native_capacity_plan_uses_stage_specific_execution_identity() -> None:
    engine = NativeStageEngine()
    workers = (
        _native_worker("node-a/worker", memory=60, rate=10, device_uuid="gpu-a"),
        _native_worker("node-b/worker", memory=60, rate=10, device_uuid="gpu-b"),
    )
    plans = await engine.candidate_plans(
        _qwen_model(),
        ClusterCapabilities(workers=workers),
        ExecutionRequest(objective="capacity", require_distributed=True),
    )

    assert len(plans) == 1
    assert plans[0].topology == "direct-stage-ring-2"
    assert plans[0].predicted_messages_per_token == 2
    assert plans[0].predicted_serial_waits_per_token == 2
    assert plans[0].execution_identity.startswith("sha256:")
