from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

import pytest
import torch

from swarm_inference.config.models import (
    Backend,
    OperationKind,
    QueueConfig,
    StageBenchmark,
    WorkerCapability,
)
from swarm_inference.config.product import ProductCoordinatorConfig
from swarm_inference.coordinator.service import (
    CoordinatorClient,
    CoordinatorCore,
    CoordinatorRpcServer,
)
from swarm_inference.execution.interfaces import StageExecutionResult, WeightOwnership
from swarm_inference.model.partition import StageAssignment
from swarm_inference.model.product import (
    ModelResolutionPolicy,
    ProductModelReference,
    ProductModelSpec,
)
from swarm_inference.protocol.messages import StreamEventType, SubmitRequest
from swarm_inference.protocol.product import (
    ModelDeployRequest,
    ModelPlanRequest,
    ModelUnloadRequest,
    PlanCandidateReport,
    PlanWorkerAssignment,
    ProductStagePlan,
    ProductTokenPublication,
    StagePlanReport,
    TopologyStatusRequest,
)
from swarm_inference.protocol.stage_ring import STAGE_RING_PROTOCOL_VERSION
from swarm_inference.protocol.stage_worker import GetStageStatusRequest, LoadStageRequest
from swarm_inference.security.identity import WorkerIdentity
from swarm_inference.transport.grpc_transport import GrpcTransport
from swarm_inference.transport.stage_tensor import unpack_tensor
from swarm_inference.worker.agent import WorkerAgent
from swarm_inference.worker.stage_runtime import PersistentStageRuntime, TokenPublication
from swarm_inference.worker.stage_service import PersistentStageWorkerService

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REAL_MODEL_PATH = REPOSITORY_ROOT / "artifacts" / "models" / "colibri" / "source-b89a7c4bc24f"
REAL_REFERENCE_PATH = (
    REPOSITORY_ROOT
    / "artifacts"
    / "runs"
    / "experiment-010-correction-work"
    / "phase-6"
    / "local-correctness-references"
    / "code-01"
    / "reference.json"
)
REAL_MODEL_ID = "allenai/OLMoE-1B-7B-0125-Instruct"
REAL_MODEL_REVISION = "b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e"
REAL_TOKENIZER_REVISION = "sha256:d1e645ebd850d79567e531a3c103ac575d8e9cf45fa941420afc584b293438ea"


def _assignment(stage_id: int) -> StageAssignment:
    return StageAssignment(
        stage_id=stage_id,
        layer_start=stage_id,
        layer_end=stage_id + 1,
        layer_ids=(stage_id,),
        weight_bytes=1_024,
        estimated_compute_ns=1,
        measured_compute_ns=1,
        kv_cache_bytes_per_token=8,
        peak_temporary_bytes=64,
        activation_bytes=4,
        device="cpu",
        owns_embeddings=stage_id == 0,
        owns_final_norm=stage_id == 1,
        owns_output_projection=stage_id == 1,
    )


class _ProcessStageExecutor:
    def __init__(self, assignment: StageAssignment) -> None:
        self.assignment = assignment
        self.positions: dict[str, int] = {}
        self.ownership = WeightOwnership(
            stage_id=assignment.stage_id,
            layer_start=assignment.layer_start,
            layer_end=assignment.layer_end,
            parameter_names=(f"model.layers.{assignment.stage_id}.test_weight",),
            parameter_bytes=assignment.weight_bytes,
            parameter_count=1,
            owns_embeddings=assignment.owns_embeddings,
            owns_final_norm=assignment.owns_final_norm,
            owns_output_projection=assignment.owns_output_projection,
            ownership_hash=f"fake-stage-{assignment.stage_id}",
        )

    def open_session(self, session_id: str) -> None:
        if session_id in self.positions:
            raise ValueError("duplicate test session")
        self.positions[session_id] = 0

    def _result(
        self,
        *,
        session_id: str,
        values: torch.Tensor,
        cache_position_start: int,
    ) -> StageExecutionResult:
        if self.positions[session_id] != cache_position_start:
            raise AssertionError("test KV position crossed a session boundary")
        sequence_length = int(values.shape[-2] if values.ndim >= 3 else values.shape[-1])
        self.positions[session_id] += sequence_length
        hidden = values.to(torch.float32)
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(-1)
        sampled = None
        all_sampled = None
        if self.assignment.owns_output_projection:
            time.sleep(0.001)
            sampled = hidden[:, -1, 0].to(torch.int64) + 1
            all_sampled = sampled.unsqueeze(0)
        return StageExecutionResult(
            hidden_states=hidden,
            stage_boundary_hidden_states=hidden,
            router_logits=(),
            final_hidden_states=(hidden if self.assignment.owns_final_norm else None),
            logits=None,
            sampled_token_ids=sampled,
            all_sampled_token_ids=all_sampled,
            cache_sequence_length=self.positions[session_id],
            compute_ns=1,
        )

    def execute_prefill(
        self,
        *,
        session_id: str,
        token_ids: torch.Tensor,
        cache_position_start: int,
    ) -> StageExecutionResult:
        return self._result(
            session_id=session_id,
            values=token_ids,
            cache_position_start=cache_position_start,
        )

    def execute_decode(
        self,
        *,
        session_id: str,
        hidden_states: torch.Tensor,
        cache_position_start: int,
    ) -> StageExecutionResult:
        return self._result(
            session_id=session_id,
            values=hidden_states,
            cache_position_start=cache_position_start,
        )

    def close_session(self, session_id: str) -> int:
        return self.cancel_session(session_id)

    def cancel_session(self, session_id: str) -> int:
        return self.positions.pop(session_id) * 8

    def kv_cache_bytes(self, session_id: str) -> int:
        return self.positions[session_id] * 8

    def close(self) -> None:
        self.positions.clear()


def _worker_capability(worker_id: str, identity: WorkerIdentity) -> WorkerCapability:
    return WorkerCapability(
        worker_id=worker_id,
        public_key=identity.public_key_b64,
        hostname="localhost",
        operating_system="test",
        architecture="test",
        backend=Backend.TORCH_CPU,
        cpu_model="test",
        logical_cpu_count=1,
        total_ram_bytes=1024**3,
        available_ram_bytes=1024**3,
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
        upload_bandwidth_bytes_s=1_000_000,
        download_bandwidth_bytes_s=1_000_000,
        coordinator_latency_ms=0.1,
        memory_limit_bytes=1024**3,
        endpoint="127.0.0.1:1",
        control_endpoint="127.0.0.1:1",
        data_plane_endpoint="127.0.0.1:1",
        device_identifier="cpu",
        stage_ring_protocol_version=STAGE_RING_PROTOCOL_VERSION,
        supported_model_adapters=["olmoe"],
        supported_stage_execution_backends=["canonical-contiguous-olmoe"],
        supported_activation_dtypes=["float32"],
        configured_memory_limit_bytes=1024**3,
        stage_runtime_enabled=True,
    )


def _worker_process(
    worker_id: str,
    coordinator_endpoint: str,
    ready_queue: Any,
    stop_event: Any,
) -> None:
    async def run() -> None:
        identity = WorkerIdentity.generate()
        capability = _worker_capability(worker_id, identity)
        agent = WorkerAgent(
            capability=capability,
            identity=identity,
            queue_config=QueueConfig(capacity=32),
        )
        coordinator = CoordinatorClient(coordinator_endpoint, timeout_s=10)

        async def publish_token(publication: TokenPublication) -> None:
            message = publication.message
            metadata = message.attributes.get("tensor")
            if not isinstance(metadata, dict):
                raise ValueError("test token result has no tensor metadata")
            tensor, _ = unpack_tensor(message.payload, dict(metadata))
            product_publication = ProductTokenPublication(
                worker_id=worker_id,
                request_id=message.request_id,
                session_id=message.session_id,
                topology_id=message.topology_id,
                route_generation=int(message.attributes["route_generation"]),
                model_revision=message.model_revision,
                token_position=message.token_position,
                token_id=int(tensor.item()),
                published_monotonic_ns=time.monotonic_ns(),
            )
            first = await coordinator.publish_token(product_publication)
            if not first.accepted:
                raise RuntimeError(first.detail)
            duplicate = await coordinator.publish_token(product_publication)
            if not duplicate.accepted or "duplicate" not in duplicate.detail:
                raise RuntimeError("coordinator did not de-duplicate token publication")

        def loader(request: LoadStageRequest, _path: Path | None) -> _ProcessStageExecutor:
            return _ProcessStageExecutor(request.assignment)

        runtime = PersistentStageRuntime(
            worker_id=worker_id,
            device="cpu",
            dtype="float32",
            memory_limit_bytes=1024**3,
            maximum_sessions=256,
            execution_queue_capacity=32,
            token_queue_capacity=32,
            capability=capability,
            loader=loader,
            token_publisher=publish_token,
        )
        service = PersistentStageWorkerService(agent=agent, stage_runtime=runtime)
        try:
            control_port, data_port = await service.start(
                control_listen_endpoint="127.0.0.1:0",
                data_listen_endpoint="127.0.0.1:0",
            )
            assert data_port is not None
            capability.endpoint = f"127.0.0.1:{control_port}"
            capability.control_endpoint = capability.endpoint
            capability.data_plane_endpoint = f"127.0.0.1:{data_port}"
            ready_queue.put(
                {
                    "worker_id": worker_id,
                    "process_id": os.getpid(),
                    "capability": capability.model_dump(mode="json"),
                }
            )
            while not stop_event.is_set():
                await asyncio.sleep(0.05)
        except BaseException as exc:
            ready_queue.put(
                {
                    "worker_id": worker_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise
        finally:
            await service.stop(grace_s=0)
            await coordinator.close()

    asyncio.run(run())


class _NoActivationRelayTransport(GrpcTransport):
    def __init__(self) -> None:
        super().__init__(timeout_s=10)
        self.activation_relay_calls = 0

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        self.activation_relay_calls += 1
        raise AssertionError("legacy coordinator activation forwarding is disabled")


def _plan(capabilities: list[WorkerCapability]) -> ProductStagePlan:
    ordered = sorted(capabilities, key=lambda item: item.worker_id)
    assignments = [
        PlanWorkerAssignment(
            stage_id=stage_id,
            worker_id=capability.worker_id,
            control_endpoint=capability.control_endpoint or "",
            data_endpoint=capability.data_plane_endpoint or "",
            device="cpu",
            effective_memory_bytes=capability.effective_memory_bytes,
            required_memory_bytes=2_048,
            assignment=_assignment(stage_id),
        )
        for stage_id, capability in enumerate(ordered)
    ]
    candidate = PlanCandidateReport(
        name="two-stage-equal-ring",
        topology="two-stage-equal-ring",
        stage_count=2,
        partition_method="equal",
        feasible=True,
        selected=True,
        worker_ids=[item.worker_id for item in assignments],
        expected_critical_path_ms=2,
        expected_utility_tokens_s=500,
    )
    return ProductStagePlan(
        plan_id="plan-process-loopback",
        topology_id="topology-process-loopback",
        generation=1,
        created_monotonic_ns=time.monotonic_ns(),
        model=ProductModelSpec(
            model_id="test/olmoe",
            model_revision="model-commit",
            tokenizer_revision="tokenizer-commit",
            adapter_id="olmoe",
            dtype="float32",
            layer_count=2,
            hidden_size=1,
            metadata_hash="process-loopback-metadata",
            resolution_policy=ModelResolutionPolicy.LOCAL_ONLY,
        ),
        stage_count=2,
        partition_method="equal",
        max_sequence_tokens=256,
        assignments=assignments,
        report=StagePlanReport(
            selected_topology="two-stage-equal-ring",
            worker_assignments=assignments,
            reason_for_selection="pinned process-loopback plan",
            candidates=[candidate],
            worker_eligibility=[],
        ),
    )


async def _collect_stream(
    client: CoordinatorClient,
    request: SubmitRequest,
) -> list[Any]:
    return [event async for event in client.submit_stream(request)]


@pytest.mark.asyncio
async def test_two_process_product_ring_persists_streams_and_never_relays_activations(
    tmp_path: Path,
) -> None:
    transport = _NoActivationRelayTransport()
    core = CoordinatorCore(
        product_config=ProductCoordinatorConfig(
            event_queue_capacity=16,
            token_ingress_capacity=16,
            request_timeout_s=10,
            control_timeout_s=10,
        ),
        state_directory=tmp_path,
        transport=transport,
    )
    server = CoordinatorRpcServer(core)
    coordinator_port = await server.start("127.0.0.1:0")
    coordinator_endpoint = f"127.0.0.1:{coordinator_port}"
    context = multiprocessing.get_context("spawn")
    ready_queue = context.Queue()
    stop_event = context.Event()
    processes = [
        context.Process(
            target=_worker_process,
            args=(f"worker-{stage_id}", coordinator_endpoint, ready_queue, stop_event),
        )
        for stage_id in range(2)
    ]
    client = CoordinatorClient(coordinator_endpoint, timeout_s=15)
    inspector = GrpcTransport(timeout_s=10)
    try:
        for process in processes:
            process.start()
        startup = [await asyncio.to_thread(ready_queue.get, True, 30) for _ in processes]
        assert not [item for item in startup if "error" in item], startup
        capabilities = [WorkerCapability.model_validate(item["capability"]) for item in startup]
        for capability in capabilities:
            core.registry.register(capability, benchmark_verified=True)
        plan = _plan(capabilities)

        deployment = await client.deploy_model(ModelDeployRequest(plan=plan))
        assert deployment.deployment.ready
        assert {item.process_id for item in deployment.deployment.workers} == {
            process.pid for process in processes
        }
        assert len({item.process_id for item in deployment.deployment.workers}) == 2

        streamed = await _collect_stream(
            client,
            SubmitRequest(
                request_id="streamed",
                prompt_token_ids=[10, 20],
                max_new_tokens=3,
                random_seed=1,
                model_id=plan.model.model_id,
                model_revision=plan.model.model_revision,
            ),
        )
        assert [event.sequence_number for event in streamed] == list(range(len(streamed)))
        assert [event.event_type for event in streamed[:4]] == [
            StreamEventType.REQUEST_ACCEPTED,
            StreamEventType.TOPOLOGY_SELECTED,
            StreamEventType.SESSION_OPENED,
            StreamEventType.PREFILL_STARTED,
        ]
        token_events = [
            event for event in streamed if event.event_type == StreamEventType.TOKEN_GENERATED
        ]
        assert [event.token_position for event in token_events] == [0, 1, 2]
        assert [event.token_id for event in token_events] == [21, 22, 23]
        assert streamed[-2].event_type == StreamEventType.SESSION_CLOSED
        assert streamed[-1].event_type == StreamEventType.REQUEST_COMPLETED
        assert streamed[-1].final_token_ids == [21, 22, 23]
        assert streamed[-1].timing_metrics["end_to_end_s"] > 0

        unary = await client.submit(
            SubmitRequest(
                request_id="unary",
                prompt_token_ids=[30],
                max_new_tokens=2,
                random_seed=1,
                model_id=plan.model.model_id,
                model_revision=plan.model.model_revision,
            )
        )
        assert unary.status == "completed"
        assert unary.verified
        assert unary.output_token_ids == [31, 32]

        for index in range(100):
            response = await client.submit(
                SubmitRequest(
                    request_id=f"reuse-{index}",
                    prompt_token_ids=[index + 1],
                    max_new_tokens=1,
                    random_seed=1,
                    model_id=plan.model.model_id,
                    model_revision=plan.model.model_revision,
                )
            )
            assert response.output_token_ids == [index + 2]

        interleaved = await asyncio.gather(
            client.submit(
                SubmitRequest(
                    request_id="interleaved-a",
                    prompt_token_ids=[100],
                    max_new_tokens=3,
                    random_seed=1,
                    model_id=plan.model.model_id,
                    model_revision=plan.model.model_revision,
                )
            ),
            client.submit(
                SubmitRequest(
                    request_id="interleaved-b",
                    prompt_token_ids=[200],
                    max_new_tokens=3,
                    random_seed=1,
                    model_id=plan.model.model_id,
                    model_revision=plan.model.model_revision,
                )
            ),
        )
        assert interleaved[0].output_token_ids == [101, 102, 103]
        assert interleaved[1].output_token_ids == [201, 202, 203]

        disconnected = client.submit_stream(
            SubmitRequest(
                request_id="disconnect",
                prompt_token_ids=[300],
                max_new_tokens=100,
                random_seed=1,
                model_id=plan.model.model_id,
                model_revision=plan.model.model_revision,
            )
        )
        async for event in disconnected:
            if event.event_type == StreamEventType.SESSION_OPENED:
                break
        await disconnected.aclose()
        deadline = asyncio.get_running_loop().time() + 5
        while core.session_controller is not None and core.session_controller.active_count:
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("disconnected stream did not clean up its session")
            await asyncio.sleep(0.02)

        statuses = []
        for assignment in plan.assignments:
            statuses.append(
                await inspector.get_stage_status(
                    assignment.control_endpoint,
                    GetStageStatusRequest(
                        worker_id=assignment.worker_id,
                        request_id=f"status-{assignment.stage_id}",
                        topology_id=plan.topology_id,
                    ),
                )
            )
        assert all(status.loaded_stage is not None for status in statuses)
        assert all(
            status.loaded_stage.load_count == 1 for status in statuses if status.loaded_stage
        )
        assert all(not status.sessions for status in statuses)
        assert {status.process_id for status in statuses} == {process.pid for process in processes}
        topology = await client.topology_status(TopologyStatusRequest())
        assert topology.deployments[0].phase.value == "ready"
        assert transport.activation_relay_calls == 0
        assert core.runtime_transport_metrics["coordinator_activation_bytes"] == 0
        assert core.runtime_transport_metrics["coordinator_input_activation_bytes"] == 0

        unloaded = await client.unload_model(ModelUnloadRequest(topology_id=plan.topology_id))
        assert unloaded.deployment is not None
        assert unloaded.deployment.phase.value == "unloaded"
    finally:
        stop_event.set()
        for process in processes:
            if process.pid is not None:
                await asyncio.to_thread(process.join, 10)
                if process.is_alive():
                    process.terminate()
                    await asyncio.to_thread(process.join, 5)
        await inspector.close()
        await client.close()
        await server.stop(grace_s=0)


def _real_worker_capability(worker_id: str, identity: WorkerIdentity) -> WorkerCapability:
    memory_limit = 16_000_000_000
    return WorkerCapability(
        worker_id=worker_id,
        public_key=identity.public_key_b64,
        hostname="localhost",
        operating_system="test",
        architecture="test",
        backend=Backend.TORCH_CUDA,
        cpu_model="test",
        logical_cpu_count=1,
        total_ram_bytes=32_000_000_000,
        available_ram_bytes=16_000_000_000,
        gpu_model="test CUDA device",
        total_vram_bytes=32_000_000_000,
        available_vram_bytes=memory_limit,
        supported_dtypes=["bfloat16"],
        stage_benchmarks=[
            StageBenchmark(
                worker_class="cuda-test",
                operation=OperationKind.DECODE,
                sequence_length=1,
                batch_size=1,
                mean_ms=1,
                p95_ms=1,
                samples=3,
                measured=True,
            )
        ],
        upload_bandwidth_bytes_s=10_000_000_000,
        download_bandwidth_bytes_s=10_000_000_000,
        coordinator_latency_ms=0.1,
        memory_limit_bytes=memory_limit,
        endpoint="127.0.0.1:1",
        control_endpoint="127.0.0.1:1",
        data_plane_endpoint="127.0.0.1:1",
        device_identifier="cuda:0",
        stage_ring_protocol_version=STAGE_RING_PROTOCOL_VERSION,
        supported_model_adapters=["olmoe"],
        supported_stage_execution_backends=["canonical-contiguous-olmoe"],
        supported_activation_dtypes=["bfloat16"],
        configured_memory_limit_bytes=memory_limit,
        stage_runtime_enabled=True,
    )


def _real_worker_process(
    worker_id: str,
    coordinator_endpoint: str,
    model_path: str,
    ready_queue: Any,
    stop_event: Any,
) -> None:
    async def run() -> None:
        identity = WorkerIdentity.generate()
        capability = _real_worker_capability(worker_id, identity)
        agent = WorkerAgent(
            capability=capability,
            identity=identity,
            queue_config=QueueConfig(capacity=8),
        )
        coordinator = CoordinatorClient(coordinator_endpoint, timeout_s=180)

        async def publish_token(publication: TokenPublication) -> None:
            message = publication.message
            metadata = message.attributes.get("tensor")
            if not isinstance(metadata, dict):
                raise ValueError("real token result has no tensor metadata")
            tensor, _ = unpack_tensor(message.payload, dict(metadata))
            response = await coordinator.publish_token(
                ProductTokenPublication(
                    worker_id=worker_id,
                    request_id=message.request_id,
                    session_id=message.session_id,
                    topology_id=message.topology_id,
                    route_generation=int(message.attributes["route_generation"]),
                    model_revision=message.model_revision,
                    token_position=message.token_position,
                    token_id=int(tensor.item()),
                    published_monotonic_ns=time.monotonic_ns(),
                )
            )
            if not response.accepted:
                raise RuntimeError(response.detail)

        runtime = PersistentStageRuntime(
            worker_id=worker_id,
            device="cuda:0",
            dtype="bfloat16",
            memory_limit_bytes=16_000_000_000,
            maximum_sessions=4,
            execution_queue_capacity=8,
            token_queue_capacity=8,
            configured_model_path=model_path,
            capability=capability,
            token_publisher=publish_token,
        )
        service = PersistentStageWorkerService(agent=agent, stage_runtime=runtime)
        try:
            control_port, data_port = await service.start(
                control_listen_endpoint="127.0.0.1:0",
                data_listen_endpoint="127.0.0.1:0",
            )
            assert data_port is not None
            capability.endpoint = f"127.0.0.1:{control_port}"
            capability.control_endpoint = capability.endpoint
            capability.data_plane_endpoint = f"127.0.0.1:{data_port}"
            ready_queue.put(
                {
                    "worker_id": worker_id,
                    "process_id": os.getpid(),
                    "capability": capability.model_dump(mode="json"),
                }
            )
            while not stop_event.is_set():
                await asyncio.sleep(0.1)
        except BaseException as exc:
            ready_queue.put(
                {
                    "worker_id": worker_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise
        finally:
            await service.stop(grace_s=0)
            await coordinator.close()

    asyncio.run(run())


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("SWARM_RUN_PRODUCT_OLMOE_CUDA") != "1"
    or not torch.cuda.is_available()
    or not REAL_MODEL_PATH.is_dir()
    or not REAL_REFERENCE_PATH.is_file(),
    reason=(
        "set SWARM_RUN_PRODUCT_OLMOE_CUDA=1 with the pinned local OLMoE snapshot "
        "for the exact product-path test"
    ),
)
@pytest.mark.asyncio
async def test_exact_olmoe_cuda_product_path_uses_two_persistent_workers(
    tmp_path: Path,
) -> None:
    reference = json.loads(REAL_REFERENCE_PATH.read_text(encoding="utf-8"))
    prompt_ids = [int(value) for value in reference["prompt_ids"]]
    expected = [int(value) for value in reference["full_ids"][len(prompt_ids) :]][:2]
    transport = _NoActivationRelayTransport()
    core = CoordinatorCore(
        product_config=ProductCoordinatorConfig(
            event_queue_capacity=16,
            token_ingress_capacity=16,
            request_timeout_s=180,
            control_timeout_s=180,
        ),
        state_directory=tmp_path,
        transport=transport,
    )
    server = CoordinatorRpcServer(core)
    coordinator_port = await server.start("127.0.0.1:0")
    coordinator_endpoint = f"127.0.0.1:{coordinator_port}"
    context = multiprocessing.get_context("spawn")
    ready_queue = context.Queue()
    stop_event = context.Event()
    processes = [
        context.Process(
            target=_real_worker_process,
            args=(
                f"cuda-worker-{stage_id}",
                coordinator_endpoint,
                str(REAL_MODEL_PATH),
                ready_queue,
                stop_event,
            ),
        )
        for stage_id in range(2)
    ]
    client = CoordinatorClient(coordinator_endpoint, timeout_s=240)
    inspector = GrpcTransport(timeout_s=180)
    try:
        for process in processes:
            process.start()
        startup = [await asyncio.to_thread(ready_queue.get, True, 60) for _ in processes]
        assert not [item for item in startup if "error" in item], startup
        capabilities = [WorkerCapability.model_validate(item["capability"]) for item in startup]
        for capability in capabilities:
            core.registry.register(capability, benchmark_verified=True)

        planned = await client.plan_model(
            ModelPlanRequest(
                reference=ProductModelReference(
                    model_id=REAL_MODEL_ID,
                    model_revision=REAL_MODEL_REVISION,
                    tokenizer_revision=REAL_TOKENIZER_REVISION,
                    adapter_id="olmoe",
                    dtype="bfloat16",
                    resolution_policy=ModelResolutionPolicy.LOCAL_ONLY,
                ),
                stage_count=2,
                partition_method="equal",
                require_distributed=True,
                max_sequence_tokens=len(prompt_ids) + 2,
            )
        )
        assert planned.plan.stage_count == 2
        assert planned.plan.report.selected_topology == "two-stage-equal-ring"
        deployed = await client.deploy_model(ModelDeployRequest(plan=planned.plan))
        worker_pids = {item.process_id for item in deployed.deployment.workers}
        assert worker_pids == {process.pid for process in processes}
        assert len(worker_pids) == 2

        events = await _collect_stream(
            client,
            SubmitRequest(
                request_id="exact-olmoe-product",
                prompt_token_ids=prompt_ids,
                max_new_tokens=2,
                random_seed=1,
                model_id=REAL_MODEL_ID,
                model_revision=REAL_MODEL_REVISION,
            ),
        )
        token_ids = [
            event.token_id
            for event in events
            if event.event_type == StreamEventType.TOKEN_GENERATED
        ]
        assert token_ids == expected
        assert events[-1].event_type == StreamEventType.REQUEST_COMPLETED
        assert events[-1].final_token_ids == expected

        statuses = []
        for assignment in planned.plan.assignments:
            statuses.append(
                await inspector.get_stage_status(
                    assignment.control_endpoint,
                    GetStageStatusRequest(
                        worker_id=assignment.worker_id,
                        request_id=f"real-status-{assignment.stage_id}",
                        topology_id=planned.plan.topology_id,
                    ),
                )
            )
        assert all(status.loaded_stage is not None for status in statuses)
        assert all(
            status.loaded_stage.load_count == 1 for status in statuses if status.loaded_stage
        )
        assert all(not status.sessions for status in statuses)
        assert {status.process_id for status in statuses} == worker_pids
        assert transport.activation_relay_calls == 0
        assert core.runtime_transport_metrics["coordinator_activation_bytes"] == 0

        unloaded = await client.unload_model(
            ModelUnloadRequest(topology_id=planned.plan.topology_id)
        )
        assert unloaded.deployment is not None
        assert unloaded.deployment.phase.value == "unloaded"
    finally:
        stop_event.set()
        for process in processes:
            if process.pid is not None:
                await asyncio.to_thread(process.join, 30)
                if process.is_alive():
                    process.terminate()
                    await asyncio.to_thread(process.join, 5)
        await inspector.close()
        await client.close()
        await server.stop(grace_s=0)
