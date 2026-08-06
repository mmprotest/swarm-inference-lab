from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import queue
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

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
from swarm_inference.exceptions import TransportError
from swarm_inference.execution.interfaces import StageExecutionResult, WeightOwnership
from swarm_inference.model.partition import StageAssignment
from swarm_inference.model.product import (
    ModelResolutionPolicy,
    ProductModelReference,
    ProductModelSpec,
)
from swarm_inference.protocol.messages import RegistrationRequest, StreamEventType, SubmitRequest
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
from swarm_inference.security.identity import (
    WorkerIdentity,
    create_identity_file,
    public_key_fingerprint,
)
from swarm_inference.security.signatures import canonical_json_bytes
from swarm_inference.security.trust_store import WorkerTrustStore
from swarm_inference.testing.process_harness import ProductCluster
from swarm_inference.transport.grpc_transport import GrpcTransport
from swarm_inference.transport.stage_ring_connection import StageRingConnectionPool
from swarm_inference.transport.stage_ring_faults import CloseConnectionBeforeSendInjector
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
INTEGRATION_CONTROL_TIMEOUT_S = 60.0
INTEGRATION_REQUEST_TIMEOUT_S = 120.0
INTEGRATION_RECOVERY_TIMEOUT_S = 120.0
INTEGRATION_CLIENT_TIMEOUT_S = 180.0
INTEGRATION_STARTUP_TIMEOUT_S = 60.0


def _write_real_gate_evidence(name: str, payload: dict[str, Any]) -> None:
    configured = os.environ.get("SWARM_ACCEPTANCE_GATE_EVIDENCE")
    if not configured:
        return
    directory = Path(configured).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _real_evidence_provenance(reference: dict[str, Any]) -> dict[str, Any]:
    metadata_hasher = hashlib.sha256()
    metadata_files = (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
    )
    for relative in metadata_files:
        path = REAL_MODEL_PATH / relative
        if path.is_file():
            metadata_hasher.update(relative.encode())
            metadata_hasher.update(path.read_bytes())
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "model_metadata_hash": f"sha256:{metadata_hasher.hexdigest()}",
        "prompt": str(reference.get("prompt", reference.get("text", ""))),
        "git_commit": git_commit,
        "git_dirty": bool(git_status),
        "git_status": git_status,
        "environment": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "os": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
        },
    }


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
    def __init__(
        self,
        assignment: StageAssignment,
        *,
        token_delay_s: float = 0.001,
        token_offset: int = 0,
    ) -> None:
        self.assignment = assignment
        self.token_delay_s = token_delay_s
        self.token_offset = token_offset
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
            time.sleep(self.token_delay_s)
            sampled = hidden[:, -1, 0].to(torch.int64) + 1 + self.token_offset
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
    token_delay_s: float = 0.001,
    token_offset: int = 0,
    data_plane_stop_event: Any | None = None,
    trusted_coordinator: tuple[str, str] | None = None,
    fault_token_position: int | None = None,
    fault_event_queue: Any | None = None,
    identity_path: str | None = None,
    registration_commands: Any | None = None,
    registration_results: Any | None = None,
) -> None:
    async def run() -> None:
        identity = (
            WorkerIdentity.load(identity_path)
            if identity_path is not None
            else WorkerIdentity.generate()
        )
        capability = _worker_capability(worker_id, identity)
        agent = WorkerAgent(
            capability=capability,
            identity=identity,
            queue_config=QueueConfig(capacity=32),
        )
        coordinator = CoordinatorClient(
            coordinator_endpoint,
            timeout_s=INTEGRATION_CONTROL_TIMEOUT_S,
        )

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
                request_generation=int(message.attributes.get("request_generation", 1)),
                replay_only=bool(message.attributes.get("replay_only", False)),
            )
            product_publication = product_publication.model_copy(
                update={
                    "signature": identity.sign(
                        canonical_json_bytes(
                            product_publication.model_dump(
                                mode="json",
                                exclude={"signature"},
                            )
                        )
                    )
                }
            )
            first = await coordinator.publish_token(product_publication)
            if not first.accepted:
                raise RuntimeError(first.detail)
            duplicate = await coordinator.publish_token(product_publication)
            if not duplicate.accepted or "duplicate" not in duplicate.detail:
                raise RuntimeError("coordinator did not de-duplicate token publication")

        def loader(request: LoadStageRequest, _path: Path | None) -> _ProcessStageExecutor:
            return _ProcessStageExecutor(
                request.assignment,
                token_delay_s=token_delay_s,
                token_offset=token_offset,
            )

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
            identity=identity,
            trusted_coordinators=(
                {trusted_coordinator[0]: trusted_coordinator[1]}
                if trusted_coordinator is not None
                else None
            ),
            require_authenticated_routes=trusted_coordinator is not None,
        )
        if fault_token_position is not None:
            await runtime.connection_pool.close()
            injector = CloseConnectionBeforeSendInjector(
                token_position=fault_token_position,
                source_stage=0,
                destination_stage=1,
                request_id="socket-recovery",
                event_sink=(fault_event_queue.put if fault_event_queue is not None else None),
            )
            runtime.connection_pool = StageRingConnectionPool(
                queue_capacity=32,
                reconnect_attempts=1,
                handshake_factory=runtime._peer_handshake_message,
                handshake_verifier=runtime._verify_peer_handshake_response,
                fault_injector=injector,
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
            data_plane_stopped = False
            while not stop_event.is_set():
                if registration_commands is not None and registration_results is not None:
                    try:
                        command = registration_commands.get_nowait()
                    except queue.Empty:
                        command = None
                    if command == "register":
                        nonce = f"{worker_id}:{uuid4().hex}"
                        signed_payload = canonical_json_bytes(
                            {
                                "capability": capability.model_dump(mode="json"),
                                "benchmark_nonce": nonce,
                            }
                        )
                        registration = RegistrationRequest(
                            capability=capability,
                            benchmark_nonce=nonce,
                            signature=identity.sign(signed_payload),
                        )
                        try:
                            response = await coordinator.register(registration)
                        except Exception as exc:
                            registration_results.put(
                                {
                                    "worker_id": worker_id,
                                    "accepted": False,
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                        else:
                            registration_results.put(
                                {
                                    "worker_id": worker_id,
                                    "accepted": response.accepted,
                                    "coordinator_fingerprint": (
                                        response.coordinator_public_key_fingerprint
                                    ),
                                }
                            )
                if (
                    not data_plane_stopped
                    and data_plane_stop_event is not None
                    and data_plane_stop_event.is_set()
                ):
                    assert service.data_server is not None
                    await service.data_server.stop()
                    data_plane_stopped = True
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
async def test_product_route_lease_and_direct_peer_handshake_are_authenticated(
    tmp_path: Path,
) -> None:
    config = ProductCoordinatorConfig(
        event_queue_capacity=16,
        token_ingress_capacity=16,
        request_timeout_s=5,
        control_timeout_s=5,
        require_trusted_workers=False,
    )
    core = CoordinatorCore(
        product_config=config,
        state_directory=tmp_path,
    )
    assert core.coordinator_identity is not None
    coordinator_trust = (
        config.coordinator_id,
        core.coordinator_identity.public_key_b64,
    )
    server = CoordinatorRpcServer(core)
    coordinator_port = await server.start("127.0.0.1:0")
    coordinator_endpoint = f"127.0.0.1:{coordinator_port}"
    cluster = ProductCluster()
    ready_queue = cluster.queue()
    stop_event = cluster.event()
    for index in range(2):
        cluster.process(
            f"worker-{index}",
            target=_worker_process,
            args=(
                f"worker-{index}",
                coordinator_endpoint,
                ready_queue,
                stop_event,
                0.01,
                0,
                None,
                coordinator_trust,
            ),
            shutdown=stop_event.set,
        )
    client = CoordinatorClient(coordinator_endpoint, timeout_s=20)
    inspector = GrpcTransport(timeout_s=5)
    try:
        cluster.start()
        startup = await cluster.wait_ready(ready_queue, count=2, timeout=30)
        assert not [item for item in startup if "error" in item], startup
        capabilities = sorted(
            (WorkerCapability.model_validate(item["capability"]) for item in startup),
            key=lambda item: item.worker_id,
        )
        for capability in capabilities:
            core.registry.register(capability, benchmark_verified=True)
        plan = _plan(capabilities)
        deployed = await client.deploy_model(ModelDeployRequest(plan=plan))
        assert deployed.deployment.ready

        for item in plan.assignments:
            status = await inspector.get_stage_status(
                item.control_endpoint,
                GetStageStatusRequest(
                    worker_id=item.worker_id,
                    request_id=f"authenticated:{item.stage_id}",
                    topology_id=plan.topology_id,
                ),
            )
            assert status.installed_route is not None
            assert status.installed_route.authenticated
            assert status.installed_route.route_lease_hash is not None

        response = await client.submit(
            SubmitRequest(
                request_id="authenticated-request",
                prompt_token_ids=[10, 20],
                max_new_tokens=2,
                random_seed=1,
                model_id=plan.model.model_id,
                model_revision=plan.model.model_revision,
            )
        )
        assert response.status == "completed"
        assert response.output_token_ids == [21, 22]
    finally:
        await inspector.close()
        await client.close()
        await cluster.close()
        await server.stop(grace_s=0)


@pytest.mark.asyncio
async def test_secure_identity_bootstrap_registration_route_and_revocation(
    tmp_path: Path,
) -> None:
    coordinator_state = tmp_path / "coordinator"
    _, coordinator_metadata = create_identity_file(
        coordinator_state / "coordinator-identity.json",
        kind="coordinator",
    )
    worker_paths = [tmp_path / "identities" / f"worker-{index}.json" for index in range(2)]
    worker_metadata = [create_identity_file(path, kind="worker")[1] for path in worker_paths]
    trust_store = WorkerTrustStore(coordinator_state / "trusted-workers.json")

    core = CoordinatorCore(
        product_config=ProductCoordinatorConfig(
            event_queue_capacity=16,
            token_ingress_capacity=16,
            request_timeout_s=5,
            control_timeout_s=5,
            worker_heartbeat_timeout_s=30,
            require_trusted_workers=True,
            trust_store_path=trust_store.path,
        ),
        state_directory=coordinator_state,
    )
    assert core.coordinator_identity is not None
    assert core.coordinator_identity.public_key_fingerprint == coordinator_metadata.fingerprint
    coordinator_trust = (
        core.product_config.coordinator_id,
        core.coordinator_identity.public_key_b64,
    )
    server = CoordinatorRpcServer(core)
    coordinator_port = await server.start("127.0.0.1:0")
    coordinator_endpoint = f"127.0.0.1:{coordinator_port}"
    cluster = ProductCluster()
    ready_queue = cluster.queue()
    registration_results = cluster.queue()
    registration_commands = [cluster.queue() for _ in range(2)]
    stop_event = cluster.event()
    for index in range(2):
        cluster.process(
            f"secure-worker-{index}",
            target=_worker_process,
            args=(
                f"worker-{index}",
                coordinator_endpoint,
                ready_queue,
                stop_event,
                0.05,
                0,
                None,
                coordinator_trust,
                None,
                None,
                str(worker_paths[index]),
                registration_commands[index],
                registration_results,
            ),
            shutdown=stop_event.set,
        )
    client = CoordinatorClient(coordinator_endpoint, timeout_s=20)
    inspector = GrpcTransport(timeout_s=5)

    async def register(index: int) -> dict[str, Any]:
        registration_commands[index].put("register")
        result = await asyncio.to_thread(registration_results.get, True, 10)
        assert result["worker_id"] == f"worker-{index}"
        return dict(result)

    try:
        cluster.start()
        startup = await cluster.wait_ready(ready_queue, count=2, timeout=30)
        capabilities = sorted(
            (WorkerCapability.model_validate(item["capability"]) for item in startup),
            key=lambda item: item.worker_id,
        )

        rejected = await register(0)
        assert rejected["accepted"] is False
        assert worker_metadata[0].fingerprint in rejected["error"]
        assert "swarm identity trust" in rejected["error"]
        assert core.registry.workers() == []

        trust_store.trust(worker_metadata[0].fingerprint, label="worker-0")
        trust_store.trust(worker_metadata[1].fingerprint, label="worker-1")
        accepted_zero = await register(0)
        accepted_one = await register(1)
        assert accepted_zero["accepted"] and accepted_one["accepted"]
        assert accepted_zero["coordinator_fingerprint"] == coordinator_metadata.fingerprint
        assert accepted_one["coordinator_fingerprint"] == coordinator_metadata.fingerprint
        assert [item.worker_id for item in core.registry.workers()] == ["worker-0", "worker-1"]

        plan = _plan(capabilities)
        deployed = await client.deploy_model(ModelDeployRequest(plan=plan))
        assert deployed.deployment.ready
        for assignment in plan.assignments:
            status = await inspector.get_stage_status(
                assignment.control_endpoint,
                GetStageStatusRequest(
                    worker_id=assignment.worker_id,
                    request_id=f"secure-route:{assignment.stage_id}",
                    topology_id=plan.topology_id,
                ),
            )
            assert status.installed_route is not None
            assert status.installed_route.authenticated

        events: list[Any] = []
        revoked = False
        async for event in client.submit_stream(
            SubmitRequest(
                request_id="secure-active-session",
                prompt_token_ids=[10, 20],
                max_new_tokens=3,
                random_seed=1,
                model_id=plan.model.model_id,
                model_revision=plan.model.model_revision,
            )
        ):
            events.append(event)
            if event.event_type == StreamEventType.TOKEN_GENERATED and not revoked:
                revoked = trust_store.untrust(worker_metadata[1].fingerprint)
        assert revoked
        assert [
            event.token_id
            for event in events
            if event.event_type == StreamEventType.TOKEN_GENERATED
        ] == [21, 22, 23]
        assert events[-1].event_type == StreamEventType.REQUEST_COMPLETED

        rejected_again = await register(1)
        assert rejected_again["accepted"] is False
        assert "is not trusted" in rejected_again["error"]
        with pytest.raises(TransportError, match="worker trust was removed or is absent"):
            await client.deploy_model(ModelDeployRequest(plan=plan))
        blocked = await client.submit(
            SubmitRequest(
                request_id="secure-new-session-blocked",
                prompt_token_ids=[1],
                max_new_tokens=1,
                random_seed=1,
                model_id=plan.model.model_id,
                model_revision=plan.model.model_revision,
            )
        )
        assert blocked.status == "failed"
        assert "worker trust was removed or is absent" in blocked.detail
    finally:
        await inspector.close()
        await client.close()
        await cluster.close()
        await server.stop(grace_s=0)


@pytest.mark.asyncio
async def test_two_process_product_ring_persists_streams_and_never_relays_activations(
    tmp_path: Path,
) -> None:
    transport = _NoActivationRelayTransport()
    core = CoordinatorCore(
        product_config=ProductCoordinatorConfig(
            event_queue_capacity=16,
            token_ingress_capacity=16,
            request_timeout_s=INTEGRATION_REQUEST_TIMEOUT_S,
            control_timeout_s=INTEGRATION_CONTROL_TIMEOUT_S,
            require_trusted_workers=False,
        ),
        state_directory=tmp_path,
        transport=transport,
    )
    server = CoordinatorRpcServer(core)
    coordinator_port = await server.start("127.0.0.1:0")
    coordinator_endpoint = f"127.0.0.1:{coordinator_port}"
    cluster = ProductCluster()
    ready_queue = cluster.queue()
    stop_event = cluster.event()
    processes = [
        cluster.process(
            f"worker-{stage_id}",
            target=_worker_process,
            args=(f"worker-{stage_id}", coordinator_endpoint, ready_queue, stop_event),
            shutdown=stop_event.set,
        )
        for stage_id in range(2)
    ]
    client = CoordinatorClient(coordinator_endpoint, timeout_s=INTEGRATION_CLIENT_TIMEOUT_S)
    inspector = GrpcTransport(timeout_s=INTEGRATION_CONTROL_TIMEOUT_S)
    try:
        cluster.start()
        startup = await cluster.wait_ready(
            ready_queue,
            count=len(processes),
            timeout=INTEGRATION_STARTUP_TIMEOUT_S,
        )
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
            assert response.status == "completed", response.model_dump(mode="json")
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
        await inspector.close()
        await client.close()
        await cluster.close()
        await server.stop(grace_s=0)


@pytest.mark.asyncio
async def test_three_worker_restart_and_replay_replaces_failed_stage_without_duplicates(
    tmp_path: Path,
) -> None:
    transport = _NoActivationRelayTransport()
    core = CoordinatorCore(
        product_config=ProductCoordinatorConfig(
            event_queue_capacity=32,
            token_ingress_capacity=32,
            request_timeout_s=INTEGRATION_REQUEST_TIMEOUT_S,
            control_timeout_s=INTEGRATION_CONTROL_TIMEOUT_S,
            recovery_timeout_s=INTEGRATION_RECOVERY_TIMEOUT_S,
            cleanup_timeout_s=15,
            worker_heartbeat_timeout_s=30,
            require_trusted_workers=False,
        ),
        state_directory=tmp_path,
        transport=transport,
    )
    server = CoordinatorRpcServer(core)
    coordinator_port = await server.start("127.0.0.1:0")
    coordinator_endpoint = f"127.0.0.1:{coordinator_port}"
    cluster = ProductCluster()
    ready_queue = cluster.queue()
    stop_event = cluster.event()
    processes = {
        f"worker-{index}": cluster.process(
            f"worker-{index}",
            target=_worker_process,
            args=(
                f"worker-{index}",
                coordinator_endpoint,
                ready_queue,
                stop_event,
                0.1,
            ),
            shutdown=stop_event.set,
        )
        for index in range(3)
    }
    client = CoordinatorClient(coordinator_endpoint, timeout_s=INTEGRATION_CLIENT_TIMEOUT_S)
    try:
        cluster.start()
        startup = await cluster.wait_ready(
            ready_queue,
            count=len(processes),
            timeout=INTEGRATION_STARTUP_TIMEOUT_S,
        )
        assert not [item for item in startup if "error" in item], startup
        capabilities = sorted(
            (WorkerCapability.model_validate(item["capability"]) for item in startup),
            key=lambda item: item.worker_id,
        )
        for capability in capabilities:
            core.registry.register(capability, benchmark_verified=True)

        plan = _plan(capabilities[:2])
        deployed = await client.deploy_model(ModelDeployRequest(plan=plan))
        assert deployed.deployment.ready
        assert [item.worker_id for item in deployed.deployment.workers] == [
            "worker-0",
            "worker-1",
        ]

        events: list[Any] = []
        failed_worker_terminated = False
        async for event in client.submit_stream(
            SubmitRequest(
                request_id="recover-after-token",
                prompt_token_ids=[10, 20],
                max_new_tokens=6,
                random_seed=1,
                model_id=plan.model.model_id,
                model_revision=plan.model.model_revision,
            )
        ):
            events.append(event)
            if event.event_type == StreamEventType.TOKEN_GENERATED and not failed_worker_terminated:
                failed_worker_terminated = True
                await asyncio.to_thread(
                    cluster.crash_process,
                    "worker-1",
                    reason="deterministic worker-process recovery injection",
                    timeout=5,
                )

        assert failed_worker_terminated
        token_events = [
            event for event in events if event.event_type == StreamEventType.TOKEN_GENERATED
        ]
        event_diagnostics = [event.model_dump(mode="json") for event in events]
        assert [event.token_position for event in token_events] == list(range(6)), event_diagnostics
        assert [event.token_id for event in token_events] == [21, 22, 23, 24, 25, 26], (
            event_diagnostics
        )
        assert events[-1].event_type == StreamEventType.REQUEST_COMPLETED
        assert events[-1].final_token_ids == [21, 22, 23, 24, 25, 26]
        assert sum(event.event_type == StreamEventType.RECOVERY_STARTED for event in events) == 1
        assert sum(event.event_type == StreamEventType.RECOVERY_COMPLETED for event in events) == 1

        topology = await client.topology_status(TopologyStatusRequest())
        assert topology.deployments[0].generation == 2
        assert [worker.worker_id for worker in topology.deployments[0].workers] == [
            "worker-0",
            "worker-2",
        ]
        status = await client.status()
        assert status.recovery_count == 1
        assert transport.activation_relay_calls == 0
        assert core.runtime_transport_metrics["coordinator_activation_bytes"] == 0
        assert core.product_telemetry is not None
        event_types = [item["event_type"] for item in core.product_telemetry.events]
        assert event_types.count("recovery_started") == 1
        assert event_types.count("replacement_selected") == 1
        assert event_types.count("replay_token_verified") >= 1
        assert event_types.count("recovery_completed") == 1
    finally:
        await client.close()
        await cluster.close()
        await server.stop(grace_s=0)


@pytest.mark.asyncio
async def test_deterministic_active_socket_closure_recovers_without_duplicate_tokens(
    tmp_path: Path,
) -> None:
    core = CoordinatorCore(
        product_config=ProductCoordinatorConfig(
            event_queue_capacity=32,
            token_ingress_capacity=32,
            request_timeout_s=5,
            control_timeout_s=2,
            recovery_timeout_s=15,
            cleanup_timeout_s=2,
            worker_heartbeat_timeout_s=30,
            require_trusted_workers=False,
        ),
        state_directory=tmp_path,
    )
    server = CoordinatorRpcServer(core)
    coordinator_port = await server.start("127.0.0.1:0")
    coordinator_endpoint = f"127.0.0.1:{coordinator_port}"
    cluster = ProductCluster()
    ready_queue = cluster.queue()
    fault_events = cluster.queue()
    stop_event = cluster.event()
    processes = {
        f"worker-{index}": cluster.process(
            f"worker-{index}",
            target=_worker_process,
            args=(
                f"worker-{index}",
                coordinator_endpoint,
                ready_queue,
                stop_event,
                0.15,
                0,
                None,
                None,
                1 if index == 0 else None,
                fault_events if index == 0 else None,
            ),
            shutdown=stop_event.set,
        )
        for index in range(3)
    }
    client = CoordinatorClient(coordinator_endpoint, timeout_s=30)
    inspector = GrpcTransport(timeout_s=5)
    try:
        cluster.start()
        startup = await cluster.wait_ready(ready_queue, count=len(processes), timeout=30)
        assert not [item for item in startup if "error" in item], startup
        capabilities = sorted(
            (WorkerCapability.model_validate(item["capability"]) for item in startup),
            key=lambda item: item.worker_id,
        )
        for capability in capabilities:
            core.registry.register(capability, benchmark_verified=True)
        plan = _plan(capabilities[:2])
        assert (await client.deploy_model(ModelDeployRequest(plan=plan))).deployment.ready

        events: list[Any] = []
        async for event in client.submit_stream(
            SubmitRequest(
                request_id="socket-recovery",
                prompt_token_ids=[10, 20],
                max_new_tokens=4,
                random_seed=1,
                model_id=plan.model.model_id,
                model_revision=plan.model.model_revision,
            )
        ):
            events.append(event)

        fault_event = await asyncio.to_thread(fault_events.get, True, 5)
        assert fault_event["event_type"] == "stage_ring_fault_injected"
        assert fault_event["action"] == "close_active_connection"
        assert fault_event["active_connection"] is True
        assert fault_event["request_id"] == "socket-recovery"
        assert fault_event["route_generation"] == 1
        assert fault_event["operation"] == "DECODE"
        assert fault_event["source_stage"] == 0
        assert fault_event["destination_stage"] == 1
        assert fault_event["token_position"] == 1
        assert processes["worker-1"].is_alive()
        control_status = await inspector.get_stage_status(
            capabilities[1].control_endpoint or "",
            GetStageStatusRequest(
                worker_id="worker-1",
                request_id="control-still-live",
                topology_id=plan.topology_id,
            ),
        )
        assert control_status.loaded_stage is not None
        token_events = [
            event for event in events if event.event_type == StreamEventType.TOKEN_GENERATED
        ]
        assert [event.token_position for event in token_events] == [0, 1, 2, 3]
        assert [event.token_id for event in token_events] == [21, 22, 23, 24]
        assert len({(event.token_position, event.token_id) for event in token_events}) == 4
        assert sum(event.event_type == StreamEventType.RECOVERY_STARTED for event in events) == 1
        assert sum(event.event_type == StreamEventType.RECOVERY_COMPLETED for event in events) == 1
        topology = await client.topology_status(TopologyStatusRequest())
        assert topology.deployments[0].generation == 2
        assert [worker.worker_id for worker in topology.deployments[0].workers] == [
            "worker-0",
            "worker-2",
        ]
        healthy, _ = core.registry.registration_health("worker-1")
        assert not healthy
        assert core.product_telemetry is not None
        event_types = [item["event_type"] for item in core.product_telemetry.events]
        assert event_types.count("recovery_started") == 1
        assert event_types.count("replacement_selected") == 1
        route_events = [
            item
            for item in core.product_telemetry.events
            if item["event_type"] == "route_generation_installed"
        ]
        assert [item["route_generation"] for item in route_events] == [1, 2]
        assert event_types.count("replay_token_verified") == 1
        assert event_types.count("recovery_completed") == 1
    finally:
        await inspector.close()
        await client.close()
        await cluster.close()
        await server.stop(grace_s=0)


@pytest.mark.asyncio
async def test_restart_and_replay_divergence_fails_before_any_duplicate_token_event(
    tmp_path: Path,
) -> None:
    core = CoordinatorCore(
        product_config=ProductCoordinatorConfig(
            event_queue_capacity=32,
            token_ingress_capacity=32,
            request_timeout_s=5,
            control_timeout_s=2,
            recovery_timeout_s=15,
            cleanup_timeout_s=2,
            worker_heartbeat_timeout_s=30,
            require_trusted_workers=False,
        ),
        state_directory=tmp_path,
    )
    server = CoordinatorRpcServer(core)
    coordinator_port = await server.start("127.0.0.1:0")
    coordinator_endpoint = f"127.0.0.1:{coordinator_port}"
    cluster = ProductCluster()
    ready_queue = cluster.queue()
    stop_event = cluster.event()
    processes = {
        f"worker-{index}": cluster.process(
            f"worker-{index}",
            target=_worker_process,
            args=(
                f"worker-{index}",
                coordinator_endpoint,
                ready_queue,
                stop_event,
                0.1,
                100 if index == 2 else 0,
            ),
            shutdown=stop_event.set,
        )
        for index in range(3)
    }
    client = CoordinatorClient(coordinator_endpoint, timeout_s=30)
    try:
        cluster.start()
        startup = await cluster.wait_ready(ready_queue, count=len(processes), timeout=30)
        assert not [item for item in startup if "error" in item], startup
        capabilities = sorted(
            (WorkerCapability.model_validate(item["capability"]) for item in startup),
            key=lambda item: item.worker_id,
        )
        for capability in capabilities:
            core.registry.register(capability, benchmark_verified=True)
        plan = _plan(capabilities[:2])
        assert (await client.deploy_model(ModelDeployRequest(plan=plan))).deployment.ready

        events: list[Any] = []
        terminated = False
        async for event in client.submit_stream(
            SubmitRequest(
                request_id="replay-divergence",
                prompt_token_ids=[10, 20],
                max_new_tokens=6,
                random_seed=1,
                model_id=plan.model.model_id,
                model_revision=plan.model.model_revision,
            )
        ):
            events.append(event)
            if event.event_type == StreamEventType.TOKEN_GENERATED and not terminated:
                terminated = True
                await asyncio.to_thread(
                    cluster.crash_process,
                    "worker-1",
                    reason="replay-divergence worker failure injection",
                    timeout=5,
                )

        assert terminated
        token_events = [
            event for event in events if event.event_type == StreamEventType.TOKEN_GENERATED
        ]
        assert [(event.token_position, event.token_id) for event in token_events] == [(0, 21)]
        assert sum(event.event_type == StreamEventType.RECOVERY_STARTED for event in events) == 1
        assert not any(event.event_type == StreamEventType.RECOVERY_COMPLETED for event in events)
        assert sum(event.event_type == StreamEventType.RECOVERY_FAILED for event in events) == 1
        assert events[-1].event_type == StreamEventType.REQUEST_FAILED
        assert "replay divergence at token 0" in events[-1].status_detail
        assert core.product_telemetry is not None
        failures = [
            event
            for event in core.product_telemetry.events
            if event["event_type"] == "recovery_failed"
        ]
        assert len(failures) == 1
        assert "replay divergence at token 0" in failures[0]["error"]
        assert core.durable_state is not None
        durable = core.durable_state.load_requests()["replay-divergence"]
        assert durable.status == "failed"
        assert durable.accepted_generated_token_ids == [21]
    finally:
        await client.close()
        await cluster.close()
        await server.stop(grace_s=0)


@pytest.mark.asyncio
async def test_product_cancel_during_prefill_and_decode_releases_kv_but_keeps_stages_loaded(
    tmp_path: Path,
) -> None:
    core = CoordinatorCore(
        product_config=ProductCoordinatorConfig(
            event_queue_capacity=32,
            token_ingress_capacity=32,
            request_timeout_s=5,
            control_timeout_s=2,
            cleanup_timeout_s=3,
            worker_heartbeat_timeout_s=30,
            require_trusted_workers=False,
        ),
        state_directory=tmp_path,
    )
    server = CoordinatorRpcServer(core)
    coordinator_port = await server.start("127.0.0.1:0")
    coordinator_endpoint = f"127.0.0.1:{coordinator_port}"
    cluster = ProductCluster()
    ready_queue = cluster.queue()
    stop_event = cluster.event()
    processes = [
        cluster.process(
            f"worker-{index}",
            target=_worker_process,
            args=(
                f"worker-{index}",
                coordinator_endpoint,
                ready_queue,
                stop_event,
                0.4,
            ),
            shutdown=stop_event.set,
        )
        for index in range(2)
    ]
    client = CoordinatorClient(coordinator_endpoint, timeout_s=20)
    inspector = GrpcTransport(timeout_s=5)
    try:
        cluster.start()
        startup = await cluster.wait_ready(ready_queue, count=len(processes), timeout=30)
        assert not [item for item in startup if "error" in item], startup
        capabilities = sorted(
            (WorkerCapability.model_validate(item["capability"]) for item in startup),
            key=lambda item: item.worker_id,
        )
        for capability in capabilities:
            core.registry.register(capability, benchmark_verified=True)
        plan = _plan(capabilities)
        assert (await client.deploy_model(ModelDeployRequest(plan=plan))).deployment.ready

        async def cancel_when(
            request_id: str,
            trigger: StreamEventType,
        ) -> tuple[list[Any], Any]:
            events: list[Any] = []
            cancellation = None
            async for event in client.submit_stream(
                SubmitRequest(
                    request_id=request_id,
                    prompt_token_ids=[10, 20],
                    max_new_tokens=20,
                    random_seed=1,
                    model_id=plan.model.model_id,
                    model_revision=plan.model.model_revision,
                )
            ):
                events.append(event)
                if event.event_type == trigger and cancellation is None:
                    cancellation = await client.cancel_request(request_id)
            assert cancellation is not None
            return events, cancellation

        prefill_events, prefill_cancel = await cancel_when(
            "cancel-prefill",
            StreamEventType.PREFILL_STARTED,
        )
        assert prefill_cancel.accepted
        assert prefill_cancel.status.value == "cancelled"
        assert (
            sum(event.event_type == StreamEventType.REQUEST_CANCELLED for event in prefill_events)
            == 1
        )

        decode_events, decode_cancel = await cancel_when(
            "cancel-decode",
            StreamEventType.TOKEN_GENERATED,
        )
        assert decode_cancel.accepted
        assert decode_cancel.released_kv_bytes > 0
        assert (
            sum(event.event_type == StreamEventType.REQUEST_CANCELLED for event in decode_events)
            == 1
        )
        assert [
            event.token_id
            for event in decode_events
            if event.event_type == StreamEventType.TOKEN_GENERATED
        ] == [21]
        repeated = await client.cancel_request("cancel-decode")
        assert repeated.accepted and repeated.idempotent

        for item in plan.assignments:
            status = await inspector.get_stage_status(
                item.control_endpoint,
                GetStageStatusRequest(
                    worker_id=item.worker_id,
                    request_id=f"post-cancel:{item.stage_id}",
                    topology_id=plan.topology_id,
                ),
            )
            assert status.sessions == []
            assert status.loaded_stage is not None
            assert status.loaded_stage.load_count == 1
            assert status.installed_route is not None
            assert core.registry.capability(item.worker_id).reliability_score == 1.0
        assert core.product_telemetry is not None
        cancellations = [
            event
            for event in core.product_telemetry.events
            if event["event_type"] == "session_cancelled"
        ]
        assert [event["request_id"] for event in cancellations] == [
            "cancel-prefill",
            "cancel-decode",
        ]
    finally:
        await inspector.close()
        await client.close()
        await cluster.close()
        await server.stop(grace_s=0)


@pytest.mark.asyncio
async def test_product_cancel_during_recovery_uses_the_same_bounded_cleanup_path(
    tmp_path: Path,
) -> None:
    core = CoordinatorCore(
        product_config=ProductCoordinatorConfig(
            event_queue_capacity=32,
            token_ingress_capacity=32,
            request_timeout_s=5,
            control_timeout_s=2,
            recovery_timeout_s=15,
            cleanup_timeout_s=3,
            worker_heartbeat_timeout_s=30,
            require_trusted_workers=False,
        ),
        state_directory=tmp_path,
    )
    server = CoordinatorRpcServer(core)
    coordinator_port = await server.start("127.0.0.1:0")
    coordinator_endpoint = f"127.0.0.1:{coordinator_port}"
    cluster = ProductCluster()
    ready_queue = cluster.queue()
    stop_event = cluster.event()
    processes = {
        f"worker-{index}": cluster.process(
            f"worker-{index}",
            target=_worker_process,
            args=(
                f"worker-{index}",
                coordinator_endpoint,
                ready_queue,
                stop_event,
                0.15,
            ),
            shutdown=stop_event.set,
        )
        for index in range(3)
    }
    client = CoordinatorClient(coordinator_endpoint, timeout_s=30)
    inspector = GrpcTransport(timeout_s=5)
    try:
        cluster.start()
        startup = await cluster.wait_ready(ready_queue, count=len(processes), timeout=30)
        assert not [item for item in startup if "error" in item], startup
        capabilities = sorted(
            (WorkerCapability.model_validate(item["capability"]) for item in startup),
            key=lambda item: item.worker_id,
        )
        for capability in capabilities:
            core.registry.register(capability, benchmark_verified=True)
        plan = _plan(capabilities[:2])
        assert (await client.deploy_model(ModelDeployRequest(plan=plan))).deployment.ready

        events: list[Any] = []
        cancellation = None
        failed_worker_terminated = False
        async for event in client.submit_stream(
            SubmitRequest(
                request_id="cancel-recovery",
                prompt_token_ids=[10, 20],
                max_new_tokens=20,
                random_seed=1,
                model_id=plan.model.model_id,
                model_revision=plan.model.model_revision,
            )
        ):
            events.append(event)
            if event.event_type == StreamEventType.TOKEN_GENERATED and not failed_worker_terminated:
                failed_worker_terminated = True
                await asyncio.to_thread(
                    cluster.crash_process,
                    "worker-1",
                    reason="cancellation-during-recovery injection",
                    timeout=5,
                )
            if event.event_type == StreamEventType.RECOVERY_STARTED and cancellation is None:
                cancellation = await client.cancel_request("cancel-recovery")

        assert failed_worker_terminated
        assert cancellation is not None and cancellation.accepted
        assert cancellation.status.value == "cancelled"
        assert sum(event.event_type == StreamEventType.RECOVERY_STARTED for event in events) == 1
        assert not any(event.event_type == StreamEventType.RECOVERY_COMPLETED for event in events)
        assert sum(event.event_type == StreamEventType.REQUEST_CANCELLED for event in events) == 1
        status = await inspector.get_stage_status(
            plan.assignments[0].control_endpoint,
            GetStageStatusRequest(
                worker_id="worker-0",
                request_id="post-recovery-cancel",
                topology_id=plan.topology_id,
            ),
        )
        assert status.sessions == []
        assert status.loaded_stage is not None
        assert status.installed_route is not None
        assert core.product_telemetry is not None
        event_types = [item["event_type"] for item in core.product_telemetry.events]
        assert event_types.count("recovery_started") == 1
        assert event_types.count("session_cancelled") == 1
    finally:
        await inspector.close()
        await client.close()
        await cluster.close()
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
                request_generation=int(message.attributes.get("request_generation", 1)),
                replay_only=bool(message.attributes.get("replay_only", False)),
            )
            product_publication = product_publication.model_copy(
                update={
                    "signature": identity.sign(
                        canonical_json_bytes(
                            product_publication.model_dump(
                                mode="json",
                                exclude={"signature"},
                            )
                        )
                    )
                }
            )
            response = await coordinator.publish_token(product_publication)
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
async def test_exact_two_stage_olmoe_cuda_baseline(tmp_path: Path) -> None:
    """Prove the unfaulted two-worker product path independently of recovery."""

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
            require_trusted_workers=False,
        ),
        state_directory=tmp_path,
        transport=transport,
    )
    server = CoordinatorRpcServer(core)
    coordinator_port = await server.start("127.0.0.1:0")
    coordinator_endpoint = f"127.0.0.1:{coordinator_port}"
    cluster = ProductCluster()
    ready_queue = cluster.queue()
    stop_event = cluster.event()
    processes = {
        f"cuda-worker-{stage_id}": cluster.process(
            f"cuda-worker-{stage_id}",
            target=_real_worker_process,
            args=(
                f"cuda-worker-{stage_id}",
                coordinator_endpoint,
                str(REAL_MODEL_PATH),
                ready_queue,
                stop_event,
            ),
            shutdown=stop_event.set,
        )
        for stage_id in range(2)
    }
    client = CoordinatorClient(coordinator_endpoint, timeout_s=240)
    inspector = GrpcTransport(timeout_s=180)
    try:
        cluster.start()
        startup = await cluster.wait_ready(ready_queue, count=2, timeout=60)
        assert not [item for item in startup if "error" in item], startup
        capabilities = [WorkerCapability.model_validate(item["capability"]) for item in startup]
        startup_pids = {str(item["worker_id"]): int(item["process_id"]) for item in startup}
        assert set(startup_pids.values()) == {process.pid for process in processes.values()}
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
        deployed = await client.deploy_model(ModelDeployRequest(plan=planned.plan))
        assert deployed.deployment.ready
        assert len(deployed.deployment.workers) == 2
        assert {item.process_id for item in deployed.deployment.workers} == set(
            startup_pids.values()
        )

        events = [
            event
            async for event in client.submit_stream(
                SubmitRequest(
                    request_id="exact-olmoe-baseline",
                    prompt_token_ids=prompt_ids,
                    max_new_tokens=2,
                    random_seed=1,
                    model_id=REAL_MODEL_ID,
                    model_revision=REAL_MODEL_REVISION,
                )
            )
        ]
        generated = [
            event.token_id
            for event in events
            if event.event_type == StreamEventType.TOKEN_GENERATED
        ]
        assert generated == expected
        assert events[-1].event_type == StreamEventType.REQUEST_COMPLETED
        assert events[-1].final_token_ids == expected
        assert not [
            event
            for event in events
            if event.event_type
            in {StreamEventType.RECOVERY_STARTED, StreamEventType.RECOVERY_COMPLETED}
        ]

        statuses = []
        for assignment in planned.plan.assignments:
            statuses.append(
                await inspector.get_stage_status(
                    assignment.control_endpoint,
                    GetStageStatusRequest(
                        worker_id=assignment.worker_id,
                        request_id=f"baseline-status-{assignment.stage_id}",
                        topology_id=planned.plan.topology_id,
                    ),
                )
            )
        assert all(status.loaded_stage is not None for status in statuses)
        assert all(
            status.loaded_stage.load_count == 1 for status in statuses if status.loaded_stage
        )
        assert {status.process_id for status in statuses} == set(startup_pids.values())
        assert all(not status.sessions for status in statuses)
        assert transport.activation_relay_calls == 0
        assert core.runtime_transport_metrics["coordinator_activation_bytes"] == 0

        _write_real_gate_evidence(
            "real-two-stage-baseline.json",
            {
                "document_type": "swarm-real-model-gate-evidence",
                "format_version": 2,
                "gate": "two-stage-baseline",
                "status": "PASS",
                "model_id": REAL_MODEL_ID,
                "model_revision": REAL_MODEL_REVISION,
                "tokenizer_revision": REAL_TOKENIZER_REVISION,
                **_real_evidence_provenance(reference),
                "prompt_token_ids": prompt_ids,
                "generated_token_ids": generated,
                "expected_token_ids": expected,
                "topology": {
                    "topology_id": planned.plan.topology_id,
                    "route_generation": planned.plan.generation,
                    "stage_count": planned.plan.stage_count,
                },
                "stage_assignments": [
                    assignment.model_dump(mode="json") for assignment in planned.plan.assignments
                ],
                "worker_identities": [
                    {
                        "worker_id": capability.worker_id,
                        "fingerprint": public_key_fingerprint(capability.public_key),
                    }
                    for capability in capabilities
                ],
                "worker_pids": startup_pids,
                "stage_load_counts": {
                    status.worker_id: status.loaded_stage.load_count
                    for status in statuses
                    if status.loaded_stage is not None
                },
                "route_generations": [planned.plan.generation],
                "bytes_transferred": dict(core.runtime_transport_metrics),
                "critical_path_timings": events[-1].timing_metrics,
                "fallback_count": 0,
                "recovery_events": [],
                "failure_injected": False,
            },
        )

        unloaded = await client.unload_model(
            ModelUnloadRequest(topology_id=planned.plan.topology_id)
        )
        assert unloaded.deployment is not None
        assert unloaded.deployment.phase.value == "unloaded"
    finally:
        await inspector.close()
        await client.close()
        await cluster.close()
        await server.stop(grace_s=0)


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
async def test_exact_olmoe_cuda_restart_and_replay_recovery_uses_third_worker(
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
            require_trusted_workers=False,
        ),
        state_directory=tmp_path,
        transport=transport,
    )
    server = CoordinatorRpcServer(core)
    coordinator_port = await server.start("127.0.0.1:0")
    coordinator_endpoint = f"127.0.0.1:{coordinator_port}"
    cluster = ProductCluster()
    ready_queue = cluster.queue()
    stop_event = cluster.event()
    processes = {
        f"cuda-worker-{stage_id}": cluster.process(
            f"cuda-worker-{stage_id}",
            target=_real_worker_process,
            args=(
                f"cuda-worker-{stage_id}",
                coordinator_endpoint,
                str(REAL_MODEL_PATH),
                ready_queue,
                stop_event,
            ),
            shutdown=stop_event.set,
        )
        for stage_id in range(3)
    }
    client = CoordinatorClient(coordinator_endpoint, timeout_s=240)
    inspector = GrpcTransport(timeout_s=180)
    try:
        cluster.start()
        startup = await cluster.wait_ready(ready_queue, count=len(processes), timeout=60)
        assert not [item for item in startup if "error" in item], startup
        capabilities = [WorkerCapability.model_validate(item["capability"]) for item in startup]
        startup_pids = {str(item["worker_id"]): int(item["process_id"]) for item in startup}
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
        assert len(worker_pids) == 2
        assert worker_pids <= {process.pid for process in processes.values()}

        failed_worker_id = planned.plan.assignments[1].worker_id
        events = []
        terminated = False
        async for event in client.submit_stream(
            SubmitRequest(
                request_id="exact-olmoe-recovery",
                prompt_token_ids=prompt_ids,
                max_new_tokens=2,
                random_seed=1,
                model_id=REAL_MODEL_ID,
                model_revision=REAL_MODEL_REVISION,
            )
        ):
            events.append(event)
            if event.event_type == StreamEventType.TOKEN_GENERATED and not terminated:
                terminated = True
                await asyncio.to_thread(
                    cluster.crash_process,
                    failed_worker_id,
                    reason="real restart-and-replay dependency interruption",
                    timeout=15,
                )
        token_ids = [
            event.token_id
            for event in events
            if event.event_type == StreamEventType.TOKEN_GENERATED
        ]
        assert token_ids == expected
        assert terminated
        assert sum(event.event_type == StreamEventType.RECOVERY_COMPLETED for event in events) == 1
        assert events[-1].event_type == StreamEventType.REQUEST_COMPLETED
        assert events[-1].final_token_ids == expected

        assert core.deployment_manager is not None
        current_plan = core.deployment_manager.ready_plan(
            model_id=REAL_MODEL_ID,
            model_revision=REAL_MODEL_REVISION,
        )
        assert current_plan.generation == planned.plan.generation + 1
        assert failed_worker_id not in {
            assignment.worker_id for assignment in current_plan.assignments
        }
        statuses = []
        for assignment in current_plan.assignments:
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
        assert len({status.process_id for status in statuses}) == 2
        assert transport.activation_relay_calls == 0
        assert core.runtime_transport_metrics["coordinator_activation_bytes"] == 0

        _write_real_gate_evidence(
            "real-two-stage-recovery.json",
            {
                "document_type": "swarm-real-model-gate-evidence",
                "format_version": 2,
                "gate": "two-stage-restart-and-replay",
                "status": "PASS",
                "model_id": REAL_MODEL_ID,
                "model_revision": REAL_MODEL_REVISION,
                "tokenizer_revision": REAL_TOKENIZER_REVISION,
                **_real_evidence_provenance(reference),
                "prompt_token_ids": prompt_ids,
                "topology": {
                    "topology_id": current_plan.topology_id,
                    "route_generation": current_plan.generation,
                    "stage_count": current_plan.stage_count,
                },
                "stage_assignments": [
                    assignment.model_dump(mode="json") for assignment in current_plan.assignments
                ],
                "worker_identities": [
                    {
                        "worker_id": capability.worker_id,
                        "fingerprint": public_key_fingerprint(capability.public_key),
                    }
                    for capability in capabilities
                ],
                "worker_pids": startup_pids,
                "generated_token_ids": token_ids,
                "expected_token_ids": expected,
                "token_ids": token_ids,
                "bytes_transferred": dict(core.runtime_transport_metrics),
                "critical_path_timings": events[-1].timing_metrics,
                "timings": events[-1].timing_metrics,
                "fallback_count": 0,
                "recovery_events": [
                    event.model_dump(mode="json")
                    for event in events
                    if event.event_type
                    in {
                        StreamEventType.RECOVERY_STARTED,
                        StreamEventType.RECOVERY_COMPLETED,
                    }
                ],
                "route_generations": [planned.plan.generation, current_plan.generation],
                "failed_worker_id": failed_worker_id,
                "failure_injected": True,
            },
        )

        unloaded = await client.unload_model(
            ModelUnloadRequest(topology_id=current_plan.topology_id)
        )
        assert unloaded.deployment is not None
        assert unloaded.deployment.phase.value == "unloaded"
    finally:
        await inspector.close()
        await client.close()
        await cluster.close()
        await server.stop(grace_s=0)
