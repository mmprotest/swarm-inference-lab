from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from swarm_inference.cluster.artifacts import (
    ArtifactManager,
    ModelArtifactBuilder,
)
from swarm_inference.cluster.models import node_id_from_fingerprint
from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.config.models import Backend, QueueConfig, WorkerCapability
from swarm_inference.config.product import ProductCoordinatorConfig
from swarm_inference.coordinator.service import (
    CoordinatorClient,
    CoordinatorCore,
    CoordinatorRpcServer,
)
from swarm_inference.engines.interfaces import (
    Deployment,
    EngineSupportReport,
    EngineSupportStatus,
    ExecutionPlan,
    InferenceEvent,
    InferenceRequest,
    PhasePlan,
)
from swarm_inference.engines.worker_control import CoordinatorAuthorizedEngineLifecycle
from swarm_inference.model.descriptor import ModelFileDescriptor, ResolvedModelDescriptor
from swarm_inference.protocol.cluster import ClusterRequestAuthentication
from swarm_inference.security.identity import WorkerIdentity
from swarm_inference.transport.grpc_transport import (
    GrpcTransport,
    WorkerRpcServer,
)
from swarm_inference.worker.agent import WorkerAgent
from swarm_inference.worker.engine_runtime import PersistentEngineRuntime


class _FakeEngine:
    engine_id = "general-test-engine"

    def __init__(self, worker_artifact_root: Path) -> None:
        self.worker_artifact_root = worker_artifact_root.resolve()
        self.prepared = 0
        self.unloaded = 0
        self.model_paths: tuple[Path, ...] = ()

    def probe(self, *_args: Any) -> EngineSupportReport:
        return EngineSupportReport(
            engine_id=self.engine_id,
            status=EngineSupportStatus.SUPPORTED,
            reason="test",
        )

    async def candidate_plans(self, *_args: Any) -> list[ExecutionPlan]:
        return []

    async def prepare(self, plan: ExecutionPlan) -> Deployment:
        self.prepared += 1
        self.model_paths = tuple(
            Path(item).resolve() for item in plan.engine_parameters["model_paths"]
        )
        assert self.model_paths
        assert all(path.is_relative_to(self.worker_artifact_root) for path in self.model_paths)
        return Deployment(
            deployment_id="worker-private-deployment",
            engine_id=self.engine_id,
            execution_identity=plan.execution_identity,
            plan=plan,
            ready=True,
            process_ids={"worker-a": 123},
        )

    async def submit(
        self,
        _deployment: Deployment,
        request: InferenceRequest,
    ) -> AsyncIterator[InferenceEvent]:
        yield InferenceEvent(
            event_type="started",
            request_id=request.request_id,
            sequence_number=0,
        )
        yield InferenceEvent(
            event_type="token",
            request_id=request.request_id,
            sequence_number=1,
            token_id=7,
            text="worker-token",
        )
        yield InferenceEvent(
            event_type="completed",
            request_id=request.request_id,
            sequence_number=2,
        )

    async def unload(self, _deployment: Deployment) -> None:
        self.unloaded += 1


class _ClusterControl:
    def __init__(self, node_id: str) -> None:
        self.node_id = node_id

    def verify_authentication(
        self,
        _authentication: ClusterRequestAuthentication,
        *,
        action: str,
        body: dict[str, Any],
    ) -> SimpleNamespace:
        assert action in {"engine-lease", "engine-artifact-lease"}
        assert body
        return SimpleNamespace(node_id=self.node_id)


def _capability(identity: WorkerIdentity) -> WorkerCapability:
    return WorkerCapability(
        worker_id="worker-a",
        public_key=identity.public_key_b64,
        hostname="localhost",
        operating_system="test",
        architecture="test",
        backend=Backend.TORCH_CPU,
        cpu_model="test",
        logical_cpu_count=1,
        total_ram_bytes=1_000_000,
        available_ram_bytes=1_000_000,
        memory_limit_bytes=1_000_000,
        upload_bandwidth_bytes_s=1_000_000,
        download_bandwidth_bytes_s=1_000_000,
        coordinator_latency_ms=0,
        execution_engines=[
            {
                "engine_id": "general-test-engine",
                "enabled": True,
                "runtime_revision": "pinned-test",
                "binary_hashes": {"server": "sha256:" + "1" * 64},
                "formats": ["gguf"],
                "roles": ["critical_path_stage"],
            }
        ],
    )


def _plan(fingerprint: str) -> ExecutionPlan:
    roles = {"worker-a": "critical_path_stage"}
    return ExecutionPlan(
        plan_id="plan-a",
        engine_id="general-test-engine",
        model_fingerprint=fingerprint,
        execution_identity="sha256:" + "2" * 64,
        objective="speed",
        topology="complete-model",
        worker_roles=roles,
        engine_parameters={"model_paths": []},
        prefill_plan=PhasePlan(phase="prefill", worker_roles=roles),
        decode_plan=PhasePlan(phase="decode", worker_roles=roles),
        predicted_ttft_ms=1,
        predicted_decode_tokens_s=10,
        predicted_aggregate_tokens_s=10,
        score=10,
    )


@pytest.mark.asyncio
async def test_general_engine_artifact_and_lifecycle_are_worker_owned(tmp_path: Path) -> None:
    requester_identity = WorkerIdentity.generate()
    requester_node_id = node_id_from_fingerprint(requester_identity.public_key_fingerprint)
    source_file = tmp_path / "source" / "model.gguf"
    source_file.parent.mkdir()
    source_file.write_bytes(b"GGUF-worker-control-test")
    source_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
    descriptor = ResolvedModelDescriptor(
        model_id=str(source_file),
        revision="sha256:" + "3" * 64,
        content_fingerprint="sha256:" + "4" * 64,
        source_type="local",
        format="gguf",
        architecture="test",
        files=(
            ModelFileDescriptor(
                relative_path=source_file.name,
                size_bytes=source_file.stat().st_size,
                sha256=source_hash,
            ),
        ),
        quantization="Q4_K_M",
        weight_bytes=source_file.stat().st_size,
        tokenizer_identity="sha256:" + "5" * 64,
        local_paths=(str(source_file),),
    )
    source_state = ClusterStateStore(tmp_path / "source-state")
    source_manager = ArtifactManager(
        state=source_state,
        node_id=requester_node_id,
        storage_limit_bytes=10_000_000,
    )
    manifest = ModelArtifactBuilder(
        artifact_root=source_state.paths.artifacts,
        temporary_root=source_state.paths.downloads,
    ).build(descriptor, engine_id="general-test-engine")
    source_manager.register(manifest.artifact_id)

    core = CoordinatorCore(
        product_config=ProductCoordinatorConfig(require_trusted_workers=False),
        state_directory=tmp_path / "coordinator",
    )
    core.cluster_control = _ClusterControl(requester_node_id)  # type: ignore[assignment]
    worker_identity = WorkerIdentity.generate()
    capability = _capability(worker_identity)
    core.registry.register(capability, benchmark_verified=True)
    coordinator_server = CoordinatorRpcServer(core)
    coordinator_port = await coordinator_server.start("127.0.0.1:0")
    coordinator_client = CoordinatorClient(f"127.0.0.1:{coordinator_port}")

    worker_state = ClusterStateStore(tmp_path / "worker-state")
    worker_manager = ArtifactManager(
        state=worker_state,
        node_id="worker-node",
        storage_limit_bytes=10_000_000,
    )
    engine = _FakeEngine(worker_state.paths.artifacts)
    engine_runtime = PersistentEngineRuntime(
        worker_id="worker-a",
        engines=(engine,),
        artifact_resolver=worker_manager.resolve,
    )
    agent = WorkerAgent(
        capability=capability,
        identity=worker_identity,
        queue_config=QueueConfig(capacity=4),
    )
    assert core.coordinator_identity is not None
    worker_server = WorkerRpcServer(
        agent=agent,
        engine_runtime=engine_runtime,
        artifact_manager=worker_manager,
        trusted_coordinator_fingerprint=(core.coordinator_identity.public_key_fingerprint),
    )
    worker_server.configure_engine_trust(
        coordinator_public_key=core.coordinator_identity.public_key_b64,
        expected_fingerprint=core.coordinator_identity.public_key_fingerprint,
    )
    worker_server.configure_artifact_trust(
        coordinator_public_key=core.coordinator_identity.public_key_b64,
        coordinator_fingerprint=core.coordinator_identity.public_key_fingerprint,
    )
    worker_port = await worker_server.start("127.0.0.1:0")
    transport = GrpcTransport()
    lifecycle = CoordinatorAuthorizedEngineLifecycle(
        coordinator=coordinator_client,
        transport=transport,
        identity=requester_identity,
        node_id=requester_node_id,
        worker_endpoints={"worker-a": f"127.0.0.1:{worker_port}"},
        artifact_manager=source_manager,
        artifact_manifest=manifest,
    )
    plan = _plan(descriptor.content_fingerprint)
    try:
        deployment = await lifecycle.prepare(plan)
        assert deployment.ready
        assert engine.prepared == 1
        assert worker_manager.resolve(manifest.artifact_id).is_dir()
        assert engine.model_paths[0].read_bytes() == source_file.read_bytes()

        # The deterministic deployment ID and worker residency make a repeated
        # prepare a warm reuse, not another backend model load.
        repeated = await lifecycle.prepare(plan)
        assert repeated.deployment_id == deployment.deployment_id
        assert engine.prepared == 1

        events = [
            event
            async for event in lifecycle.submit(
                repeated,
                InferenceRequest(request_id="request-a", prompt="hello", max_new_tokens=1),
            )
        ]
        assert [event.event_type for event in events] == ["started", "token", "completed"]
        assert events[1].token_id == 7
        assert transport.metrics.streams_created == 1
        await lifecycle.unload(repeated)
        assert engine.unloaded == 1
    finally:
        await transport.close()
        await coordinator_client.close()
        await worker_server.stop(0)
        await coordinator_server.stop(0)
