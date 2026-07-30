from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from swarm_inference.config.loader import load_experiment_config
from swarm_inference.config.models import (
    Backend,
    OperationKind,
    QueueConfig,
    StageBenchmark,
    WorkerCapability,
)
from swarm_inference.coordinator.service import CoordinatorCore
from swarm_inference.protocol.messages import RegistrationRequest, SubmitRequest
from swarm_inference.security.identity import WorkerIdentity
from swarm_inference.security.signatures import canonical_json_bytes
from swarm_inference.transport.fault_proxy import EndpointFault, FaultProxy
from swarm_inference.transport.grpc_transport import GrpcTransport, WorkerRpcServer
from swarm_inference.worker.agent import WorkerAgent


def make_capability(
    worker_id: str,
    endpoint: str,
    identity: WorkerIdentity,
    memory_limit: int,
) -> WorkerCapability:
    return WorkerCapability(
        worker_id=worker_id,
        public_key=identity.public_key_b64,
        hostname="localhost",
        operating_system="test",
        architecture="test",
        backend=Backend.SYNTHETIC,
        cpu_model="test",
        logical_cpu_count=1,
        total_ram_bytes=memory_limit,
        available_ram_bytes=memory_limit,
        supported_dtypes=["float32"],
        supported_quantisation_formats=[],
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
        upload_bandwidth_bytes_s=1e9,
        download_bandwidth_bytes_s=1e9,
        coordinator_latency_ms=0.1,
        reliability_score=1,
        memory_limit_bytes=memory_limit,
        endpoint=endpoint,
        last_heartbeat=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_failed_stage_uses_backup_and_exact_replay(repository_root) -> None:
    config = load_experiment_config(repository_root / "configs/experiments/scaling_loopback.yaml")
    inner = GrpcTransport()
    proxy = FaultProxy(inner)
    core = CoordinatorCore(config=config, transport=proxy)
    servers = []
    identities = []
    largest_stage = max(stage.required_memory_bytes for stage in core.stages)
    try:
        for index in range(4):
            identity = WorkerIdentity.generate()
            identities.append(identity)
            placeholder = make_capability(
                f"worker-{index}",
                "127.0.0.1:0",
                identity,
                largest_stage + 1024,
            )
            agent = WorkerAgent(
                capability=placeholder,
                identity=identity,
                queue_config=QueueConfig(capacity=8),
            )
            server = WorkerRpcServer(agent=agent)
            port = await server.start("127.0.0.1:0")
            endpoint = f"127.0.0.1:{port}"
            placeholder.endpoint = endpoint
            servers.append(server)
            nonce = f"nonce-{index}"
            payload = canonical_json_bytes(
                {
                    "capability": placeholder.model_dump(mode="json"),
                    "benchmark_nonce": nonce,
                }
            )
            response = await core.register(
                RegistrationRequest(
                    capability=placeholder,
                    benchmark_nonce=nonce,
                    signature=identity.sign(payload),
                )
            )
            assert response.accepted
        await core.wait_for_coverage(minimum_replicas=2)
        selected = core._choose_route()[0]
        assert selected.endpoint is not None
        proxy.configure(selected.endpoint, EndpointFault(timeout_next=True))
        result = await core.submit(
            SubmitRequest(
                request_id="recovery-request",
                prompt_token_ids=[1, 2, 3],
                max_new_tokens=3,
                random_seed=1,
            )
        )
        assert result.status == "completed", result.detail
        assert result.verified
        recovery = [event for event in core.events if event["event_type"] == "stage_recovered"]
        assert recovery
        assert recovery[0]["replacement_worker_id"] != recovery[0]["failed_worker_id"]
        assert core.request_metrics[-1]["route_changes"] == 1
    finally:
        await core.close()
        await asyncio.gather(*(server.stop(0) for server in servers))
