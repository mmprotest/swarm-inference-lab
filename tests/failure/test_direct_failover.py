from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime

import numpy as np
import pytest

from swarm_inference.config.loader import load_experiment_config
from swarm_inference.config.models import (
    Backend,
    OperationKind,
    QueueConfig,
    StageBenchmark,
    WorkerCapability,
)
from swarm_inference.coordinator.service import CoordinatorCore, CoordinatorRpcServer
from swarm_inference.model.synthetic import SyntheticStageModule, synthetic_activation
from swarm_inference.protocol.messages import RegistrationRequest, SubmitRequest
from swarm_inference.security.identity import WorkerIdentity
from swarm_inference.security.signatures import canonical_json_bytes
from swarm_inference.simulation.model import build_synthetic_stages
from swarm_inference.transport.grpc_transport import GrpcTransport, WorkerRpcServer
from swarm_inference.worker.agent import WorkerAgent


def _capability(
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
        stage_benchmarks=[
            StageBenchmark(
                worker_class="test",
                operation=OperationKind.DECODE,
                sequence_length=1,
                batch_size=1,
                mean_ms=1,
                p95_ms=1,
                samples=3,
            )
        ],
        upload_bandwidth_bytes_s=1e9,
        download_bandwidth_bytes_s=1e9,
        coordinator_latency_ms=0.1,
        memory_limit_bytes=memory_limit,
        endpoint=endpoint,
        last_heartbeat=datetime.now(UTC),
    )


def _expected_tokens(config, request: SubmitRequest) -> list[int]:
    modules = [
        SyntheticStageModule(config=config.model, stage=stage)
        for stage in build_synthetic_stages(config.model)
    ]
    outputs: list[int] = []
    for position in range(request.max_new_tokens):
        operation = OperationKind.PREFILL if position == 0 else OperationKind.DECODE
        token_ids = request.prompt_token_ids if position == 0 else [outputs[-1]]
        activation = synthetic_activation(
            token_ids,
            hidden_size=config.model.hidden_size,
            dtype=config.model.activation_dtype,
        )
        for module in modules:
            activation = module.execute(
                activation,
                request_id=request.request_id,
                operation=operation,
                token_position=position,
                sequence_length=len(token_ids),
                cache_generation=0,
            )
        digest = hashlib.sha256(
            np.ascontiguousarray(activation).tobytes()
            + request.random_seed.to_bytes(8, "little", signed=True)
            + position.to_bytes(8, "little")
        ).digest()
        outputs.append(int.from_bytes(digest[:4], "little") % 151_936)
    return outputs


@pytest.mark.asyncio
async def test_killed_direct_replica_replays_and_resumes_from_committed_state(
    repository_root,
    monkeypatch,
) -> None:
    config = load_experiment_config(
        repository_root / "configs" / "experiments" / "experiment_001_replica_scaling.yaml"
    )
    config.matrix = None
    config.model.layer_count = 2
    config.model.stage_count = 2
    config.model.bytes_per_layer = 1024 * 1024
    config.model.hidden_size = 32
    config.model.cpu_work_units = 1
    config.model.cpu_kernel_buffer_bytes = 1024
    config.worker.logical_memory_limit_bytes = 1024 * 1024 + 1024

    transport = GrpcTransport(timeout_s=5)
    core = CoordinatorCore(config=config, transport=transport)
    coordinator = CoordinatorRpcServer(core)
    await coordinator.start("127.0.0.1:0")
    worker_servers: dict[str, WorkerRpcServer] = {}
    largest_stage = max(stage.required_memory_bytes for stage in core.stages)
    try:
        for index in range(4):
            worker_id = f"worker-{index}"
            identity = WorkerIdentity.generate()
            capability = _capability(
                worker_id,
                "127.0.0.1:0",
                identity,
                largest_stage + 1024,
            )
            agent = WorkerAgent(
                capability=capability,
                identity=identity,
                queue_config=QueueConfig(capacity=8),
                reconnect_attempts=2,
                reconnect_initial_backoff_ms=1,
                reconnect_max_backoff_ms=5,
            )
            server = WorkerRpcServer(agent=agent)
            port = await server.start("127.0.0.1:0")
            capability.endpoint = f"127.0.0.1:{port}"
            worker_servers[worker_id] = server
            nonce = f"nonce-{index}"
            signed = canonical_json_bytes(
                {
                    "capability": capability.model_dump(mode="json"),
                    "benchmark_nonce": nonce,
                }
            )
            registration = await core.register(
                RegistrationRequest(
                    capability=capability,
                    benchmark_nonce=nonce,
                    signature=identity.sign(signed),
                )
            )
            assert registration.accepted
        await core.wait_for_coverage(minimum_replicas=2)

        original_call = core._call_direct_route
        killed_worker: str | None = None

        async def kill_after_first_commit(**kwargs):
            nonlocal killed_worker
            result = await original_call(**kwargs)
            if kwargs["token_position"] == 0 and killed_worker is None:
                killed_worker = kwargs["route"].assignments[1].worker_id
                await worker_servers[killed_worker].stop(0)
            return result

        monkeypatch.setattr(core, "_call_direct_route", kill_after_first_commit)
        request = SubmitRequest(
            request_id="direct-failover",
            prompt_token_ids=[1],
            max_new_tokens=4,
            random_seed=7,
            model_id=config.model_id,
            model_revision=config.model_revision,
        )
        expected = _expected_tokens(config, request)
        result = await core.submit(request)

        assert result.status == "completed", result.detail
        assert result.verified
        assert result.output_token_ids == expected
        recovery = [
            event
            for event in core.events
            if event["event_type"] == "stage_recovered" and event.get("data_plane_mode") == "direct"
        ]
        assert len(recovery) == 1
        assert recovery[0]["failed_worker_ids"] == [killed_worker]
        assert recovery[0]["route_generation"] == 2
        assert recovery[0]["replay_bytes"] > 0
        assert core.request_metrics[-1]["route_changes"] == 1
        committed = {
            position: generation
            for (request_id, position), generation in (core._committed_route_generations.items())
            if request_id == request.request_id
        }
        assert committed == {0: 1, 1: 2, 2: 2, 3: 2}
        assert core.runtime_transport_metrics["coordinator_activation_bytes"] == 0
    finally:
        await coordinator.stop(0)
        await asyncio.gather(
            *(server.stop(0) for server in worker_servers.values()),
            return_exceptions=True,
        )
