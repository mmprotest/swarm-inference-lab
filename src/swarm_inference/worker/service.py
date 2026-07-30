"""Physical/loopback worker process entrypoint."""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import suppress
from pathlib import Path

from swarm_inference.config.models import Backend, QueueConfig
from swarm_inference.coordinator.service import CoordinatorClient
from swarm_inference.protocol.messages import Heartbeat, RegistrationRequest
from swarm_inference.security.identity import WorkerIdentity
from swarm_inference.security.signatures import canonical_json_bytes
from swarm_inference.transport.grpc_transport import WorkerRpcServer
from swarm_inference.worker.agent import WorkerAgent
from swarm_inference.worker.capabilities import (
    measure_capabilities,
    measure_coordinator_latency_ms,
)


async def run_worker(
    *,
    coordinator_endpoint: str,
    listen_endpoint: str,
    advertised_endpoint: str,
    backend: Backend,
    memory_limit_bytes: int,
    identity_path: str | Path,
    total_memory_limit_bytes: int | None = None,
    worker_id: str | None = None,
    model_shard_root: str | Path | None = None,
    queue_config: QueueConfig | None = None,
    stop_event: asyncio.Event | None = None,
    outbound_queue_capacity: int = 1024,
    inbound_queue_capacity: int = 1024,
    max_inflight_operations: int = 256,
    reconnect_attempts: int = 5,
    reconnect_initial_backoff_ms: float = 25.0,
    reconnect_max_backoff_ms: float = 1000.0,
) -> None:
    identity = WorkerIdentity.load_or_create(identity_path)
    coordinator_latency_ms = measure_coordinator_latency_ms(coordinator_endpoint)
    capability = measure_capabilities(
        backend=backend,
        identity=identity,
        worker_id=worker_id,
        endpoint=advertised_endpoint,
        memory_limit_bytes=memory_limit_bytes,
        coordinator_latency_ms=coordinator_latency_ms,
    )
    try:
        import psutil

        capability.cpu_affinity = list(psutil.Process().cpu_affinity())
    except (AttributeError, OSError, ValueError):
        capability.cpu_affinity = []
    capability.single_thread_environment = {
        name: os.environ.get(name, "")
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    agent = WorkerAgent(
        capability=capability,
        identity=identity,
        queue_config=queue_config or QueueConfig(),
        total_memory_limit_bytes=total_memory_limit_bytes,
        outbound_queue_capacity=outbound_queue_capacity,
        inbound_queue_capacity=inbound_queue_capacity,
        max_inflight_operations=max_inflight_operations,
        reconnect_attempts=reconnect_attempts,
        reconnect_initial_backoff_ms=reconnect_initial_backoff_ms,
        reconnect_max_backoff_ms=reconnect_max_backoff_ms,
    )
    server = WorkerRpcServer(
        agent=agent,
        model_shard_root=str(model_shard_root) if model_shard_root else None,
    )
    await server.start(listen_endpoint)
    client = CoordinatorClient(coordinator_endpoint)
    nonce = f"{capability.worker_id}:{time.monotonic_ns()}"
    registration_payload = canonical_json_bytes(
        {
            "capability": capability.model_dump(mode="json"),
            "benchmark_nonce": nonce,
        }
    )
    response = await client.register(
        RegistrationRequest(
            capability=capability,
            benchmark_nonce=nonce,
            signature=identity.sign(registration_payload),
        )
    )
    if not response.accepted:
        await server.stop()
        await client.close()
        raise RuntimeError(f"coordinator rejected worker: {response.reason}")

    async def heartbeat_loop() -> None:
        while True:
            payload = {
                "worker_id": capability.worker_id,
                "queue_depth": agent.execution.queue_depth,
                "assignments": sorted(agent.shards.modules),
                "monotonic_ns": time.monotonic_ns(),
            }
            from datetime import UTC, datetime

            timestamp = datetime.now(UTC)
            signed = canonical_json_bytes({**payload, "timestamp": timestamp.isoformat()})
            await client.heartbeat(
                Heartbeat(
                    **payload,
                    timestamp=timestamp,
                    signature=identity.sign(signed),
                )
            )
            await asyncio.sleep(response.heartbeat_interval_s)

    heartbeat_task = asyncio.create_task(heartbeat_loop(), name=f"heartbeat:{capability.worker_id}")
    try:
        if stop_event is None:
            await server.wait_for_termination()
        else:
            await stop_event.wait()
    finally:
        heartbeat_task.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat_task
        await client.close()
        await server.stop()
