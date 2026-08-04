"""Physical/loopback worker process entrypoint."""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from swarm_inference.config.models import Backend, QueueConfig
from swarm_inference.coordinator.service import CoordinatorClient
from swarm_inference.host import is_wildcard_host, split_endpoint
from swarm_inference.protocol.messages import Heartbeat, RegistrationRequest
from swarm_inference.protocol.product import ProductTokenPublication
from swarm_inference.runtime.telemetry import lifecycle_observer
from swarm_inference.security.identity import WorkerIdentity
from swarm_inference.security.signatures import canonical_json_bytes
from swarm_inference.transport.stage_tensor import unpack_tensor
from swarm_inference.worker.agent import WorkerAgent
from swarm_inference.worker.capabilities import (
    measure_capabilities,
    measure_coordinator_latency_ms,
)
from swarm_inference.worker.stage_service import PersistentStageWorkerService

if TYPE_CHECKING:
    from swarm_inference.worker.stage_runtime import PersistentStageRuntime


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
    stage_runtime_enabled: bool = False,
    data_listen_endpoint: str | None = None,
    data_advertised_endpoint: str | None = None,
    device: str | None = None,
    dtype: str = "bfloat16",
    model_cache_dir: str | Path | None = None,
    configured_model_path: str | Path | None = None,
    allow_model_download: bool = False,
    max_stage_sessions: int = 256,
    stage_execution_queue_capacity: int = 256,
    token_publication_queue_capacity: int = 256,
    upload_bandwidth_bytes_s: float = 0.0,
    download_bandwidth_bytes_s: float = 0.0,
    network_rates_measured: bool = False,
) -> None:
    if stage_runtime_enabled:
        if data_listen_endpoint is None or data_advertised_endpoint is None:
            raise ValueError("stage runtime requires both data listen and advertised endpoints")
        advertised_host, advertised_port = split_endpoint(data_advertised_endpoint)
        if is_wildcard_host(advertised_host) or advertised_port == 0:
            raise ValueError("stage data endpoint cannot advertise a wildcard or zero port")
        if device is None:
            raise ValueError("stage runtime requires an explicit device")
        device_type = device.split(":", 1)[0].lower()
        expected_device = {
            Backend.TORCH_CPU: "cpu",
            Backend.TORCH_CUDA: "cuda",
            Backend.TORCH_MPS: "mps",
        }.get(backend)
        if expected_device is None or device_type != expected_device:
            raise ValueError(
                f"backend {backend.value} is incompatible with stage device {device!r}"
            )
    identity = WorkerIdentity.load_or_create(identity_path)
    coordinator_latency_ms = measure_coordinator_latency_ms(coordinator_endpoint)
    capability = measure_capabilities(
        backend=backend,
        identity=identity,
        worker_id=worker_id,
        endpoint=advertised_endpoint,
        control_endpoint=advertised_endpoint,
        data_plane_endpoint=data_advertised_endpoint if stage_runtime_enabled else None,
        device_identifier=device,
        stage_runtime_enabled=stage_runtime_enabled,
        memory_limit_bytes=memory_limit_bytes,
        coordinator_latency_ms=coordinator_latency_ms,
        upload_bandwidth_bytes_s=upload_bandwidth_bytes_s,
        download_bandwidth_bytes_s=download_bandwidth_bytes_s,
        network_rates_measured=network_rates_measured,
    )
    requested_dtype = {"bf16": "bfloat16", "f16": "float16", "f32": "float32"}.get(
        dtype.lower(), dtype.lower()
    )
    if stage_runtime_enabled and requested_dtype not in capability.supported_activation_dtypes:
        raise ValueError(
            f"stage dtype {dtype!r} did not pass execution probing on device {device!r}"
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
    client = CoordinatorClient(coordinator_endpoint)
    stage_runtime: PersistentStageRuntime | None = None
    if stage_runtime_enabled:
        from swarm_inference.worker.stage_runtime import PersistentStageRuntime

        async def publish_token(publication: object) -> None:
            assert stage_runtime is not None
            from swarm_inference.worker.stage_runtime import TokenPublication

            if not isinstance(publication, TokenPublication):
                raise TypeError("stage token publisher received an invalid publication")
            message = publication.message
            metadata = message.attributes.get("tensor")
            if not isinstance(metadata, dict):
                raise ValueError("stage token publication has no tensor metadata")
            token_tensor, _ = unpack_tensor(message.payload, dict(metadata))
            if token_tensor.numel() != 1:
                raise ValueError("stage token publication must contain exactly one token")
            token_id = int(token_tensor.item())
            response = await client.publish_token(
                ProductTokenPublication(
                    worker_id=capability.worker_id,
                    request_id=message.request_id,
                    session_id=message.session_id,
                    topology_id=message.topology_id,
                    route_generation=int(message.attributes["route_generation"]),
                    model_revision=message.model_revision,
                    token_position=message.token_position,
                    token_id=token_id,
                    decoded_text_fragment=stage_runtime.decode_token_id(token_id),
                    published_monotonic_ns=time.monotonic_ns(),
                )
            )
            if not response.accepted:
                raise RuntimeError(f"coordinator rejected token publication: {response.detail}")

        stage_runtime = PersistentStageRuntime(
            worker_id=capability.worker_id,
            device=device or "cpu",
            dtype=dtype,
            memory_limit_bytes=memory_limit_bytes,
            maximum_sessions=max_stage_sessions,
            execution_queue_capacity=stage_execution_queue_capacity,
            token_queue_capacity=token_publication_queue_capacity,
            model_cache_dir=model_cache_dir,
            configured_model_path=configured_model_path,
            allow_model_download=allow_model_download,
            capability=capability,
            token_publisher=publish_token,
        )
    service = PersistentStageWorkerService(
        agent=agent,
        stage_runtime=stage_runtime,
        model_shard_root=str(model_shard_root) if model_shard_root else None,
        data_queue_capacity=stage_execution_queue_capacity,
    )
    try:
        await service.start(
            control_listen_endpoint=listen_endpoint,
            data_listen_endpoint=data_listen_endpoint if stage_runtime_enabled else None,
        )
    except BaseException:
        await client.close()
        raise
    nonce = f"{capability.worker_id}:{time.monotonic_ns()}"
    registration_payload = canonical_json_bytes(
        {
            "capability": capability.model_dump(mode="json"),
            "benchmark_nonce": nonce,
        }
    )
    recorder = lifecycle_observer()
    registration_started = time.monotonic_ns()
    if recorder is not None:
        recorder.emit("worker_registration_started", monotonic_ns=registration_started)
    try:
        response = await client.register(
            RegistrationRequest(
                capability=capability,
                benchmark_nonce=nonce,
                signature=identity.sign(registration_payload),
            )
        )
    except BaseException:
        await client.close()
        await service.stop()
        raise
    registration_completed = time.monotonic_ns()
    if recorder is not None:
        recorder.emit(
            "worker_registered",
            monotonic_ns=registration_completed,
            duration_ns=registration_completed - registration_started,
            details={
                "coordinator_endpoint": coordinator_endpoint,
                "registration_includes_assignment_ack": True,
            },
        )
    if not response.accepted:
        await service.stop()
        await client.close()
        raise RuntimeError(f"coordinator rejected worker: {response.reason}")

    async def heartbeat_loop() -> None:
        while True:
            if stage_runtime is not None:
                stage_runtime.refresh_capability()
            payload = {
                "worker_id": capability.worker_id,
                "queue_depth": agent.execution.queue_depth,
                "assignments": sorted(
                    {
                        *agent.shards.modules,
                        *(
                            [stage_runtime.loaded_executor.ownership.stage_id]
                            if stage_runtime is not None
                            and stage_runtime.loaded_executor is not None
                            else []
                        ),
                    }
                ),
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

    if recorder is not None:
        recorder.emit(
            "worker_routable",
            details={
                "loaded_stage_ids": sorted(agent.shards.modules),
                "stage_local_warmup": os.environ.get("SWARM_STAGE_LOCAL_WARMUP") == "1",
            },
        )
    heartbeat_task = asyncio.create_task(heartbeat_loop(), name=f"heartbeat:{capability.worker_id}")
    try:
        if stop_event is None:
            await service.wait_for_termination()
        else:
            await stop_event.wait()
    finally:
        if recorder is not None:
            recorder.emit("worker_shutdown_started")
        heartbeat_task.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat_task
        await client.close()
        await service.stop()
