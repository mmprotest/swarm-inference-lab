"""Physical/loopback worker process entrypoint."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from swarm_inference.config.models import Backend, QueueConfig, WorkerCapability, WorkerRole
from swarm_inference.coordinator.service import CoordinatorClient
from swarm_inference.host import is_wildcard_host, split_endpoint
from swarm_inference.protocol.messages import Heartbeat, RegistrationRequest
from swarm_inference.protocol.product import ProductTokenPublication
from swarm_inference.runtime.shutdown import (
    install_shutdown_signal_handlers,
    wait_for_service_shutdown,
)
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
    from swarm_inference.cluster.artifacts import ArtifactManager
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
    artifact_storage_limit_bytes: int | None = None,
    artifact_manager: ArtifactManager | None = None,
    configured_model_path: str | Path | None = None,
    allow_model_download: bool = False,
    max_stage_sessions: int = 256,
    stage_execution_queue_capacity: int = 256,
    token_publication_queue_capacity: int = 256,
    upload_bandwidth_bytes_s: float = 0.0,
    download_bandwidth_bytes_s: float = 0.0,
    network_rates_measured: bool = False,
    trusted_coordinator_fingerprint: str | None = None,
    worker_roles: set[WorkerRole] | None = None,
    expert_manifest_path: str | Path | None = None,
    expert_data_listen_endpoint: str | None = None,
    expert_data_advertised_endpoint: str | None = None,
    expert_residency_budget_bytes: int = 0,
    expert_cache_budget_bytes: int = 0,
    expert_queue_capacity: int = 64,
    expert_max_concurrent_requests: int = 1,
    startup_future: asyncio.Future[WorkerCapability] | None = None,
    service_mode: str = "foreground",
    platform_support_status: str = "unknown",
) -> None:
    roles = set(worker_roles or ({WorkerRole.CONTIGUOUS_STAGE} if stage_runtime_enabled else set()))
    expert_roles = roles & {
        WorkerRole.WHOLE_EXPERT,
        WorkerRole.EXPERT_MICROSHARD,
        WorkerRole.REDUCER,
    }
    if WorkerRole.CONTIGUOUS_STAGE in roles:
        stage_runtime_enabled = True
    stage_memory_limit_bytes = memory_limit_bytes
    if expert_roles:
        if trusted_coordinator_fingerprint is None:
            raise ValueError("expert roles require an explicitly pinned coordinator fingerprint")
        if expert_manifest_path is None:
            raise ValueError("expert roles require an expert ownership manifest")
        if expert_data_listen_endpoint is None or expert_data_advertised_endpoint is None:
            raise ValueError("expert roles require expert data listen and advertised endpoints")
        if expert_residency_budget_bytes <= 0 or expert_cache_budget_bytes < 0:
            raise ValueError("expert residency/cache budgets are invalid")
        if expert_residency_budget_bytes > memory_limit_bytes:
            raise ValueError("expert residency budget exceeds the worker memory limit")
        expert_host, expert_port = split_endpoint(expert_data_advertised_endpoint)
        if is_wildcard_host(expert_host) or expert_port == 0:
            raise ValueError("expert data endpoint cannot advertise a wildcard or zero port")
    if stage_runtime_enabled:
        if trusted_coordinator_fingerprint is None:
            raise ValueError("stage runtime requires an explicitly pinned coordinator fingerprint")
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
        if expert_roles:
            stage_memory_limit_bytes -= expert_residency_budget_bytes
            if stage_memory_limit_bytes <= 0:
                raise ValueError("combined stage and expert roles leave no stage memory budget")
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
        benchmark_dtype=dtype,
        service_mode=service_mode,
        platform_support_status=platform_support_status,
    )
    capability.roles = sorted(roles, key=lambda item: item.value)
    if stage_runtime_enabled and expert_roles:
        # ``configured_memory_limit_bytes`` retains the process-wide limit;
        # stage planning sees only the capacity left after the explicit expert
        # residency reservation.
        capability.memory_limit_bytes = stage_memory_limit_bytes
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
    expert_runtime = None
    expert_server = None
    if expert_roles:
        from swarm_inference.execution.expert import (
            ExpertStore,
            npz_expert_loader,
            safetensors_expert_loader,
        )
        from swarm_inference.worker.expert_service import (
            ExpertWorkerRuntime,
            ExpertWorkerServer,
        )

        assert expert_manifest_path is not None
        assert expert_data_listen_endpoint is not None
        assert expert_data_advertised_endpoint is not None
        manifest_path = Path(expert_manifest_path).expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = {
            "model_id",
            "model_revision",
            "model_fingerprint",
            "quantization_fingerprint",
            "owned_experts",
        }
        missing = sorted(required - set(manifest))
        if missing:
            raise ValueError(f"expert ownership manifest is missing fields: {missing}")
        owned_entries = list(manifest["owned_experts"])
        owned = {(int(item["layer_id"]), int(item["expert_id"])) for item in owned_entries}
        owned_microshards = list(manifest.get("owned_microshards", []))
        microshard_keys = {
            (int(item["layer_id"]), int(item["expert_id"])) for item in owned_microshards
        }
        loader_type = str(manifest.get("loader_type", "npz"))
        if loader_type == "npz":
            files = {}
            for item in owned_entries:
                source = Path(str(item["path"])).expanduser()
                if not source.is_absolute():
                    source = manifest_path.parent / source
                files[(int(item["layer_id"]), int(item["expert_id"]))] = source
            loader = npz_expert_loader(files)
        elif loader_type == "safetensors":
            if WorkerRole.EXPERT_MICROSHARD in expert_roles:
                raise ValueError(
                    "native microshard roles require physically sliced npz ownership files"
                )
            model_path = Path(str(manifest["model_path"])).expanduser()
            if not model_path.is_absolute():
                model_path = manifest_path.parent / model_path
            loader = safetensors_expert_loader(model_path)
        else:
            raise ValueError(f"unsupported canonical expert loader {loader_type!r}")
        store = ExpertStore(
            owned=owned,
            loader=loader,
            residency_budget_bytes=expert_residency_budget_bytes,
            cache_budget_bytes=expert_cache_budget_bytes,
        )
        expert_runtime = ExpertWorkerRuntime(
            worker_id=capability.worker_id,
            identity=identity,
            model_id=str(manifest["model_id"]),
            model_revision=str(manifest["model_revision"]),
            model_fingerprint=str(manifest["model_fingerprint"]),
            quantization_fingerprint=str(manifest["quantization_fingerprint"]),
            store=store,
            roles={item.value for item in expert_roles},
            owned_microshards=owned_microshards,
            maximum_queue_depth=expert_queue_capacity,
            maximum_concurrent_requests=expert_max_concurrent_requests,
        )
        expert_listen_host, expert_listen_port = split_endpoint(expert_data_listen_endpoint)
        expert_server = ExpertWorkerServer(
            expert_runtime, host=expert_listen_host, port=expert_listen_port
        )
        await expert_server.start()
        capability.expert_data_plane_endpoint = expert_data_advertised_endpoint
        by_layer: dict[str, list[int]] = {}
        if WorkerRole.WHOLE_EXPERT in expert_roles:
            for layer_id, expert_id in sorted(owned - microshard_keys):
                by_layer.setdefault(str(layer_id), []).append(expert_id)
        capability.owned_experts = by_layer
        capability.owned_microshards = owned_microshards
        capability.expert_content_hashes = {
            f"{int(item['layer_id'])}:{int(item['expert_id'])}": str(item.get("content_hash", ""))
            for item in owned_entries
        }
        capability.expert_memory_budget_bytes = expert_residency_budget_bytes
        capability.expert_cache_budget_bytes = expert_cache_budget_bytes
        capability.model_fingerprint = str(manifest["model_fingerprint"])
        capability.quantisation_fingerprint = str(manifest["quantization_fingerprint"])
        capability.supported_expert_codecs = ["raw_fp32"]
        capability.supported_reduction_modes = ["fixed_order_fp32"]
        capability.measured_expert_service_rates = {
            str(key): float(value)
            for key, value in dict(manifest.get("measured_service_rates", {})).items()
        }
    stage_runtime: PersistentStageRuntime | None = None
    if stage_runtime_enabled:
        if (
            artifact_manager is None
            and model_cache_dir is not None
            and artifact_storage_limit_bytes is not None
        ):
            from swarm_inference.cluster.artifacts import ArtifactManager
            from swarm_inference.cluster.state import ClusterStateStore

            artifact_cache_root = Path(model_cache_dir).expanduser().resolve()
            artifact_manager = ArtifactManager(
                state=ClusterStateStore(artifact_cache_root.parent),
                node_id=capability.node_id or capability.worker_id.split("/", 1)[0],
                storage_limit_bytes=artifact_storage_limit_bytes,
            )
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
            publication = ProductTokenPublication(
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
                request_generation=int(message.attributes["request_generation"]),
                replay_only=bool(message.attributes["replay_only"]),
                expert_trace=list(message.attributes.get("expert_trace", [])),
                expert_metrics=dict(message.attributes.get("expert_metrics", {})),
            )
            publication = publication.model_copy(
                update={
                    "signature": identity.sign(
                        canonical_json_bytes(
                            publication.model_dump(mode="json", exclude={"signature"})
                        )
                    )
                }
            )
            response = await client.publish_token(publication)
            if not response.accepted:
                raise RuntimeError(f"coordinator rejected token publication: {response.detail}")

        stage_runtime = PersistentStageRuntime(
            worker_id=capability.worker_id,
            device=device or "cpu",
            dtype=dtype,
            memory_limit_bytes=stage_memory_limit_bytes,
            maximum_sessions=max_stage_sessions,
            execution_queue_capacity=stage_execution_queue_capacity,
            token_queue_capacity=token_publication_queue_capacity,
            model_cache_dir=model_cache_dir,
            configured_model_path=configured_model_path,
            allow_model_download=allow_model_download,
            capability=capability,
            token_publisher=publish_token,
            identity=identity,
            artifact_resolver=(artifact_manager.resolve if artifact_manager is not None else None),
            artifact_lease_acquirer=(
                lambda artifact_id, owner: (
                    artifact_manager.lease(
                        artifact_id,
                        owner=owner,
                        purpose="loaded-stage",
                    ).lease_id
                    if artifact_manager is not None
                    else ""
                )
            )
            if artifact_manager is not None
            else None,
            artifact_lease_releaser=(
                artifact_manager.release if artifact_manager is not None else None
            ),
        )
    service = PersistentStageWorkerService(
        agent=agent,
        stage_runtime=stage_runtime,
        artifact_manager=artifact_manager,
        trusted_coordinator_fingerprint=trusted_coordinator_fingerprint,
        model_shard_root=str(model_shard_root) if model_shard_root else None,
        data_queue_capacity=stage_execution_queue_capacity,
    )
    try:
        await service.start(
            control_listen_endpoint=listen_endpoint,
            data_listen_endpoint=data_listen_endpoint if stage_runtime_enabled else None,
        )
    except BaseException:
        if expert_server is not None:
            await expert_server.close()
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
        if expert_server is not None:
            await expert_server.close()
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
        if expert_server is not None:
            await expert_server.close()
        raise RuntimeError(f"coordinator rejected worker: {response.reason}")
    if stage_runtime is not None:
        if (
            response.coordinator_identity is None
            or response.coordinator_public_key is None
            or response.coordinator_public_key_fingerprint is None
        ):
            await service.stop()
            await client.close()
            if expert_server is not None:
                await expert_server.close()
            raise RuntimeError("coordinator did not provide an authenticated product identity")
        stage_runtime.configure_route_trust(
            coordinator_identity=response.coordinator_identity,
            coordinator_public_key=response.coordinator_public_key,
            expected_fingerprint=trusted_coordinator_fingerprint,
        )
        service.configure_artifact_trust(
            coordinator_public_key=response.coordinator_public_key,
            coordinator_fingerprint=response.coordinator_public_key_fingerprint,
        )
    if expert_runtime is not None:
        if (
            response.coordinator_identity is None
            or response.coordinator_public_key is None
            or response.coordinator_public_key_fingerprint is None
        ):
            await service.stop()
            await client.close()
            if expert_server is not None:
                await expert_server.close()
            raise RuntimeError("coordinator did not provide an authenticated product identity")
        expert_runtime.configure_route_trust(
            coordinator_identity=response.coordinator_identity,
            coordinator_public_key=response.coordinator_public_key,
            expected_fingerprint=trusted_coordinator_fingerprint,
        )

    # The reusable WorkerRuntime waits for this exact point: the control/data
    # services are live, registration has succeeded, and route trust is pinned.
    # Direct callers that predate the lifecycle class need no synchronization
    # object and retain their existing behavior.
    if startup_future is not None and not startup_future.done():
        startup_future.set_result(capability.model_copy(deep=True))

    async def heartbeat_loop() -> None:
        while True:
            expert_queue_depth = 0
            if stage_runtime is not None:
                stage_runtime.refresh_capability()
            if expert_runtime is not None:
                expert_status = expert_runtime.status()
                expert_queue_depth = int(expert_status["queue_depth"])
                capability.expert_cache_resident_bytes = int(expert_status["cache_resident_bytes"])
                capability.expert_cache_hits = int(expert_status["cache_hits"])
                capability.expert_cache_misses = int(expert_status["cache_misses"])
                capability.remote_expert_calls = int(expert_status["remote_whole_expert_calls"])
                capability.remote_microshard_calls = int(expert_status["remote_microshard_calls"])
                capability.expert_bytes_transferred = int(expert_status["bytes_received"]) + int(
                    expert_status["bytes_sent"]
                )
                capability.expert_critical_path_ns = int(expert_status["compute_ns"])
                whole_calls = int(expert_status["remote_whole_expert_calls"])
                if whole_calls:
                    capability.measured_expert_service_rates["whole_expert_ms"] = (
                        int(expert_status["whole_expert_compute_ns"]) / whole_calls / 1e6
                    )
                microshard_calls = int(expert_status["remote_microshard_calls"])
                if microshard_calls:
                    capability.measured_expert_service_rates["microshard_ms"] = (
                        int(expert_status["microshard_compute_ns"]) / microshard_calls / 1e6
                    )
            payload = {
                "worker_id": capability.worker_id,
                "queue_depth": agent.execution.queue_depth + expert_queue_depth,
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
    shutdown_event = stop_event or asyncio.Event()
    restore_signal_handlers = (
        install_shutdown_signal_handlers(shutdown_event) if stop_event is None else lambda: None
    )
    shutdown_started = False

    async def shutdown_service() -> None:
        nonlocal shutdown_started
        if shutdown_started:
            return
        shutdown_started = True
        if recorder is not None:
            recorder.emit("worker_shutdown_started")
        heartbeat_task.cancel()
        await asyncio.gather(heartbeat_task, return_exceptions=True)
        await service.stop()

    try:
        await wait_for_service_shutdown(
            service.wait_for_termination(),
            shutdown_event,
            shutdown=shutdown_service,
        )
    finally:
        restore_signal_handlers()
        try:
            await shutdown_service()
        finally:
            try:
                if expert_server is not None:
                    await expert_server.close()
            finally:
                await client.close()
