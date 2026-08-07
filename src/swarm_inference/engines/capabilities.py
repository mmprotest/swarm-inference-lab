"""Translate durable worker reports into engine/device planning facts."""

from __future__ import annotations

from typing import Any

from swarm_inference.config.models import Backend, WorkerCapability
from swarm_inference.engines.interfaces import (
    ClusterCapabilities,
    ExecutionDevice,
    ExecutionEngineCapability,
    WorkerExecutionCapability,
)


def _legacy_native_capability(worker: WorkerCapability) -> ExecutionEngineCapability | None:
    if not worker.stage_runtime_enabled:
        return None
    device_type = {
        Backend.TORCH_CUDA: "cuda",
        Backend.TORCH_MPS: "mps",
    }.get(worker.backend, "cpu")
    total_memory = (
        worker.total_vram_bytes if device_type in {"cuda", "mps"} else worker.total_ram_bytes
    )
    usable_memory = worker.effective_memory_bytes
    decode_rate = max(
        (
            1000.0 / item.median_ms
            for item in worker.stage_benchmarks
            if item.median_ms is not None and item.median_ms > 0
        ),
        default=None,
    )
    return ExecutionEngineCapability(
        engine_id="native-stage",
        enabled=True,
        runtime_revision=worker.runtime_version,
        formats=("safetensors",),
        devices=(
            ExecutionDevice(
                device_id=worker.device_identifier or device_type,
                device_type=device_type,
                name=worker.gpu_model or worker.cpu_model,
                total_memory_bytes=total_memory,
                usable_memory_bytes=usable_memory,
                measured_decode_tokens_s=decode_rate,
                features=tuple(worker.supported_stage_execution_backends),
            ),
        ),
        adapters=tuple(worker.supported_model_adapters),
        fast_paths=tuple(
            sorted(
                {
                    feature
                    for item in worker.execution_engines
                    for feature in item.get("fast_paths", [])
                    if isinstance(feature, str)
                }
            )
        ),
        roles=tuple(role.value for role in worker.roles) or ("stage", "idle"),
        detail="migrated legacy stage capability",
    )


def worker_execution_capability(
    worker: WorkerCapability,
    *,
    network_latency_ms: dict[str, float] | None = None,
    network_bandwidth_bytes_s: dict[str, float] | None = None,
    resident_model_fingerprints: tuple[str, ...] = (),
    storage_available_bytes: int = 0,
) -> WorkerExecutionCapability:
    engines: list[ExecutionEngineCapability] = []
    for raw in worker.execution_engines:
        if not isinstance(raw, dict):
            raise ValueError("worker execution-engine capability must be an object")
        engines.append(ExecutionEngineCapability.model_validate(raw))
    if not engines:
        legacy = _legacy_native_capability(worker)
        if legacy is not None:
            engines.append(legacy)
    if len({item.engine_id for item in engines}) != len(engines):
        raise ValueError("worker advertises duplicate execution-engine capabilities")
    return WorkerExecutionCapability(
        worker_id=worker.worker_id,
        node_id=worker.node_id or worker.worker_id.split("/", 1)[0],
        engines=tuple(sorted(engines, key=lambda item: item.engine_id)),
        queue_depth=worker.current_queue_depth,
        reliability=worker.reliability_score,
        network_latency_ms=network_latency_ms or {},
        network_bandwidth_bytes_s=network_bandwidth_bytes_s or {},
        resident_model_fingerprints=resident_model_fingerprints,
        storage_available_bytes=max(0, storage_available_bytes),
    )


def cluster_execution_capabilities(
    workers: list[WorkerCapability] | tuple[WorkerCapability, ...],
    *,
    latency_by_worker: dict[str, dict[str, float]] | None = None,
    bandwidth_by_worker: dict[str, dict[str, float]] | None = None,
) -> ClusterCapabilities:
    latency = latency_by_worker or {}
    bandwidth = bandwidth_by_worker or {}
    return ClusterCapabilities(
        workers=tuple(
            worker_execution_capability(
                worker,
                network_latency_ms=latency.get(worker.worker_id, {}),
                network_bandwidth_bytes_s=bandwidth.get(worker.worker_id, {}),
            )
            for worker in workers
        )
    )


def capability_payload(capability: ExecutionEngineCapability) -> dict[str, Any]:
    """Stable helper used by worker heartbeat and diagnostics code."""

    return capability.model_dump(mode="json")


__all__ = [
    "capability_payload",
    "cluster_execution_capabilities",
    "worker_execution_capability",
]
