"""Coordinator model discovery without coordinator-side weight loading."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from swarm_inference.config.models import Backend, WorkerCapability
from swarm_inference.coordinator.registry import WorkerRegistry
from swarm_inference.host import is_wildcard_host, split_endpoint
from swarm_inference.model.product import (
    ProductModelMetadata,
    ProductModelReference,
    ProductModelSpec,
)
from swarm_inference.protocol.product import (
    WorkerEligibilityReport,
    WorkerModelProbeRequest,
    WorkerModelProbeResponse,
)
from swarm_inference.protocol.stage_ring import STAGE_RING_PROTOCOL_VERSION
from swarm_inference.protocol.stage_worker import (
    GetStageCapabilitiesRequest,
    GetStageCapabilitiesResponse,
)


class ModelProbeTransport(Protocol):
    async def get_stage_capabilities(
        self,
        endpoint: str,
        request: GetStageCapabilitiesRequest,
    ) -> GetStageCapabilitiesResponse: ...

    async def inspect_stage_model(
        self,
        endpoint: str,
        request: WorkerModelProbeRequest,
    ) -> WorkerModelProbeResponse: ...


@dataclass(frozen=True, slots=True)
class InspectedProductModel:
    spec: ProductModelSpec
    metadata: ProductModelMetadata
    capabilities: dict[str, WorkerCapability]
    eligibility: tuple[WorkerEligibilityReport, ...]


def _endpoint_rejection(endpoint: str | None, *, name: str) -> str | None:
    if endpoint is None:
        return f"worker has no {name} endpoint"
    try:
        host, port = split_endpoint(endpoint)
    except ValueError as exc:
        return f"worker {name} endpoint is invalid: {exc}"
    if is_wildcard_host(host) or port == 0:
        return f"worker {name} endpoint is not reachable"
    return None


class ProductModelCatalog:
    """Resolve an exact metadata consensus across eligible stage workers."""

    def __init__(
        self,
        *,
        registry: WorkerRegistry,
        transport: ModelProbeTransport,
        maximum_active_sessions_per_worker: int,
    ) -> None:
        self.registry = registry
        self.transport = transport
        self.maximum_active_sessions_per_worker = maximum_active_sessions_per_worker

    def _base_rejections(
        self,
        capability: WorkerCapability,
        reference: ProductModelReference,
        *,
        healthy_registration: bool,
    ) -> list[str]:
        reasons: list[str] = []
        if not healthy_registration:
            reasons.append("registration heartbeat is unhealthy")
        if not capability.stage_runtime_enabled:
            reasons.append("persistent stage runtime is disabled")
        if reference.adapter_id not in capability.supported_model_adapters:
            reasons.append(f"model adapter {reference.adapter_id!r} is unsupported")
        normalised_dtype = {
            "bf16": "bfloat16",
            "f16": "float16",
            "f32": "float32",
        }.get(reference.dtype.lower(), reference.dtype.lower())
        if normalised_dtype not in capability.supported_activation_dtypes:
            reasons.append(f"activation dtype {normalised_dtype!r} is unsupported")
        expected_device = {
            Backend.TORCH_CPU: "cpu",
            Backend.TORCH_CUDA: "cuda",
            Backend.TORCH_MPS: "mps",
        }.get(capability.backend)
        actual_device = (
            capability.device_identifier.split(":", 1)[0].lower()
            if capability.device_identifier
            else None
        )
        if expected_device is None or actual_device != expected_device:
            reasons.append("worker backend and stage device are incompatible")
        if capability.stage_ring_protocol_version != STAGE_RING_PROTOCOL_VERSION:
            reasons.append(
                "stage-ring protocol version is incompatible "
                f"(worker={capability.stage_ring_protocol_version}, "
                f"coordinator={STAGE_RING_PROTOCOL_VERSION})"
            )
        for endpoint, name in (
            (capability.control_endpoint or capability.endpoint, "control"),
            (capability.data_plane_endpoint, "data"),
        ):
            rejection = _endpoint_rejection(endpoint, name=name)
            if rejection is not None:
                reasons.append(rejection)
        if capability.active_session_count >= self.maximum_active_sessions_per_worker:
            reasons.append("worker active load is above the admission threshold")
        if capability.current_queue_depth > max(4, capability.max_concurrent_stage_operations * 4):
            reasons.append("worker execution queue load is above the admission threshold")
        if not any(
            benchmark.measured and benchmark.mean_ms > 0
            for benchmark in capability.stage_benchmarks
        ):
            reasons.append("worker has no measured positive stage benchmark")
        return reasons

    async def inspect(self, reference: ProductModelReference) -> InspectedProductModel:
        registered = sorted(self.registry.workers(), key=lambda item: item.worker_id)
        if not registered:
            raise RuntimeError("no workers are registered")

        async def inspect_worker(
            registered_capability: WorkerCapability,
        ) -> tuple[WorkerCapability, list[str], WorkerModelProbeResponse | None]:
            healthy, _ = self.registry.registration_health(registered_capability.worker_id)
            reasons = self._base_rejections(
                registered_capability,
                reference,
                healthy_registration=healthy,
            )
            endpoint = registered_capability.control_endpoint or registered_capability.endpoint
            capability = registered_capability
            if endpoint is None or reasons:
                return capability, reasons, None
            request_id = f"catalog-{uuid4().hex}"
            try:
                advertised = await self.transport.get_stage_capabilities(
                    endpoint,
                    GetStageCapabilitiesRequest(
                        worker_id=capability.worker_id,
                        request_id=request_id,
                    ),
                )
                capability = advertised.capability
            except Exception as exc:
                reasons.append(f"stage capability probe failed: {type(exc).__name__}: {exc}")
                return capability, reasons, None
            reasons.extend(
                self._base_rejections(
                    capability,
                    reference,
                    healthy_registration=healthy,
                )
            )
            if reasons:
                return capability, reasons, None
            try:
                probe = await self.transport.inspect_stage_model(
                    endpoint,
                    WorkerModelProbeRequest(
                        worker_id=capability.worker_id,
                        request_id=request_id,
                        reference=reference,
                    ),
                )
            except Exception as exc:
                reasons.append(f"exact model probe failed: {type(exc).__name__}: {exc}")
                return capability, reasons, None
            if not probe.available:
                reasons.append(f"exact model identity unavailable: {probe.detail}")
            elif probe.metadata is not None:
                minimum_stage_bytes = min(
                    cost.weight_bytes
                    + cost.peak_temporary_bytes
                    + cost.kv_bytes_per_token
                    + (probe.metadata.embedding_weight_bytes if cost.layer_id == 0 else 0)
                    + (
                        probe.metadata.final_weight_bytes
                        if cost.layer_id == len(probe.metadata.layer_costs) - 1
                        else 0
                    )
                    for cost in probe.metadata.layer_costs
                )
                if capability.effective_memory_bytes < minimum_stage_bytes:
                    reasons.append(
                        f"effective memory {capability.effective_memory_bytes} bytes cannot host "
                        f"even the smallest valid stage ({minimum_stage_bytes} bytes)"
                    )
            return capability, reasons, probe

        inspected = await asyncio.gather(*(inspect_worker(worker) for worker in registered))
        available = [
            (capability, probe)
            for capability, reasons, probe in inspected
            if not reasons and probe is not None and probe.available
        ]
        if not available:
            details = "; ".join(
                f"{capability.worker_id}: {', '.join(reasons) or 'unavailable'}"
                for capability, reasons, _ in inspected
            )
            raise RuntimeError(f"no worker resolved the exact model identity; {details}")

        identity_groups: dict[tuple[str, str], list[str]] = {}
        for capability, available_probe in available:
            assert available_probe.spec is not None and available_probe.metadata is not None
            key = (
                available_probe.spec.metadata_hash,
                available_probe.metadata.model_dump_json(),
            )
            identity_groups.setdefault(key, []).append(capability.worker_id)
        selected_identity, consensus_workers = max(
            identity_groups.items(),
            key=lambda item: (len(item[1]), item[0][0]),
        )
        selected_probe = next(
            available_probe
            for capability, available_probe in available
            if capability.worker_id in consensus_workers
            and available_probe.spec is not None
            and available_probe.spec.metadata_hash == selected_identity[0]
        )
        assert selected_probe.spec is not None and selected_probe.metadata is not None

        capabilities: dict[str, WorkerCapability] = {}
        reports: list[WorkerEligibilityReport] = []
        for capability, reasons, inspected_probe in inspected:
            exact = (
                not reasons
                and inspected_probe is not None
                and inspected_probe.available
                and inspected_probe.spec is not None
                and inspected_probe.spec.metadata_hash == selected_probe.spec.metadata_hash
                and capability.worker_id in consensus_workers
            )
            if not reasons and not exact:
                reasons.append("model or tokenizer metadata differs from worker consensus")
            eligible = not reasons and exact
            if eligible:
                capabilities[capability.worker_id] = capability
            reports.append(
                WorkerEligibilityReport(
                    worker_id=capability.worker_id,
                    eligible=eligible,
                    rejection_reasons=reasons,
                    effective_memory_bytes=capability.effective_memory_bytes,
                    active_session_count=capability.active_session_count,
                    exact_model_identity=exact,
                    measured_profile=any(
                        item.measured and item.mean_ms > 0 for item in capability.stage_benchmarks
                    ),
                )
            )
        return InspectedProductModel(
            spec=selected_probe.spec,
            metadata=selected_probe.metadata,
            capabilities=capabilities,
            eligibility=tuple(reports),
        )


__all__ = [
    "InspectedProductModel",
    "ModelProbeTransport",
    "ProductModelCatalog",
]
