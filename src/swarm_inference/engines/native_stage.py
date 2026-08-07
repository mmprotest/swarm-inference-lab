"""Primary local and direct-stage-ring native execution engine."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import AsyncIterator, Mapping
from itertools import pairwise
from typing import Protocol

from swarm_inference.engines.cost_model import PlanCostInputs, score_costs, stable_plan_id
from swarm_inference.engines.interfaces import (
    ClusterCapabilities,
    Deployment,
    EngineSupportReport,
    EngineSupportStatus,
    ExecutionDevice,
    ExecutionEngineCapability,
    ExecutionPlan,
    ExecutionRequest,
    InferenceEvent,
    InferenceRequest,
    PhasePlan,
    WorkerExecutionCapability,
)
from swarm_inference.engines.topology import summarize_network_path
from swarm_inference.model.adapter import (
    NativeModelAdapterRegistry,
    default_native_adapter_registry,
)
from swarm_inference.model.descriptor import ResolvedModelDescriptor


class NativeStageRuntimeBridge(Protocol):
    async def prepare(self, plan: ExecutionPlan) -> Deployment: ...

    def submit(
        self, deployment: Deployment, request: InferenceRequest
    ) -> AsyncIterator[InferenceEvent]: ...

    async def unload(self, deployment: Deployment) -> None: ...


def _stage_capable(capability: ExecutionEngineCapability) -> bool:
    return bool({"stage", "critical_path_stage", "contiguous-stage"}.intersection(capability.roles))


def _stage_runtime_facts(
    worker: WorkerExecutionCapability,
    *,
    ownership: Mapping[str, object],
    fast_path: str,
) -> dict[str, object]:
    capability, device = _engine_device(worker)
    return {
        "runtime_revision": capability.runtime_revision,
        "binary_hashes": capability.binary_hashes,
        "device": {
            "identity": device.uuid or f"{device.device_id}:{device.name}",
            "type": device.device_type,
            "runtime_version": device.runtime_version,
            "driver_version": device.driver_version,
        },
        "ownership": ownership,
        "fast_path": fast_path,
    }


def _identity(
    model: ResolvedModelDescriptor,
    *,
    adapter_id: str,
    adapter_version: str,
    stages: tuple[dict[str, object], ...],
) -> str:
    payload = json.dumps(
        {
            "model": model.content_fingerprint,
            "engine": "native-stage",
            "engine_version": NativeStageEngine.engine_version,
            "adapter": adapter_id,
            "adapter_version": adapter_version,
            "stages": stages,
            "quantization": model.quantization,
            "tokenizer": model.tokenizer_identity,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _engine_device(
    worker: WorkerExecutionCapability,
) -> tuple[ExecutionEngineCapability, ExecutionDevice]:
    capability = worker.engine("native-stage")
    assert capability is not None and capability.devices
    device = max(
        capability.devices,
        key=lambda item: (
            item.measured_decode_tokens_s or 0,
            item.usable_memory_bytes,
            item.device_id,
        ),
    )
    return capability, device


def _fast_path_mode(capability: object, adapter_id: str, device_type: str) -> str:
    mappings = getattr(capability, "adapter_fast_paths", ())
    if device_type == "cuda" and any(item.adapter_id == adapter_id for item in mappings):
        return "auto-exact"
    return "eager"


def _worker_requested(worker: WorkerExecutionCapability, request: ExecutionRequest) -> bool:
    identities = {worker.worker_id, worker.node_id}
    return not identities.intersection(request.excluded_nodes) and (
        not request.requested_nodes or bool(identities.intersection(request.requested_nodes))
    )


class NativeStageEngine:
    engine_id = "native-stage"
    engine_version = "2"

    def __init__(
        self,
        *,
        adapters: NativeModelAdapterRegistry | None = None,
        runtime: NativeStageRuntimeBridge | None = None,
    ) -> None:
        self.adapters = adapters or default_native_adapter_registry()
        self.runtime = runtime

    def probe(
        self,
        model: ResolvedModelDescriptor,
        cluster: ClusterCapabilities,
    ) -> EngineSupportReport:
        reports = self.adapters.probe_all(model)
        supported = [item for item in reports if item.supported]
        if not supported:
            status = (
                EngineSupportStatus.UNSUPPORTED_FORMAT
                if all(item.status.value == "UNSUPPORTED_FORMAT" for item in reports)
                else EngineSupportStatus.UNSUPPORTED_ARCHITECTURE
            )
            return EngineSupportReport(
                engine_id=self.engine_id,
                status=status,
                reason="; ".join(f"{item.adapter_id}: {item.reason}" for item in reports),
                model_architecture=model.architecture,
                model_format=model.format,
                required_runtime="PyTorch native stage runtime and a registered adapter",
            )
        adapter = supported[0]
        workers = [
            worker
            for worker in cluster.workers_for_engine(self.engine_id)
            if (capability := worker.engine(self.engine_id)) is not None
            and adapter.adapter_id in capability.adapters
            and capability.devices
            and _stage_capable(capability)
        ]
        if not workers:
            return EngineSupportReport(
                engine_id=self.engine_id,
                status=EngineSupportStatus.MISSING_RUNTIME,
                reason=f"no worker advertises native adapter {adapter.adapter_id}",
                adapter_id=adapter.adapter_id,
                model_architecture=model.architecture,
                model_format=model.format,
                required_runtime=f"native-stage adapter {adapter.adapter_id}",
            )
        memory = sum(
            max(device.usable_memory_bytes for device in _engine_device(worker)[0].devices)
            for worker in workers
        )
        if memory < model.weight_bytes:
            return EngineSupportReport(
                engine_id=self.engine_id,
                status=EngineSupportStatus.INSUFFICIENT_MEMORY,
                reason="aggregate native-stage memory cannot own the immutable checkpoint",
                supported_worker_ids=tuple(worker.worker_id for worker in workers),
                adapter_id=adapter.adapter_id,
                model_architecture=model.architecture,
                model_format=model.format,
                required_runtime=f"native-stage adapter {adapter.adapter_id}",
            )
        return EngineSupportReport(
            engine_id=self.engine_id,
            status=EngineSupportStatus.SUPPORTED,
            reason="native adapter and stage-capable workers are available",
            supported_worker_ids=tuple(worker.worker_id for worker in workers),
            adapter_id=adapter.adapter_id,
            runtime_identity={"engine_version": self.engine_version},
            model_architecture=model.architecture,
            model_format=model.format,
            required_runtime=f"native-stage adapter {adapter.adapter_id}",
        )

    def probe_model_support(
        self,
        model: ResolvedModelDescriptor,
        cluster: ClusterCapabilities,
    ) -> EngineSupportReport:
        return self.probe(model, cluster)

    async def candidate_plans(
        self,
        model: ResolvedModelDescriptor,
        cluster: ClusterCapabilities,
        request: ExecutionRequest,
    ) -> list[ExecutionPlan]:
        adapter = self.adapters.resolve(model)
        workers = [
            worker
            for worker in cluster.workers_for_engine(self.engine_id)
            if _worker_requested(worker, request)
            and (capability := worker.engine(self.engine_id)) is not None
            and adapter.adapter_id in capability.adapters
            and capability.devices
            and _stage_capable(capability)
        ]
        if not workers:
            return []
        workers.sort(
            key=lambda item: (
                -max(
                    (device.measured_decode_tokens_s or 0)
                    for device in item.engine(self.engine_id).devices  # type: ignore[union-attr]
                ),
                item.queue_depth,
                item.worker_id,
            )
        )
        plans: list[ExecutionPlan] = []
        local_rates: dict[str, float] = {}
        for worker in workers:
            capability, device = _engine_device(worker)
            if device.usable_memory_bytes < model.weight_bytes:
                continue
            rate = float(device.measured_decode_tokens_s or 1.0) / (1 + worker.queue_depth)
            local_rates[worker.worker_id] = rate
            roles = {worker.worker_id: "critical_path_stage"}
            fast_path = _fast_path_mode(capability, adapter.adapter_id, device.device_type)
            assignment: dict[str, object] = {
                "stage_id": 0,
                "layer_start": 0,
                "layer_end": model.layer_count,
                "model_bytes": model.weight_bytes,
                "expected_memory_bytes": model.weight_bytes,
                "execution_device": device.device_id,
            }
            execution_identity = _identity(
                model,
                adapter_id=adapter.adapter_id,
                adapter_version=adapter.adapter_version,
                stages=(
                    _stage_runtime_facts(
                        worker,
                        ownership=assignment,
                        fast_path=fast_path,
                    ),
                ),
            )
            idle = {
                item.worker_id: "no positive single-request utility"
                for item in workers
                if item.worker_id != worker.worker_id
            }
            costs = score_costs(
                PlanCostInputs(
                    measured_prefill_tokens_s=device.measured_prefill_tokens_s,
                    measured_decode_tokens_s=device.measured_decode_tokens_s,
                    queue_depth=worker.queue_depth,
                    reliability=worker.reliability,
                    usable_memory_bytes=device.usable_memory_bytes,
                    required_memory_bytes=model.weight_bytes,
                    resident_model_bytes=(
                        model.weight_bytes
                        if model.content_fingerprint in worker.resident_model_fingerprints
                        else 0
                    ),
                    concurrency=request.concurrency,
                    request_priority=request.priority,
                    network_latency_ms=0.0,
                    network_jitter_ms=0.0,
                    messages_per_token=0.0,
                    bytes_per_token=0.0,
                    serial_waits_per_token=0.0,
                ),
                objective=request.objective,
            )
            topology = "local-one-stage"
            plan_identity: dict[str, object] = {
                "model": model.content_fingerprint,
                "execution": execution_identity,
                "topology": topology,
                "worker": worker.worker_id,
                "adapter": adapter.adapter_id,
            }
            plans.append(
                ExecutionPlan(
                    plan_id=stable_plan_id("native-local", plan_identity),
                    engine_id=self.engine_id,
                    model_fingerprint=model.content_fingerprint,
                    execution_identity=execution_identity,
                    objective=request.objective,
                    topology=topology,
                    worker_roles=roles,
                    idle_workers=idle,
                    stage_assignments=({**assignment, "worker_id": worker.worker_id},),
                    fast_paths={worker.worker_id: fast_path},
                    optional_mechanisms={
                        "lossless_activation_compression": False,
                        "prefetch": False,
                        "speculation": False,
                    },
                    prefill_plan=PhasePlan(phase="prefill", worker_roles=roles),
                    decode_plan=PhasePlan(phase="decode", worker_roles=roles),
                    predicted_ttft_ms=costs.predicted_ttft_ms,
                    predicted_decode_tokens_s=costs.predicted_decode_tokens_s,
                    predicted_aggregate_tokens_s=costs.predicted_aggregate_tokens_s,
                    predicted_network_bytes=0,
                    predicted_messages_per_token=0.0,
                    predicted_bytes_per_token=0.0,
                    predicted_serial_waits_per_token=0.0,
                    number_of_wan_stage_boundaries=0,
                    persistent_connections=False,
                    network_cost_confidence="measured",
                    network_cost_provenance="no network boundary",
                    required_memory_bytes=model.weight_bytes,
                    score=costs.score,
                    explanation=("one stage avoids network and acquisition costs",),
                    engine_parameters={
                        "adapter_id": adapter.adapter_id,
                        "cost_components": costs.components,
                        "unmeasured_inputs": costs.unmeasured_inputs,
                    },
                )
            )
        best_local = max(local_rates.values(), default=0.0)
        maximum_stages = min(len(workers), model.layer_count or len(workers))
        for stage_count in range(2, maximum_stages + 1):
            selected = workers[:stage_count]
            per_stage = math.ceil(model.weight_bytes / stage_count)
            devices = [_engine_device(worker)[1] for worker in selected]
            if any(device.usable_memory_bytes < per_stage for device in devices):
                continue
            rates = [float(device.measured_decode_tokens_s or 1.0) for device in devices]
            stage_rate = min(rate * stage_count for rate in rates)
            links = tuple(left.link_to(right.worker_id) for left, right in pairwise(selected))
            network = summarize_network_path(links)
            predicted = (
                1000
                / (
                    1000 / max(stage_rate, 0.001)
                    + float(network.aggregate_rtt_ms)
                    + float(network.aggregate_jitter_ms or 0.0)
                )
                if network.aggregate_rtt_ms is not None
                else None
            )
            capacity_required = not local_rates
            positive = predicted is not None and predicted > best_local
            if (
                request.objective == "speed"
                and not request.require_distributed
                and not (capacity_required or positive)
            ):
                continue
            roles = {item.worker_id: "critical_path_stage" for item in selected}
            idle = {
                item.worker_id: "outside the positive-utility contiguous partition"
                for item in workers[stage_count:]
            }
            assignments: list[dict[str, object]] = []
            fast_paths: dict[str, str] = {}
            layers = model.layer_count or stage_count
            for index, worker in enumerate(selected):
                start = index * layers // stage_count
                end = (index + 1) * layers // stage_count
                capability, device = _engine_device(worker)
                assignments.append(
                    {
                        "stage_id": index,
                        "worker_id": worker.worker_id,
                        "layer_start": start,
                        "layer_end": end,
                        "model_bytes": (
                            (model.weight_bytes * (index + 1) // stage_count)
                            - (model.weight_bytes * index // stage_count)
                        ),
                        "expected_memory_bytes": per_stage,
                        "execution_device": device.device_id,
                    }
                )
                fast_paths[worker.worker_id] = _fast_path_mode(
                    capability,
                    adapter.adapter_id,
                    device.device_type,
                )
            execution_identity = _identity(
                model,
                adapter_id=adapter.adapter_id,
                adapter_version=adapter.adapter_version,
                stages=tuple(
                    _stage_runtime_facts(
                        worker,
                        ownership={
                            key: value for key, value in assignment.items() if key != "worker_id"
                        },
                        fast_path=fast_paths[worker.worker_id],
                    )
                    for worker, assignment in zip(selected, assignments, strict=True)
                ),
            )
            reliability = min(item.reliability for item in selected)
            usable_memory = sum(device.usable_memory_bytes for device in devices)
            bytes_per_token = (
                float(model.hidden_size * model.activation_dtype_bytes * (stage_count - 1))
                if model.hidden_size is not None and model.activation_dtype_bytes is not None
                else None
            )
            costs = score_costs(
                PlanCostInputs(
                    measured_decode_tokens_s=max(stage_rate, 1e-9),
                    queue_depth=max(item.queue_depth for item in selected),
                    reliability=reliability,
                    usable_memory_bytes=usable_memory,
                    required_memory_bytes=model.weight_bytes,
                    network_latency_ms=network.aggregate_rtt_ms,
                    network_jitter_ms=network.aggregate_jitter_ms,
                    messages_per_token=float(stage_count),
                    bytes_per_token=bytes_per_token,
                    serial_waits_per_token=float(stage_count),
                    concurrency=request.concurrency,
                    request_priority=request.priority,
                ),
                objective=request.objective,
            )
            topology = f"direct-stage-ring-{stage_count}"
            plan_identity = {
                "model": model.content_fingerprint,
                "execution": execution_identity,
                "topology": topology,
                "assignments": assignments,
                "adapter": adapter.adapter_id,
            }
            plans.append(
                ExecutionPlan(
                    plan_id=stable_plan_id(f"native-ring-{stage_count}", plan_identity),
                    engine_id=self.engine_id,
                    model_fingerprint=model.content_fingerprint,
                    execution_identity=execution_identity,
                    objective=request.objective,
                    topology=topology,
                    worker_roles=roles,
                    idle_workers=idle,
                    stage_assignments=tuple(assignments),
                    fast_paths=fast_paths,
                    optional_mechanisms={
                        "lossless_activation_compression": False,
                        "prefetch": False,
                        "speculation": False,
                    },
                    prefill_plan=PhasePlan(phase="prefill", worker_roles=roles),
                    decode_plan=PhasePlan(phase="decode", worker_roles=roles),
                    predicted_ttft_ms=costs.predicted_ttft_ms,
                    predicted_decode_tokens_s=costs.predicted_decode_tokens_s,
                    predicted_aggregate_tokens_s=costs.predicted_aggregate_tokens_s,
                    predicted_network_bytes=(
                        int(bytes_per_token * request.max_new_tokens)
                        if bytes_per_token is not None
                        else None
                    ),
                    predicted_messages_per_token=float(stage_count),
                    predicted_bytes_per_token=bytes_per_token,
                    predicted_serial_waits_per_token=float(stage_count),
                    number_of_wan_stage_boundaries=network.wan_boundaries,
                    persistent_connections=True,
                    network_cost_confidence=network.confidence.value,
                    network_cost_provenance=network.provenance,
                    required_memory_bytes=model.weight_bytes,
                    score=costs.score,
                    explanation=(
                        "persistent workers own contiguous stages",
                        "adjacent stages use direct persistent tensor connections",
                        "coordinator has zero hidden-state relay edges",
                        "network domains: " + ", ".join(item.value for item in network.domains),
                    ),
                    engine_parameters={
                        "adapter_id": adapter.adapter_id,
                        "cost_components": costs.components,
                        "unmeasured_inputs": costs.unmeasured_inputs,
                        "network_links": [item.model_dump(mode="json") for item in links],
                    },
                )
            )
        if request.objective == "throughput" and request.concurrency > 1:
            replicas = [item for item in workers if item.worker_id in local_rates]
            if len(replicas) > 1:
                roles = {replicas[0].worker_id: "critical_path_stage"}
                roles.update({item.worker_id: "background_replica" for item in replicas[1:]})
                fast_paths = {
                    item.worker_id: _fast_path_mode(
                        _engine_device(item)[0],
                        adapter.adapter_id,
                        _engine_device(item)[1].device_type,
                    )
                    for item in replicas
                }
                execution_identity = _identity(
                    model,
                    adapter_id=adapter.adapter_id,
                    adapter_version=adapter.adapter_version,
                    stages=tuple(
                        _stage_runtime_facts(
                            item,
                            ownership={
                                "replica": index,
                                "layer_start": 0,
                                "layer_end": model.layer_count,
                            },
                            fast_path=fast_paths[item.worker_id],
                        )
                        for index, item in enumerate(replicas)
                    ),
                )
                aggregate = sum(local_rates[item.worker_id] for item in replicas)
                topology = f"replicated-{len(replicas)}"
                plan_identity = {
                    "model": model.content_fingerprint,
                    "execution": execution_identity,
                    "topology": topology,
                    "workers": [item.worker_id for item in replicas],
                    "adapter": adapter.adapter_id,
                }
                plans.append(
                    ExecutionPlan(
                        plan_id=stable_plan_id("native-replicated", plan_identity),
                        engine_id=self.engine_id,
                        model_fingerprint=model.content_fingerprint,
                        execution_identity=execution_identity,
                        objective=request.objective,
                        topology=topology,
                        worker_roles=roles,
                        fast_paths=fast_paths,
                        optional_mechanisms={"background_inference": True},
                        prefill_plan=PhasePlan(phase="prefill", worker_roles=roles),
                        decode_plan=PhasePlan(phase="decode", worker_roles=roles),
                        predicted_ttft_ms=1000 / max(local_rates[replicas[0].worker_id], 0.001),
                        predicted_decode_tokens_s=local_rates[replicas[0].worker_id],
                        predicted_aggregate_tokens_s=aggregate,
                        predicted_network_bytes=0,
                        predicted_messages_per_token=0.0,
                        predicted_bytes_per_token=0.0,
                        predicted_serial_waits_per_token=0.0,
                        number_of_wan_stage_boundaries=0,
                        persistent_connections=False,
                        network_cost_confidence="measured",
                        network_cost_provenance="independent complete-model replicas",
                        required_memory_bytes=model.weight_bytes * len(replicas),
                        score=aggregate,
                        explanation=(
                            "replicas add independent service throughput without joining one critical path",
                        ),
                        engine_parameters={"adapter_id": adapter.adapter_id},
                    )
                )
        return plans

    async def prepare(self, plan: ExecutionPlan) -> Deployment:
        if plan.engine_id != self.engine_id:
            raise ValueError("native engine cannot prepare another engine's plan")
        if self.runtime is None:
            raise RuntimeError("native stage runtime bridge is not configured")
        return await self.runtime.prepare(plan)

    async def submit(
        self,
        deployment: Deployment,
        request: InferenceRequest,
    ) -> AsyncIterator[InferenceEvent]:
        if self.runtime is None:
            raise RuntimeError("native stage runtime bridge is not configured")
        async for event in self.runtime.submit(deployment, request):
            yield event

    async def unload(self, deployment: Deployment) -> None:
        if self.runtime is None:
            raise RuntimeError("native stage runtime bridge is not configured")
        await self.runtime.unload(deployment)


__all__ = ["NativeStageEngine", "NativeStageRuntimeBridge"]
