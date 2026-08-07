"""One model interface over measured, interchangeable execution engines."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, NonNegativeInt

from swarm_inference.config.models import StrictModel
from swarm_inference.coordinator.canonical_planner import (
    CanonicalPlanner,
    CanonicalPlanningDecision,
    MechanismEvidence,
)
from swarm_inference.engines.interfaces import (
    ClusterCapabilities,
    Deployment,
    EngineSupportReport,
    ExecutionRequest,
    InferenceEvent,
    InferenceRequest,
)
from swarm_inference.engines.registry import ExecutionEngineRegistry, default_engine_registry
from swarm_inference.model.descriptor import ResolvedModelDescriptor
from swarm_inference.model.resolver import (
    ModelResolution,
    ModelSourceResolver,
    ResolutionResources,
)
from swarm_inference.model.variants import VariantCandidate


class VariantInspection(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    variant: str
    quantization: str
    bytes: NonNegativeInt
    feasible: bool
    reason: str
    score: float
    selected: bool = False
    files: tuple[str, ...] = ()


class CanonicalModelInspection(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: ResolvedModelDescriptor
    variants: tuple[VariantInspection, ...] = ()
    engine_support: tuple[EngineSupportReport, ...]
    automatic_variant: str | None = None
    automatic_engine: str | None = None
    download_bytes: NonNegativeInt = 0


class CanonicalDryRun(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    inspection: CanonicalModelInspection
    decision: CanonicalPlanningDecision


def _variant_inspections(resolution: ModelResolution) -> tuple[VariantInspection, ...]:
    selected = resolution.descriptor.variant
    candidates: tuple[VariantCandidate, ...] = resolution.variant_candidates
    return tuple(
        VariantInspection(
            variant=item.variant.variant_id,
            quantization=item.variant.quantization,
            bytes=item.variant.total_bytes,
            feasible=item.feasible,
            reason=item.reason,
            score=item.score,
            selected=item.variant.variant_id == selected,
            files=tuple(file.relative_path for file in item.variant.files),
        )
        for item in candidates
    )


class CanonicalRuntime:
    """Resolve, plan, acquire, prepare, and reuse canonical deployments."""

    def __init__(
        self,
        *,
        resolver: ModelSourceResolver,
        engines: ExecutionEngineRegistry | None = None,
        planner: CanonicalPlanner | None = None,
    ) -> None:
        self.resolver = resolver
        self.engines = engines or default_engine_registry()
        self.planner = planner or CanonicalPlanner(self.engines)
        self._deployments: dict[str, Deployment] = {}
        self._deployment_engines: dict[str, str] = {}
        self._lock = asyncio.Lock()

    def inspect(
        self,
        source: str | Path,
        cluster: ClusterCapabilities,
        *,
        revision: str | None = None,
        variant: str | None = None,
        quantization: str | None = None,
        objective: Literal["speed", "throughput", "capacity", "balanced"] = "balanced",
    ) -> tuple[ModelResolution, CanonicalModelInspection]:
        local_fast = max(
            (
                device.usable_memory_bytes
                for worker in cluster.workers
                for engine in worker.engines
                if engine.enabled
                for device in engine.devices
                if device.device_type in {"cuda", "metal", "mps", "rocm", "vulkan"}
            ),
            default=0,
        )
        resolution = self.resolver.inspect(
            source,
            revision=revision,
            variant=variant,
            quantization=quantization,
            objective=objective,
            resources=ResolutionResources(
                aggregate_usable_memory_bytes=cluster.aggregate_usable_memory_bytes,
                local_fast_memory_bytes=local_fast,
            ),
        )
        support = self.engines.probe_all(resolution.descriptor, cluster)
        supported = [item.engine_id for item in support if item.supported]
        inspection = CanonicalModelInspection(
            model=resolution.descriptor,
            variants=_variant_inspections(resolution),
            engine_support=support,
            automatic_variant=resolution.descriptor.variant,
            automatic_engine=supported[0] if len(supported) == 1 else None,
            download_bytes=(
                0
                if resolution.descriptor.source_type == "local"
                else resolution.descriptor.weight_bytes
            ),
        )
        return resolution, inspection

    async def dry_run(
        self,
        source: str | Path,
        cluster: ClusterCapabilities,
        request: ExecutionRequest,
        *,
        revision: str | None = None,
        variant: str | None = None,
        quantization: str | None = None,
        mechanism_evidence: tuple[MechanismEvidence, ...] = (),
    ) -> CanonicalDryRun:
        resolution, inspection = await asyncio.to_thread(
            self.inspect,
            source,
            cluster,
            revision=revision,
            variant=variant,
            quantization=quantization,
            objective=request.objective,
        )
        decision = await self.planner.plan(
            resolution.descriptor,
            cluster,
            request,
            mechanism_evidence=mechanism_evidence,
        )
        inspection = inspection.model_copy(update={"automatic_engine": decision.selected.engine_id})
        return CanonicalDryRun(inspection=inspection, decision=decision)

    async def _acquire_and_replan(
        self,
        resolution: ModelResolution,
        cluster: ClusterCapabilities,
        request: ExecutionRequest,
        mechanism_evidence: tuple[MechanismEvidence, ...],
    ) -> tuple[ResolvedModelDescriptor, CanonicalPlanningDecision]:
        paths = await self.resolver.acquire_async(resolution.descriptor)
        descriptor = resolution.descriptor.model_copy(
            update={"local_paths": tuple(str(item.resolve()) for item in paths)}
        )
        for engine in self.engines.engines():
            binder = getattr(engine, "bind_acquired_model", None)
            if callable(binder):
                binder(descriptor, paths)
        decision = await self.planner.plan(
            descriptor,
            cluster,
            request,
            mechanism_evidence=mechanism_evidence,
        )
        return descriptor, decision

    async def prepare(
        self,
        resolution: ModelResolution,
        cluster: ClusterCapabilities,
        request: ExecutionRequest,
        *,
        mechanism_evidence: tuple[MechanismEvidence, ...] = (),
    ) -> tuple[ResolvedModelDescriptor, CanonicalPlanningDecision, Deployment]:
        descriptor, decision = await self._acquire_and_replan(
            resolution,
            cluster,
            request,
            mechanism_evidence,
        )
        plan = decision.selected
        key = plan.execution_identity + ":" + plan.topology
        async with self._lock:
            existing = self._deployments.get(key)
            if existing is not None and existing.ready:
                return descriptor, decision, existing
            engine = self.engines.get(plan.engine_id)
            deployment = await engine.prepare(plan)
            if not deployment.ready:
                raise RuntimeError("execution engine returned an unready deployment")
            if deployment.execution_identity != plan.execution_identity:
                raise RuntimeError("deployment changed immutable execution identity")
            self._deployments[key] = deployment
            self._deployment_engines[deployment.deployment_id] = plan.engine_id
            return descriptor, decision, deployment

    async def submit(
        self,
        deployment: Deployment,
        request: InferenceRequest,
    ) -> AsyncIterator[InferenceEvent]:
        engine_id = self._deployment_engines.get(deployment.deployment_id)
        if engine_id is None or engine_id != deployment.engine_id:
            raise RuntimeError("deployment is not owned by this canonical runtime")
        engine = self.engines.get(engine_id)
        sequence = -1
        async for event in engine.submit(deployment, request):
            if event.request_id != request.request_id or event.sequence_number <= sequence:
                raise RuntimeError("engine emitted an invalid inference event sequence")
            sequence = event.sequence_number
            yield event

    async def unload(self, deployment_id: str) -> bool:
        async with self._lock:
            pair = next(
                (
                    (key, value)
                    for key, value in self._deployments.items()
                    if value.deployment_id == deployment_id
                ),
                None,
            )
            if pair is None:
                return False
            key, deployment = pair
            engine_id = self._deployment_engines.pop(deployment_id)
            await self.engines.get(engine_id).unload(deployment)
            del self._deployments[key]
            return True

    async def close(self) -> None:
        for deployment_id in list(self._deployment_engines):
            await self.unload(deployment_id)


__all__ = [
    "CanonicalDryRun",
    "CanonicalModelInspection",
    "CanonicalRuntime",
    "VariantInspection",
]
