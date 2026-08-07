"""Colibri execution engine and sparse-MoE component capability provider."""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Callable
from typing import Protocol, TypeVar
from uuid import uuid4

from swarm_inference.backends.colibri.adapters import default_colibri_adapter_registry
from swarm_inference.backends.colibri.architecture import (
    ColibriArchitectureAdapter,
    ColibriExecutionProfile,
    ColibriRuntimeCapabilities,
)
from swarm_inference.backends.colibri.backend import ColibriBackend
from swarm_inference.engines.cost_model import PlanCostInputs, score_costs, stable_plan_id
from swarm_inference.engines.interfaces import (
    ClusterCapabilities,
    Deployment,
    EngineSupportReport,
    EngineSupportStatus,
    ExecutionEngineCapability,
    ExecutionPlan,
    ExecutionProfileCapability,
    ExecutionRequest,
    InferenceEvent,
    InferenceRequest,
    MechanismEvidence,
    PhasePlan,
    WorkerExecutionCapability,
)
from swarm_inference.model.descriptor import ResolvedModelDescriptor
from swarm_inference.worker.abi import (
    GenerationParameters,
    TokenPayload,
    WorkerJob,
    WorkerJobStatus,
    WorkerJobType,
)


def _adapter_for_model(
    model: ResolvedModelDescriptor,
    capability: ExecutionEngineCapability,
) -> ColibriArchitectureAdapter | None:
    """Resolve exact checkpoint metadata, then require an installed adapter runtime."""

    try:
        adapter = default_colibri_adapter_registry().resolve_model(model)
    except LookupError:
        return None
    if adapter is None or adapter.adapter_id not in capability.adapters:
        return None
    return adapter


def _runtime_capabilities(
    capability: ExecutionEngineCapability,
) -> ColibriRuntimeCapabilities:
    return ColibriRuntimeCapabilities(
        installed=capability.enabled,
        runtime_version=capability.runtime_revision,
        binary_hashes=capability.binary_hashes,
        adapters=capability.adapters,
        formats=capability.formats,
        quantizations=capability.quantizations,
        device_types=tuple(sorted({item.device_type for item in capability.devices})),
        features=tuple(
            sorted(
                {
                    *capability.required_features,
                    *capability.fast_paths,
                    *(feature for item in capability.devices for feature in item.features),
                }
            )
        ),
    )


class ColibriLifecycle(Protocol):
    async def prepare(self, plan: ExecutionPlan) -> Deployment: ...

    def submit(
        self, deployment: Deployment, request: InferenceRequest
    ) -> AsyncIterator[InferenceEvent]: ...

    async def unload(self, deployment: Deployment) -> None: ...


class LocalColibriLifecycle:
    """Own a persistent Colibri backend for the lifetime of one deployment."""

    def __init__(self, backend_factory: Callable[[ExecutionPlan], ColibriBackend]) -> None:
        self.backend_factory = backend_factory
        self._backends: dict[str, ColibriBackend] = {}

    async def prepare(self, plan: ExecutionPlan) -> Deployment:
        active = [worker for worker, role in plan.worker_roles.items() if role != "idle"]
        if len(active) != 1:
            raise RuntimeError("the complete-model Colibri engine requires one owning worker")
        backend = self.backend_factory(plan)
        capabilities = await asyncio_to_thread(backend.capabilities)
        deployment_id = f"colibri-{uuid4().hex}"
        self._backends[deployment_id] = backend
        return Deployment(
            deployment_id=deployment_id,
            engine_id=plan.engine_id,
            execution_identity=plan.execution_identity,
            plan=plan,
            ready=True,
            metadata={
                "capabilities": capabilities.model_dump(mode="json"),
                "routing_telemetry": True,
            },
        )

    async def submit(
        self,
        deployment: Deployment,
        request: InferenceRequest,
    ) -> AsyncIterator[InferenceEvent]:
        try:
            backend = self._backends[deployment.deployment_id]
        except KeyError as exc:
            raise RuntimeError("Colibri deployment is not active") from exc
        yield InferenceEvent(event_type="started", request_id=request.request_id, sequence_number=0)
        prompt_payload = await asyncio_to_thread(
            lambda: backend.prompt_payload(
                request.prompt,
                tokenizer_hash=(
                    str(deployment.plan.engine_parameters["tokenizer_identity"])
                    if deployment.plan.engine_parameters.get("tokenizer_identity")
                    else None
                ),
            )
        )
        job = WorkerJob(
            job_id=f"job-{uuid4().hex}",
            request_id=request.request_id,
            role=WorkerJobType.GENERATE,
            model_id=str(deployment.plan.engine_parameters["model_id"]),
            model_revision=str(deployment.plan.engine_parameters["model_revision"]),
            input_payload=prompt_payload,
            generation_parameters=GenerationParameters(
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=1.0,
                top_k=1 if request.temperature == 0 else 40,
                seed=request.seed,
            ),
            deadline_ms=3_600_000,
        )
        result = await backend.execute(job)
        if result.status != WorkerJobStatus.ACCEPTED:
            yield InferenceEvent(
                event_type="failed",
                request_id=request.request_id,
                sequence_number=1,
                detail=result.detail,
            )
            return
        output = result.output_payload
        tokens = output.token_ids if isinstance(output, TokenPayload) else []
        text = output.text or "" if isinstance(output, TokenPayload) else ""
        if tokens and not text:
            text = str(await asyncio_to_thread(lambda: backend.decode_tokens(tokens)))
        for index, token in enumerate(tokens, start=1):
            yield InferenceEvent(
                event_type="token",
                request_id=request.request_id,
                sequence_number=index,
                token_id=token,
            )
        yield InferenceEvent(
            event_type="completed",
            request_id=request.request_id,
            sequence_number=len(tokens) + 1,
            text=text,
            telemetry=result.metrics,
        )

    async def unload(self, deployment: Deployment) -> None:
        backend = self._backends.pop(deployment.deployment_id, None)
        if backend is not None:
            await backend.shutdown()


_T = TypeVar("_T")


async def asyncio_to_thread(function: Callable[[], _T]) -> _T:
    import asyncio

    return await asyncio.to_thread(function)


class ColibriExecutionEngine:
    engine_id = "colibri"

    def __init__(self, *, lifecycle: ColibriLifecycle | None = None) -> None:
        self.lifecycle = lifecycle

    def probe(
        self,
        model: ResolvedModelDescriptor,
        cluster: ClusterCapabilities,
    ) -> EngineSupportReport:
        try:
            adapter = default_colibri_adapter_registry().resolve_model(model)
        except LookupError as exc:
            return EngineSupportReport(
                engine_id=self.engine_id,
                status=EngineSupportStatus.UNSUPPORTED_ARCHITECTURE,
                reason=str(exc),
                model_architecture=model.architecture,
                model_format=model.format,
                architecture_supported=False,
                format_supported=model.format == "safetensors",
                quantization_supported=False,
                hardware_supported=False,
                required_runtime="pinned Colibri runtime with an architecture adapter",
                confidence=1.0,
            )
        if adapter is None:
            return EngineSupportReport(
                engine_id=self.engine_id,
                status=EngineSupportStatus.UNSUPPORTED_ARCHITECTURE,
                reason="no Colibri architecture adapter validates the checkpoint metadata",
                model_architecture=model.architecture,
                model_format=model.format,
                architecture_supported=False,
                format_supported=model.format == "safetensors",
                quantization_supported=False,
                hardware_supported=False,
                required_runtime="pinned Colibri runtime with an architecture adapter",
                confidence=1.0,
            )

        # Separate the Swarm-owned adapter contract from the physical runtime
        # inventory. This keeps format/quantization/scope facts visible even
        # on a coordinator that has no Colibri worker installed.
        implementation_support = adapter.supports(
            model,
            ColibriRuntimeCapabilities(
                installed=True,
                runtime_version="adapter-contract",
                binary_hashes={"adapter-contract": adapter.adapter_version},
                adapters=(adapter.adapter_id,),
                formats=("safetensors",),
                device_types=("cpu", "cuda"),
            ),
        )
        implementation_profile = (
            adapter.build_execution_profile(model, cluster)
            if all(
                (
                    implementation_support.architecture_supported,
                    implementation_support.model_format_supported,
                    implementation_support.quantization_supported,
                )
            )
            else None
        )
        implementation_capabilities = (
            implementation_profile.component_capabilities
            if implementation_profile is not None and implementation_profile.component_capabilities
            else adapter.component_capabilities
            or ("sparse-moe-model-execution", "expert-storage-tiering")
        )
        support_scope = "complete_model" if adapter.complete_model else "component"
        if not all(
            (
                implementation_support.architecture_supported,
                implementation_support.model_format_supported,
                implementation_support.quantization_supported,
            )
        ):
            status = (
                EngineSupportStatus.UNSUPPORTED_ARCHITECTURE
                if not implementation_support.architecture_supported
                else EngineSupportStatus.UNSUPPORTED_FORMAT
                if not implementation_support.model_format_supported
                else EngineSupportStatus.UNSUPPORTED_QUANTIZATION
            )
            return EngineSupportReport(
                engine_id=self.engine_id,
                status=status,
                reason="; ".join(implementation_support.reasons),
                adapter_id=adapter.adapter_id,
                model_architecture=model.architecture,
                model_format=model.format,
                architecture_supported=implementation_support.architecture_supported,
                format_supported=implementation_support.model_format_supported,
                quantization_supported=implementation_support.quantization_supported,
                hardware_supported=False,
                capabilities=implementation_capabilities,
                limitations=implementation_support.limitations,
                expected_memory_cost=(
                    implementation_profile.required_memory_bytes
                    if implementation_profile is not None
                    else None
                ),
                required_runtime=f"pinned Colibri adapter {adapter.adapter_id}",
                required_features=implementation_support.required_features,
                confidence=1.0,
                support_scope=support_scope,
            )

        runtime_workers: list[WorkerExecutionCapability] = []
        broken: list[str] = []
        support_results = []
        feasible: list[tuple[WorkerExecutionCapability, ColibriExecutionProfile]] = []
        runtime_identity: dict[str, object] = {}
        for worker in cluster.workers_for_engine(self.engine_id):
            capability = worker.engine(self.engine_id)
            assert capability is not None
            if not capability.runtime_revision or not capability.binary_hashes:
                broken.append(worker.worker_id)
                continue
            runtime_workers.append(worker)
            result = adapter.supports(model, _runtime_capabilities(capability))
            support_results.append(result)
            if not result.supported:
                continue
            profile = adapter.build_execution_profile(model, cluster)
            memory = sum(item.usable_memory_bytes for item in capability.devices)
            if memory >= profile.required_memory_bytes:
                feasible.append((worker, profile))
                runtime_identity = {
                    "revision": capability.runtime_revision,
                    "binary_hashes": capability.binary_hashes,
                    "adapter_version": adapter.adapter_version,
                }
        if not runtime_workers:
            return EngineSupportReport(
                engine_id=self.engine_id,
                status=(
                    EngineSupportStatus.BROKEN_RUNTIME
                    if broken
                    else EngineSupportStatus.MISSING_RUNTIME
                ),
                reason=(
                    "Colibri capability lacks pinned runtime hash evidence"
                    if broken
                    else "no worker advertises a pinned Colibri runtime"
                ),
                adapter_id=adapter.adapter_id,
                model_architecture=model.architecture,
                model_format=model.format,
                architecture_supported=True,
                format_supported=model.format == "safetensors",
                quantization_supported=True,
                hardware_supported=False,
                capabilities=implementation_capabilities,
                limitations=tuple(
                    dict.fromkeys(
                        (
                            *implementation_support.limitations,
                            *(
                                implementation_profile.limitations
                                if implementation_profile is not None
                                else ()
                            ),
                        )
                    )
                ),
                expected_memory_cost=(
                    implementation_profile.required_memory_bytes
                    if implementation_profile is not None
                    else None
                ),
                required_runtime="pinned Colibri runtime",
                required_features=implementation_support.required_features,
                confidence=1.0,
                support_scope=support_scope,
            )
        if not support_results:
            return EngineSupportReport(
                engine_id=self.engine_id,
                status=EngineSupportStatus.MISSING_RUNTIME,
                reason=(
                    f"pinned Colibri workers do not install architecture adapter "
                    f"{adapter.adapter_id!r}"
                ),
                adapter_id=adapter.adapter_id,
                model_architecture=model.architecture,
                model_format=model.format,
                architecture_supported=True,
                format_supported=model.format == "safetensors",
                quantization_supported=True,
                hardware_supported=False,
                capabilities=implementation_capabilities,
                limitations=implementation_support.limitations,
                expected_memory_cost=(
                    implementation_profile.required_memory_bytes
                    if implementation_profile is not None
                    else None
                ),
                required_runtime=f"pinned Colibri adapter {adapter.adapter_id}",
                required_features=(adapter.adapter_id,),
                confidence=1.0,
                support_scope=support_scope,
            )
        supported_result = next((item for item in support_results if item.supported), None)
        if supported_result is None:
            format_supported = any(item.model_format_supported for item in support_results)
            quantization_supported = any(item.quantization_supported for item in support_results)
            architecture_supported = any(item.architecture_supported for item in support_results)
            status = (
                EngineSupportStatus.UNSUPPORTED_FORMAT
                if not format_supported
                else EngineSupportStatus.UNSUPPORTED_QUANTIZATION
                if not quantization_supported
                else EngineSupportStatus.MISSING_RUNTIME
            )
            reasons = tuple(
                dict.fromkeys(reason for item in support_results for reason in item.reasons)
            )
            limitations = tuple(
                dict.fromkeys(
                    limitation for item in support_results for limitation in item.limitations
                )
            )
            return EngineSupportReport(
                engine_id=self.engine_id,
                status=status,
                reason="; ".join(reasons),
                adapter_id=adapter.adapter_id,
                model_architecture=model.architecture,
                model_format=model.format,
                architecture_supported=architecture_supported,
                format_supported=format_supported,
                quantization_supported=quantization_supported,
                hardware_supported=False,
                limitations=limitations,
                required_runtime=f"pinned Colibri adapter {adapter.adapter_id}",
                required_features=tuple(
                    dict.fromkeys(
                        feature for item in support_results for feature in item.required_features
                    )
                ),
                confidence=1.0,
            )
        assert implementation_profile is not None
        profile = implementation_profile
        capabilities = profile.component_capabilities or (
            "sparse-moe-model-execution",
            "expert-storage-tiering",
        )
        if not feasible:
            return EngineSupportReport(
                engine_id=self.engine_id,
                status=EngineSupportStatus.INSUFFICIENT_MEMORY,
                reason=(
                    "no Colibri worker satisfies the adapter-derived resident working-set "
                    f"requirement ({profile.required_memory_bytes} bytes)"
                ),
                supported_worker_ids=tuple(item.worker_id for item in runtime_workers),
                adapter_id=adapter.adapter_id,
                model_architecture=model.architecture,
                model_format=model.format,
                architecture_supported=True,
                format_supported=True,
                quantization_supported=True,
                hardware_supported=False,
                capabilities=capabilities,
                limitations=tuple((*supported_result.limitations, *profile.limitations)),
                expected_memory_cost=profile.required_memory_bytes,
                required_runtime=f"pinned Colibri adapter {adapter.adapter_id}",
                confidence=0.9,
                support_scope="complete_model" if profile.complete_model else "component",
            )
        limitations = tuple(dict.fromkeys((*supported_result.limitations, *profile.limitations)))
        measured_rates = [
            device.measured_decode_tokens_s
            for worker, _worker_profile in feasible
            for device in worker.engine(self.engine_id).devices  # type: ignore[union-attr]
            if device.measured_decode_tokens_s is not None
        ]
        return EngineSupportReport(
            engine_id=self.engine_id,
            status=(
                EngineSupportStatus.SUPPORTED
                if profile.complete_model
                else EngineSupportStatus.COMPONENT_SUPPORTED
            ),
            reason=(
                f"pinned Colibri adapter {adapter.adapter_id} provides a complete execution path"
                if profile.complete_model
                else f"pinned Swarm Colibri adapter {adapter.adapter_id} provides routed-MoE "
                "components for hybrid planning"
            ),
            supported_worker_ids=tuple(item.worker_id for item, _profile in feasible),
            adapter_id=adapter.adapter_id,
            runtime_identity=runtime_identity,
            model_architecture=model.architecture,
            model_format=model.format,
            architecture_supported=True,
            format_supported=True,
            quantization_supported=True,
            hardware_supported=True,
            capabilities=capabilities,
            limitations=limitations,
            expected_compute_cost=(
                1.0 / max(measured_rates) if measured_rates and max(measured_rates) > 0 else None
            ),
            expected_network_cost=0.0 if profile.complete_model else None,
            expected_memory_cost=profile.required_memory_bytes,
            required_runtime=f"pinned Colibri adapter {adapter.adapter_id}",
            required_features=supported_result.required_features,
            confidence=0.95 if measured_rates else 0.7,
            support_scope="complete_model" if profile.complete_model else "component",
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
        try:
            architecture_adapter = default_colibri_adapter_registry().resolve_model(model)
        except LookupError:
            return []
        if architecture_adapter is None:
            return []

        plans: list[ExecutionPlan] = []
        for worker in cluster.workers_for_engine(self.engine_id):
            identities = {worker.worker_id, worker.node_id}
            if identities.intersection(request.excluded_nodes) or (
                request.requested_nodes and not identities.intersection(request.requested_nodes)
            ):
                continue
            capability = worker.engine(self.engine_id)
            assert capability is not None
            adapter = _adapter_for_model(model, capability)
            if adapter is None:
                continue
            support = adapter.supports(model, _runtime_capabilities(capability))
            if not support.supported:
                continue
            profile = adapter.build_execution_profile(model, cluster)
            # Component adapters participate in hybrid planning.  They are not a
            # valid complete-model candidate until a composable planner supplies
            # every other required model capability.
            if not profile.complete_model:
                continue
            memory = sum(item.usable_memory_bytes for item in capability.devices)
            if memory < profile.required_memory_bytes:
                continue
            decode_rates = [
                item.measured_decode_tokens_s
                for item in capability.devices
                if item.measured_decode_tokens_s
            ]
            prefill_rates = [
                item.measured_prefill_tokens_s
                for item in capability.devices
                if item.measured_prefill_tokens_s
            ]
            costs = PlanCostInputs(
                measured_prefill_tokens_s=max(prefill_rates) if prefill_rates else None,
                measured_decode_tokens_s=max(decode_rates) if decode_rates else None,
                queue_depth=worker.queue_depth,
                reliability=worker.reliability,
                usable_memory_bytes=memory,
                required_memory_bytes=profile.required_memory_bytes,
                resident_model_bytes=(
                    profile.required_memory_bytes
                    if model.content_fingerprint in worker.resident_model_fingerprints
                    else 0
                ),
                artifact_transfer_bytes=model.weight_bytes,
                concurrency=request.concurrency,
                request_priority=request.priority,
                network_latency_ms=0.0,
                network_jitter_ms=0.0,
                messages_per_token=0.0,
                bytes_per_token=0.0,
                serial_waits_per_token=0.0,
            )
            utility = score_costs(costs, objective=request.objective)
            roles = {worker.worker_id: "critical_path_stage"}
            routing_profiles = tuple(
                item
                for item in capability.execution_profiles
                if item.mechanism == "routing_aware_placement"
                and item.adapter_id == adapter.adapter_id
                and item.model_fingerprint == model.content_fingerprint
                and item.exactness_passed
                and item.measured_utility > 0
            )
            policy_candidates: list[tuple[str, ExecutionProfileCapability | None]] = [
                ("backend-native", None)
            ]
            if "routing-aware-placement" in capability.fast_paths:
                policy_candidates.extend(
                    ("routing-aware-placement", profile) for profile in routing_profiles
                )
            for fast_path, routing_profile in policy_candidates:
                routing_aware = routing_profile is not None
                identity_payload = json.dumps(
                    {
                        "model": model.content_fingerprint,
                        "runtime": capability.runtime_revision,
                        "binaries": capability.binary_hashes,
                        "adapter": adapter.adapter_id,
                        "routing_profile": (
                            routing_profile.content_fingerprint if routing_profile else None
                        ),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                identity = "sha256:" + hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()
                plan_identity = {
                    "execution_identity": identity,
                    "objective": request.objective,
                    "worker": worker.worker_id,
                    "fast_path": fast_path,
                    "routing_profile": (
                        routing_profile.content_fingerprint if routing_profile else None
                    ),
                    "context": request.max_context_tokens,
                    "concurrency": request.concurrency,
                }
                plans.append(
                    ExecutionPlan(
                        plan_id=stable_plan_id("colibri", plan_identity),
                        engine_id=self.engine_id,
                        model_fingerprint=model.content_fingerprint,
                        execution_identity=identity,
                        objective=request.objective,
                        topology=profile.topology,
                        worker_roles=roles,
                        idle_workers={
                            item.worker_id: "complete-model Colibri request has no positive synchronous role"
                            for item in cluster.workers
                            if item.worker_id != worker.worker_id
                        },
                        fast_paths={worker.worker_id: fast_path},
                        optional_mechanisms={
                            "routing_aware_placement": routing_aware,
                            "prefetch": profile.persistent_expert_residency,
                            "tensor_microshards": profile.tensor_microshards,
                            "direct_peer_model_data": profile.direct_peer_model_data,
                            "persistent_expert_residency": (profile.persistent_expert_residency),
                        },
                        mechanism_evidence=(
                            (
                                MechanismEvidence(
                                    mechanism="routing_aware_placement",
                                    exactness_passed=routing_profile.exactness_passed,
                                    measured_utility=routing_profile.measured_utility,
                                    evidence_fingerprint=(routing_profile.evidence_fingerprint),
                                    runtime_fingerprint=identity,
                                ),
                            )
                            if routing_profile is not None
                            else ()
                        ),
                        prefill_plan=PhasePlan(phase="prefill", worker_roles=roles),
                        decode_plan=PhasePlan(phase="decode", worker_roles=roles),
                        predicted_ttft_ms=utility.predicted_ttft_ms,
                        predicted_decode_tokens_s=utility.predicted_decode_tokens_s,
                        predicted_aggregate_tokens_s=utility.predicted_aggregate_tokens_s,
                        predicted_network_bytes=0,
                        predicted_messages_per_token=0.0,
                        predicted_bytes_per_token=0.0,
                        predicted_serial_waits_per_token=0.0,
                        number_of_wan_stage_boundaries=0,
                        persistent_connections=profile.persistent_worker,
                        network_cost_confidence="measured",
                        network_cost_provenance="complete model executes on one worker",
                        required_memory_bytes=profile.required_memory_bytes,
                        score=utility.score,
                        explanation=(
                            "Colibri candidate is admitted only from its physical capability probe",
                            "resident memory is adapter-derived rather than total checkpoint size",
                            "rejected prefetch variants remain disabled",
                            *profile.limitations,
                            *(
                                ("unmeasured inputs: " + ", ".join(utility.unmeasured_inputs),)
                                if utility.unmeasured_inputs
                                else ()
                            ),
                        ),
                        engine_parameters={
                            "model_id": model.model_id,
                            "model_revision": model.revision,
                            "model_family": adapter.adapter_id,
                            "model_paths": list(model.local_paths),
                            "tokenizer_identity": model.tokenizer_identity,
                            **(
                                {"routing_profile_id": routing_profile.profile_id}
                                if routing_profile is not None
                                else {}
                            ),
                        },
                    )
                )
        return plans

    async def prepare(self, plan: ExecutionPlan) -> Deployment:
        if self.lifecycle is None:
            raise RuntimeError("Colibri lifecycle is not configured")
        return await self.lifecycle.prepare(plan)

    async def submit(
        self,
        deployment: Deployment,
        request: InferenceRequest,
    ) -> AsyncIterator[InferenceEvent]:
        if self.lifecycle is None:
            raise RuntimeError("Colibri lifecycle is not configured")
        async for event in self.lifecycle.submit(deployment, request):
            yield event

    async def unload(self, deployment: Deployment) -> None:
        if self.lifecycle is None:
            raise RuntimeError("Colibri lifecycle is not configured")
        await self.lifecycle.unload(deployment)


__all__ = ["ColibriExecutionEngine", "ColibriLifecycle", "LocalColibriLifecycle"]
