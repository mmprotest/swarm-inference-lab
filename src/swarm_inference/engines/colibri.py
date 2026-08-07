"""First-class Colibri complete-model execution engine."""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Callable
from typing import Protocol, TypeVar
from uuid import uuid4

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
from swarm_inference.model.architecture import normalize_model_architecture
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
) -> str | None:
    """Match an advertised backend adapter without embedding model-family policy."""

    architecture = normalize_model_architecture(model.architecture)
    matches = [
        adapter
        for adapter in capability.adapters
        if normalize_model_architecture(adapter) == architecture
    ]
    if not matches:
        return None
    return sorted(matches)[0]


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
        runtime_workers: list[WorkerExecutionCapability] = []
        architecture_workers: list[tuple[WorkerExecutionCapability, str]] = []
        workers: list[tuple[WorkerExecutionCapability, str]] = []
        broken: list[str] = []
        for worker in cluster.workers_for_engine(self.engine_id):
            capability = worker.engine(self.engine_id)
            assert capability is not None
            if not capability.runtime_revision or not capability.binary_hashes:
                broken.append(worker.worker_id)
                continue
            runtime_workers.append(worker)
            adapter_id = _adapter_for_model(model, capability)
            if adapter_id is None:
                continue
            architecture_workers.append((worker, adapter_id))
            if model.format.casefold() in {item.casefold() for item in capability.formats}:
                workers.append((worker, adapter_id))
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
                model_architecture=model.architecture,
                model_format=model.format,
                required_runtime="pinned Colibri runtime",
            )
        if not architecture_workers:
            return EngineSupportReport(
                engine_id=self.engine_id,
                status=EngineSupportStatus.UNSUPPORTED_ARCHITECTURE,
                reason="no Colibri runtime advertises an adapter matching the model",
                model_architecture=model.architecture,
                model_format=model.format,
                required_runtime="pinned Colibri runtime with a matching model adapter",
            )
        if not workers:
            return EngineSupportReport(
                engine_id=self.engine_id,
                status=EngineSupportStatus.UNSUPPORTED_FORMAT,
                reason=f"Colibri adapters do not consume model format {model.format!r}",
                adapter_id=architecture_workers[0][1],
                model_architecture=model.architecture,
                model_format=model.format,
                required_runtime="pinned Colibri runtime",
            )
        feasible: list[tuple[WorkerExecutionCapability, str]] = []
        for worker, adapter_id in workers:
            capability = worker.engine(self.engine_id)
            assert capability is not None
            if sum(item.usable_memory_bytes for item in capability.devices) >= model.weight_bytes:
                feasible.append((worker, adapter_id))
        selected_adapter = workers[0][1]
        if not feasible:
            return EngineSupportReport(
                engine_id=self.engine_id,
                status=EngineSupportStatus.INSUFFICIENT_MEMORY,
                reason="no complete-model Colibri worker passes memory admission",
                supported_worker_ids=tuple(item.worker_id for item, _adapter in workers),
                adapter_id=selected_adapter,
                model_architecture=model.architecture,
                model_format=model.format,
                required_runtime="pinned Colibri runtime with complete-model memory",
            )
        return EngineSupportReport(
            engine_id=self.engine_id,
            status=EngineSupportStatus.SUPPORTED,
            reason=f"pinned Colibri runtime supports adapter {selected_adapter}",
            supported_worker_ids=tuple(item.worker_id for item, _adapter in feasible),
            adapter_id=selected_adapter,
            model_architecture=model.architecture,
            model_format=model.format,
            required_runtime="pinned Colibri runtime",
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
        plans: list[ExecutionPlan] = []
        for worker in cluster.workers_for_engine(self.engine_id):
            identities = {worker.worker_id, worker.node_id}
            if identities.intersection(request.excluded_nodes) or (
                request.requested_nodes and not identities.intersection(request.requested_nodes)
            ):
                continue
            capability = worker.engine(self.engine_id)
            assert capability is not None
            adapter_id = _adapter_for_model(model, capability)
            if adapter_id is None or model.format.casefold() not in {
                item.casefold() for item in capability.formats
            }:
                continue
            memory = sum(item.usable_memory_bytes for item in capability.devices)
            if memory < model.weight_bytes:
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
                required_memory_bytes=model.weight_bytes,
                resident_model_bytes=(
                    model.weight_bytes
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
                and item.adapter_id == adapter_id
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
                        "adapter": adapter_id,
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
                        topology="colibri-complete-model",
                        worker_roles=roles,
                        idle_workers={
                            item.worker_id: "complete-model Colibri request has no positive synchronous role"
                            for item in cluster.workers
                            if item.worker_id != worker.worker_id
                        },
                        fast_paths={worker.worker_id: fast_path},
                        optional_mechanisms={
                            "routing_aware_placement": routing_aware,
                            "prefetch": False,
                            "tensor_microshards": False,
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
                        persistent_connections=False,
                        network_cost_confidence="measured",
                        network_cost_provenance="complete model executes on one worker",
                        required_memory_bytes=model.weight_bytes,
                        score=utility.score,
                        explanation=(
                            "Colibri candidate is admitted only from its physical capability probe",
                            "rejected prefetch variants remain disabled",
                            *(
                                ("unmeasured inputs: " + ", ".join(utility.unmeasured_inputs),)
                                if utility.unmeasured_inputs
                                else ()
                            ),
                        ),
                        engine_parameters={
                            "model_id": model.model_id,
                            "model_revision": model.revision,
                            "model_family": adapter_id,
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
