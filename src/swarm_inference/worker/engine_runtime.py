"""Persistent authenticated lifecycle owner for registered execution engines."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from swarm_inference.cluster.artifacts import load_artifact_manifest
from swarm_inference.engines.interfaces import Deployment, ExecutionEngine, InferenceEvent
from swarm_inference.protocol.engine_worker import (
    EngineActionResponse,
    EngineControlAction,
    EngineInferenceResponse,
    PrepareEngineRequest,
    SubmitEngineRequest,
    UnloadEngineRequest,
    verify_engine_deployment_lease,
)
from swarm_inference.protocol.routes import BoundedNonceCache
from swarm_inference.security.identity import public_key_fingerprint


class PersistentEngineRuntime:
    """Keep engine deployments resident and reject unsigned lifecycle actions."""

    def __init__(
        self,
        *,
        worker_id: str,
        engines: tuple[ExecutionEngine, ...] = (),
        artifact_resolver: Callable[[str], Path] | None = None,
        maximum_deployments: int = 8,
        maximum_cached_responses: int = 1024,
        nonce_cache_capacity: int = 4096,
    ) -> None:
        if not worker_id:
            raise ValueError("engine runtime worker ID cannot be empty")
        if maximum_deployments <= 0 or maximum_cached_responses <= 0:
            raise ValueError("engine runtime bounds must be positive")
        self.worker_id = worker_id
        self.artifact_resolver = artifact_resolver
        self.maximum_deployments = maximum_deployments
        self.maximum_cached_responses = maximum_cached_responses
        self._engines: dict[str, ExecutionEngine] = {}
        for engine in engines:
            self.register(engine)
        self._deployments: dict[str, tuple[ExecutionEngine, Deployment]] = {}
        self._responses: OrderedDict[
            str, EngineActionResponse | EngineInferenceResponse
        ] = OrderedDict()
        self._nonce_cache = BoundedNonceCache(capacity=nonce_cache_capacity)
        self._coordinator_public_key: str | None = None
        self._coordinator_fingerprint: str | None = None
        self._lock = asyncio.Lock()

    def register(self, engine: ExecutionEngine) -> None:
        engine_id = engine.engine_id.strip()
        if not engine_id:
            raise ValueError("execution engine ID cannot be empty")
        if engine_id in self._engines:
            raise ValueError(f"execution engine {engine_id!r} is already registered")
        self._engines[engine_id] = engine

    @property
    def engine_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._engines))

    @property
    def deployment_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._deployments))

    def configure_trust(
        self,
        *,
        coordinator_public_key: str,
        expected_fingerprint: str | None,
    ) -> None:
        actual = public_key_fingerprint(coordinator_public_key)
        if expected_fingerprint is None or actual != expected_fingerprint:
            raise ValueError("engine runtime coordinator fingerprint is not pinned")
        self._coordinator_public_key = coordinator_public_key
        self._coordinator_fingerprint = actual

    def _cached(
        self, request_id: str
    ) -> EngineActionResponse | EngineInferenceResponse | None:
        response = self._responses.get(request_id)
        if response is not None:
            self._responses.move_to_end(request_id)
        return response

    def _remember(
        self,
        request_id: str,
        response: EngineActionResponse | EngineInferenceResponse,
    ) -> None:
        self._responses[request_id] = response
        self._responses.move_to_end(request_id)
        while len(self._responses) > self.maximum_cached_responses:
            self._responses.popitem(last=False)

    def _verify(
        self,
        request: PrepareEngineRequest | SubmitEngineRequest | UnloadEngineRequest,
        *,
        action: EngineControlAction,
    ) -> None:
        public_key = self._coordinator_public_key
        fingerprint = self._coordinator_fingerprint
        if public_key is None or fingerprint is None:
            raise PermissionError("engine runtime coordinator trust is not configured")
        lease = request.lease
        plan = request.plan
        deployment_id = request.deployment_id
        verify_engine_deployment_lease(
            lease,
            action=action,
            worker_id=self.worker_id,
            deployment_id=deployment_id,
            engine_id=plan.engine_id,
            execution_identity=plan.execution_identity,
            plan=plan,
            trusted_coordinator_public_key=public_key,
            trusted_coordinator_fingerprint=fingerprint,
        )
        self._nonce_cache.add(lease.nonce)

    def _worker_plan(self, request: PrepareEngineRequest):
        plan = request.plan
        if not request.artifact_ids:
            return plan
        if self.artifact_resolver is None:
            raise RuntimeError("engine artifacts were requested but no resolver is configured")
        model_paths: list[str] = []
        for artifact_id in request.artifact_ids:
            root = self.artifact_resolver(artifact_id)
            manifest = load_artifact_manifest(root)
            if manifest.model_fingerprint != plan.model_fingerprint:
                raise RuntimeError("engine artifact model identity differs from the plan")
            model_paths.extend(str((root / item.relative_path).resolve()) for item in manifest.files)
        parameters = {**plan.engine_parameters, "model_paths": model_paths}
        return plan.model_copy(update={"engine_parameters": parameters})

    async def prepare(self, request: PrepareEngineRequest) -> EngineActionResponse:
        if request.worker_id != self.worker_id:
            raise PermissionError("engine prepare request targets another worker")
        cached = self._cached(request.request_id)
        if cached is not None:
            if not isinstance(cached, EngineActionResponse):
                raise RuntimeError("engine request ID was reused for another response type")
            return cached.model_copy(deep=True)
        self._verify(request, action="prepare")
        assigned = request.plan.worker_roles.get(self.worker_id)
        if assigned != request.assigned_role or assigned in {None, "idle"}:
            raise RuntimeError("engine plan does not assign the declared active worker role")
        try:
            engine = self._engines[request.plan.engine_id]
        except KeyError as exc:
            raise RuntimeError(
                f"worker does not own engine {request.plan.engine_id!r}"
            ) from exc
        async with self._lock:
            existing = self._deployments.get(request.deployment_id)
            if existing is not None:
                _, deployment = existing
                if deployment.execution_identity != request.plan.execution_identity:
                    raise RuntimeError("deployment ID is already bound to another identity")
            else:
                if len(self._deployments) >= self.maximum_deployments:
                    raise RuntimeError("worker engine deployment capacity is full")
                worker_plan = self._worker_plan(request)
                deployment = await engine.prepare(worker_plan)
                if not deployment.ready:
                    raise RuntimeError("execution engine returned an unready deployment")
                self._deployments[request.deployment_id] = (engine, deployment)
            public_deployment = deployment.model_copy(
                update={
                    "deployment_id": request.deployment_id,
                    "plan": request.plan,
                }
            )
            response = EngineActionResponse(
                worker_id=self.worker_id,
                request_id=request.request_id,
                deployment_id=request.deployment_id,
                accepted=True,
                detail="engine deployment is resident",
                deployment=public_deployment,
                process_ids=public_deployment.process_ids,
                endpoints=public_deployment.endpoints,
            )
            self._remember(request.request_id, response)
            return response.model_copy(deep=True)

    async def stream(self, request: SubmitEngineRequest) -> AsyncIterator[InferenceEvent]:
        """Validate once, then expose bounded events without buffering the response."""

        if request.worker_id != self.worker_id:
            raise PermissionError("engine submit request targets another worker")
        cached = self._cached(request.request_id)
        if cached is not None:
            if not isinstance(cached, EngineInferenceResponse):
                raise RuntimeError("engine request ID was reused for another response type")
            for event in cached.events:
                yield event.model_copy(deep=True)
            return
        self._verify(request, action="submit")
        try:
            engine, deployment = self._deployments[request.deployment_id]
        except KeyError as exc:
            raise RuntimeError("engine deployment is not resident") from exc
        if deployment.execution_identity != request.plan.execution_identity:
            raise RuntimeError("engine submission identity differs from the resident deployment")
        events: list[InferenceEvent] = []
        terminal: InferenceEvent | None = None
        previous_sequence = -1
        async for event in engine.submit(deployment, request.inference):
            if event.request_id != request.inference.request_id:
                raise RuntimeError("engine event request identity differs from the submission")
            if event.sequence_number <= previous_sequence:
                raise RuntimeError("engine event sequence is not strictly increasing")
            previous_sequence = event.sequence_number
            if terminal is not None:
                raise RuntimeError("engine emitted an event after its terminal event")
            events.append(event)
            if len(events) > request.inference.max_new_tokens + 3:
                raise RuntimeError("engine emitted more events than the bounded request permits")
            if event.event_type in {"completed", "failed"}:
                terminal = event
            else:
                yield event.model_copy(deep=True)
        if terminal is None or events[-1] is not terminal:
            raise RuntimeError("engine stream must contain exactly one final terminal event")
        response = EngineInferenceResponse(
            worker_id=self.worker_id,
            request_id=request.request_id,
            deployment_id=request.deployment_id,
            accepted=terminal.event_type == "completed",
            events=tuple(events),
            detail=terminal.detail,
        )
        self._remember(request.request_id, response)
        yield terminal.model_copy(deep=True)

    async def submit(self, request: SubmitEngineRequest) -> EngineInferenceResponse:
        """Compatibility unary facade; canonical callers use ``stream``."""

        async for _event in self.stream(request):
            pass
        response = self._cached(request.request_id)
        if not isinstance(response, EngineInferenceResponse):
            raise RuntimeError("engine stream completed without a cached terminal response")
        return response.model_copy(deep=True)

    async def unload(self, request: UnloadEngineRequest) -> EngineActionResponse:
        if request.worker_id != self.worker_id:
            raise PermissionError("engine unload request targets another worker")
        cached = self._cached(request.request_id)
        if cached is not None:
            if not isinstance(cached, EngineActionResponse):
                raise RuntimeError("engine request ID was reused for another response type")
            return cached.model_copy(deep=True)
        self._verify(request, action="unload")
        async with self._lock:
            value = self._deployments.pop(request.deployment_id, None)
            if value is not None:
                engine, deployment = value
                if deployment.execution_identity != request.plan.execution_identity:
                    self._deployments[request.deployment_id] = value
                    raise RuntimeError("engine unload identity differs from the deployment")
                await engine.unload(deployment)
            response = EngineActionResponse(
                worker_id=self.worker_id,
                request_id=request.request_id,
                deployment_id=request.deployment_id,
                accepted=True,
                detail="engine deployment unloaded" if value is not None else "already unloaded",
            )
            self._remember(request.request_id, response)
            return response.model_copy(deep=True)

    async def close(self) -> None:
        async with self._lock:
            deployments = list(self._deployments.values())
            self._deployments.clear()
        results = await asyncio.gather(
            *(engine.unload(deployment) for engine, deployment in deployments),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise RuntimeError(f"{len(errors)} engine deployment(s) failed to unload")


__all__ = ["PersistentEngineRuntime"]
