from __future__ import annotations

import time
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest

from swarm_inference.engines.interfaces import (
    Deployment,
    EngineSupportReport,
    EngineSupportStatus,
    ExecutionPlan,
    InferenceEvent,
    InferenceRequest,
    PhasePlan,
)
from swarm_inference.exceptions import IntegrityError
from swarm_inference.protocol.engine_worker import (
    EngineDeploymentLease,
    PrepareEngineRequest,
    SubmitEngineRequest,
    UnloadEngineRequest,
    execution_plan_hash,
    sign_engine_deployment_lease,
)
from swarm_inference.security.identity import CoordinatorIdentity
from swarm_inference.worker.engine_runtime import PersistentEngineRuntime


def _plan() -> ExecutionPlan:
    roles = {"worker-a": "critical_path_stage"}
    return ExecutionPlan(
        plan_id="plan-a",
        engine_id="fake-engine",
        model_fingerprint="sha256:" + "1" * 64,
        execution_identity="sha256:" + "2" * 64,
        objective="speed",
        topology="local-complete-model",
        worker_roles=roles,
        prefill_plan=PhasePlan(phase="prefill", worker_roles=roles),
        decode_plan=PhasePlan(phase="decode", worker_roles=roles),
        predicted_ttft_ms=1,
        predicted_decode_tokens_s=10,
        predicted_aggregate_tokens_s=10,
        score=10,
    )


def _lease(
    identity: CoordinatorIdentity,
    plan: ExecutionPlan,
    *,
    action: str,
    deployment_id: str = "deployment-a",
    nonce: str | None = None,
) -> EngineDeploymentLease:
    now = time.time_ns()
    unsigned = EngineDeploymentLease(
        action=action,
        worker_id="worker-a",
        deployment_id=deployment_id,
        engine_id=plan.engine_id,
        execution_identity=plan.execution_identity,
        plan_hash=execution_plan_hash(plan),
        issued_at_unix_ns=now,
        expires_at_unix_ns=now + 60_000_000_000,
        nonce=nonce or uuid4().hex,
        coordinator_identity="coordinator",
        coordinator_public_key=identity.public_key_b64,
        coordinator_fingerprint=identity.public_key_fingerprint,
    )
    return sign_engine_deployment_lease(unsigned, identity)


class _FakeEngine:
    engine_id = "fake-engine"

    def __init__(self) -> None:
        self.prepared = 0
        self.unloaded = 0

    def probe(self, *_args, **_kwargs) -> EngineSupportReport:
        return EngineSupportReport(
            engine_id=self.engine_id,
            status=EngineSupportStatus.SUPPORTED,
            reason="test",
        )

    async def candidate_plans(self, *_args, **_kwargs) -> list[ExecutionPlan]:
        return [_plan()]

    async def prepare(self, plan: ExecutionPlan) -> Deployment:
        self.prepared += 1
        return Deployment(
            deployment_id="engine-private-id",
            engine_id=self.engine_id,
            execution_identity=plan.execution_identity,
            plan=plan,
            ready=True,
            process_ids={"worker-a": 123},
        )

    async def submit(
        self, deployment: Deployment, request: InferenceRequest
    ) -> AsyncIterator[InferenceEvent]:
        assert deployment.deployment_id == "engine-private-id"
        yield InferenceEvent(event_type="started", request_id=request.request_id, sequence_number=0)
        yield InferenceEvent(
            event_type="token",
            request_id=request.request_id,
            sequence_number=1,
            token_id=7,
            text="x",
        )
        yield InferenceEvent(
            event_type="completed", request_id=request.request_id, sequence_number=2
        )

    async def unload(self, deployment: Deployment) -> None:
        assert deployment.deployment_id == "engine-private-id"
        self.unloaded += 1


@pytest.mark.asyncio
async def test_engine_runtime_requires_signed_plan_bound_actions_and_is_idempotent() -> None:
    identity = CoordinatorIdentity.generate()
    engine = _FakeEngine()
    runtime = PersistentEngineRuntime(worker_id="worker-a", engines=(engine,))
    plan = _plan()
    prepare = PrepareEngineRequest(
        worker_id="worker-a",
        request_id="prepare-a",
        deployment_id="deployment-a",
        assigned_role="critical_path_stage",
        plan=plan,
        lease=_lease(identity, plan, action="prepare"),
    )
    with pytest.raises(PermissionError, match="trust is not configured"):
        await runtime.prepare(prepare)

    runtime.configure_trust(
        coordinator_public_key=identity.public_key_b64,
        expected_fingerprint=identity.public_key_fingerprint,
    )
    prepared = await runtime.prepare(prepare)
    repeated = await runtime.prepare(prepare)
    assert prepared == repeated
    assert prepared.deployment is not None
    assert prepared.deployment.deployment_id == "deployment-a"
    assert engine.prepared == 1

    inference = InferenceRequest(request_id="inference-a", prompt="hello")
    submitted = await runtime.submit(
        SubmitEngineRequest(
            worker_id="worker-a",
            request_id="submit-a",
            deployment_id="deployment-a",
            plan=plan,
            inference=inference,
            lease=_lease(identity, plan, action="submit"),
        )
    )
    assert submitted.accepted
    assert [item.event_type for item in submitted.events] == [
        "started",
        "token",
        "completed",
    ]

    unloaded = await runtime.unload(
        UnloadEngineRequest(
            worker_id="worker-a",
            request_id="unload-a",
            deployment_id="deployment-a",
            plan=plan,
            lease=_lease(identity, plan, action="unload"),
        )
    )
    assert unloaded.accepted
    assert runtime.deployment_ids == ()
    assert engine.unloaded == 1


@pytest.mark.asyncio
async def test_engine_runtime_rejects_plan_tampering_and_signed_nonce_replay() -> None:
    identity = CoordinatorIdentity.generate()
    runtime = PersistentEngineRuntime(worker_id="worker-a", engines=(_FakeEngine(),))
    runtime.configure_trust(
        coordinator_public_key=identity.public_key_b64,
        expected_fingerprint=identity.public_key_fingerprint,
    )
    plan = _plan()
    nonce = uuid4().hex
    lease = _lease(identity, plan, action="prepare", nonce=nonce)
    tampered = plan.model_copy(update={"score": 999.0})
    with pytest.raises(IntegrityError, match="does not authorize"):
        await runtime.prepare(
            PrepareEngineRequest(
                worker_id="worker-a",
                request_id="tampered",
                deployment_id="deployment-a",
                assigned_role="critical_path_stage",
                plan=tampered,
                lease=lease,
            )
        )

    request = PrepareEngineRequest(
        worker_id="worker-a",
        request_id="valid",
        deployment_id="deployment-a",
        assigned_role="critical_path_stage",
        plan=plan,
        lease=lease,
    )
    await runtime.prepare(request)
    with pytest.raises(IntegrityError, match="replayed"):
        await runtime.prepare(request.model_copy(update={"request_id": "replay"}))
    await runtime.close()
