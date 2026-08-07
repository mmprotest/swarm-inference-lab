"""Authenticated control messages for worker-owned execution engines."""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Literal

from pydantic import ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator

from swarm_inference.config.models import StrictModel
from swarm_inference.engines.interfaces import (
    Deployment,
    ExecutionPlan,
    InferenceEvent,
    InferenceRequest,
)
from swarm_inference.exceptions import IntegrityError
from swarm_inference.security.identity import CoordinatorIdentity, public_key_fingerprint
from swarm_inference.security.signatures import canonical_json_bytes, verify_signature

EngineControlAction = Literal["prepare", "submit", "unload"]


def execution_plan_hash(plan: ExecutionPlan) -> str:
    payload = canonical_json_bytes(plan.model_dump(mode="json"))
    return "sha256:" + hashlib.sha256(payload).hexdigest()


class EngineDeploymentLease(StrictModel):
    """Finite coordinator authorization for one engine-control action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: EngineControlAction
    worker_id: str
    deployment_id: str
    engine_id: str
    execution_identity: str
    plan_hash: str
    issued_at_unix_ns: PositiveInt
    expires_at_unix_ns: PositiveInt
    nonce: str
    coordinator_identity: str
    coordinator_public_key: str
    coordinator_fingerprint: str
    signature: str = ""

    @model_validator(mode="after")
    def validate_identity(self) -> EngineDeploymentLease:
        if self.expires_at_unix_ns <= self.issued_at_unix_ns:
            raise ValueError("engine deployment lease expiry must follow issue time")
        if not self.plan_hash.startswith("sha256:") or len(self.plan_hash) != 71:
            raise ValueError("engine deployment lease requires a SHA-256 plan hash")
        actual = public_key_fingerprint(self.coordinator_public_key)
        if not hmac.compare_digest(actual, self.coordinator_fingerprint):
            raise ValueError("engine deployment coordinator fingerprint mismatch")
        return self


def _lease_payload(lease: EngineDeploymentLease) -> bytes:
    return canonical_json_bytes(lease.model_dump(mode="json", exclude={"signature"}))


def sign_engine_deployment_lease(
    lease: EngineDeploymentLease,
    identity: CoordinatorIdentity,
) -> EngineDeploymentLease:
    if lease.coordinator_public_key != identity.public_key_b64 or not hmac.compare_digest(
        lease.coordinator_fingerprint,
        identity.public_key_fingerprint,
    ):
        raise IntegrityError("engine deployment signer differs from its lease identity")
    return lease.model_copy(update={"signature": identity.sign(_lease_payload(lease))})


def verify_engine_deployment_lease(
    lease: EngineDeploymentLease,
    *,
    action: EngineControlAction,
    worker_id: str,
    deployment_id: str,
    engine_id: str,
    execution_identity: str,
    plan: ExecutionPlan,
    trusted_coordinator_public_key: str,
    trusted_coordinator_fingerprint: str,
    now_unix_ns: int | None = None,
) -> None:
    if not lease.signature:
        raise IntegrityError("engine deployment lease signature is missing")
    expected = (
        action,
        worker_id,
        deployment_id,
        engine_id,
        execution_identity,
        execution_plan_hash(plan),
    )
    actual = (
        lease.action,
        lease.worker_id,
        lease.deployment_id,
        lease.engine_id,
        lease.execution_identity,
        lease.plan_hash,
    )
    if actual != expected:
        raise IntegrityError("engine deployment lease does not authorize this action")
    if lease.coordinator_public_key != trusted_coordinator_public_key or not hmac.compare_digest(
        lease.coordinator_fingerprint,
        trusted_coordinator_fingerprint,
    ):
        raise IntegrityError("engine deployment coordinator is not pinned")
    now = time.time_ns() if now_unix_ns is None else now_unix_ns
    if lease.expires_at_unix_ns <= now:
        raise IntegrityError("engine deployment lease has expired")
    if lease.issued_at_unix_ns > now + 30_000_000_000:
        raise IntegrityError("engine deployment lease is future-dated")
    verify_signature(
        trusted_coordinator_public_key,
        _lease_payload(lease),
        lease.signature,
    )


class PrepareEngineRequest(StrictModel):
    worker_id: str
    request_id: str
    deployment_id: str
    assigned_role: str
    plan: ExecutionPlan
    artifact_ids: tuple[str, ...] = ()
    lease: EngineDeploymentLease


class SubmitEngineRequest(StrictModel):
    worker_id: str
    request_id: str
    deployment_id: str
    plan: ExecutionPlan
    inference: InferenceRequest
    lease: EngineDeploymentLease


class UnloadEngineRequest(StrictModel):
    worker_id: str
    request_id: str
    deployment_id: str
    plan: ExecutionPlan
    force: bool = False
    lease: EngineDeploymentLease


class EngineActionResponse(StrictModel):
    worker_id: str
    request_id: str
    deployment_id: str
    accepted: bool
    detail: str = ""
    deployment: Deployment | None = None
    process_ids: dict[str, int] = Field(default_factory=dict)
    endpoints: dict[str, str] = Field(default_factory=dict)


class EngineInferenceResponse(StrictModel):
    worker_id: str
    request_id: str
    deployment_id: str
    accepted: bool
    events: tuple[InferenceEvent, ...] = ()
    detail: str = ""
    network_bytes: NonNegativeInt = 0
    completed_at_unix_ns: PositiveInt = Field(default_factory=time.time_ns)


class EngineInferenceChunk(StrictModel):
    """One authenticated deployment event on the worker-to-client stream."""

    worker_id: str
    request_id: str
    deployment_id: str
    event: InferenceEvent
    network_bytes: NonNegativeInt = 0


__all__ = [
    "EngineActionResponse",
    "EngineControlAction",
    "EngineDeploymentLease",
    "EngineInferenceChunk",
    "EngineInferenceResponse",
    "PrepareEngineRequest",
    "SubmitEngineRequest",
    "UnloadEngineRequest",
    "execution_plan_hash",
    "sign_engine_deployment_lease",
    "verify_engine_deployment_lease",
]
