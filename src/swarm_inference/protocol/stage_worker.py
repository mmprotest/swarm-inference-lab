"""Typed control-plane messages for the persistent stage worker.

The runtime transports these records in protobuf ``Any`` envelopes through the
existing generic gRPC service.  Tensor payloads never use these messages; they
remain on the direct binary stage-ring data plane.
"""

from __future__ import annotations

import hmac
import time
from typing import Any

from pydantic import Field, NonNegativeInt, PositiveInt, model_validator

from swarm_inference.cluster.models import ArtifactChunk, ArtifactManifest
from swarm_inference.config.models import StrictModel, WorkerCapability
from swarm_inference.exceptions import IntegrityError
from swarm_inference.model.partition import StageAssignment
from swarm_inference.protocol.expert import SignedExpertRouteLease
from swarm_inference.protocol.routes import SignedRouteLease
from swarm_inference.security.identity import WorkerIdentity, public_key_fingerprint
from swarm_inference.security.signatures import canonical_json_bytes, verify_signature


class _StageControlModel(StrictModel):
    @model_validator(mode="after")
    def _reject_empty_strings(self) -> _StageControlModel:
        for name, value in self.__dict__.items():
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"stage control field {name!r} cannot be empty")
        return self


class StageRouteEndpoint(_StageControlModel):
    """Identity and advertised address of one adjacent stage."""

    worker_id: str
    stage_id: NonNegativeInt
    data_endpoint: str
    assignment: StageAssignment | None = None


class GetStageCapabilitiesRequest(_StageControlModel):
    worker_id: str
    request_id: str
    deadline_unix_ns: PositiveInt | None = None


class GetStageCapabilitiesResponse(_StageControlModel):
    worker_id: str
    request_id: str
    capability: WorkerCapability


class ArtifactTransferLease(_StageControlModel):
    """Finite coordinator authorization for one exact worker artifact."""

    artifact_id: str
    destination_worker_id: str
    source_node_id: str
    issued_at_unix_ns: PositiveInt
    expires_at_unix_ns: PositiveInt
    nonce: str
    coordinator_identity: str
    coordinator_public_key: str
    coordinator_fingerprint: str
    signature: str | None = None

    @model_validator(mode="after")
    def validate_lease(self) -> ArtifactTransferLease:
        if self.expires_at_unix_ns <= self.issued_at_unix_ns:
            raise ValueError("artifact transfer lease expiry must follow issue time")
        actual = public_key_fingerprint(self.coordinator_public_key)
        if not hmac.compare_digest(actual, self.coordinator_fingerprint):
            raise ValueError("artifact transfer coordinator fingerprint mismatch")
        return self


def _artifact_lease_payload(lease: ArtifactTransferLease) -> bytes:
    return canonical_json_bytes(lease.model_dump(mode="json", exclude={"signature"}))


def sign_artifact_transfer_lease(
    lease: ArtifactTransferLease,
    identity: WorkerIdentity,
) -> ArtifactTransferLease:
    if lease.coordinator_public_key != identity.public_key_b64 or not hmac.compare_digest(
        lease.coordinator_fingerprint,
        identity.public_key_fingerprint,
    ):
        raise IntegrityError("artifact transfer signer does not match its coordinator identity")
    return lease.model_copy(update={"signature": identity.sign(_artifact_lease_payload(lease))})


def verify_artifact_transfer_lease(
    lease: ArtifactTransferLease,
    *,
    trusted_coordinator_public_key: str,
    trusted_coordinator_fingerprint: str,
    destination_worker_id: str,
    artifact_id: str,
    now_unix_ns: int | None = None,
) -> None:
    if not lease.signature:
        raise IntegrityError("artifact transfer lease signature is missing")
    if lease.destination_worker_id != destination_worker_id:
        raise IntegrityError("artifact transfer lease targets another worker")
    if lease.artifact_id != artifact_id:
        raise IntegrityError("artifact transfer lease targets another artifact")
    if lease.coordinator_public_key != trusted_coordinator_public_key or not hmac.compare_digest(
        lease.coordinator_fingerprint,
        trusted_coordinator_fingerprint,
    ):
        raise IntegrityError("artifact transfer lease coordinator is not pinned")
    now = time.time_ns() if now_unix_ns is None else now_unix_ns
    if lease.expires_at_unix_ns <= now:
        raise IntegrityError("artifact transfer lease has expired")
    if lease.issued_at_unix_ns > now + 30_000_000_000:
        raise IntegrityError("artifact transfer lease is future-dated")
    verify_signature(
        trusted_coordinator_public_key,
        _artifact_lease_payload(lease),
        lease.signature,
    )


class PrepareArtifactRequest(_StageControlModel):
    worker_id: str
    request_id: str
    manifest: ArtifactManifest
    chunks_total: NonNegativeInt
    lease: ArtifactTransferLease


class WriteArtifactChunkRequest(_StageControlModel):
    worker_id: str
    request_id: str
    transfer_id: str
    chunk: ArtifactChunk
    payload: bytes
    lease: ArtifactTransferLease


class CompleteArtifactRequest(_StageControlModel):
    worker_id: str
    request_id: str
    transfer_id: str
    manifest: ArtifactManifest
    lease: ArtifactTransferLease


class VerifyArtifactRequest(_StageControlModel):
    worker_id: str
    request_id: str
    artifact_id: str
    lease: ArtifactTransferLease


class ArtifactTransferResponse(_StageControlModel):
    worker_id: str
    request_id: str
    artifact_id: str
    accepted: bool
    transfer_id: str | None = None
    bytes_completed: NonNegativeInt = 0
    chunks_completed: NonNegativeInt = 0
    complete: bool = False
    verified: bool = False
    detail: str = ""


class LoadStageRequest(_StageControlModel):
    worker_id: str
    request_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    topology_id: str
    route_generation: NonNegativeInt = 0
    stage_count: PositiveInt = 1
    assignment: StageAssignment
    device: str
    dtype: str
    artifact_id: str | None = None
    model_path: str | None = Field(default=None, json_schema_extra={"deprecated": True})
    allow_download: bool = False
    lease_expiry_unix_ns: PositiveInt | None = None
    deadline_unix_ns: PositiveInt | None = None
    expert_plan: dict[str, Any] | None = None
    expert_model_fingerprint: str | None = None
    expert_quantization_fingerprint: str | None = None

    @model_validator(mode="after")
    def validate_model_source(self) -> LoadStageRequest:
        if self.artifact_id is not None and self.model_path is not None:
            raise ValueError("artifact_id and the deprecated model_path override are exclusive")
        return self


class UnloadStageRequest(_StageControlModel):
    worker_id: str
    request_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    topology_id: str
    route_generation: NonNegativeInt
    stage_count: PositiveInt = 1
    assignment: StageAssignment
    device: str
    dtype: str
    force: bool = False
    deadline_unix_ns: PositiveInt | None = None


class InstallStageRouteRequest(_StageControlModel):
    worker_id: str
    request_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    topology_id: str
    route_generation: PositiveInt
    assignment: StageAssignment
    device: str
    dtype: str
    previous_stage: StageRouteEndpoint | None
    next_stage: StageRouteEndpoint | None
    stage_count: PositiveInt
    stage_zero_publication_destination: str | None = None
    lease_expiry_unix_ns: PositiveInt
    deadline_unix_ns: PositiveInt | None = None
    replace: bool = False
    route_lease: SignedRouteLease | None = None
    expert_route_lease: SignedExpertRouteLease | None = None


class RemoveStageRouteRequest(_StageControlModel):
    worker_id: str
    request_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    topology_id: str
    route_generation: PositiveInt
    stage_id: NonNegativeInt
    device: str
    dtype: str
    deadline_unix_ns: PositiveInt | None = None


class VerifyStageRouteRequest(_StageControlModel):
    worker_id: str
    request_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    topology_id: str
    route_generation: PositiveInt
    stage_id: NonNegativeInt
    device: str
    dtype: str
    deadline_unix_ns: PositiveInt | None = None


class OpenStageSessionRequest(_StageControlModel):
    worker_id: str
    request_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    topology_id: str
    route_generation: PositiveInt
    stage_id: NonNegativeInt
    device: str
    dtype: str
    session_id: str
    request_generation: PositiveInt = 1
    lease_expiry_unix_ns: PositiveInt | None = None
    deadline_unix_ns: PositiveInt | None = None


class CloseStageSessionRequest(OpenStageSessionRequest):
    pass


class CancelStageSessionRequest(OpenStageSessionRequest):
    pass


class GetStageStatusRequest(_StageControlModel):
    worker_id: str
    request_id: str
    topology_id: str | None = None
    deadline_unix_ns: PositiveInt | None = None


class TokenizeStageRequest(_StageControlModel):
    worker_id: str
    request_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    topology_id: str
    route_generation: PositiveInt
    stage_id: NonNegativeInt = 0
    device: str
    dtype: str
    text: str
    add_special_tokens: bool = True
    deadline_unix_ns: PositiveInt | None = None


class TokenizeStageResponse(_StageControlModel):
    worker_id: str
    request_id: str
    token_ids: list[NonNegativeInt]


class DrainWorkerRequest(_StageControlModel):
    worker_id: str
    request_id: str
    deadline_unix_ns: PositiveInt | None = None
    cancel_active_sessions: bool = False


class StageActionResponse(_StageControlModel):
    worker_id: str
    request_id: str
    accepted: bool
    detail: str
    idempotent: bool = False
    released_kv_bytes: NonNegativeInt = 0


class StageSessionStatus(_StageControlModel):
    topology_id: str
    session_id: str
    model_revision: str
    route_generation: PositiveInt
    request_generation: PositiveInt = 1
    stage_id: NonNegativeInt
    cache_position: NonNegativeInt
    kv_cache_bytes: NonNegativeInt
    opened_monotonic_ns: PositiveInt
    last_operation_monotonic_ns: PositiveInt
    cancelled: bool = False


class LoadedStageStatus(_StageControlModel):
    model_id: str
    model_revision: str
    tokenizer_revision: str
    topology_id: str
    assignment: StageAssignment
    device: str
    dtype: str
    model_path: str
    artifact_id: str | None = None
    ownership: dict[str, Any]
    loaded_monotonic_ns: PositiveInt
    load_count: PositiveInt
    process_rss_before_bytes: NonNegativeInt
    process_rss_after_bytes: NonNegativeInt
    cuda_allocated_before_bytes: NonNegativeInt
    cuda_allocated_after_bytes: NonNegativeInt
    cuda_reserved_before_bytes: NonNegativeInt
    cuda_reserved_after_bytes: NonNegativeInt


class InstalledStageRouteStatus(_StageControlModel):
    topology_id: str
    route_generation: PositiveInt
    previous_stage: StageRouteEndpoint | None
    next_stage: StageRouteEndpoint | None
    stage_count: PositiveInt
    stage_zero_publication_destination: str | None
    lease_expiry_unix_ns: PositiveInt
    authenticated: bool = False
    route_lease_hash: str | None = None


class StageStatusResponse(_StageControlModel):
    worker_id: str
    request_id: str
    process_id: PositiveInt
    draining: bool
    loaded_stage: LoadedStageStatus | None
    installed_route: InstalledStageRouteStatus | None
    sessions: list[StageSessionStatus] = Field(default_factory=list)
    execution_queue_depth: NonNegativeInt
    execution_queue_capacity: PositiveInt
    token_queue_depth: NonNegativeInt
    token_queue_capacity: PositiveInt
    dropped_token_publications: NonNegativeInt
    expert_status: dict[str, Any] = Field(default_factory=dict)


# Concise aliases mirror the operation names used by the documented RPC API.
GetStageCapabilities = GetStageCapabilitiesRequest
LoadStage = LoadStageRequest
UnloadStage = UnloadStageRequest
InstallStageRoute = InstallStageRouteRequest
RemoveStageRoute = RemoveStageRouteRequest
VerifyStageRoute = VerifyStageRouteRequest
OpenStageSession = OpenStageSessionRequest
CloseStageSession = CloseStageSessionRequest
CancelStageSession = CancelStageSessionRequest
GetStageStatus = GetStageStatusRequest
TokenizeStage = TokenizeStageRequest
DrainWorker = DrainWorkerRequest


__all__ = [
    "ArtifactTransferLease",
    "ArtifactTransferResponse",
    "CancelStageSession",
    "CancelStageSessionRequest",
    "CloseStageSession",
    "CloseStageSessionRequest",
    "CompleteArtifactRequest",
    "DrainWorker",
    "DrainWorkerRequest",
    "GetStageCapabilities",
    "GetStageCapabilitiesRequest",
    "GetStageCapabilitiesResponse",
    "GetStageStatus",
    "GetStageStatusRequest",
    "InstallStageRoute",
    "InstallStageRouteRequest",
    "InstalledStageRouteStatus",
    "LoadStage",
    "LoadStageRequest",
    "LoadedStageStatus",
    "OpenStageSession",
    "OpenStageSessionRequest",
    "PrepareArtifactRequest",
    "RemoveStageRoute",
    "RemoveStageRouteRequest",
    "StageActionResponse",
    "StageRouteEndpoint",
    "StageSessionStatus",
    "StageStatusResponse",
    "TokenizeStage",
    "TokenizeStageRequest",
    "TokenizeStageResponse",
    "UnloadStage",
    "UnloadStageRequest",
    "VerifyArtifactRequest",
    "VerifyStageRoute",
    "VerifyStageRouteRequest",
    "WriteArtifactChunkRequest",
    "sign_artifact_transfer_lease",
    "verify_artifact_transfer_lease",
]
