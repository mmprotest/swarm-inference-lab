"""Versioned control messages for pairing and authenticated cluster operations."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, NonNegativeInt, PositiveInt, field_validator, model_validator

from swarm_inference.cluster.models import (
    ArtifactCacheEntry,
    ArtifactLease,
    ArtifactManifest,
    ArtifactTransferStatus,
    ClusterMetadata,
    NetworkLinkMeasurement,
    NodeMembership,
    NodeMetadata,
    NodeRevocation,
    NodeRuntimeMetadata,
    VersionCompatibility,
)
from swarm_inference.config.models import StrictModel
from swarm_inference.security.identity import public_key_fingerprint
from swarm_inference.security.trust_store import normalize_fingerprint

CLUSTER_RPC_SCHEMA_VERSION: Literal[1] = 1


def _base64(value: str, *, field: str, decoded_length: int | None = None) -> str:
    try:
        raw = base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise ValueError(f"{field} must be canonical base64") from exc
    if decoded_length is not None and len(raw) != decoded_length:
        raise ValueError(f"{field} must decode to {decoded_length} bytes")
    if base64.b64encode(raw).decode("ascii") != value:
        raise ValueError(f"{field} must use canonical padded base64")
    return value


class PairingHello(StrictModel):
    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    session_id: str
    node_id: str
    node_public_key: str
    node_fingerprint: str
    node_ephemeral_public_key: str
    client_nonce: str
    agent_version: str
    runtime_version: str
    build_id: str
    product_protocol_major: PositiveInt
    product_protocol_minor: NonNegativeInt
    artifact_format_versions: list[PositiveInt]

    @field_validator("node_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        return normalize_fingerprint(value)

    @field_validator("node_ephemeral_public_key")
    @classmethod
    def validate_ephemeral_key(cls, value: str) -> str:
        return _base64(value, field="node_ephemeral_public_key", decoded_length=32)

    @field_validator("client_nonce")
    @classmethod
    def validate_client_nonce(cls, value: str) -> str:
        return _base64(value, field="client_nonce", decoded_length=16)


class PairingChallenge(StrictModel):
    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    session_id: str
    coordinator_public_key: str
    coordinator_fingerprint: str
    coordinator_ephemeral_public_key: str
    server_nonce: str
    expires_at_unix_ns: PositiveInt
    encryption_nonce: str
    encrypted_payload: str

    @field_validator("coordinator_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        return normalize_fingerprint(value)

    @field_validator("coordinator_ephemeral_public_key")
    @classmethod
    def validate_ephemeral_key(cls, value: str) -> str:
        return _base64(value, field="coordinator_ephemeral_public_key", decoded_length=32)

    @field_validator("server_nonce")
    @classmethod
    def validate_server_nonce(cls, value: str) -> str:
        return _base64(value, field="server_nonce", decoded_length=16)

    @field_validator("encryption_nonce")
    @classmethod
    def validate_encryption_nonce(cls, value: str) -> str:
        return _base64(value, field="encryption_nonce", decoded_length=12)

    @field_validator("encrypted_payload")
    @classmethod
    def validate_encrypted_payload(cls, value: str) -> str:
        return _base64(value, field="encrypted_payload")


class PairingCompleteRequest(StrictModel):
    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    session_id: str
    hello_hash: str
    encryption_nonce: str
    encrypted_payload: str

    @field_validator("encryption_nonce")
    @classmethod
    def validate_encryption_nonce(cls, value: str) -> str:
        return _base64(value, field="encryption_nonce", decoded_length=12)

    @field_validator("encrypted_payload")
    @classmethod
    def validate_encrypted_payload(cls, value: str) -> str:
        return _base64(value, field="encrypted_payload")


class PairingCompleteResponse(StrictModel):
    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    session_id: str
    consumed: bool
    encryption_nonce: str
    encrypted_payload: str

    @field_validator("encryption_nonce")
    @classmethod
    def validate_encryption_nonce(cls, value: str) -> str:
        return _base64(value, field="encryption_nonce", decoded_length=12)

    @field_validator("encrypted_payload")
    @classmethod
    def validate_encrypted_payload(cls, value: str) -> str:
        return _base64(value, field="encrypted_payload")


class PairingChallengePayload(StrictModel):
    """Decrypted only in memory; never written to status, audit, or logs."""

    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    transcript_hash: str
    challenge: str
    coordinator_secret_proof: str
    coordinator_signature: str


class PairingNodeCompletionPayload(StrictModel):
    """Encrypted node proof and metadata sent during completion."""

    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    transcript_hash: str
    challenge: str
    node_secret_proof: str
    node_signature: str
    node_metadata: NodeMetadata


class PairingCoordinatorCompletionPayload(StrictModel):
    """Encrypted final coordinator proof and pinned membership records."""

    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    transcript_hash: str
    coordinator_secret_proof: str
    coordinator_signature: str
    cluster: ClusterMetadata
    membership: NodeMembership


class PairingResult(StrictModel):
    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    cluster: ClusterMetadata
    membership: NodeMembership


class ClusterRequestAuthentication(StrictModel):
    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    node_id: str
    timestamp_unix_ns: PositiveInt
    nonce: str
    signature: str


class PairingCreateRequest(StrictModel):
    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    authentication: ClusterRequestAuthentication
    ttl_seconds: PositiveInt = Field(default=600, le=3600)


class PairingCreateResponse(StrictModel):
    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    session_id: str
    pairing_uri: str = Field(repr=False)
    redacted_uri: str
    expires_at_unix_ns: PositiveInt


class PairingDeliveryResult(StrictModel):
    """Public delivery receipt; the secret-bearing URI is intentionally absent."""

    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    status: Literal["ready"] = "ready"
    session_id: str
    expires_at_unix_ns: PositiveInt
    redacted_uri: str
    delivery: Literal["interactive", "protected-file"]
    invitation_file: Path | None = None
    permission_protection: (
        Literal["posix-0600", "windows-user-acl", "user-scoped-best-effort"] | None
    ) = None
    permission_limitation: str | None = None

    @model_validator(mode="after")
    def validate_delivery(self) -> Self:
        if self.delivery == "protected-file" and self.invitation_file is None:
            raise ValueError("protected-file pairing delivery requires an invitation path")
        if self.delivery == "protected-file" and self.permission_protection is None:
            raise ValueError("protected-file pairing delivery requires a protection result")
        if self.delivery == "interactive" and self.invitation_file is not None:
            raise ValueError("interactive pairing delivery cannot claim an invitation file")
        return self


class ClusterStatusRequest(StrictModel):
    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    authentication: ClusterRequestAuthentication
    include_artifacts: bool = True
    include_network: bool = True


class ClusterNodeStatus(StrictModel):
    metadata: NodeMetadata
    runtime: NodeRuntimeMetadata | None = None
    inclusion_reason: str | None = None
    exclusion_reason: str | None = None


class ClusterStatusResponse(StrictModel):
    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    cluster: ClusterMetadata
    nodes: list[ClusterNodeStatus]
    network_links: list[NetworkLinkMeasurement] = Field(default_factory=list)
    artifact_entries: list[ArtifactCacheEntry] = Field(default_factory=list)
    artifact_transfers: list[ArtifactTransferStatus] = Field(default_factory=list)
    revocations: list[NodeRevocation] = Field(default_factory=list)
    generated_at_unix_ns: PositiveInt


class ClusterRevokeRequest(StrictModel):
    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    authentication: ClusterRequestAuthentication
    node_id: str
    reason: str


class ClusterRevokeResponse(StrictModel):
    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    revoked: bool
    revocation: NodeRevocation | None = None


class NodeLeaveRequest(StrictModel):
    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    authentication: ClusterRequestAuthentication
    node_id: str


class NodeLeaveResponse(StrictModel):
    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    left: bool
    node_id: str


class NodeUpdateRequest(StrictModel):
    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    authentication: ClusterRequestAuthentication
    metadata: NodeMetadata
    runtime: NodeRuntimeMetadata | None = None


class NodeUpdateResponse(StrictModel):
    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    accepted: bool
    node_id: str
    endpoint_changed: bool


class ReachabilityCheckRequest(StrictModel):
    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    authentication: ClusterRequestAuthentication
    node_id: str
    timeout_ms: PositiveInt = 3000


class ReachabilityCheckResponse(StrictModel):
    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    node_id: str
    control_endpoint: str
    data_endpoint: str
    control_reachable: bool
    data_reachable: bool
    probe_endpoint: str | None = None
    probe_reachable: bool | None = None
    coordinator_source_address: str | None = None
    detail: str


class NetworkProbeTicket(StrictModel):
    """Coordinator-signed, short-lived authorization for one directed probe."""

    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    ticket_id: str
    cluster_id: str
    source_node_id: str
    source_worker_id: str
    source_public_key: str
    source_fingerprint: str
    source_endpoint: str | None = None
    destination_node_id: str
    destination_worker_id: str
    destination_public_key: str
    destination_fingerprint: str
    destination_endpoint: str
    destination_interface: str | None = None
    issued_at_unix_ns: PositiveInt
    expires_at_unix_ns: PositiveInt
    payload_sizes: list[PositiveInt]
    sample_count: PositiveInt
    maximum_bytes: PositiveInt
    coordinator_public_key: str
    coordinator_fingerprint: str
    signature: str

    @field_validator(
        "source_fingerprint",
        "destination_fingerprint",
        "coordinator_fingerprint",
    )
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        return normalize_fingerprint(value)

    @model_validator(mode="after")
    def validate_ticket(self) -> Self:
        if self.expires_at_unix_ns <= self.issued_at_unix_ns:
            raise ValueError("network probe ticket expiry must follow issuance")
        if not self.payload_sizes or self.payload_sizes != sorted(set(self.payload_sizes)):
            raise ValueError("network probe payload sizes must be non-empty, unique, and sorted")
        wire_bytes = 2 * sum(self.payload_sizes) * self.sample_count
        if wire_bytes > self.maximum_bytes:
            raise ValueError("network probe ticket exceeds its byte bound")
        for public_key, fingerprint, role in (
            (self.source_public_key, self.source_fingerprint, "source"),
            (self.destination_public_key, self.destination_fingerprint, "destination"),
            (self.coordinator_public_key, self.coordinator_fingerprint, "coordinator"),
        ):
            if public_key_fingerprint(public_key) != fingerprint:
                raise ValueError(f"network probe {role} identity binding is invalid")
        return self


class DirectNetworkProbeRequest(StrictModel):
    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    ticket: NetworkProbeTicket
    timestamp_unix_ns: PositiveInt
    nonce: str
    sample_index: NonNegativeInt
    payload_size: PositiveInt
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_seed: str
    signature: str


class DirectNetworkProbeAck(StrictModel):
    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    ticket_id: str
    request_nonce: str
    destination_node_id: str
    destination_worker_id: str
    received_at_unix_ns: PositiveInt
    signature: str


class DirectNetworkProbeResponse(StrictModel):
    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    ticket_id: str
    request_nonce: str
    destination_node_id: str
    destination_worker_id: str
    payload_size: PositiveInt
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sent_at_unix_ns: PositiveInt
    signature: str


class NetworkProbeControlRequest(StrictModel):
    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    authentication: ClusterRequestAuthentication
    operation: Literal["issue", "record"] = "issue"
    source_worker_id: str
    destination_worker_id: str
    payload_sizes: list[PositiveInt] = Field(default_factory=list)
    sample_count: PositiveInt = 3
    maximum_bytes: PositiveInt = 16 * 1024 * 1024
    timeout_ms: PositiveInt = 10_000
    measurement: NetworkLinkMeasurement | None = None


class NetworkProbeControlResponse(StrictModel):
    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    accepted: bool
    ticket: NetworkProbeTicket | None = None
    measurement: NetworkLinkMeasurement | None = None
    detail: str | None = None


class ArtifactOperationRequest(StrictModel):
    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    authentication: ClusterRequestAuthentication
    operation: Literal["locate", "prepare", "transfer", "status", "lease", "release"]
    artifact_id: str
    manifest: ArtifactManifest | None = None
    source_node_id: str | None = None
    destination_node_id: str | None = None
    maximum_bytes: PositiveInt | None = None
    chunks_total: NonNegativeInt | None = None
    lease_id: str | None = None
    lease_purpose: Literal["loaded-stage", "deployment", "transfer", "pinned"] = "transfer"
    lease_expires_at_unix_ns: PositiveInt | None = None


class ArtifactOperationResponse(StrictModel):
    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    accepted: bool
    locations: list[str] = Field(default_factory=list)
    transfer: ArtifactTransferStatus | None = None
    lease: ArtifactLease | None = None
    detail: str | None = None


class VersionCompatibilityRequest(StrictModel):
    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    authentication: ClusterRequestAuthentication | None = None
    compatibility: VersionCompatibility


class VersionCompatibilityResponse(StrictModel):
    schema_version: Literal[1] = CLUSTER_RPC_SCHEMA_VERSION
    compatible: bool
    coordinator_compatibility: VersionCompatibility
    reasons: list[str] = Field(default_factory=list)


def verify_hello_identity(hello: PairingHello) -> None:
    from swarm_inference.cluster.models import node_id_from_fingerprint

    fingerprint = public_key_fingerprint(hello.node_public_key)
    if fingerprint != hello.node_fingerprint:
        raise ValueError("pairing hello public key and fingerprint do not match")
    if hello.node_id != node_id_from_fingerprint(fingerprint):
        raise ValueError("pairing hello node ID is not derived from its identity")


__all__ = [
    "CLUSTER_RPC_SCHEMA_VERSION",
    "ArtifactOperationRequest",
    "ArtifactOperationResponse",
    "ClusterNodeStatus",
    "ClusterRequestAuthentication",
    "ClusterRevokeRequest",
    "ClusterRevokeResponse",
    "ClusterStatusRequest",
    "ClusterStatusResponse",
    "DirectNetworkProbeAck",
    "DirectNetworkProbeRequest",
    "DirectNetworkProbeResponse",
    "NetworkProbeControlRequest",
    "NetworkProbeControlResponse",
    "NetworkProbeTicket",
    "NodeLeaveRequest",
    "NodeLeaveResponse",
    "NodeUpdateRequest",
    "NodeUpdateResponse",
    "PairingChallenge",
    "PairingChallengePayload",
    "PairingCompleteRequest",
    "PairingCompleteResponse",
    "PairingCoordinatorCompletionPayload",
    "PairingCreateRequest",
    "PairingCreateResponse",
    "PairingDeliveryResult",
    "PairingHello",
    "PairingNodeCompletionPayload",
    "PairingResult",
    "ReachabilityCheckRequest",
    "ReachabilityCheckResponse",
    "VersionCompatibilityRequest",
    "VersionCompatibilityResponse",
    "verify_hello_identity",
]
