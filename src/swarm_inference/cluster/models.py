"""Strict, versioned durable models for the universal cluster product."""

from __future__ import annotations

import re
from typing import Literal, Self

from pydantic import (
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

from swarm_inference.config.models import Backend, StrictModel
from swarm_inference.security.identity import public_key_fingerprint
from swarm_inference.security.trust_store import normalize_fingerprint

CLUSTER_SCHEMA_VERSION: Literal[1] = 1
CLUSTER_DOCUMENT_VERSION: Literal[1] = 1
PRODUCT_PROTOCOL_MAJOR = 1
PRODUCT_PROTOCOL_MINOR = 0
ARTIFACT_FORMAT_VERSION = 1
TRUSTED_LAN_SECURITY_CLASSIFICATION: Literal[
    "trusted-lan-private-network-unencrypted-data-plane"
] = "trusted-lan-private-network-unencrypted-data-plane"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NODE_ID = re.compile(r"^node-[0-9a-f]{8}$")


def _sha256(value: str, *, field: str) -> str:
    normalized = value.removeprefix("sha256:").lower()
    if not _SHA256.fullmatch(normalized):
        raise ValueError(f"{field} must be a SHA-256 hex digest")
    return normalized


def node_id_from_fingerprint(fingerprint: str) -> str:
    """Derive the stable node namespace from its Ed25519 fingerprint."""

    return f"node-{normalize_fingerprint(fingerprint)[:8]}"


class VersionCompatibility(StrictModel):
    schema_version: Literal[1] = CLUSTER_SCHEMA_VERSION
    document_version: Literal[1] = CLUSTER_DOCUMENT_VERSION
    minimum_runtime_version: str
    maximum_runtime_version_exclusive: str
    product_protocol_major: PositiveInt = PRODUCT_PROTOCOL_MAJOR
    minimum_product_protocol_minor: NonNegativeInt = PRODUCT_PROTOCOL_MINOR
    maximum_product_protocol_minor: NonNegativeInt = PRODUCT_PROTOCOL_MINOR
    artifact_format_versions: list[PositiveInt] = Field(
        default_factory=lambda: [ARTIFACT_FORMAT_VERSION]
    )

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        if self.minimum_product_protocol_minor > self.maximum_product_protocol_minor:
            raise ValueError("product protocol minor compatibility range is inverted")
        if not self.artifact_format_versions:
            raise ValueError("at least one artifact format version is required")
        if self.artifact_format_versions != sorted(set(self.artifact_format_versions)):
            raise ValueError("artifact format versions must be unique and sorted")
        return self


class ClusterMetadata(StrictModel):
    schema_version: Literal[1] = CLUSTER_SCHEMA_VERSION
    document_version: Literal[1] = CLUSTER_DOCUMENT_VERSION
    cluster_id: str
    name: str
    coordinator_id: str
    coordinator_endpoint: str
    coordinator_public_key: str
    coordinator_fingerprint: str
    created_at_unix_ns: PositiveInt
    product_protocol_major: PositiveInt = PRODUCT_PROTOCOL_MAJOR
    product_protocol_minor: NonNegativeInt = PRODUCT_PROTOCOL_MINOR
    runtime_compatibility: VersionCompatibility
    security_classification: Literal["trusted-lan-private-network-unencrypted-data-plane"] = (
        TRUSTED_LAN_SECURITY_CLASSIFICATION
    )

    @field_validator("coordinator_fingerprint")
    @classmethod
    def validate_coordinator_fingerprint(cls, value: str) -> str:
        return normalize_fingerprint(value)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or len(normalized) > 80:
            raise ValueError("cluster name must contain between 1 and 80 characters")
        return normalized

    @field_validator("coordinator_public_key")
    @classmethod
    def validate_coordinator_public_key(cls, value: str, info: object) -> str:
        del info
        public_key_fingerprint(value)
        return value

    def verify_identity_binding(self) -> None:
        if public_key_fingerprint(self.coordinator_public_key) != self.coordinator_fingerprint:
            raise ValueError("cluster coordinator public key and fingerprint do not match")

    @model_validator(mode="after")
    def validate_identity_binding(self) -> Self:
        self.verify_identity_binding()
        return self


NodeValidationStatus = Literal[
    "pending",
    "validated",
    "implemented-unvalidated",
    "unsupported",
    "failed",
]
NodeServiceMode = Literal[
    "foreground",
    "windows-task",
    "systemd-user",
    "launch-agent",
    "unavailable",
]


class NodeMetadata(StrictModel):
    schema_version: Literal[1] = CLUSTER_SCHEMA_VERSION
    document_version: Literal[1] = CLUSTER_DOCUMENT_VERSION
    node_id: str
    public_key: str
    fingerprint: str
    hostname: str
    operating_system: str
    architecture: str
    agent_version: str
    runtime_version: str
    build_id: str
    package_lock_hash: str
    product_protocol_major: PositiveInt = PRODUCT_PROTOCOL_MAJOR
    product_protocol_minor: NonNegativeInt = PRODUCT_PROTOCOL_MINOR
    artifact_format_versions: list[PositiveInt] = Field(
        default_factory=lambda: [ARTIFACT_FORMAT_VERSION]
    )
    selected_backend: Backend | None = None
    selected_device: str | None = None
    worker_ids: list[str] = Field(default_factory=list)
    control_endpoint: str | None = None
    data_endpoint: str | None = None
    probe_endpoint: str | None = None
    joined_at_unix_ns: PositiveInt
    last_seen_at_unix_ns: PositiveInt
    service_mode: NodeServiceMode = "foreground"
    validation_status: NodeValidationStatus = "pending"
    platform_support_status: NodeValidationStatus = "pending"
    revoked: bool = False
    revoked_at_unix_ns: PositiveInt | None = None
    revocation_reason: str | None = None

    @field_validator("fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        return normalize_fingerprint(value)

    @field_validator("package_lock_hash")
    @classmethod
    def validate_package_lock_hash(cls, value: str) -> str:
        return _sha256(value, field="package_lock_hash")

    @field_validator("node_id")
    @classmethod
    def validate_node_id_shape(cls, value: str) -> str:
        if not _NODE_ID.fullmatch(value):
            raise ValueError("node_id must use node-<8 fingerprint hex> format")
        return value

    def verify_identity_binding(self) -> None:
        fingerprint = public_key_fingerprint(self.public_key)
        if fingerprint != self.fingerprint:
            raise ValueError("node public key and fingerprint do not match")
        if self.node_id != node_id_from_fingerprint(fingerprint):
            raise ValueError("node_id is not derived from the node fingerprint")
        invalid_workers = [
            worker_id
            for worker_id in self.worker_ids
            if not worker_id.startswith(f"{self.node_id}/")
        ]
        if invalid_workers:
            raise ValueError("worker IDs must be namespaced by their owning node ID")

    @model_validator(mode="after")
    def validate_identity_and_revocation(self) -> Self:
        self.verify_identity_binding()
        if self.revoked and self.revoked_at_unix_ns is None:
            raise ValueError("revoked node metadata requires a revocation timestamp")
        if not self.revoked and (
            self.revoked_at_unix_ns is not None or self.revocation_reason is not None
        ):
            raise ValueError("active node metadata cannot contain revocation fields")
        return self


NodeAgentState = Literal["ready", "degraded", "blocked", "stopped", "failed"]


class BackendCandidateRecord(StrictModel):
    backend: Backend
    device: str
    detected: bool
    operational: bool
    reason: str
    device_name: str | None = None
    total_memory_bytes: NonNegativeInt = 0
    available_memory_bytes: NonNegativeInt = 0
    supported_dtypes: list[str] = Field(default_factory=list)


class BackendSelectionReport(StrictModel):
    schema_version: Literal[1] = CLUSTER_SCHEMA_VERSION
    document_version: Literal[1] = CLUSTER_DOCUMENT_VERSION
    candidates: list[BackendCandidateRecord]
    selected_backend: Backend
    selected_device: str
    selected_dtype: str
    reason: str
    measured_at_unix_ns: PositiveInt


class MemoryBudget(StrictModel):
    backend: Backend
    available_bytes: PositiveInt
    total_bytes: PositiveInt
    reserve_bytes: NonNegativeInt
    limit_bytes: PositiveInt
    source: Literal["automatic", "explicit-bytes", "explicit-percent"]
    fraction_of_available: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_limit(self) -> Self:
        if self.limit_bytes + self.reserve_bytes > self.available_bytes:
            raise ValueError("memory limit and reserve exceed currently available memory")
        return self


class EndpointSelection(StrictModel):
    control_listen_endpoint: str
    control_advertised_endpoint: str
    data_listen_endpoint: str
    data_advertised_endpoint: str
    probe_listen_endpoint: str | None = None
    probe_advertised_endpoint: str | None = None
    source_address: str
    interface_name: str | None = None
    selected_at_unix_ns: PositiveInt
    selection_reason: str
    network_fingerprint: str


class NodeConfiguration(StrictModel):
    schema_version: Literal[1] = CLUSTER_SCHEMA_VERSION
    document_version: Literal[1] = CLUSTER_DOCUMENT_VERSION
    node_id: str
    cluster_id: str
    coordinator_endpoint: str
    coordinator_fingerprint: str
    backend_override: Backend | None = None
    memory_limit_override_bytes: PositiveInt | None = None
    memory_percent_override: float | None = Field(default=None, gt=0, le=100)
    storage_limit_bytes: PositiveInt
    control_endpoint_override: str | None = None
    data_endpoint_override: str | None = None
    interface_override: str | None = None
    backend_selection: BackendSelectionReport
    memory_budget: MemoryBudget
    endpoints: EndpointSelection
    service_mode: NodeServiceMode
    allow_model_download: bool = True
    updated_at_unix_ns: PositiveInt

    @field_validator("coordinator_fingerprint")
    @classmethod
    def validate_coordinator_fingerprint(cls, value: str) -> str:
        return normalize_fingerprint(value)

    @model_validator(mode="after")
    def validate_overrides(self) -> Self:
        if (
            self.memory_limit_override_bytes is not None
            and self.memory_percent_override is not None
        ):
            raise ValueError("memory byte and percentage overrides are mutually exclusive")
        return self


class NodeRuntimeMetadata(StrictModel):
    schema_version: Literal[1] = CLUSTER_SCHEMA_VERSION
    document_version: Literal[1] = CLUSTER_DOCUMENT_VERSION
    node_id: str
    cluster_id: str | None = None
    state: NodeAgentState
    reason: str | None = None
    service_state: str
    process_id: NonNegativeInt | None = None
    selected_backend: Backend | None = None
    selected_device: str | None = None
    selected_dtype: str | None = None
    memory_limit_bytes: PositiveInt | None = None
    storage_limit_bytes: PositiveInt | None = None
    control_endpoint: str | None = None
    data_endpoint: str | None = None
    probe_endpoint: str | None = None
    coordinator_reachable: bool = False
    data_reachable: bool = False
    probe_reachable: bool | None = None
    artifact_cache_bytes: NonNegativeInt = 0
    loaded_stage_ids: list[NonNegativeInt] = Field(default_factory=list)
    current_role: str = "idle"
    last_refresh_unix_ns: PositiveInt
    last_error: str | None = None
    error_category: (
        Literal[
            "permission",
            "connectivity",
            "compatibility",
            "capacity",
            "artifact-integrity",
            "execution",
        ]
        | None
    ) = None


class NodeMembership(StrictModel):
    schema_version: Literal[1] = CLUSTER_SCHEMA_VERSION
    document_version: Literal[1] = CLUSTER_DOCUMENT_VERSION
    cluster_id: str
    node_id: str
    node_public_key: str
    node_fingerprint: str
    coordinator_public_key: str
    coordinator_fingerprint: str
    joined_at_unix_ns: PositiveInt
    status: Literal["active", "revoked", "left"] = "active"
    membership_generation: PositiveInt = 1
    last_authenticated_at_unix_ns: PositiveInt | None = None

    @field_validator("node_fingerprint", "coordinator_fingerprint")
    @classmethod
    def validate_fingerprints(cls, value: str) -> str:
        return normalize_fingerprint(value)

    def verify_identity_bindings(self) -> None:
        if public_key_fingerprint(self.node_public_key) != self.node_fingerprint:
            raise ValueError("membership node identity binding is invalid")
        if self.node_id != node_id_from_fingerprint(self.node_fingerprint):
            raise ValueError("membership node ID is invalid")
        if public_key_fingerprint(self.coordinator_public_key) != self.coordinator_fingerprint:
            raise ValueError("membership coordinator identity binding is invalid")

    @model_validator(mode="after")
    def validate_identity_bindings(self) -> Self:
        self.verify_identity_bindings()
        return self


class NodeRevocation(StrictModel):
    schema_version: Literal[1] = CLUSTER_SCHEMA_VERSION
    document_version: Literal[1] = CLUSTER_DOCUMENT_VERSION
    revocation_id: str
    cluster_id: str
    node_id: str
    node_fingerprint: str
    revoked_at_unix_ns: PositiveInt
    revoked_by_node_id: str
    reason: str
    generation: PositiveInt

    @field_validator("node_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        return normalize_fingerprint(value)


class PairingSession(StrictModel):
    """Non-secret session tombstone; ephemeral keys and secrets are never persisted."""

    schema_version: Literal[1] = CLUSTER_SCHEMA_VERSION
    document_version: Literal[1] = CLUSTER_DOCUMENT_VERSION
    session_id: str
    coordinator_endpoint: str
    coordinator_ephemeral_public_key: str
    created_at_unix_ns: PositiveInt
    expires_at_unix_ns: PositiveInt
    state: Literal["active", "consumed", "expired", "invalidated", "rejected"] = "active"
    attempts: NonNegativeInt = 0
    maximum_attempts: PositiveInt = 5
    consumed_at_unix_ns: PositiveInt | None = None
    node_id: str | None = None
    last_rejection_reason: str | None = None

    @model_validator(mode="after")
    def validate_time_range(self) -> Self:
        if self.expires_at_unix_ns <= self.created_at_unix_ns:
            raise ValueError("pairing expiry must be after creation")
        if self.attempts > self.maximum_attempts and self.state == "active":
            raise ValueError("active pairing session exceeded its attempt bound")
        if self.state == "consumed" and (self.consumed_at_unix_ns is None or self.node_id is None):
            raise ValueError("consumed pairing session requires timestamp and node ID")
        return self


class NetworkLinkMeasurement(StrictModel):
    schema_version: Literal[1] = CLUSTER_SCHEMA_VERSION
    document_version: Literal[1] = CLUSTER_DOCUMENT_VERSION
    source_worker_id: str
    destination_worker_id: str
    source_node_id: str | None = None
    destination_node_id: str | None = None
    measured_at_unix_ns: PositiveInt
    round_trip_latency_ms: NonNegativeFloat
    one_way_estimate_ms: NonNegativeFloat | None = None
    upload_bytes_per_s: NonNegativeFloat
    download_bytes_per_s: NonNegativeFloat
    payload_sizes: list[PositiveInt]
    sample_count: PositiveInt
    p95_transfer_ms: NonNegativeFloat
    source_endpoint: str | None = None
    destination_endpoint: str | None = None
    source_interface: str | None = None
    destination_interface: str | None = None
    medium: str | None = None
    mtu: PositiveInt | None = None
    measured: bool
    probe_ticket_id: str | None = None
    authentication_verified: bool = False
    payload_checksums_verified: bool = False

    @model_validator(mode="after")
    def validate_directed_link(self) -> Self:
        if self.source_worker_id == self.destination_worker_id:
            raise ValueError("network measurement must connect two distinct workers")
        if not self.payload_sizes:
            raise ValueError("network measurement requires payload-size evidence")
        return self


class ArtifactFile(StrictModel):
    relative_path: str
    size_bytes: NonNegativeInt
    sha256: str
    media_type: str = "application/octet-stream"
    tensor_names: list[str] = Field(default_factory=list)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        return _sha256(value, field="artifact file sha256")

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        if not normalized or normalized.startswith("/") or ".." in normalized.split("/"):
            raise ValueError("artifact paths must be safe relative paths")
        return normalized


class ArtifactManifest(StrictModel):
    schema_version: Literal[1] = CLUSTER_SCHEMA_VERSION
    document_version: Literal[1] = CLUSTER_DOCUMENT_VERSION
    artifact_format_version: PositiveInt = ARTIFACT_FORMAT_VERSION
    artifact_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    adapter_id: Literal["olmoe"] = "olmoe"
    source_hashes: dict[str, str]
    stage_assignment_id: str
    dtype: str
    quantization: str = "none"
    content_hash: str
    layer_start: NonNegativeInt
    layer_end: PositiveInt
    owns_embeddings: bool
    owns_final_norm: bool
    owns_output_projection: bool
    tied_tensor_groups: list[list[str]] = Field(default_factory=list)
    files: list[ArtifactFile]
    total_size_bytes: NonNegativeInt
    created_at_unix_ns: PositiveInt

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: str) -> str:
        return _sha256(value, field="artifact content_hash")

    @field_validator("source_hashes")
    @classmethod
    def validate_source_hashes(cls, values: dict[str, str]) -> dict[str, str]:
        return {key: _sha256(value, field=f"source hash {key}") for key, value in values.items()}

    @model_validator(mode="after")
    def validate_manifest_totals(self) -> Self:
        if self.layer_end <= self.layer_start:
            raise ValueError("artifact layer range must be non-empty")
        if len({item.relative_path for item in self.files}) != len(self.files):
            raise ValueError("artifact file paths must be unique")
        if sum(item.size_bytes for item in self.files) != self.total_size_bytes:
            raise ValueError("artifact total size does not match its files")
        if self.artifact_id not in {self.content_hash, f"sha256:{self.content_hash}"}:
            raise ValueError("artifact ID must be its complete content hash")
        return self


class ArtifactChunk(StrictModel):
    artifact_id: str
    relative_path: str
    chunk_index: NonNegativeInt
    offset_bytes: NonNegativeInt
    size_bytes: PositiveInt
    sha256: str
    final_chunk: bool = False

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        return _sha256(value, field="artifact chunk sha256")


class ArtifactLease(StrictModel):
    schema_version: Literal[1] = CLUSTER_SCHEMA_VERSION
    document_version: Literal[1] = CLUSTER_DOCUMENT_VERSION
    lease_id: str
    artifact_id: str
    owner: str
    purpose: Literal["loaded-stage", "deployment", "transfer", "pinned"]
    created_at_unix_ns: PositiveInt
    expires_at_unix_ns: PositiveInt | None = None


class ArtifactCacheEntry(StrictModel):
    schema_version: Literal[1] = CLUSTER_SCHEMA_VERSION
    document_version: Literal[1] = CLUSTER_DOCUMENT_VERSION
    artifact_id: str
    manifest: ArtifactManifest
    relative_directory: str
    size_bytes: NonNegativeInt
    created_at_unix_ns: PositiveInt
    last_accessed_at_unix_ns: PositiveInt
    verified_at_unix_ns: PositiveInt
    pinned: bool = False
    active_lease_ids: list[str] = Field(default_factory=list)


class ArtifactTransferStatus(StrictModel):
    schema_version: Literal[1] = CLUSTER_SCHEMA_VERSION
    document_version: Literal[1] = CLUSTER_DOCUMENT_VERSION
    transfer_id: str
    artifact_id: str
    source: str
    destination_node_id: str
    state: Literal[
        "queued",
        "transferring",
        "verifying",
        "complete",
        "failed",
        "cancelled",
    ]
    bytes_total: NonNegativeInt
    bytes_completed: NonNegativeInt
    chunks_total: NonNegativeInt
    chunks_completed: NonNegativeInt
    started_at_unix_ns: PositiveInt
    updated_at_unix_ns: PositiveInt
    last_error: str | None = None

    @model_validator(mode="after")
    def validate_progress_bounds(self) -> Self:
        if self.bytes_completed > self.bytes_total:
            raise ValueError("artifact transfer byte progress exceeds its total")
        if self.chunks_completed > self.chunks_total:
            raise ValueError("artifact transfer chunk progress exceeds its total")
        return self


class ClusterAuditEvent(StrictModel):
    schema_version: Literal[1] = CLUSTER_SCHEMA_VERSION
    event_id: str
    event_type: str
    timestamp_unix_ns: PositiveInt
    cluster_id: str | None = None
    node_id: str | None = None
    worker_id: str | None = None
    pairing_session_id: str | None = None
    category: str | None = None
    detail: str | None = None


class NodeRegistryDocument(StrictModel):
    schema_version: Literal[1] = CLUSTER_SCHEMA_VERSION
    document_version: Literal[1] = CLUSTER_DOCUMENT_VERSION
    nodes: list[NodeMetadata] = Field(default_factory=list)


class MembershipRegistryDocument(StrictModel):
    schema_version: Literal[1] = CLUSTER_SCHEMA_VERSION
    document_version: Literal[1] = CLUSTER_DOCUMENT_VERSION
    memberships: list[NodeMembership] = Field(default_factory=list)


class RevocationRegistryDocument(StrictModel):
    schema_version: Literal[1] = CLUSTER_SCHEMA_VERSION
    document_version: Literal[1] = CLUSTER_DOCUMENT_VERSION
    revocations: list[NodeRevocation] = Field(default_factory=list)


class PairingSessionRegistryDocument(StrictModel):
    schema_version: Literal[1] = CLUSTER_SCHEMA_VERSION
    document_version: Literal[1] = CLUSTER_DOCUMENT_VERSION
    sessions: list[PairingSession] = Field(default_factory=list)


class NetworkMeasurementRegistryDocument(StrictModel):
    schema_version: Literal[1] = CLUSTER_SCHEMA_VERSION
    document_version: Literal[1] = CLUSTER_DOCUMENT_VERSION
    measurements: list[NetworkLinkMeasurement] = Field(default_factory=list)


class ArtifactCacheDocument(StrictModel):
    schema_version: Literal[1] = CLUSTER_SCHEMA_VERSION
    document_version: Literal[1] = CLUSTER_DOCUMENT_VERSION
    entries: list[ArtifactCacheEntry] = Field(default_factory=list)
    transfers: list[ArtifactTransferStatus] = Field(default_factory=list)
    leases: list[ArtifactLease] = Field(default_factory=list)


__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "CLUSTER_DOCUMENT_VERSION",
    "CLUSTER_SCHEMA_VERSION",
    "PRODUCT_PROTOCOL_MAJOR",
    "PRODUCT_PROTOCOL_MINOR",
    "TRUSTED_LAN_SECURITY_CLASSIFICATION",
    "ArtifactCacheDocument",
    "ArtifactCacheEntry",
    "ArtifactChunk",
    "ArtifactFile",
    "ArtifactLease",
    "ArtifactManifest",
    "ArtifactTransferStatus",
    "BackendCandidateRecord",
    "BackendSelectionReport",
    "ClusterAuditEvent",
    "ClusterMetadata",
    "EndpointSelection",
    "MembershipRegistryDocument",
    "MemoryBudget",
    "NetworkLinkMeasurement",
    "NetworkMeasurementRegistryDocument",
    "NodeConfiguration",
    "NodeMembership",
    "NodeMetadata",
    "NodeRegistryDocument",
    "NodeRevocation",
    "NodeRuntimeMetadata",
    "PairingSession",
    "PairingSessionRegistryDocument",
    "RevocationRegistryDocument",
    "VersionCompatibility",
    "node_id_from_fingerprint",
]
