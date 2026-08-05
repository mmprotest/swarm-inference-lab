"""Canonical product protocol for stage-internal MoE execution.

``SWARMEX1`` is the version-one expert data-plane protocol.  It carries only
selected expert activations and results; transformer stage boundaries continue
to use the stage-ring protocol.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, PositiveInt, field_validator, model_validator

from swarm_inference.config.models import StrictModel
from swarm_inference.exceptions import IntegrityError
from swarm_inference.protocol.routes import BoundedNonceCache
from swarm_inference.security.identity import WorkerIdentity, public_key_fingerprint
from swarm_inference.security.signatures import canonical_json_bytes, verify_signature


class ExpertProtocolVersion(StrEnum):
    V1 = "1.0"


SUPPORTED_EXPERT_PROTOCOL_VERSIONS = (ExpertProtocolVersion.V1,)


def negotiate_expert_protocol(
    offered: Sequence[ExpertProtocolVersion | str],
) -> ExpertProtocolVersion:
    """Select the newest mutually supported expert protocol version."""

    normalized = {ExpertProtocolVersion(item) for item in offered}
    for version in reversed(SUPPORTED_EXPERT_PROTOCOL_VERSIONS):
        if version in normalized:
            return version
    raise ValueError("no mutually supported expert protocol version")


class DataPlane(StrEnum):
    IN_PROCESS = "in_process"
    SHARED_MEMORY = "shared_memory"
    DIRECT_TCP = "direct_tcp"
    RELAYED_TCP = "relayed_tcp"


class ExpertExecutionMode(StrEnum):
    WHOLE_EXPERT = "whole_expert"
    MICROSHARD = "microshard"


class DeterminismMode(StrEnum):
    EXACT = "exact"
    QUALITY_BOUNDED = "quality_bounded"


class ExpertResponseMode(StrEnum):
    PER_EXPERT_EXACT = "per_expert_exact"
    PER_WORKER_FAST = "per_worker_fast"


class TransportCodec(StrEnum):
    RAW_FP32 = "raw_fp32"
    RAW_FP16 = "raw_fp16"
    INT8_PER_VECTOR = "int8_per_vector"
    LOSSLESS_GENERAL = "lossless_general"


class ReductionMode(StrEnum):
    FIXED_ORDER_FP32 = "fixed_order_fp32"
    TREE_FP32 = "tree_fp32"
    FAST_BACKEND_NATIVE = "fast_backend_native"


class TensorWireMetadata(StrictModel):
    name: str
    envelope: Literal["raw", "SWARMT01"] = "raw"
    dtype: Literal["float32", "float16", "int8", "uint8"]
    shape: list[int] = Field(min_length=1)
    codec: TransportCodec = TransportCodec.RAW_FP32
    payload_index: int = Field(ge=0)
    raw_bytes: int = Field(ge=0)
    encoded_bytes: int = Field(ge=0)
    scale: float | list[float] | None = None
    checksum: str

    @field_validator("shape")
    @classmethod
    def positive_shape(cls, value: list[int]) -> list[int]:
        if any(item <= 0 for item in value):
            raise ValueError("tensor dimensions must be positive")
        return value


class ExpertExecutionRequest(StrictModel):
    """One exact whole-expert or native-microshard operation.

    The legacy routing fields remain part of protocol v1 for Experiment 010
    wire compatibility.  Product callers also bind every operation to its
    stage session, token, sequence, and installed route generation.
    """

    schema_version: Literal["1.0"] = "1.0"
    request_id: str
    session_id: str = "legacy"
    token_position: int = Field(default=0, ge=0)
    sequence_id: int = Field(default=0, ge=0)
    route_generation: int = Field(default=0, ge=0)
    topology_id: str = "legacy"
    model_id: str
    model_revision: str
    model_fingerprint: str = ""
    quantization_fingerprint: str
    layer_id: int = Field(ge=0)
    batch_rows: int = Field(gt=0)
    latent_dimension: int = Field(gt=0)
    expert_ids: list[int] = Field(default_factory=list)
    expert_hashes: dict[int, str] = Field(default_factory=dict)
    routing_weights: list[float] = Field(default_factory=list)
    top_k: int | None = Field(default=None, gt=0)
    expert_ids_by_row: list[list[int]] | None = None
    routing_weights_by_row: list[list[float]] | None = None
    selected_rank_by_row: list[list[int]] | None = None
    # Product dispatch may pack only selected rows.  These fields restore the
    # original flattened-token and top-k positions at the owning stage.
    token_indices: list[int] | None = None
    selected_ranks: list[int] | None = None
    response_mode: ExpertResponseMode = ExpertResponseMode.PER_WORKER_FAST
    activations: dict[str, Any]
    deadline_ns: int = Field(gt=0)
    execution_mode: ExpertExecutionMode = ExpertExecutionMode.WHOLE_EXPERT
    determinism_mode: DeterminismMode = DeterminismMode.EXACT
    compression: TransportCodec = TransportCodec.RAW_FP32
    hidden_start: int | None = Field(default=None, ge=0)
    hidden_end: int | None = Field(default=None, gt=0)
    down_accumulators: dict[str, Any] | None = None
    microshard_final: bool = False
    reduction_mode: ReductionMode = ReductionMode.FIXED_ORDER_FP32
    challenge: bool = False
    authentication: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_request(self) -> ExpertExecutionRequest:
        if not self.request_id or not self.session_id:
            raise ValueError("expert request and session identities are required")
        if len(self.routing_weights) != len(self.expert_ids):
            raise ValueError("expert IDs and routing weights must have equal lengths")
        if any(expert < 0 for expert in self.expert_ids):
            raise ValueError("expert IDs must be non-negative")
        per_row = (
            self.expert_ids_by_row,
            self.routing_weights_by_row,
            self.selected_rank_by_row,
        )
        if any(value is not None for value in per_row):
            if any(value is None for value in per_row):
                raise ValueError("per-row routing requires IDs, weights, and selected ranks")
            assert self.expert_ids_by_row is not None
            assert self.routing_weights_by_row is not None
            assert self.selected_rank_by_row is not None
            if len(self.expert_ids_by_row) != self.batch_rows:
                raise ValueError("per-row routing row count does not match batch_rows")
            inferred_top_k = len(self.expert_ids_by_row[0])
            if inferred_top_k < 1 or (self.top_k is not None and self.top_k != inferred_top_k):
                raise ValueError("top_k does not match per-row routing width")
            for row, (experts, weights, ranks) in enumerate(
                zip(
                    self.expert_ids_by_row,
                    self.routing_weights_by_row,
                    self.selected_rank_by_row,
                    strict=True,
                )
            ):
                if len(experts) != inferred_top_k or len(weights) != inferred_top_k:
                    raise ValueError(f"routing width differs at batch row {row}")
                if ranks != list(range(inferred_top_k)):
                    raise ValueError(f"selected ranks are not rank-preserving at batch row {row}")
                if any(expert < 0 for expert in experts):
                    raise ValueError("expert IDs must be non-negative")
            if self.expert_ids or self.routing_weights:
                raise ValueError("request cannot mix flat and per-row routing")
        else:
            if not self.expert_ids:
                raise ValueError("request requires flat or per-row expert routing")
            if self.top_k is not None and self.top_k != len(self.expert_ids):
                raise ValueError("top_k does not match flat routing width")
        if (self.token_indices is None) != (self.selected_ranks is None):
            raise ValueError("packed selected rows require token indices and selected ranks")
        if self.token_indices is not None:
            if (
                len(self.token_indices) != self.batch_rows
                or len(self.selected_ranks or []) != self.batch_rows
            ):
                raise ValueError("packed selected-row identities must match batch_rows")
            if any(index < 0 for index in self.token_indices):
                raise ValueError("packed token indices must be non-negative")
        if self.determinism_mode == DeterminismMode.EXACT:
            if self.compression != TransportCodec.RAW_FP32:
                raise ValueError("exact mode requires raw_fp32 transport")
            if self.reduction_mode != ReductionMode.FIXED_ORDER_FP32:
                raise ValueError("exact mode requires fixed_order_fp32 reduction")
        if self.execution_mode == ExpertExecutionMode.MICROSHARD:
            if self.hidden_start is None or self.hidden_end is None:
                raise ValueError("microshard execution requires a hidden range")
            if self.hidden_end <= self.hidden_start:
                raise ValueError("microshard hidden range must be non-empty")
            exact_chain = self.response_mode == ExpertResponseMode.PER_EXPERT_EXACT
            if (self.hidden_start == 0 or not exact_chain) and self.down_accumulators is not None:
                raise ValueError("this microshard request cannot carry a down accumulator")
            if exact_chain and self.hidden_start > 0 and self.down_accumulators is None:
                raise ValueError("non-initial exact microshard requires a down accumulator")
            if not exact_chain and self.microshard_final:
                raise ValueError("fast microshards do not carry exact chain state")
        elif self.hidden_start is not None or self.hidden_end is not None:
            raise ValueError("whole-expert execution cannot carry a hidden range")
        elif self.down_accumulators is not None or self.microshard_final:
            raise ValueError("whole-expert execution cannot carry microshard chain state")
        return self

    @property
    def effective_top_k(self) -> int:
        if self.expert_ids_by_row is not None:
            return len(self.expert_ids_by_row[0])
        return len(self.expert_ids)

    @property
    def all_expert_ids(self) -> set[int]:
        if self.expert_ids_by_row is None:
            return set(self.expert_ids)
        return {expert for row in self.expert_ids_by_row for expert in row}

    def routing_for_row(self, row: int) -> tuple[list[int], list[float], list[int]]:
        if row < 0 or row >= self.batch_rows:
            raise IndexError(row)
        if self.expert_ids_by_row is not None:
            assert self.routing_weights_by_row is not None
            assert self.selected_rank_by_row is not None
            return (
                self.expert_ids_by_row[row],
                self.routing_weights_by_row[row],
                self.selected_rank_by_row[row],
            )
        return self.expert_ids, self.routing_weights, list(range(len(self.expert_ids)))


class ExpertExecutionMetadata(StrictModel):
    experts_executed: list[int]
    bytes_read: int = Field(ge=0)
    bytes_received: int = Field(ge=0)
    bytes_sent: int = Field(ge=0)
    cache_hits: int = Field(ge=0)
    cache_misses: int = Field(ge=0)
    compute_ns: int = Field(ge=0)
    queue_ns: int = Field(ge=0)
    transfer_ns: int = Field(ge=0)
    serialisation_ns: int = Field(default=0, ge=0)
    copy_ns: int = Field(default=0, ge=0)
    kernel_transition_ns: int = Field(default=0, ge=0)
    backend: str = "cpu"
    device: str = "cpu"
    resident_tensor_bytes: int = Field(default=0, ge=0)
    expert_resident_bytes: int = Field(default=0, ge=0)
    fallback_events: list[dict[str, Any]] = Field(default_factory=list)


class ResultIntegrity(StrictModel):
    result_hash: str
    model_fingerprint: str
    worker_signature: str
    expert_hashes: dict[int, str] = Field(default_factory=dict)


class ExpertExecutionResponse(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    request_id: str
    session_id: str = "legacy"
    token_position: int = Field(default=0, ge=0)
    sequence_id: int = Field(default=0, ge=0)
    route_generation: int = Field(default=0, ge=0)
    worker_id: str
    model_revision: str
    quantization_fingerprint: str = ""
    layer_id: int = Field(ge=0)
    result: dict[str, Any]
    execution_metadata: ExpertExecutionMetadata
    integrity: ResultIntegrity
    status: Literal["ok", "error"] = "ok"
    error: str | None = None

    @model_validator(mode="after")
    def error_contract(self) -> ExpertExecutionResponse:
        if self.status == "error" and not self.error:
            raise ValueError("error responses require a reason")
        if self.status == "ok" and self.error is not None:
            raise ValueError("successful responses cannot carry an error")
        return self


class WorkerBudget(StrictModel):
    worker_id: str
    memory_budget_bytes: int = Field(gt=0)
    expert_residency_budget_bytes: int = Field(gt=0)
    cache_budget_bytes: int = Field(ge=0)
    thread_count: int = Field(gt=0)
    cpu_affinity: list[int] = Field(min_length=1)
    storage_directory: str
    device: str
    backend: str
    physical_memory_limit: bool = False


class WorkerManifest(StrictModel):
    worker_id: str
    process_id: int = Field(gt=0)
    endpoint: str
    control_endpoint: str | None = None
    universal_worker_abi: dict[str, Any] = Field(default_factory=dict)
    model_id: str
    model_revision: str
    quantization_fingerprint: str
    model_fingerprint: str
    bridge_version: str
    owned_experts: dict[str, list[int]] = Field(default_factory=dict)
    owned_microshards: list[dict[str, Any]] = Field(default_factory=list)
    tensor_hashes: dict[str, str] = Field(default_factory=dict)
    resident_tensor_bytes: int = Field(ge=0)
    expert_bytes: int = Field(ge=0)
    cache_bytes: int = Field(ge=0)
    peak_rss_bytes: int = Field(ge=0)
    roles: list[str] = Field(default_factory=list)


class ExpertRouteParticipant(StrictModel):
    worker_id: str
    worker_public_key: str
    worker_public_key_fingerprint: str
    endpoint: str
    roles: list[Literal["contiguous-stage", "whole-expert", "expert-microshard", "reducer"]]
    owned_experts: dict[int, list[int]] = Field(default_factory=dict)
    owned_microshards: list[dict[str, Any]] = Field(default_factory=list)
    model_fingerprint: str
    quantization_fingerprint: str

    @model_validator(mode="after")
    def validate_identity_and_ownership(self) -> ExpertRouteParticipant:
        actual = public_key_fingerprint(self.worker_public_key)
        if not hmac.compare_digest(actual, self.worker_public_key_fingerprint):
            raise ValueError("expert participant public-key fingerprint mismatch")
        if not self.roles:
            raise ValueError("expert participant requires at least one role")
        return self


class SignedExpertRouteLease(StrictModel):
    topology_id: str
    route_generation: PositiveInt
    model_id: str
    model_revision: str
    model_fingerprint: str
    quantization_fingerprint: str
    participants: list[ExpertRouteParticipant]
    lease_issued_unix_ns: PositiveInt
    lease_expiry_unix_ns: PositiveInt
    nonce: str
    coordinator_identity: str
    coordinator_public_key: str
    coordinator_public_key_fingerprint: str
    signature: str = ""

    @model_validator(mode="after")
    def validate_lease(self) -> SignedExpertRouteLease:
        if self.lease_expiry_unix_ns <= self.lease_issued_unix_ns:
            raise ValueError("expert route lease expiry must follow its issue time")
        workers = [participant.worker_id for participant in self.participants]
        if len(workers) != len(set(workers)):
            raise ValueError("expert route workers must be unique")
        actual = public_key_fingerprint(self.coordinator_public_key)
        if not hmac.compare_digest(actual, self.coordinator_public_key_fingerprint):
            raise ValueError("coordinator public-key fingerprint mismatch")
        return self


class ExpertPeerHandshake(StrictModel):
    protocol_versions: list[ExpertProtocolVersion]
    selected_version: ExpertProtocolVersion | None = None
    worker_id: str
    public_key_fingerprint: str
    topology_id: str
    route_generation: PositiveInt
    peer_worker_id: str
    model_revision: str
    quantization_fingerprint: str
    nonce: str
    timestamp_unix_ns: PositiveInt
    route_lease_hash: str
    signature: str = ""


def _signed_payload(model: StrictModel) -> bytes:
    return canonical_json_bytes(model.model_dump(mode="json", exclude={"signature"}))


def sign_expert_route_lease(
    lease: SignedExpertRouteLease, identity: WorkerIdentity
) -> SignedExpertRouteLease:
    if lease.coordinator_public_key != identity.public_key_b64:
        raise IntegrityError("expert route coordinator public key does not match signer")
    return lease.model_copy(update={"signature": identity.sign(_signed_payload(lease))})


def verify_expert_route_lease(
    lease: SignedExpertRouteLease,
    trusted_coordinators: dict[str, str] | set[str],
    *,
    now_unix_ns: int | None = None,
    last_route_generation: int | None = None,
    nonce_cache: BoundedNonceCache | None = None,
) -> None:
    if not lease.signature:
        raise IntegrityError("expert route lease signature is missing")
    if last_route_generation is not None and lease.route_generation <= last_route_generation:
        raise IntegrityError("stale route generation")
    if isinstance(trusted_coordinators, dict):
        key = trusted_coordinators.get(lease.coordinator_identity)
        if key is None or key != lease.coordinator_public_key:
            raise IntegrityError("unknown coordinator identity")
    else:
        if (
            lease.coordinator_public_key not in trusted_coordinators
            and lease.coordinator_public_key_fingerprint not in trusted_coordinators
        ):
            raise IntegrityError("unknown coordinator identity")
        key = lease.coordinator_public_key
    verify_signature(key, _signed_payload(lease), lease.signature)
    now = time.time_ns() if now_unix_ns is None else now_unix_ns
    if lease.lease_expiry_unix_ns <= now:
        raise IntegrityError("expert route lease has expired")
    if lease.lease_issued_unix_ns > now + 30_000_000_000:
        raise IntegrityError("expert route lease is future-dated outside tolerance")
    if nonce_cache is not None:
        nonce_cache.add(lease.nonce)


def expert_route_lease_hash(lease: SignedExpertRouteLease) -> str:
    return hashlib.sha256(canonical_json_bytes(lease.model_dump(mode="json"))).hexdigest()


def sign_expert_peer_handshake(
    handshake: ExpertPeerHandshake, identity: WorkerIdentity
) -> ExpertPeerHandshake:
    if handshake.public_key_fingerprint != identity.public_key_fingerprint:
        raise IntegrityError("expert peer handshake fingerprint does not match signer")
    return handshake.model_copy(update={"signature": identity.sign(_signed_payload(handshake))})


def verify_expert_peer_handshake(
    handshake: ExpertPeerHandshake,
    lease: SignedExpertRouteLease,
    *,
    expected_worker_id: str,
    expected_peer_worker_id: str,
    now_unix_ns: int | None = None,
    nonce_cache: BoundedNonceCache | None = None,
) -> ExpertProtocolVersion:
    if not handshake.signature:
        raise IntegrityError("expert peer handshake signature is missing")
    if (
        handshake.worker_id != expected_worker_id
        or handshake.peer_worker_id != expected_peer_worker_id
        or handshake.topology_id != lease.topology_id
        or handshake.route_generation != lease.route_generation
        or handshake.model_revision != lease.model_revision
        or handshake.quantization_fingerprint != lease.quantization_fingerprint
        or handshake.route_lease_hash != expert_route_lease_hash(lease)
    ):
        raise IntegrityError("expert peer handshake route identity mismatch")
    participant = next(
        (item for item in lease.participants if item.worker_id == expected_worker_id), None
    )
    if participant is None:
        raise IntegrityError("expert peer is not authorized by the installed route")
    if not hmac.compare_digest(
        participant.worker_public_key_fingerprint, handshake.public_key_fingerprint
    ):
        raise IntegrityError("expert peer public-key fingerprint mismatch")
    now = time.time_ns() if now_unix_ns is None else now_unix_ns
    if abs(now - handshake.timestamp_unix_ns) > 30_000_000_000:
        raise IntegrityError("expert peer handshake timestamp is stale or future-dated")
    verify_signature(participant.worker_public_key, _signed_payload(handshake), handshake.signature)
    if nonce_cache is not None:
        nonce_cache.add(handshake.nonce)
    selected = negotiate_expert_protocol(handshake.protocol_versions)
    if handshake.selected_version is not None and handshake.selected_version != selected:
        raise IntegrityError("expert peer selected an unsupported protocol version")
    return selected


__all__ = [
    "SUPPORTED_EXPERT_PROTOCOL_VERSIONS",
    "DataPlane",
    "DeterminismMode",
    "ExpertExecutionMetadata",
    "ExpertExecutionMode",
    "ExpertExecutionRequest",
    "ExpertExecutionResponse",
    "ExpertPeerHandshake",
    "ExpertProtocolVersion",
    "ExpertResponseMode",
    "ExpertRouteParticipant",
    "ReductionMode",
    "ResultIntegrity",
    "SignedExpertRouteLease",
    "TensorWireMetadata",
    "TransportCodec",
    "WorkerBudget",
    "WorkerManifest",
    "expert_route_lease_hash",
    "negotiate_expert_protocol",
    "sign_expert_peer_handshake",
    "sign_expert_route_lease",
    "verify_expert_peer_handshake",
    "verify_expert_route_lease",
]
