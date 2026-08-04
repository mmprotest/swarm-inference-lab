"""Typed control-plane and activation-envelope messages."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from enum import Enum, StrEnum
from typing import Any, Literal, TypeVar

from google.protobuf.any_pb2 import Any as ProtoAny
from pydantic import Field, model_validator

from swarm_inference.config.models import (
    DataPlaneMode,
    ModelManifest,
    OperationKind,
    StageDefinition,
    StrictModel,
    SyntheticModelConfig,
    TensorSpec,
    WorkerCapability,
)
from swarm_inference.exceptions import IntegrityError


class RegistrationRequest(StrictModel):
    capability: WorkerCapability
    benchmark_nonce: str
    signature: str


class RegistrationResponse(StrictModel):
    accepted: bool
    reason: str = ""
    heartbeat_interval_s: float = 5.0


class Heartbeat(StrictModel):
    worker_id: str
    queue_depth: int = Field(ge=0)
    assignments: list[int] = Field(default_factory=list)
    monotonic_ns: int = Field(ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    signature: str


class StageAssignmentMessage(StrictModel):
    worker_id: str
    stage: StageDefinition
    shard_path: str
    shard_hash: str
    model_id: str
    model_revision: str
    synthetic_model: SyntheticModelConfig | None = None
    architecture_config: dict[str, Any] | None = None
    model_manifest: ModelManifest | None = None
    dtype: str | None = None
    data_plane_mode: DataPlaneMode = DataPlaneMode.COORDINATOR_RELAY
    coordinator_data_endpoint: str | None = None
    route_signing_key: str | None = None


class ActivationMetadata(StrictModel):
    request_id: str
    tensor_id: str
    stage_id: int = Field(ge=0)
    operation: OperationKind
    token_position: int = Field(ge=0)
    sequence_length: int = Field(gt=0)
    cache_generation: int = Field(ge=0)
    route_generation: int = Field(default=0, ge=0)
    model_id: str
    model_revision: str
    deadline_monotonic_ns: int | None = Field(default=None, ge=0)
    audit: bool = False


class ActivationRequest(StrictModel):
    metadata: ActivationMetadata
    tensor_payload: bytes


class ActivationResult(StrictModel):
    metadata: ActivationMetadata
    tensor_payload: bytes
    worker_id: str
    execution_ms: float = Field(ge=0)
    queue_ms: float = Field(ge=0)
    checksum: str
    signature: str = ""


class RouteHop(StrictModel):
    stage_id: int = Field(ge=0)
    worker_id: str
    worker_data_endpoint: str
    expected_shard_hash: str
    expected_input_spec: TensorSpec
    expected_output_spec: TensorSpec


class RoutePlan(StrictModel):
    route_id: str
    route_generation: int = Field(ge=1)
    request_id: str
    model_id: str
    model_revision: str
    assignments: list[RouteHop]
    route_lease_expiry_unix_ns: int = Field(gt=0)
    workload_class: str
    cancellation_generation: int = Field(ge=0)
    integrity_policy: str
    final_result_destination: str
    signature: str = ""

    @model_validator(mode="after")
    def validate_order(self) -> RoutePlan:
        stages = [hop.stage_id for hop in self.assignments]
        if stages != list(range(len(stages))):
            raise ValueError("route assignments must be ordered and contiguous from stage zero")
        if len({hop.worker_id for hop in self.assignments}) != len(self.assignments):
            raise ValueError("one worker cannot own multiple hops in one route")
        return self


class RouteInstallRequest(StrictModel):
    worker_id: str
    route: RoutePlan


class HopTelemetry(StrictModel):
    stage_id: int = Field(ge=0)
    worker_id: str
    source_worker: str
    destination_worker: str
    execution_ms: float = Field(ge=0)
    queue_ms: float = Field(ge=0)
    serialisation_ms: float = Field(ge=0)
    deserialisation_ms: float = Field(ge=0)
    integrity_validation_ms: float = Field(ge=0)
    cache_update_ms: float = Field(ge=0)
    stream_queue_ms: float = Field(ge=0)
    transfer_ms: float = Field(ge=0)
    hop_end_to_end_ms: float = Field(ge=0)
    payload_bytes: int = Field(ge=0)

    @property
    def effective_service_ms(self) -> float:
        return (
            self.execution_ms
            + self.queue_ms
            + self.serialisation_ms
            + self.deserialisation_ms
            + self.integrity_validation_ms
            + self.cache_update_ms
            + self.stream_queue_ms
            + self.transfer_ms
        )


class DataPlaneEnvelope(StrictModel):
    message_id: str
    route_id: str
    route_generation: int = Field(ge=1)
    request_id: str
    stage_id: int = Field(ge=0)
    source_worker: str
    destination_worker: str
    token_position: int = Field(ge=0)
    operation: OperationKind
    tensor_metadata: ActivationMetadata
    tensor_payload: bytes
    payload_length: int = Field(ge=0)
    payload_checksum: str
    sequence_number: int = Field(ge=0)
    timestamp_unix_ns: int = Field(gt=0)
    hop_telemetry: list[HopTelemetry] = Field(default_factory=list)
    replay_only: bool = False
    integrity_metadata: str = "hmac-sha256"
    signature: str = ""


class DataPlaneAck(StrictModel):
    message_id: str
    status: Literal[
        "accepted",
        "backpressured",
        "duplicate",
        "stale_route",
        "invalid_checksum",
        "invalid_stage_transition",
        "unknown_request",
        "destination_unavailable",
    ]
    detail: str = ""
    accepted_timestamp_unix_ns: int = Field(gt=0)

    @property
    def accepted(self) -> bool:
        return self.status in {"accepted", "duplicate"}


class FinalResultMessage(StrictModel):
    message_id: str
    route_id: str
    route_generation: int = Field(ge=1)
    request_id: str
    token_position: int = Field(ge=0)
    result: ActivationResult
    hop_telemetry: list[HopTelemetry]
    payload_checksum: str
    timestamp_unix_ns: int = Field(gt=0)
    signature: str = ""


class HealthResponse(StrictModel):
    worker_id: str
    healthy: bool
    queue_depth: int = Field(ge=0)
    loaded_stages: list[int] = Field(default_factory=list)
    detail: str = ""
    proof: dict[str, Any] = Field(default_factory=dict)


class CancelRequest(StrictModel):
    request_id: str
    model_revision: str


class CacheControlRequest(StrictModel):
    worker_id: str
    request_id: str
    model_revision: str
    stage_id: int = Field(ge=0)
    action: Literal["inspect", "clear", "replay", "clear-and-replay"]


class CacheControlResponse(StrictModel):
    accepted: bool
    worker_id: str
    request_id: str
    stage_id: int = Field(ge=0)
    action: str
    replay_input_count: int = Field(default=0, ge=0)
    replay_bytes: int = Field(default=0, ge=0)
    replay_duration_s: float = Field(default=0.0, ge=0)
    cache_before: list[dict[str, Any]] = Field(default_factory=list)
    cache_after: list[dict[str, Any]] = Field(default_factory=list)
    detail: str = ""


class Ack(StrictModel):
    accepted: bool
    detail: str = ""


class WireChunk(StrictModel):
    message_id: str
    chunk_index: int = Field(ge=0)
    chunk_count: int = Field(gt=0)
    total_length: int = Field(ge=0)
    payload: bytes
    checksum: str


class SubmitRequest(StrictModel):
    request_id: str
    prompt: str | None = None
    prompt_token_ids: list[int] = Field(default_factory=list)
    max_new_tokens: int = Field(gt=0)
    random_seed: int
    workload_class: str = "standard"
    model_id: str = "synthetic"
    model_revision: str = "synthetic-v1"
    cache_replay_stage_id: int | None = Field(default=None, ge=0)
    cache_replay_after_tokens: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def require_prompt(self) -> SubmitRequest:
        if self.prompt is None and not self.prompt_token_ids:
            raise ValueError("either prompt or prompt_token_ids is required")
        if (self.cache_replay_stage_id is None) != (self.cache_replay_after_tokens is None):
            raise ValueError(
                "cache_replay_stage_id and cache_replay_after_tokens must be supplied together"
            )
        return self


class SubmitResponse(StrictModel):
    request_id: str
    output_token_ids: list[int] = Field(default_factory=list)
    status: Literal["completed", "failed", "cancelled"]
    verified: bool
    aggregate_experiment_metric: bool = False
    time_to_first_token_s: float | None = Field(default=None, ge=0)
    end_to_end_s: float = Field(ge=0)
    detail: str = ""


class StreamEventType(StrEnum):
    REQUEST_ACCEPTED = "REQUEST_ACCEPTED"
    TOPOLOGY_SELECTED = "TOPOLOGY_SELECTED"
    SESSION_OPENED = "SESSION_OPENED"
    PREFILL_STARTED = "PREFILL_STARTED"
    TOKEN_GENERATED = "TOKEN_GENERATED"
    SESSION_CLOSED = "SESSION_CLOSED"
    REQUEST_COMPLETED = "REQUEST_COMPLETED"
    REQUEST_FAILED = "REQUEST_FAILED"
    REQUEST_CANCELLED = "REQUEST_CANCELLED"


class SubmitStreamEvent(StrictModel):
    """One strictly ordered publication in a coordinator submit stream."""

    event_type: StreamEventType
    request_id: str
    sequence_number: int = Field(ge=0)
    monotonic_timestamp_ns: int = Field(gt=0)
    session_id: str | None = None
    topology_id: str | None = None
    model_revision: str | None = None
    token_position: int | None = Field(default=None, ge=0)
    token_id: int | None = Field(default=None, ge=0)
    decoded_text_fragment: str = ""
    status_detail: str = ""
    final_token_ids: list[int] = Field(default_factory=list)
    timing_metrics: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_event_payload(self) -> SubmitStreamEvent:
        if self.event_type == StreamEventType.TOKEN_GENERATED and (
            self.token_position is None or self.token_id is None
        ):
            raise ValueError("TOKEN_GENERATED requires token position and token ID")
        return self


WireMessage = (
    RegistrationRequest
    | RegistrationResponse
    | Heartbeat
    | StageAssignmentMessage
    | ActivationRequest
    | ActivationResult
    | RoutePlan
    | RouteInstallRequest
    | DataPlaneEnvelope
    | DataPlaneAck
    | FinalResultMessage
    | HealthResponse
    | CancelRequest
    | CacheControlRequest
    | CacheControlResponse
    | Ack
    | WireChunk
    | SubmitRequest
    | SubmitResponse
    | SubmitStreamEvent
)
MessageT = TypeVar("MessageT", bound=StrictModel)


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"__bytes_b64__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _restore(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"__bytes_b64__"}:
        return base64.b64decode(value["__bytes_b64__"], validate=True)
    if isinstance(value, dict):
        return {key: _restore(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore(item) for item in value]
    return value


def to_protobuf(message: StrictModel, *, kind: str | None = None) -> ProtoAny:
    """Pack a validated message in a protobuf ``Any`` wire envelope."""

    payload = message.model_dump(mode="python")
    packed = ProtoAny()
    packed.type_url = f"type.swarm-inference.dev/{kind or type(message).__name__}"
    packed.value = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return packed


def from_protobuf(payload: ProtoAny, model: type[MessageT]) -> MessageT:
    expected = model.__name__
    actual = payload.type_url.rsplit("/", 1)[-1]
    if actual != expected:
        raise IntegrityError(f"wire message kind mismatch: expected={expected} actual={actual}")
    try:
        raw = _restore(json.loads(payload.value.decode("utf-8")))
        return model.model_validate(raw)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"invalid {expected} protobuf envelope: {exc}") from exc


def serialize_message(message: StrictModel) -> bytes:
    return to_protobuf(message).SerializeToString()


def parse_message(data: bytes, model: type[MessageT]) -> MessageT:
    envelope = ProtoAny.FromString(data)
    return from_protobuf(envelope, model)
