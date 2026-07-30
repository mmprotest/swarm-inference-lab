"""Typed control-plane and activation-envelope messages."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal, TypeVar

from google.protobuf.any_pb2 import Any as ProtoAny
from pydantic import Field, model_validator

from swarm_inference.config.models import (
    ModelManifest,
    OperationKind,
    StageDefinition,
    StrictModel,
    SyntheticModelConfig,
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


class ActivationMetadata(StrictModel):
    request_id: str
    tensor_id: str
    stage_id: int = Field(ge=0)
    operation: OperationKind
    token_position: int = Field(ge=0)
    sequence_length: int = Field(gt=0)
    cache_generation: int = Field(ge=0)
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

    @model_validator(mode="after")
    def require_prompt(self) -> SubmitRequest:
        if self.prompt is None and not self.prompt_token_ids:
            raise ValueError("either prompt or prompt_token_ids is required")
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


WireMessage = (
    RegistrationRequest
    | RegistrationResponse
    | Heartbeat
    | StageAssignmentMessage
    | ActivationRequest
    | ActivationResult
    | HealthResponse
    | CancelRequest
    | Ack
    | WireChunk
    | SubmitRequest
    | SubmitResponse
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
