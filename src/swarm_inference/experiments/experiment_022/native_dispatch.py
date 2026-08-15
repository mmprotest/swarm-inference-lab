"""Authenticated production EXECUTE_SHARD dispatch for persistent K3 workers.

Unlike the E020/E021 lifecycle mock, this path cannot manufacture a placeholder
result.  Every accepted assignment must have a resident primitive registered at
startup and dispatch increments that primitive's native invocation receipt.
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Protocol

import numpy as np

from swarm_inference.experiments.experiment_020.transport import Frame, MessageType


class ShardTaskType(StrEnum):
    WHOLE_LAYER = "WHOLE_LAYER"
    KDA_SHARD = "KDA_SHARD"
    MLA_SHARD = "MLA_SHARD"
    EXPERT_STRIPE = "EXPERT_STRIPE"
    WHOLE_EXPERT_GROUP = "WHOLE_EXPERT_GROUP"
    SHARED_EXPERT_SHARD = "SHARED_EXPERT_SHARD"
    PROJECTION_SHARD = "PROJECTION_SHARD"
    REDUCTION_CONTRIBUTION = "REDUCTION_CONTRIBUTION"
    ORDERED_LAYER_DAG = "ORDERED_LAYER_DAG"
    ENDPOINT = "ENDPOINT"


@dataclass(frozen=True, slots=True)
class ShardRequest:
    assignment_id: str
    task_type: ShardTaskType
    layer: int
    shard_index: int
    degree: int
    rows: int
    input_shape: tuple[int, ...]
    input_dtype: str = "float32"
    state_id: str = ""
    exact: bool = True
    whole_layer_fallback: bool = False

    def __post_init__(self) -> None:
        if not self.assignment_id or self.layer < 0 or self.shard_index < 0:
            raise ValueError("invalid shard assignment identity")
        if self.degree <= 0 or self.shard_index >= self.degree or self.rows <= 0:
            raise ValueError("invalid shard geometry")
        if not self.exact or self.whole_layer_fallback:
            raise ValueError("E022 production shard dispatch forbids mathematical fallback")
        if self.input_dtype != "float32":
            raise ValueError("E022 native boundary currently requires float32")


@dataclass(frozen=True, slots=True)
class ShardResult:
    assignment_id: str
    task_type: ShardTaskType
    output_shape: tuple[int, ...]
    output_dtype: str
    output_sha256: str
    native_primitive: str
    native_invocation_count: int
    wall_ms: float
    state_id: str


class ResidentPrimitive(Protocol):
    native_primitive: str
    native: bool
    invocation_count: int

    def __call__(self, values: np.ndarray, request: ShardRequest) -> np.ndarray: ...


@dataclass(slots=True)
class CallableResidentPrimitive:
    """Bind a prepared native handle/callable to one immutable assignment."""

    native_primitive: str
    operation: Callable[[np.ndarray, ShardRequest], np.ndarray]
    native: bool = True
    invocation_count: int = 0

    def __call__(self, values: np.ndarray, request: ShardRequest) -> np.ndarray:
        output = self.operation(values, request)
        self.invocation_count += 1
        return np.ascontiguousarray(output, dtype=np.float32)


def _pack(metadata: dict[str, Any], payload: bytes) -> bytes:
    header = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return struct.pack("!I", len(header)) + header + payload


def _unpack(payload: bytes) -> tuple[dict[str, Any], bytes]:
    if len(payload) < 4:
        raise ValueError("shard payload is missing its metadata length")
    length = struct.unpack("!I", payload[:4])[0]
    if length <= 0 or 4 + length > len(payload):
        raise ValueError("shard payload has an invalid metadata length")
    metadata = json.loads(payload[4 : 4 + length])
    if not isinstance(metadata, dict):
        raise ValueError("shard metadata must be an object")
    return metadata, payload[4 + length :]


def encode_shard_request(request: ShardRequest, values: np.ndarray) -> bytes:
    source = np.ascontiguousarray(values, dtype=np.float32)
    if tuple(source.shape) != request.input_shape:
        raise ValueError("request shape does not match its binary input")
    return _pack(asdict(request), source.tobytes())


def decode_shard_result(payload: bytes) -> tuple[ShardResult, np.ndarray]:
    metadata, raw = _unpack(payload)
    result = ShardResult(
        assignment_id=str(metadata["assignment_id"]),
        task_type=ShardTaskType(metadata["task_type"]),
        output_shape=tuple(int(value) for value in metadata["output_shape"]),
        output_dtype=str(metadata["output_dtype"]),
        output_sha256=str(metadata["output_sha256"]),
        native_primitive=str(metadata["native_primitive"]),
        native_invocation_count=int(metadata["native_invocation_count"]),
        wall_ms=float(metadata["wall_ms"]),
        state_id=str(metadata["state_id"]),
    )
    values = np.frombuffer(raw, dtype="<f4").reshape(result.output_shape).copy()
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    if digest != result.output_sha256:
        raise ValueError("SHARD_RESULT output digest mismatch")
    return result, values


class NativeShardDispatcher:
    """Dispatch authenticated frames only to prepared resident primitives."""

    def __init__(self, worker_id: str) -> None:
        if not worker_id:
            raise ValueError("native dispatcher requires worker identity")
        self.worker_id = worker_id
        self._primitives: dict[tuple[str, ShardTaskType], ResidentPrimitive] = {}
        self.audit: list[dict[str, Any]] = []

    def register(
        self,
        assignment_id: str,
        task_type: ShardTaskType,
        primitive: ResidentPrimitive,
    ) -> None:
        if not primitive.native:
            raise ValueError("production assignments require a native primitive")
        key = (assignment_id, task_type)
        if key in self._primitives:
            raise ValueError(f"duplicate resident assignment {key}")
        self._primitives[key] = primitive

    @property
    def assignment_count(self) -> int:
        return len(self._primitives)

    def execute_payload(self, payload: bytes) -> bytes:
        metadata, raw = _unpack(payload)
        request = ShardRequest(
            assignment_id=str(metadata["assignment_id"]),
            task_type=ShardTaskType(metadata["task_type"]),
            layer=int(metadata["layer"]),
            shard_index=int(metadata["shard_index"]),
            degree=int(metadata["degree"]),
            rows=int(metadata["rows"]),
            input_shape=tuple(int(value) for value in metadata["input_shape"]),
            input_dtype=str(metadata.get("input_dtype", "float32")),
            state_id=str(metadata.get("state_id", "")),
            exact=bool(metadata.get("exact", True)),
            whole_layer_fallback=bool(metadata.get("whole_layer_fallback", False)),
        )
        expected_bytes = int(np.prod(request.input_shape, dtype=np.int64)) * 4
        if len(raw) != expected_bytes:
            raise ValueError("EXECUTE_SHARD binary input length mismatch")
        try:
            primitive = self._primitives[(request.assignment_id, request.task_type)]
        except KeyError as exc:
            raise KeyError(
                f"worker {self.worker_id} has no resident native primitive for "
                f"{request.assignment_id}/{request.task_type.value}"
            ) from exc
        values = np.frombuffer(raw, dtype="<f4").reshape(request.input_shape)
        before = primitive.invocation_count
        started = time.perf_counter_ns()
        output = primitive(values, request)
        wall_ms = (time.perf_counter_ns() - started) / 1e6
        if primitive.invocation_count != before + 1:
            raise RuntimeError("resident primitive did not record a native invocation")
        output = np.ascontiguousarray(output, dtype=np.float32)
        output_raw = output.tobytes()
        result = ShardResult(
            assignment_id=request.assignment_id,
            task_type=request.task_type,
            output_shape=tuple(int(value) for value in output.shape),
            output_dtype="float32",
            output_sha256="sha256:" + hashlib.sha256(output_raw).hexdigest(),
            native_primitive=primitive.native_primitive,
            native_invocation_count=primitive.invocation_count,
            wall_ms=wall_ms,
            state_id=request.state_id,
        )
        self.audit.append(
            {
                "worker_id": self.worker_id,
                "assignment_id": request.assignment_id,
                "task_type": request.task_type.value,
                "native_primitive": primitive.native_primitive,
                "wall_ms": wall_ms,
                "input_bytes": len(raw),
                "output_bytes": len(output_raw),
                "whole_layer_fallback": False,
                "native_invocation_count": primitive.invocation_count,
            }
        )
        return _pack(asdict(result), output_raw)

    def execute_frame(self, frame: Frame) -> Frame:
        if frame.message_type is not MessageType.EXECUTE_SHARD:
            raise ValueError("native dispatcher accepts only EXECUTE_SHARD")
        if frame.worker_id != self.worker_id:
            raise ValueError("EXECUTE_SHARD addressed to a different worker")
        payload = self.execute_payload(frame.payload)
        return Frame(
            MessageType.SHARD_RESULT,
            frame.request_id,
            frame.chunk_id,
            frame.worker_id,
            frame.state_id,
            payload,
        )


__all__ = [
    "CallableResidentPrimitive",
    "NativeShardDispatcher",
    "ResidentPrimitive",
    "ShardRequest",
    "ShardResult",
    "ShardTaskType",
    "decode_shard_result",
    "encode_shard_request",
]
