"""Versioned binary messages for direct stage-ring communication.

Frames use a fixed little-endian header, canonical JSON metadata, and a
separate opaque payload.  This module deliberately provides integrity and
ordering checks only; authentication, encryption, and network policy belong
at higher layers.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import struct
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Protocol

STAGE_RING_PROTOCOL_VERSION = 1
"""Current product stage-ring wire-protocol version."""

# Compatibility-friendly short name for callers that deal only in this
# protocol.  The authoritative name above makes its scope unambiguous.
PROTOCOL_VERSION = STAGE_RING_PROTOCOL_VERSION
MAGIC = b"SWRING01"
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_PAYLOAD_BYTES = 1024 * 1024 * 1024

# magic, version, operation, flags, metadata length, payload length, sequence,
# token position, source stage, destination stage, sha256(metadata || payload)
HEADER = struct.Struct("<8sHHIIQQihh32s")


class Operation(IntEnum):
    """Explicit stage-ring operation codes."""

    HELLO = 1
    CAPABILITIES = 2
    LOAD_STAGE = 3
    OPEN_SESSION = 4
    PREFILL = 5
    DECODE = 6
    VERIFY_CANDIDATES = 7
    TOKEN_RESULT = 8
    SESSION_CHECKPOINT = 9
    CLOSE_SESSION = 10
    CANCEL_SESSION = 11
    HEALTH = 12
    ERROR = 13


CONTROL_OPERATIONS = frozenset(
    {
        Operation.HELLO,
        Operation.CAPABILITIES,
        Operation.LOAD_STAGE,
        Operation.OPEN_SESSION,
        Operation.SESSION_CHECKPOINT,
        Operation.CLOSE_SESSION,
        Operation.CANCEL_SESSION,
        Operation.HEALTH,
        Operation.ERROR,
    }
)

_PRE_SESSION_OPERATIONS = frozenset({Operation.HELLO, Operation.CAPABILITIES, Operation.LOAD_STAGE})
_METADATA_FIELDS = frozenset(
    {
        "protocol_version",
        "model_revision",
        "tokenizer_revision",
        "stage_topology_id",
        "stage_id",
        "layer_range",
        "session_id",
        "request_id",
        "sequence_number",
        "token_position",
        "source_stage",
        "destination_stage",
        "message_type",
        "tensor_shape",
        "tensor_dtype",
        "compression_mode",
        "payload_length",
        "status",
        "attributes",
    }
)


class SocketLike(Protocol):
    """Small synchronous socket surface required by the frame helpers."""

    def send(self, data: bytes | memoryview) -> int: ...

    def recv_into(self, buffer: memoryview, nbytes: int = 0) -> int: ...


TelemetryCallback = Callable[[str, dict[str, Any]], None]


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"stage {name} must be an integer")
    return value


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"stage {name} must be a non-empty string")
    return value


def _validate_json_value(value: Any, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    raise ValueError(f"{path} contains a non-JSON value of type {type(value).__name__}")


def _validate_stage_index(value: Any, name: str, *, allow_coordinator: bool) -> int:
    stage = _require_int(value, name)
    minimum = -1 if allow_coordinator else 0
    if not minimum <= stage <= 32767:
        raise ValueError(f"stage {name} is outside the signed 16-bit stage range")
    return stage


def _validate_shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("stage tensor shape must be an array")
    shape: list[int] = []
    for dimension in value:
        parsed = _require_int(dimension, "tensor dimension")
        if parsed < 0:
            raise ValueError("stage tensor dimensions cannot be negative")
        shape.append(parsed)
    return tuple(shape)


@dataclass(frozen=True, slots=True)
class StageMessage:
    operation: Operation
    model_revision: str
    tokenizer_revision: str
    topology_id: str
    stage_id: int
    layer_start: int
    layer_end: int
    session_id: str
    request_id: str
    sequence_number: int
    token_position: int
    source_stage: int
    destination_stage: int
    tensor_shape: tuple[int, ...] = ()
    tensor_dtype: str = "none"
    compression_mode: str = "none"
    payload: bytes = b""
    status: str = "OK"
    flags: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        """Return the canonical semantic metadata carried by a frame."""

        return {
            "protocol_version": STAGE_RING_PROTOCOL_VERSION,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "stage_topology_id": self.topology_id,
            "stage_id": self.stage_id,
            "layer_range": [self.layer_start, self.layer_end],
            "session_id": self.session_id,
            "request_id": self.request_id,
            "sequence_number": self.sequence_number,
            "token_position": self.token_position,
            "source_stage": self.source_stage,
            "destination_stage": self.destination_stage,
            "message_type": self.operation.name,
            "tensor_shape": list(self.tensor_shape),
            "tensor_dtype": self.tensor_dtype,
            "compression_mode": self.compression_mode,
            "payload_length": len(self.payload),
            "status": self.status,
            "attributes": self.attributes,
        }


@dataclass(frozen=True, slots=True)
class EncodedFrame:
    frame: bytes
    metadata_bytes: int
    payload_bytes: int
    wire_bytes: int
    checksum: str


@dataclass(frozen=True, slots=True)
class FrameHeader:
    """Validated fixed-header values available before body allocation."""

    operation: Operation
    flags: int
    metadata_length: int
    payload_length: int
    sequence_number: int
    token_position: int
    source_stage: int
    destination_stage: int
    checksum: bytes


def _validate_message(message: StageMessage) -> None:
    if not isinstance(message.operation, Operation):
        raise ValueError("stage operation must be an Operation")
    _require_nonempty_string(message.model_revision, "model revision")
    _require_nonempty_string(message.tokenizer_revision, "tokenizer revision")
    _require_nonempty_string(message.topology_id, "topology identity")
    _validate_stage_index(message.stage_id, "ID", allow_coordinator=False)
    layer_start = _require_int(message.layer_start, "layer start")
    layer_end = _require_int(message.layer_end, "layer end")
    if layer_start < 0 or layer_end <= layer_start:
        raise ValueError("stage layer range must be a non-empty half-open interval")
    _require_nonempty_string(message.session_id, "session ID")
    _require_nonempty_string(message.request_id, "request ID")
    sequence = _require_int(message.sequence_number, "sequence number")
    if not 0 <= sequence <= (1 << 64) - 1:
        raise ValueError("stage sequence number is outside the unsigned 64-bit range")
    token_position = _require_int(message.token_position, "token position")
    if not -1 <= token_position <= (1 << 31) - 1:
        raise ValueError("stage token position is outside the signed 32-bit protocol range")
    _validate_stage_index(message.source_stage, "source", allow_coordinator=True)
    _validate_stage_index(message.destination_stage, "destination", allow_coordinator=True)
    _validate_shape(message.tensor_shape)
    _require_nonempty_string(message.tensor_dtype, "tensor dtype")
    _require_nonempty_string(message.compression_mode, "compression mode")
    _require_nonempty_string(message.status, "status")
    flags = _require_int(message.flags, "flags")
    if not 0 <= flags <= (1 << 32) - 1:
        raise ValueError("stage flags are outside the unsigned 32-bit range")
    if not isinstance(message.payload, bytes):
        raise ValueError("stage payload must be bytes")
    if not isinstance(message.attributes, dict):
        raise ValueError("stage attributes must be an object")
    _validate_json_value(message.attributes, path="stage attributes")


def encode_message(message: StageMessage) -> EncodedFrame:
    """Encode one message using canonical JSON and the fixed frame header."""

    _validate_message(message)
    metadata = json.dumps(
        message.metadata(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(metadata) > MAX_METADATA_BYTES:
        raise ValueError("stage metadata exceeds the protocol limit")
    if len(message.payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("stage payload exceeds the protocol limit")
    digest = hashlib.sha256(metadata + message.payload).digest()
    header = HEADER.pack(
        MAGIC,
        STAGE_RING_PROTOCOL_VERSION,
        int(message.operation),
        message.flags,
        len(metadata),
        len(message.payload),
        message.sequence_number,
        message.token_position,
        message.source_stage,
        message.destination_stage,
        digest,
    )
    frame = header + metadata + message.payload
    return EncodedFrame(
        frame=frame,
        metadata_bytes=len(metadata),
        payload_bytes=len(message.payload),
        wire_bytes=len(frame),
        checksum=digest.hex(),
    )


def _unpack_header(
    header: bytes | bytearray | memoryview,
) -> tuple[int, int, int, int, int, int, int, int, bytes]:
    if len(header) != HEADER.size:
        raise ValueError("truncated stage frame header")
    (
        magic,
        version,
        operation_code,
        flags,
        metadata_length,
        payload_length,
        sequence_number,
        token_position,
        source_stage,
        destination_stage,
        expected_digest,
    ) = HEADER.unpack(header)
    if magic != MAGIC:
        raise ValueError("invalid stage frame magic")
    if version != STAGE_RING_PROTOCOL_VERSION:
        raise ValueError(f"unsupported stage protocol version {version}")
    try:
        Operation(operation_code)
    except ValueError as exc:
        raise ValueError(f"unsupported stage operation {operation_code}") from exc
    if metadata_length > MAX_METADATA_BYTES or payload_length > MAX_PAYLOAD_BYTES:
        raise ValueError("stage frame declares an oversized component")
    if sequence_number > (1 << 64) - 1:
        raise ValueError("invalid stage sequence number")
    _validate_stage_index(source_stage, "source", allow_coordinator=True)
    _validate_stage_index(destination_stage, "destination", allow_coordinator=True)
    return (
        operation_code,
        flags,
        metadata_length,
        payload_length,
        sequence_number,
        token_position,
        source_stage,
        destination_stage,
        expected_digest,
    )


def inspect_frame_header(header: bytes | bytearray | memoryview) -> FrameHeader:
    """Validate a fixed header without allocating its declared frame body."""

    (
        operation_code,
        flags,
        metadata_length,
        payload_length,
        sequence_number,
        token_position,
        source_stage,
        destination_stage,
        expected_digest,
    ) = _unpack_header(header)
    return FrameHeader(
        operation=Operation(operation_code),
        flags=flags,
        metadata_length=metadata_length,
        payload_length=payload_length,
        sequence_number=sequence_number,
        token_position=token_position,
        source_stage=source_stage,
        destination_stage=destination_stage,
        checksum=expected_digest,
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"stage metadata contains invalid JSON constant {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"stage metadata repeats field {key!r}")
        value[key] = item
    return value


def _decode_metadata(metadata_bytes: bytes) -> dict[str, Any]:
    try:
        metadata = json.loads(
            metadata_bytes.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"stage metadata is not valid canonical JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("stage metadata must be an object")
    fields = frozenset(metadata)
    if fields != _METADATA_FIELDS:
        missing = sorted(_METADATA_FIELDS - fields)
        unexpected = sorted(fields - _METADATA_FIELDS)
        raise ValueError(
            f"stage metadata fields do not match the protocol; missing={missing} "
            f"unexpected={unexpected}"
        )
    return metadata


def decode_message(frame: bytes | bytearray | memoryview) -> StageMessage:
    """Validate and decode exactly one complete stage-ring frame."""

    view = memoryview(frame)
    if len(view) < HEADER.size:
        raise ValueError("truncated stage frame header")
    (
        operation_code,
        flags,
        metadata_length,
        payload_length,
        sequence_number,
        token_position,
        source_stage,
        destination_stage,
        expected_digest,
    ) = _unpack_header(view[: HEADER.size])
    expected_size = HEADER.size + metadata_length + payload_length
    if len(view) != expected_size:
        raise ValueError("stage frame length does not match its header")
    payload_start = HEADER.size + metadata_length
    metadata_bytes = bytes(view[HEADER.size : payload_start])
    payload = bytes(view[payload_start:])
    actual_digest = hashlib.sha256(metadata_bytes + payload).digest()
    if not hmac.compare_digest(expected_digest, actual_digest):
        raise ValueError("stage frame checksum mismatch")
    metadata = _decode_metadata(metadata_bytes)
    operation = Operation(operation_code)
    if metadata["message_type"] != operation.name:
        raise ValueError("stage operation disagrees with semantic metadata")
    if (
        _require_int(metadata["protocol_version"], "protocol version")
        != STAGE_RING_PROTOCOL_VERSION
    ):
        raise ValueError("stage protocol version disagrees with semantic metadata")
    if _require_int(metadata["payload_length"], "payload length") != payload_length:
        raise ValueError("stage payload length disagrees with semantic metadata")
    if _require_int(metadata["sequence_number"], "sequence number") != sequence_number:
        raise ValueError("stage sequence disagrees with semantic metadata")
    if _require_int(metadata["token_position"], "token position") != token_position:
        raise ValueError("stage token position disagrees with semantic metadata")
    if _require_int(metadata["source_stage"], "source") != source_stage:
        raise ValueError("stage source disagrees with semantic metadata")
    if _require_int(metadata["destination_stage"], "destination") != destination_stage:
        raise ValueError("stage destination disagrees with semantic metadata")
    layer_range = metadata["layer_range"]
    if not isinstance(layer_range, list) or len(layer_range) != 2:
        raise ValueError("stage layer range is missing or malformed")
    layer_start = _require_int(layer_range[0], "layer start")
    layer_end = _require_int(layer_range[1], "layer end")
    shape = _validate_shape(metadata["tensor_shape"])
    attributes = metadata["attributes"]
    if not isinstance(attributes, dict):
        raise ValueError("stage attributes must be an object")
    _validate_json_value(attributes, path="stage attributes")
    message = StageMessage(
        operation=operation,
        model_revision=_require_nonempty_string(metadata["model_revision"], "model revision"),
        tokenizer_revision=_require_nonempty_string(
            metadata["tokenizer_revision"], "tokenizer revision"
        ),
        topology_id=_require_nonempty_string(metadata["stage_topology_id"], "topology identity"),
        stage_id=_validate_stage_index(metadata["stage_id"], "ID", allow_coordinator=False),
        layer_start=layer_start,
        layer_end=layer_end,
        session_id=_require_nonempty_string(metadata["session_id"], "session ID"),
        request_id=_require_nonempty_string(metadata["request_id"], "request ID"),
        sequence_number=sequence_number,
        token_position=token_position,
        source_stage=source_stage,
        destination_stage=destination_stage,
        tensor_shape=shape,
        tensor_dtype=_require_nonempty_string(metadata["tensor_dtype"], "tensor dtype"),
        compression_mode=_require_nonempty_string(metadata["compression_mode"], "compression mode"),
        payload=payload,
        status=_require_nonempty_string(metadata["status"], "status"),
        flags=flags,
        attributes=attributes,
    )
    _validate_message(message)
    return message


class BufferPool:
    """Bounded reusable bytearray pool for socket receive buffers."""

    def __init__(self, *, capacity: int = 8, initial_size: int = 64 * 1024) -> None:
        if capacity < 1 or initial_size < HEADER.size:
            raise ValueError("invalid buffer-pool geometry")
        self.capacity = capacity
        self.initial_size = initial_size
        self._buffers: deque[bytearray] = deque(bytearray(initial_size) for _ in range(capacity))
        self._condition = threading.Condition()
        self.allocations = capacity
        self.reuses = 0

    def acquire(self, minimum_size: int, timeout_s: float = 30.0) -> bytearray:
        if minimum_size < 0 or minimum_size > MAX_METADATA_BYTES + MAX_PAYLOAD_BYTES:
            raise ValueError("invalid receive buffer size")
        if timeout_s < 0:
            raise ValueError("buffer-pool timeout cannot be negative")
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while not self._buffers:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._condition.wait(remaining):
                    raise TimeoutError("receive buffer pool exhausted")
            buffer = self._buffers.popleft()
            self.reuses += 1
        if len(buffer) < minimum_size:
            buffer.extend(b"\0" * (minimum_size - len(buffer)))
        return buffer

    def release(self, buffer: bytearray) -> None:
        if not isinstance(buffer, bytearray):
            raise TypeError("receive buffer must be a bytearray")
        with self._condition:
            if len(self._buffers) >= self.capacity:
                raise RuntimeError("receive buffer returned more than once")
            self._buffers.append(buffer)
            self._condition.notify()


def _recv_exact_into(connection: SocketLike, target: memoryview, byte_count: int) -> None:
    offset = 0
    while offset < byte_count:
        received = connection.recv_into(target[offset:byte_count], byte_count - offset)
        if received <= 0:
            raise ConnectionError("stage socket closed before the frame completed")
        if received > byte_count - offset:
            raise ValueError("stage socket reported an impossible receive length")
        offset += received


def recv_message(
    connection: SocketLike,
    *,
    pool: BufferPool | None = None,
    telemetry: TelemetryCallback | None = None,
) -> StageMessage:
    """Receive one frame, handling partial reads and bounded buffer reuse."""

    started = time.perf_counter_ns()
    header = bytearray(HEADER.size)
    _recv_exact_into(connection, memoryview(header), HEADER.size)
    unpacked = _unpack_header(header)
    metadata_length = unpacked[2]
    payload_length = unpacked[3]
    remaining = metadata_length + payload_length
    receive_pool = pool or BufferPool(capacity=1, initial_size=max(remaining, HEADER.size))
    buffer = receive_pool.acquire(remaining)
    try:
        _recv_exact_into(connection, memoryview(buffer), remaining)
        frame = bytes(header) + bytes(memoryview(buffer)[:remaining])
    finally:
        receive_pool.release(buffer)
    message = decode_message(frame)
    if telemetry is not None:
        telemetry(
            "socket_receive_end",
            {
                "duration_ns": time.perf_counter_ns() - started,
                "wire_bytes": len(frame),
                "payload_bytes": len(message.payload),
                "message_type": message.operation.name,
                "sequence_number": message.sequence_number,
            },
        )
    return message


def send_encoded_frame(
    connection: SocketLike,
    encoded: EncodedFrame,
    *,
    telemetry: TelemetryCallback | None = None,
) -> None:
    """Write a pre-encoded frame, handling partial socket writes."""

    started = time.perf_counter_ns()
    view = memoryview(encoded.frame)
    sent = 0
    while sent < len(view):
        count = connection.send(view[sent:])
        if count <= 0:
            raise ConnectionError("stage socket closed during send")
        if count > len(view) - sent:
            raise ValueError("stage socket reported an impossible send length")
        sent += count
    if telemetry is not None:
        telemetry(
            "socket_send_end",
            {
                "socket_ns": time.perf_counter_ns() - started,
                "wire_bytes": encoded.wire_bytes,
                "payload_bytes": encoded.payload_bytes,
                "checksum": encoded.checksum,
            },
        )


def send_message(
    connection: SocketLike,
    message: StageMessage,
    *,
    telemetry: TelemetryCallback | None = None,
) -> EncodedFrame:
    """Encode and write one message without applying network policy."""

    serialisation_started = time.perf_counter_ns()
    encoded = encode_message(message)
    serialisation_ns = time.perf_counter_ns() - serialisation_started
    captured: dict[str, Any] = {}

    def capture(_: str, values: dict[str, Any]) -> None:
        captured.update(values)

    send_encoded_frame(connection, encoded, telemetry=capture)
    if telemetry is not None:
        telemetry(
            "socket_send_end",
            {
                "serialisation_ns": serialisation_ns,
                "socket_ns": int(captured["socket_ns"]),
                "wire_bytes": encoded.wire_bytes,
                "payload_bytes": encoded.payload_bytes,
                "message_type": message.operation.name,
                "sequence_number": message.sequence_number,
                "checksum": encoded.checksum,
            },
        )
    return encoded


class MessageSequenceValidator:
    """Reject duplicates, stale frames, and gaps per directed session edge."""

    def __init__(self) -> None:
        self._last: dict[tuple[str, int, int], int] = {}
        self._lock = threading.Lock()

    def validate(self, message: StageMessage) -> None:
        key = (message.session_id, message.source_stage, message.destination_stage)
        with self._lock:
            previous = self._last.get(key)
            if previous is not None:
                if message.sequence_number <= previous:
                    reason = "duplicate" if message.sequence_number == previous else "stale"
                    raise ValueError(f"{reason} stage message sequence")
                if message.sequence_number != previous + 1:
                    raise ValueError("out-of-order stage message sequence")
            self._last[key] = message.sequence_number

    def reset_session(self, session_id: str) -> None:
        with self._lock:
            for key in [key for key in self._last if key[0] == session_id]:
                del self._last[key]


class SessionValidator:
    """Track active sessions and enforce model, topology, and optional route identity."""

    def __init__(
        self,
        *,
        model_revision: str,
        topology_id: str,
        tokenizer_revision: str | None = None,
        stage_id: int | None = None,
        layer_range: tuple[int, int] | None = None,
    ) -> None:
        self.model_revision = _require_nonempty_string(model_revision, "model revision")
        self.topology_id = _require_nonempty_string(topology_id, "topology identity")
        self.tokenizer_revision = tokenizer_revision
        self.stage_id = stage_id
        self.layer_range = layer_range
        if tokenizer_revision is not None:
            _require_nonempty_string(tokenizer_revision, "tokenizer revision")
        if stage_id is not None:
            _validate_stage_index(stage_id, "ID", allow_coordinator=False)
        if layer_range is not None:
            start, end = layer_range
            if start < 0 or end <= start:
                raise ValueError("invalid expected stage layer range")
        self._sessions: set[str] = set()
        self._lock = threading.Lock()

    def open(self, session_id: str) -> None:
        _require_nonempty_string(session_id, "session ID")
        with self._lock:
            if session_id in self._sessions:
                raise ValueError("session is already open")
            self._sessions.add(session_id)

    def close(self, session_id: str) -> None:
        with self._lock:
            if session_id not in self._sessions:
                raise ValueError("session is not open")
            self._sessions.remove(session_id)

    def validate(self, message: StageMessage) -> None:
        if message.model_revision != self.model_revision:
            raise ValueError("wrong model revision")
        if (
            self.tokenizer_revision is not None
            and message.tokenizer_revision != self.tokenizer_revision
        ):
            raise ValueError("wrong tokenizer revision")
        if message.topology_id != self.topology_id:
            raise ValueError("wrong stage topology")
        if self.stage_id is not None and (
            message.stage_id != self.stage_id or message.destination_stage != self.stage_id
        ):
            raise ValueError("wrong destination stage")
        if (
            self.layer_range is not None
            and (
                message.layer_start,
                message.layer_end,
            )
            != self.layer_range
        ):
            raise ValueError("wrong destination layer ownership")
        if message.operation not in _PRE_SESSION_OPERATIONS:
            with self._lock:
                if message.session_id not in self._sessions:
                    raise ValueError("wrong or closed session")


def message_wire_identity(message: StageMessage) -> str:
    """Return a stable identity linking send and receive trace events."""

    value = (
        f"{message.session_id}:{message.source_stage}:{message.destination_stage}:"
        f"{message.sequence_number}:{message.operation.name}:{message.token_position}"
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class SequenceAllocator:
    """Allocate monotonically increasing sequence numbers per directed edge."""

    def __init__(self) -> None:
        self._values: defaultdict[tuple[str, int, int], int] = defaultdict(int)
        self._lock = threading.Lock()

    def next(self, session_id: str, source_stage: int, destination_stage: int) -> int:
        _require_nonempty_string(session_id, "session ID")
        _validate_stage_index(source_stage, "source", allow_coordinator=True)
        _validate_stage_index(destination_stage, "destination", allow_coordinator=True)
        key = (session_id, source_stage, destination_stage)
        with self._lock:
            value = self._values[key]
            self._values[key] += 1
            return value

    def reset_session(self, session_id: str) -> None:
        with self._lock:
            for key in [key for key in self._values if key[0] == session_id]:
                del self._values[key]


__all__ = [
    "CONTROL_OPERATIONS",
    "HEADER",
    "MAGIC",
    "MAX_METADATA_BYTES",
    "MAX_PAYLOAD_BYTES",
    "PROTOCOL_VERSION",
    "STAGE_RING_PROTOCOL_VERSION",
    "BufferPool",
    "EncodedFrame",
    "FrameHeader",
    "MessageSequenceValidator",
    "Operation",
    "SequenceAllocator",
    "SessionValidator",
    "SocketLike",
    "StageMessage",
    "TelemetryCallback",
    "decode_message",
    "encode_message",
    "inspect_frame_header",
    "message_wire_identity",
    "recv_message",
    "send_encoded_frame",
    "send_message",
]
