"""Versioned binary protocol for Experiment 011 stage-ring messages.

The fixed little-endian header is intentionally small and boring.  Semantic
fields are canonical JSON and tensor data is a separate contiguous byte blob;
pickle is never accepted on this path.
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

from swarm_inference.experiments.experiment_011 import PROTOCOL_VERSION

MAGIC = b"SWRING11"
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_PAYLOAD_BYTES = 1024 * 1024 * 1024

# magic, version, operation, flags, metadata length, payload length, sequence,
# token position, source stage, destination stage, sha256(metadata || payload)
HEADER = struct.Struct("<8sHHIIQQihh32s")


class Operation(IntEnum):
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


CONTROL_OPERATIONS = {
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


class SocketLike(Protocol):
    def send(self, data: bytes | memoryview) -> int: ...

    def recv_into(self, buffer: memoryview, nbytes: int = 0) -> int: ...


TelemetryCallback = Callable[[str, dict[str, Any]], None]


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
        return {
            "protocol_version": PROTOCOL_VERSION,
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


def encode_message(message: StageMessage) -> EncodedFrame:
    if message.sequence_number < 0:
        raise ValueError("sequence number must be non-negative")
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
        PROTOCOL_VERSION,
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


def decode_message(frame: bytes | bytearray | memoryview) -> StageMessage:
    view = memoryview(frame)
    if len(view) < HEADER.size:
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
    ) = HEADER.unpack_from(view)
    if magic != MAGIC:
        raise ValueError("invalid stage frame magic")
    if version != PROTOCOL_VERSION:
        raise ValueError(f"unsupported stage protocol version {version}")
    if metadata_length > MAX_METADATA_BYTES or payload_length > MAX_PAYLOAD_BYTES:
        raise ValueError("stage frame declares an oversized component")
    expected_size = HEADER.size + metadata_length + payload_length
    if len(view) != expected_size:
        raise ValueError("stage frame length does not match its header")
    metadata_start = HEADER.size
    payload_start = metadata_start + metadata_length
    metadata_bytes = bytes(view[metadata_start:payload_start])
    payload = bytes(view[payload_start:])
    actual_digest = hashlib.sha256(metadata_bytes + payload).digest()
    if not hmac.compare_digest(expected_digest, actual_digest):
        raise ValueError("stage frame checksum mismatch")
    metadata = json.loads(metadata_bytes.decode("utf-8"))
    operation = Operation(operation_code)
    if metadata.get("message_type") != operation.name:
        raise ValueError("stage operation disagrees with semantic metadata")
    if int(metadata.get("protocol_version", -1)) != version:
        raise ValueError("stage protocol version disagrees with semantic metadata")
    if int(metadata.get("payload_length", -1)) != payload_length:
        raise ValueError("stage payload length disagrees with semantic metadata")
    if int(metadata.get("sequence_number", -1)) != sequence_number:
        raise ValueError("stage sequence disagrees with semantic metadata")
    if int(metadata.get("token_position", -2)) != token_position:
        raise ValueError("stage token position disagrees with semantic metadata")
    if int(metadata.get("source_stage", -2)) != source_stage:
        raise ValueError("stage source disagrees with semantic metadata")
    if int(metadata.get("destination_stage", -2)) != destination_stage:
        raise ValueError("stage destination disagrees with semantic metadata")
    layer_range = metadata.get("layer_range")
    if not isinstance(layer_range, list) or len(layer_range) != 2:
        raise ValueError("stage layer range is missing or malformed")
    shape = metadata.get("tensor_shape", [])
    if not isinstance(shape, list) or any(int(value) < 0 for value in shape):
        raise ValueError("stage tensor shape is malformed")
    attributes = metadata.get("attributes", {})
    if not isinstance(attributes, dict):
        raise ValueError("stage attributes must be an object")
    return StageMessage(
        operation=operation,
        model_revision=str(metadata["model_revision"]),
        tokenizer_revision=str(metadata["tokenizer_revision"]),
        topology_id=str(metadata["stage_topology_id"]),
        stage_id=int(metadata["stage_id"]),
        layer_start=int(layer_range[0]),
        layer_end=int(layer_range[1]),
        session_id=str(metadata["session_id"]),
        request_id=str(metadata["request_id"]),
        sequence_number=sequence_number,
        token_position=token_position,
        source_stage=source_stage,
        destination_stage=destination_stage,
        tensor_shape=tuple(int(value) for value in shape),
        tensor_dtype=str(metadata.get("tensor_dtype", "none")),
        compression_mode=str(metadata.get("compression_mode", "none")),
        payload=payload,
        status=str(metadata.get("status", "OK")),
        flags=flags,
        attributes=attributes,
    )


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
        offset += received


def recv_message(
    connection: SocketLike,
    *,
    pool: BufferPool | None = None,
    telemetry: TelemetryCallback | None = None,
) -> StageMessage:
    started = time.perf_counter_ns()
    header = bytearray(HEADER.size)
    _recv_exact_into(connection, memoryview(header), HEADER.size)
    unpacked = HEADER.unpack(header)
    metadata_length = int(unpacked[4])
    payload_length = int(unpacked[5])
    if metadata_length > MAX_METADATA_BYTES or payload_length > MAX_PAYLOAD_BYTES:
        raise ValueError("stage socket frame exceeds configured limits")
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


def send_message(
    connection: SocketLike,
    message: StageMessage,
    *,
    shaper: Any | None = None,
    timeout_s: float = 30.0,
    telemetry: TelemetryCallback | None = None,
) -> EncodedFrame:
    serialisation_started = time.perf_counter_ns()
    encoded = encode_message(message)
    serialisation_ns = time.perf_counter_ns() - serialisation_started
    shaping_ns = 0
    if shaper is not None:
        shaping_started = time.perf_counter_ns()
        with shaper.flow(timeout_s):
            shaper.enforce(encoded.wire_bytes, direction="stage_send")
        shaping_ns = time.perf_counter_ns() - shaping_started
    socket_started = time.perf_counter_ns()
    view = memoryview(encoded.frame)
    sent = 0
    while sent < len(view):
        count = connection.send(view[sent:])
        if count <= 0:
            raise ConnectionError("stage socket closed during send")
        sent += count
    socket_ns = time.perf_counter_ns() - socket_started
    if telemetry is not None:
        telemetry(
            "socket_send_end",
            {
                "serialisation_ns": serialisation_ns,
                "shaping_ns": shaping_ns,
                "socket_ns": socket_ns,
                "wire_bytes": encoded.wire_bytes,
                "payload_bytes": encoded.payload_bytes,
                "message_type": message.operation.name,
                "sequence_number": message.sequence_number,
                "checksum": encoded.checksum,
            },
        )
    return encoded


class MessageSequenceValidator:
    """Reject duplicates, stale frames and sequence gaps per directed session edge."""

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
    """Tracks active sessions and enforces topology/model identity."""

    def __init__(self, *, model_revision: str, topology_id: str) -> None:
        self.model_revision = model_revision
        self.topology_id = topology_id
        self._sessions: set[str] = set()
        self._lock = threading.Lock()

    def open(self, session_id: str) -> None:
        if not session_id:
            raise ValueError("session ID cannot be empty")
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
        if message.topology_id != self.topology_id:
            raise ValueError("wrong stage topology")
        if message.operation not in {Operation.HELLO, Operation.CAPABILITIES, Operation.LOAD_STAGE}:
            with self._lock:
                if message.session_id not in self._sessions:
                    raise ValueError("wrong or closed session")


def message_wire_identity(message: StageMessage) -> str:
    """Stable identity used to link send/receive events in a dependency trace."""

    value = (
        f"{message.session_id}:{message.source_stage}:{message.destination_stage}:"
        f"{message.sequence_number}:{message.operation.name}:{message.token_position}"
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class SequenceAllocator:
    def __init__(self) -> None:
        self._values: defaultdict[tuple[str, int, int], int] = defaultdict(int)
        self._lock = threading.Lock()

    def next(self, session_id: str, source_stage: int, destination_stage: int) -> int:
        key = (session_id, source_stage, destination_stage)
        with self._lock:
            value = self._values[key]
            self._values[key] += 1
            return value
