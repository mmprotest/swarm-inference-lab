"""Bounded authenticated E025 wire payloads and persistent TLS connections."""

from __future__ import annotations

import hashlib
import json
import socket
import ssl
import struct
import threading
from collections.abc import Mapping
from contextlib import suppress
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.experiments.experiment_020.transport import (
    MAX_METADATA_BYTES,
    MAX_PAYLOAD_BYTES,
    Frame,
    decode_frame,
    encode_frame,
)

_TRANSPORT_HEADER = struct.Struct("!4sBBHII32s32s")
_PAYLOAD_HEADER = struct.Struct("!I")
_ALLOWED_DTYPES = {"float32": np.dtype("<f4"), "int32": np.dtype("<i4"), "int64": np.dtype("<i8")}


class Action(StrEnum):
    REGISTER = "REGISTER"
    OPEN_SESSION = "OPEN_SESSION"
    EXECUTE_STAGE = "EXECUTE_STAGE"
    EXECUTE_EXPERT_PARTITION = "EXECUTE_EXPERT_PARTITION"
    DISABLE_EXECUTION = "DISABLE_EXECUTION"
    ENABLE_EXECUTION = "ENABLE_EXECUTION"
    CLOSE_SESSION = "CLOSE_SESSION"
    HEALTH = "HEALTH"
    SHUTDOWN = "SHUTDOWN"


def _recv_exact(channel: socket.socket, count: int) -> bytes:
    if count < 0:
        raise ValueError("negative frame read")
    result = bytearray(count)
    view = memoryview(result)
    offset = 0
    while offset < count:
        received = channel.recv_into(view[offset:])
        if received <= 0:
            raise EOFError("E025 connection closed during a frame")
        offset += received
    return bytes(result)


def recv_frame(channel: socket.socket, credential: bytes) -> tuple[Frame, int]:
    header = _recv_exact(channel, _TRANSPORT_HEADER.size)
    _, _, _, _, metadata_length, payload_length, _, _ = _TRANSPORT_HEADER.unpack(header)
    if metadata_length > MAX_METADATA_BYTES or payload_length > MAX_PAYLOAD_BYTES:
        raise ValueError("E025 peer declared an oversized frame")
    body = _recv_exact(channel, metadata_length + payload_length)
    encoded = header + body
    return decode_frame(encoded, credential), len(encoded)


def send_frame(channel: socket.socket, frame: Frame, credential: bytes) -> int:
    encoded = encode_frame(frame, credential)
    channel.sendall(encoded)
    return len(encoded)


def pack_payload(
    action: Action,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray] | None = None,
) -> bytes:
    descriptors: dict[str, dict[str, Any]] = {}
    chunks: list[bytes] = []
    offset = 0
    for name, value in sorted((arrays or {}).items()):
        source = np.asarray(value)
        logical_dtype = source.dtype.name
        if logical_dtype not in _ALLOWED_DTYPES:
            raise ValueError(f"E025 wire rejects array dtype {source.dtype}")
        contiguous = np.ascontiguousarray(source, dtype=_ALLOWED_DTYPES[logical_dtype])
        raw = contiguous.tobytes(order="C")
        descriptors[name] = {
            "dtype": logical_dtype,
            "shape": [int(dimension) for dimension in contiguous.shape],
            "offset": offset,
            "nbytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        chunks.append(raw)
        offset += len(raw)
    header = {
        "action": action.value,
        "metadata": dict(metadata),
        "arrays": descriptors,
        "binary_bytes": offset,
    }
    encoded = json.dumps(
        header,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > MAX_METADATA_BYTES:
        raise ValueError("E025 payload metadata exceeds its bound")
    payload = _PAYLOAD_HEADER.pack(len(encoded)) + encoded + b"".join(chunks)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("E025 payload exceeds its bound")
    return payload


def unpack_payload(payload: bytes) -> tuple[Action, dict[str, Any], dict[str, np.ndarray]]:
    if len(payload) < _PAYLOAD_HEADER.size:
        raise ValueError("E025 payload is truncated")
    header_length = _PAYLOAD_HEADER.unpack_from(payload)[0]
    if header_length <= 0 or header_length > MAX_METADATA_BYTES:
        raise ValueError("E025 payload header length is invalid")
    binary_start = _PAYLOAD_HEADER.size + header_length
    if binary_start > len(payload):
        raise ValueError("E025 payload header is truncated")
    try:
        header = json.loads(payload[_PAYLOAD_HEADER.size:binary_start])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("E025 payload header is invalid JSON") from exc
    if not isinstance(header, dict) or set(header) != {
        "action",
        "metadata",
        "arrays",
        "binary_bytes",
    }:
        raise ValueError("E025 payload header fields are invalid")
    metadata = header["metadata"]
    descriptors = header["arrays"]
    if not isinstance(metadata, dict) or not isinstance(descriptors, dict):
        raise ValueError("E025 payload metadata or arrays are invalid")
    raw = memoryview(payload)[binary_start:]
    if int(header["binary_bytes"]) != len(raw):
        raise ValueError("E025 payload binary length differs")
    arrays: dict[str, np.ndarray] = {}
    covered = 0
    for name, descriptor in sorted(descriptors.items()):
        if not isinstance(name, str) or not isinstance(descriptor, dict):
            raise ValueError("E025 array descriptor is invalid")
        dtype_name = str(descriptor["dtype"])
        try:
            dtype = _ALLOWED_DTYPES[dtype_name]
        except KeyError as exc:
            raise ValueError(f"E025 array dtype is unsupported: {dtype_name}") from exc
        offset = int(descriptor["offset"])
        nbytes = int(descriptor["nbytes"])
        shape = tuple(int(value) for value in descriptor["shape"])
        if offset != covered or nbytes < 0 or offset + nbytes > len(raw):
            raise ValueError("E025 array ranges are not canonical and contiguous")
        item_count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if item_count * dtype.itemsize != nbytes:
            raise ValueError("E025 array shape and byte count differ")
        encoded = bytes(raw[offset : offset + nbytes])
        if hashlib.sha256(encoded).hexdigest() != str(descriptor["sha256"]):
            raise ValueError("E025 array digest differs")
        arrays[name] = np.frombuffer(encoded, dtype=dtype).reshape(shape).copy()
        covered += nbytes
    if covered != len(raw):
        raise ValueError("E025 payload contains unowned binary bytes")
    return Action(str(header["action"])), metadata, arrays


class AuthenticatedConnection:
    """One pinned-certificate persistent data-plane connection."""

    def __init__(
        self,
        host: str,
        port: int,
        credential: bytes,
        certificate: Path,
        *,
        timeout_seconds: float = 60.0,
    ) -> None:
        if len(credential) < 32:
            raise ValueError("E025 run credential must contain at least 32 bytes")
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=str(certificate))
        context.check_hostname = False
        raw = socket.create_connection((host, port), timeout=timeout_seconds)
        raw.settimeout(timeout_seconds)
        self.channel = context.wrap_socket(raw, server_hostname=None)
        self.credential = credential
        self.lock = threading.Lock()
        self.sent_bytes = 0
        self.received_bytes = 0
        self.closed = False

    def request(self, frame: Frame) -> Frame:
        if self.closed:
            raise RuntimeError("E025 connection is closed")
        with self.lock:
            self.sent_bytes += send_frame(self.channel, frame, self.credential)
            response, received = recv_frame(self.channel, self.credential)
            self.received_bytes += received
            if response.request_id != frame.request_id or response.worker_id != frame.worker_id:
                raise ValueError("E025 response identity differs from its request")
            return response

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        with suppress(OSError):
            self.channel.shutdown(socket.SHUT_RDWR)
        self.channel.close()

    def __enter__(self) -> AuthenticatedConnection:
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def server_ssl_context(certificate: Path, private_key: Path) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(str(certificate), str(private_key))
    return context


__all__ = [
    "Action",
    "AuthenticatedConnection",
    "pack_payload",
    "recv_frame",
    "send_frame",
    "server_ssl_context",
    "unpack_payload",
]
