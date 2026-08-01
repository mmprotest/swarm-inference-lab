"""Language-neutral binary activation-tensor codec.

The format is:

``SWARMT01`` | uint32 big-endian JSON-header length | canonical UTF-8 JSON |
raw C-order tensor bytes.

The header carries dtype, shape, byte order, identifiers, token position,
sequence length, payload length, and SHA-256 checksum. No pickle or Python
object serialization is involved.
"""

from __future__ import annotations

import json
import struct
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from swarm_inference.exceptions import IntegrityError
from swarm_inference.protocol.checksums import sha256_bytes

MAGIC = b"SWARMT01"
HEADER_LENGTH = struct.Struct(">I")


@dataclass(frozen=True, slots=True)
class ActivationTensor:
    tensor_id: str
    request_id: str
    stage_id: int
    token_position: int
    sequence_length: int
    array: np.ndarray
    logical_dtype: str | None = None
    model_revision: str = ""
    partition_hash: str = ""
    route_generation: int = 0


@dataclass(frozen=True, slots=True)
class TensorChunk:
    tensor_id: str
    chunk_index: int
    chunk_count: int
    total_length: int
    payload: bytes
    checksum: str


def _canonical_header(header: dict[str, Any]) -> bytes:
    return json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")


def encode_tensor(tensor: ActivationTensor) -> bytes:
    array = np.ascontiguousarray(tensor.array)
    raw = array.tobytes(order="C")
    dtype = array.dtype
    logical_dtype = tensor.logical_dtype
    if logical_dtype == "bfloat16" and dtype != np.dtype(np.uint16):
        raise IntegrityError("bfloat16 transport requires a uint16 raw-bit array")
    byte_order = dtype.byteorder
    if byte_order == "=":
        byte_order = "<" if sys.byteorder == "little" else ">"
    if byte_order == "|":
        byte_order = "<"
    header = {
        "dtype": logical_dtype or dtype.str,
        "shape": list(array.shape),
        "strides": [int(value) for value in array.strides],
        "byte_order": "little" if byte_order == "<" else "big",
        "tensor_id": tensor.tensor_id,
        "request_id": tensor.request_id,
        "stage_id": tensor.stage_id,
        "token_position": tensor.token_position,
        "sequence_length": tensor.sequence_length,
        "model_revision": tensor.model_revision,
        "partition_hash": tensor.partition_hash,
        "route_generation": tensor.route_generation,
        "payload_length": len(raw),
        "checksum": sha256_bytes(raw),
    }
    encoded_header = _canonical_header(header)
    return MAGIC + HEADER_LENGTH.pack(len(encoded_header)) + encoded_header + raw


def decode_tensor(payload: bytes, *, copy: bool = True) -> ActivationTensor:
    minimum = len(MAGIC) + HEADER_LENGTH.size
    if len(payload) < minimum or payload[: len(MAGIC)] != MAGIC:
        raise IntegrityError("invalid activation tensor magic or truncated envelope")
    header_length = HEADER_LENGTH.unpack_from(payload, len(MAGIC))[0]
    header_start = minimum
    header_end = header_start + header_length
    if header_end > len(payload):
        raise IntegrityError("activation header length exceeds envelope size")
    try:
        header = json.loads(payload[header_start:header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"invalid activation header: {exc}") from exc
    raw = payload[header_end:]
    if len(raw) != int(header.get("payload_length", -1)):
        raise IntegrityError(
            f"activation payload length mismatch: header={header.get('payload_length')} "
            f"actual={len(raw)}"
        )
    actual_checksum = sha256_bytes(raw)
    if actual_checksum != header.get("checksum"):
        raise IntegrityError(
            f"activation checksum mismatch: expected={header.get('checksum')} "
            f"actual={actual_checksum}"
        )
    try:
        logical_dtype = str(header["dtype"])
        dtype = np.dtype("<u2") if logical_dtype == "bfloat16" else np.dtype(logical_dtype)
        shape = tuple(int(dimension) for dimension in header["shape"])
        array = np.frombuffer(raw, dtype=dtype).reshape(shape)
    except (KeyError, TypeError, ValueError) as exc:
        raise IntegrityError(f"invalid dtype or shape in activation header: {exc}") from exc
    if copy:
        array = array.copy()
    return ActivationTensor(
        tensor_id=str(header["tensor_id"]),
        request_id=str(header["request_id"]),
        stage_id=int(header["stage_id"]),
        token_position=int(header["token_position"]),
        sequence_length=int(header["sequence_length"]),
        array=array,
        logical_dtype=("bfloat16" if str(header["dtype"]) == "bfloat16" else None),
        model_revision=str(header.get("model_revision", "")),
        partition_hash=str(header.get("partition_hash", "")),
        route_generation=int(header.get("route_generation", 0)),
    )


def split_chunks(payload: bytes, *, tensor_id: str, max_chunk_bytes: int) -> list[TensorChunk]:
    if max_chunk_bytes <= 0:
        raise ValueError("max_chunk_bytes must be positive")
    if not payload:
        parts = [b""]
    else:
        parts = [
            payload[offset : offset + max_chunk_bytes]
            for offset in range(0, len(payload), max_chunk_bytes)
        ]
    return [
        TensorChunk(
            tensor_id=tensor_id,
            chunk_index=index,
            chunk_count=len(parts),
            total_length=len(payload),
            payload=part,
            checksum=sha256_bytes(part),
        )
        for index, part in enumerate(parts)
    ]


def reassemble_chunks(chunks: list[TensorChunk]) -> bytes:
    if not chunks:
        raise IntegrityError("cannot reassemble an empty chunk list")
    ordered = sorted(chunks, key=lambda item: item.chunk_index)
    first = ordered[0]
    if [chunk.chunk_index for chunk in ordered] != list(range(first.chunk_count)):
        raise IntegrityError("tensor chunks are missing, duplicated, or non-contiguous")
    for chunk in ordered:
        if (
            chunk.tensor_id != first.tensor_id
            or chunk.chunk_count != first.chunk_count
            or chunk.total_length != first.total_length
        ):
            raise IntegrityError("tensor chunk metadata is inconsistent")
        if sha256_bytes(chunk.payload) != chunk.checksum:
            raise IntegrityError(f"tensor chunk {chunk.chunk_index} checksum mismatch")
    payload = b"".join(chunk.payload for chunk in ordered)
    if len(payload) != first.total_length:
        raise IntegrityError(
            f"reassembled tensor length mismatch: expected={first.total_length} actual={len(payload)}"
        )
    return payload
