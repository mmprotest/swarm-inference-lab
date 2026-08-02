"""Canonical binary framing for expert requests and responses."""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from swarm_inference.experiments.experiment_010.codecs import decode_array, encode_array
from swarm_inference.experiments.experiment_010.schemas import (
    ExpertExecutionRequest,
    ExpertExecutionResponse,
    TensorWireMetadata,
    TransportCodec,
)
from swarm_inference.protocol.checksums import sha256_bytes
from swarm_inference.protocol.tensor_codec import (
    ActivationTensor,
    decode_tensor,
    encode_tensor,
)

MAGIC = b"SWARMEX1"
_HEADER = struct.Struct(">II")
_LENGTH = struct.Struct(">Q")
MAX_FRAME_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ExpertPacket:
    kind: Literal["request", "response", "control"]
    semantic: dict[str, Any]
    blobs: tuple[bytes, ...]


def encode_packet(packet: ExpertPacket) -> bytes:
    header = json.dumps(
        {"kind": packet.kind, "semantic": packet.semantic, "blob_count": len(packet.blobs)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    lengths = b"".join(_LENGTH.pack(len(blob)) for blob in packet.blobs)
    payload = (
        MAGIC + _HEADER.pack(len(header), len(lengths)) + header + lengths + b"".join(packet.blobs)
    )
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError(f"expert frame exceeds {MAX_FRAME_BYTES} bytes")
    return payload


def decode_packet(payload: bytes) -> ExpertPacket:
    minimum = len(MAGIC) + _HEADER.size
    if len(payload) < minimum or payload[: len(MAGIC)] != MAGIC:
        raise ValueError("invalid or truncated expert frame")
    header_length, lengths_length = _HEADER.unpack_from(payload, len(MAGIC))
    header_start = minimum
    header_end = header_start + header_length
    lengths_end = header_end + lengths_length
    if lengths_end > len(payload) or lengths_length % _LENGTH.size:
        raise ValueError("expert frame length table is invalid")
    header = json.loads(payload[header_start:header_end].decode("utf-8"))
    count = int(header["blob_count"])
    if lengths_length != count * _LENGTH.size:
        raise ValueError("expert frame blob count does not match its length table")
    lengths = [
        _LENGTH.unpack_from(payload, header_end + index * _LENGTH.size)[0] for index in range(count)
    ]
    cursor = lengths_end
    blobs = []
    for length in lengths:
        end = cursor + length
        if end > len(payload):
            raise ValueError("expert frame blob is truncated")
        blobs.append(payload[cursor:end])
        cursor = end
    if cursor != len(payload):
        raise ValueError("expert frame has trailing bytes")
    kind = str(header["kind"])
    if kind not in {"request", "response", "control"}:
        raise ValueError(f"unknown expert frame kind {kind!r}")
    semantic = header["semantic"]
    if not isinstance(semantic, dict):
        raise ValueError("expert semantic payload must be an object")
    return ExpertPacket(kind=kind, semantic=semantic, blobs=tuple(blobs))  # type: ignore[arg-type]


def encode_request(request: ExpertExecutionRequest, activation: np.ndarray) -> tuple[bytes, int]:
    if request.compression == TransportCodec.RAW_FP32:
        started = time.perf_counter_ns()
        source = np.ascontiguousarray(activation, dtype=np.float32)
        payload = encode_tensor(
            ActivationTensor(
                tensor_id=f"{request.request_id}:expert-activation",
                request_id=request.request_id,
                stage_id=request.layer_id,
                token_position=0,
                sequence_length=request.batch_rows,
                array=source,
                logical_dtype="float32",
                model_revision=request.model_revision,
                partition_hash=request.quantization_fingerprint,
                route_generation=0,
            )
        )
        metadata = TensorWireMetadata(
            name="activations",
            envelope="SWARMT01",
            dtype="float32",
            shape=list(source.shape),
            codec=TransportCodec.RAW_FP32,
            payload_index=0,
            raw_bytes=source.nbytes,
            encoded_bytes=len(payload),
            checksum=sha256_bytes(payload),
        )
        semantic = request.model_copy(
            update={"activations": metadata.model_dump(mode="json")}
        ).model_dump(mode="json")
        return (
            encode_packet(ExpertPacket(kind="request", semantic=semantic, blobs=(payload,))),
            time.perf_counter_ns() - started,
        )
    encoded = encode_array(activation, name="activations", codec=request.compression)
    semantic = request.model_copy(
        update={"activations": encoded.metadata.model_dump(mode="json")}
    ).model_dump(mode="json")
    return encode_packet(
        ExpertPacket(kind="request", semantic=semantic, blobs=(encoded.payload,))
    ), encoded.encode_ns


def decode_request(payload: bytes) -> tuple[ExpertExecutionRequest, np.ndarray, int]:
    packet = decode_packet(payload)
    if packet.kind != "request" or len(packet.blobs) != 1:
        raise ValueError("expected one-blob expert request")
    metadata = TensorWireMetadata.model_validate(packet.semantic.get("activations"))
    request = ExpertExecutionRequest.model_validate(packet.semantic)
    if metadata.envelope == "SWARMT01":
        started = time.perf_counter_ns()
        blob = packet.blobs[metadata.payload_index]
        if len(blob) != metadata.encoded_bytes or sha256_bytes(blob) != metadata.checksum:
            raise ValueError("activation tensor envelope failed outer integrity validation")
        tensor = decode_tensor(blob)
        if (
            tensor.request_id != request.request_id
            or tensor.stage_id != request.layer_id
            or tensor.model_revision != request.model_revision
            or tensor.partition_hash != request.quantization_fingerprint
        ):
            raise ValueError("activation tensor identity does not match expert request")
        activation = np.ascontiguousarray(tensor.array, dtype=np.float32)
        decode_ns = time.perf_counter_ns() - started
    else:
        decoded = decode_array(metadata, packet.blobs[metadata.payload_index])
        activation = decoded.array
        decode_ns = decoded.decode_ns
    if list(activation.shape) != [request.batch_rows, request.latent_dimension]:
        raise ValueError("activation tensor shape does not match request geometry")
    return request, activation, decode_ns


def encode_response(
    response: ExpertExecutionResponse,
    result: np.ndarray,
) -> tuple[bytes, int]:
    codec = response.result.get("codec", "raw_fp32")
    if TransportCodec(codec) == TransportCodec.RAW_FP32:
        started = time.perf_counter_ns()
        source = np.ascontiguousarray(result, dtype=np.float32)
        payload = encode_tensor(
            ActivationTensor(
                tensor_id=f"{response.request_id}:expert-result",
                request_id=response.request_id,
                stage_id=response.layer_id,
                token_position=0,
                sequence_length=int(source.shape[0]),
                array=source,
                logical_dtype="float32",
                model_revision=response.model_revision,
                partition_hash=response.integrity.model_fingerprint,
                route_generation=0,
            )
        )
        metadata = TensorWireMetadata(
            name="result",
            envelope="SWARMT01",
            dtype="float32",
            shape=list(source.shape),
            codec=TransportCodec.RAW_FP32,
            payload_index=0,
            raw_bytes=source.nbytes,
            encoded_bytes=len(payload),
            checksum=sha256_bytes(payload),
        )
        semantic = response.model_copy(
            update={"result": metadata.model_dump(mode="json")}
        ).model_dump(mode="json")
        return (
            encode_packet(ExpertPacket(kind="response", semantic=semantic, blobs=(payload,))),
            time.perf_counter_ns() - started,
        )
    encoded = encode_array(result, name="result", codec=codec)
    semantic = response.model_copy(
        update={"result": encoded.metadata.model_dump(mode="json")}
    ).model_dump(mode="json")
    return encode_packet(
        ExpertPacket(kind="response", semantic=semantic, blobs=(encoded.payload,))
    ), encoded.encode_ns


def decode_response(payload: bytes) -> tuple[ExpertExecutionResponse, np.ndarray, int]:
    packet = decode_packet(payload)
    if packet.kind != "response" or len(packet.blobs) != 1:
        raise ValueError("expected one-blob expert response")
    metadata = TensorWireMetadata.model_validate(packet.semantic.get("result"))
    response = ExpertExecutionResponse.model_validate(packet.semantic)
    blob = packet.blobs[metadata.payload_index]
    if metadata.envelope == "SWARMT01":
        started = time.perf_counter_ns()
        if len(blob) != metadata.encoded_bytes or sha256_bytes(blob) != metadata.checksum:
            raise ValueError("result tensor envelope failed outer integrity validation")
        tensor = decode_tensor(blob)
        if (
            tensor.request_id != response.request_id
            or tensor.stage_id != response.layer_id
            or tensor.model_revision != response.model_revision
            or tensor.partition_hash != response.integrity.model_fingerprint
        ):
            raise ValueError("result tensor identity does not match expert response")
        result = np.ascontiguousarray(tensor.array, dtype=np.float32)
        decode_ns = time.perf_counter_ns() - started
    else:
        decoded = decode_array(metadata, blob)
        result = decoded.array
        decode_ns = decoded.decode_ns
    return response, result, decode_ns


def frame_with_length(payload: bytes) -> bytes:
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError("expert socket frame is too large")
    return _LENGTH.pack(len(payload)) + payload


async def read_length_frame(reader: Any) -> bytes:
    length = _LENGTH.unpack(await reader.readexactly(_LENGTH.size))[0]
    if length > MAX_FRAME_BYTES:
        raise ValueError("expert socket frame is too large")
    return await reader.readexactly(length)
