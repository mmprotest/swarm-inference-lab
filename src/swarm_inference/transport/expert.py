"""SWARMEX1 framing, codecs, and the canonical expert transport client."""

from __future__ import annotations

import json
import socket
import struct
import time
import zlib
from dataclasses import asdict, dataclass
from typing import Any, Literal, overload

import numpy as np

from swarm_inference.protocol.checksums import sha256_bytes
from swarm_inference.protocol.expert import (
    DataPlane,
    ExpertExecutionRequest,
    ExpertExecutionResponse,
    TensorWireMetadata,
    TransportCodec,
)
from swarm_inference.protocol.tensor_codec import ActivationTensor, decode_tensor, encode_tensor
from swarm_inference.security.tls import TlsClientConfig, require_tls_for_endpoint

MAGIC = b"SWARMEX1"
_HEADER = struct.Struct(">II")
_LENGTH = struct.Struct(">Q")
MAX_FRAME_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ExpertPacket:
    kind: Literal["request", "response", "control"]
    semantic: dict[str, Any]
    blobs: tuple[bytes, ...]


@dataclass(frozen=True, slots=True)
class EncodedTensor:
    metadata: TensorWireMetadata
    payload: bytes
    encode_ns: int


@dataclass(frozen=True, slots=True)
class DecodedTensor:
    array: np.ndarray
    decode_ns: int


def _vectors(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array.reshape(-1, array.shape[-1])


def encode_array(
    array: np.ndarray,
    *,
    name: str,
    codec: TransportCodec | str,
    payload_index: int = 0,
) -> EncodedTensor:
    selected = TransportCodec(codec)
    started = time.perf_counter_ns()
    source = np.ascontiguousarray(array, dtype=np.float32)
    raw = source.tobytes(order="C")
    scales: float | list[float] | None = None
    dtype: Literal["float32", "float16", "int8", "uint8"] = "float32"
    if selected == TransportCodec.RAW_FP32:
        payload = raw
    elif selected == TransportCodec.RAW_FP16:
        payload = np.ascontiguousarray(source, dtype=np.float16).tobytes(order="C")
        dtype = "float16"
    elif selected == TransportCodec.INT8_PER_VECTOR:
        vectors = _vectors(source)
        maxima = np.max(np.abs(vectors), axis=1)
        vector_scales = np.where(maxima > 0, maxima / 127.0, 1.0).astype(np.float32)
        quantized = np.rint(vectors / vector_scales[:, None]).clip(-127, 127).astype(np.int8)
        payload = np.ascontiguousarray(quantized.reshape(source.shape)).tobytes(order="C")
        scales = [float(value) for value in vector_scales]
        dtype = "int8"
    elif selected == TransportCodec.LOSSLESS_GENERAL:
        payload = zlib.compress(raw, level=3)
        dtype = "float32"
    else:  # pragma: no cover
        raise ValueError(f"unsupported transport codec {selected}")
    return EncodedTensor(
        metadata=TensorWireMetadata(
            name=name,
            dtype=dtype,
            shape=[int(value) for value in source.shape],
            codec=selected,
            payload_index=payload_index,
            raw_bytes=len(raw),
            encoded_bytes=len(payload),
            scale=scales,
            checksum=sha256_bytes(payload),
        ),
        payload=payload,
        encode_ns=time.perf_counter_ns() - started,
    )


def decode_array(metadata: TensorWireMetadata, payload: bytes) -> DecodedTensor:
    started = time.perf_counter_ns()
    element_count = 1
    for dimension in metadata.shape:
        element_count *= dimension
    expected_raw_bytes = element_count * np.dtype(np.float32).itemsize
    if metadata.raw_bytes != expected_raw_bytes:
        raise ValueError("tensor raw byte count does not match its float32 geometry")
    if len(payload) != metadata.encoded_bytes:
        raise ValueError(
            f"encoded tensor length mismatch: expected {metadata.encoded_bytes}, got {len(payload)}"
        )
    if sha256_bytes(payload) != metadata.checksum:
        raise ValueError("encoded tensor checksum mismatch")
    shape = tuple(metadata.shape)
    if metadata.codec == TransportCodec.RAW_FP32:
        if metadata.encoded_bytes != expected_raw_bytes:
            raise ValueError("raw_fp32 tensor length does not match its geometry")
        array = np.frombuffer(payload, dtype=np.float32).reshape(shape).copy()
    elif metadata.codec == TransportCodec.RAW_FP16:
        if metadata.encoded_bytes != element_count * np.dtype(np.float16).itemsize:
            raise ValueError("raw_fp16 tensor length does not match its geometry")
        array = np.frombuffer(payload, dtype=np.float16).reshape(shape).astype(np.float32)
    elif metadata.codec == TransportCodec.INT8_PER_VECTOR:
        if metadata.encoded_bytes != element_count:
            raise ValueError("int8 tensor length does not match its geometry")
        if not isinstance(metadata.scale, list):
            raise ValueError("int8_per_vector tensor requires one scale per vector")
        quantized = np.frombuffer(payload, dtype=np.int8).reshape(shape)
        vectors = _vectors(quantized)
        scales = np.asarray(metadata.scale, dtype=np.float32)
        if len(scales) != vectors.shape[0]:
            raise ValueError("int8_per_vector scale count does not match tensor vectors")
        array = (vectors.astype(np.float32) * scales[:, None]).reshape(shape)
    elif metadata.codec == TransportCodec.LOSSLESS_GENERAL:
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(payload, expected_raw_bytes + 1)
        if (
            not decompressor.eof
            or decompressor.unconsumed_tail
            or decompressor.unused_data
            or len(raw) > expected_raw_bytes
        ):
            raise ValueError("lossless tensor exceeds its declared bounded geometry")
        if len(raw) != metadata.raw_bytes:
            raise ValueError("lossless tensor decompressed length mismatch")
        array = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    else:  # pragma: no cover
        raise ValueError(f"unsupported transport codec {metadata.codec}")
    return DecodedTensor(
        array=np.ascontiguousarray(array), decode_ns=time.perf_counter_ns() - started
    )


def numerical_error(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    expected = np.asarray(reference, dtype=np.float64)
    observed = np.asarray(actual, dtype=np.float64)
    if expected.shape != observed.shape:
        raise ValueError("numerical comparison requires identical shapes")
    difference = observed - expected
    absolute = np.abs(difference)
    denominator = float(np.linalg.norm(expected.ravel()))
    expected_norm = denominator if denominator > 0 else 1.0
    actual_flat = observed.ravel()
    expected_flat = expected.ravel()
    cosine_denominator = float(np.linalg.norm(expected_flat) * np.linalg.norm(actual_flat))
    cosine = (
        float(np.dot(expected_flat, actual_flat) / cosine_denominator)
        if cosine_denominator > 0
        else 1.0
        if np.array_equal(expected, observed)
        else 0.0
    )
    return {
        "maximum_absolute_error": float(absolute.max(initial=0.0)),
        "mean_absolute_error": float(absolute.mean()) if absolute.size else 0.0,
        "relative_l2_error": float(np.linalg.norm(difference.ravel()) / expected_norm),
        "cosine_similarity": cosine,
        "exact": bool(np.array_equal(expected, observed)),
    }


def codec_break_even(
    *,
    raw_bytes: int,
    encoded_bytes: int,
    encode_ns: int,
    decode_ns: int,
    bandwidth_bps: float,
) -> dict[str, float | bool]:
    if raw_bytes < 0 or encoded_bytes < 0 or bandwidth_bps <= 0:
        raise ValueError("codec break-even inputs must be non-negative with positive bandwidth")
    raw_transfer_ns = raw_bytes * 8 / bandwidth_bps * 1e9
    encoded_transfer_ns = encoded_bytes * 8 / bandwidth_bps * 1e9
    encoded_total_ns = encode_ns + encoded_transfer_ns + decode_ns
    return {
        "raw_total_ns": raw_transfer_ns,
        "encoded_total_ns": encoded_total_ns,
        "beneficial": encoded_total_ns < raw_transfer_ns,
        "saved_ns": raw_transfer_ns - encoded_total_ns,
    }


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
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError(f"expert frame exceeds {MAX_FRAME_BYTES} bytes")
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


def _tensor_metadata(
    *, name: str, source: np.ndarray, payload: bytes, payload_index: int
) -> TensorWireMetadata:
    return TensorWireMetadata(
        name=name,
        envelope="SWARMT01",
        dtype="float32",
        shape=list(source.shape),
        codec=TransportCodec.RAW_FP32,
        payload_index=payload_index,
        raw_bytes=source.nbytes,
        encoded_bytes=len(payload),
        checksum=sha256_bytes(payload),
    )


def encode_request(
    request: ExpertExecutionRequest,
    activation: np.ndarray,
    down_accumulators: np.ndarray | None = None,
) -> tuple[bytes, int]:
    if request.compression != TransportCodec.RAW_FP32:
        if down_accumulators is not None or request.down_accumulators is not None:
            raise ValueError("microshard down accumulators require raw_fp32 transport")
        encoded = encode_array(activation, name="activations", codec=request.compression)
        semantic = request.model_copy(
            update={"activations": encoded.metadata.model_dump(mode="json")}
        ).model_dump(mode="json")
        return (
            encode_packet(
                ExpertPacket(kind="request", semantic=semantic, blobs=(encoded.payload,))
            ),
            encoded.encode_ns,
        )
    started = time.perf_counter_ns()
    source = np.ascontiguousarray(activation, dtype=np.float32)
    payload = encode_tensor(
        ActivationTensor(
            tensor_id=f"{request.request_id}:expert-activation",
            request_id=request.request_id,
            stage_id=request.layer_id,
            token_position=int(getattr(request, "token_position", 0)),
            sequence_length=request.batch_rows,
            array=source,
            logical_dtype="float32",
            model_revision=request.model_revision,
            partition_hash=request.quantization_fingerprint,
            route_generation=int(getattr(request, "route_generation", 0)),
        )
    )
    blobs = [payload]
    update: dict[str, Any] = {
        "activations": _tensor_metadata(
            name="activations", source=source, payload=payload, payload_index=0
        ).model_dump(mode="json")
    }
    if down_accumulators is not None:
        accumulator = np.ascontiguousarray(down_accumulators, dtype=np.float32)
        expected = (request.batch_rows, request.effective_top_k, request.latent_dimension)
        if accumulator.shape != expected:
            raise ValueError("down accumulator tensor shape does not match request geometry")
        accumulator_payload = encode_tensor(
            ActivationTensor(
                tensor_id=f"{request.request_id}:down-accumulators",
                request_id=request.request_id,
                stage_id=request.layer_id,
                token_position=int(getattr(request, "token_position", 0)),
                sequence_length=request.batch_rows,
                array=accumulator,
                logical_dtype="float32",
                model_revision=request.model_revision,
                partition_hash=request.quantization_fingerprint,
                route_generation=int(getattr(request, "route_generation", 0)),
            )
        )
        update["down_accumulators"] = _tensor_metadata(
            name="down_accumulators",
            source=accumulator,
            payload=accumulator_payload,
            payload_index=1,
        ).model_dump(mode="json")
        blobs.append(accumulator_payload)
    elif request.down_accumulators is not None:
        raise ValueError("request declares down accumulators but none were supplied")
    semantic = request.model_copy(update=update).model_dump(mode="json")
    return (
        encode_packet(ExpertPacket(kind="request", semantic=semantic, blobs=tuple(blobs))),
        time.perf_counter_ns() - started,
    )


def _validate_tensor_identity(tensor: ActivationTensor, request: ExpertExecutionRequest) -> None:
    product_identity = request.topology_id != "legacy"
    if (
        tensor.request_id != request.request_id
        or tensor.stage_id != request.layer_id
        or (product_identity and tensor.token_position != request.token_position)
        or tensor.model_revision != request.model_revision
        or tensor.partition_hash != request.quantization_fingerprint
        or (product_identity and tensor.route_generation != request.route_generation)
    ):
        raise ValueError("activation tensor identity does not match expert request")


@overload
def decode_request(
    payload: bytes,
    *,
    include_down_accumulators: Literal[False] = False,
) -> tuple[ExpertExecutionRequest, np.ndarray, int]: ...


@overload
def decode_request(
    payload: bytes,
    *,
    include_down_accumulators: Literal[True],
) -> tuple[ExpertExecutionRequest, np.ndarray, np.ndarray | None, int]: ...


def decode_request(
    payload: bytes,
    *,
    include_down_accumulators: bool = False,
) -> (
    tuple[ExpertExecutionRequest, np.ndarray, int]
    | tuple[ExpertExecutionRequest, np.ndarray, np.ndarray | None, int]
):
    packet = decode_packet(payload)
    if packet.kind != "request" or len(packet.blobs) not in {1, 2}:
        raise ValueError("expected one- or two-blob expert request")
    metadata = TensorWireMetadata.model_validate(packet.semantic.get("activations"))
    request = ExpertExecutionRequest.model_validate(packet.semantic)
    blob = packet.blobs[metadata.payload_index]
    if metadata.envelope == "SWARMT01":
        started = time.perf_counter_ns()
        if len(blob) != metadata.encoded_bytes or sha256_bytes(blob) != metadata.checksum:
            raise ValueError("activation tensor envelope failed outer integrity validation")
        tensor = decode_tensor(blob)
        _validate_tensor_identity(tensor, request)
        activation = np.ascontiguousarray(tensor.array, dtype=np.float32)
        decode_ns = time.perf_counter_ns() - started
    else:
        decoded = decode_array(metadata, blob)
        activation = decoded.array
        decode_ns = decoded.decode_ns
    if list(activation.shape) != [request.batch_rows, request.latent_dimension]:
        raise ValueError("activation tensor shape does not match request geometry")
    accumulator: np.ndarray | None = None
    accumulator_semantic = packet.semantic.get("down_accumulators")
    if accumulator_semantic is not None:
        accumulator_metadata = TensorWireMetadata.model_validate(accumulator_semantic)
        if len(packet.blobs) != 2 or accumulator_metadata.payload_index != 1:
            raise ValueError("down accumulator tensor has an invalid payload index")
        started = time.perf_counter_ns()
        accumulator_blob = packet.blobs[1]
        if (
            len(accumulator_blob) != accumulator_metadata.encoded_bytes
            or sha256_bytes(accumulator_blob) != accumulator_metadata.checksum
        ):
            raise ValueError("down accumulator envelope failed outer integrity validation")
        tensor = decode_tensor(accumulator_blob)
        _validate_tensor_identity(tensor, request)
        accumulator = np.ascontiguousarray(tensor.array, dtype=np.float32)
        expected = [request.batch_rows, request.effective_top_k, request.latent_dimension]
        if list(accumulator.shape) != expected:
            raise ValueError("down accumulator tensor shape does not match request geometry")
        decode_ns += time.perf_counter_ns() - started
    elif len(packet.blobs) != 1:
        raise ValueError("expert request has an undeclared tensor blob")
    if include_down_accumulators:
        return request, activation, accumulator, decode_ns
    return request, activation, decode_ns


def encode_response(response: ExpertExecutionResponse, result: np.ndarray) -> tuple[bytes, int]:
    codec = response.result.get("codec", "raw_fp32")
    if TransportCodec(codec) != TransportCodec.RAW_FP32:
        encoded = encode_array(result, name="result", codec=codec)
        semantic = response.model_copy(
            update={"result": encoded.metadata.model_dump(mode="json")}
        ).model_dump(mode="json")
        return (
            encode_packet(
                ExpertPacket(kind="response", semantic=semantic, blobs=(encoded.payload,))
            ),
            encoded.encode_ns,
        )
    started = time.perf_counter_ns()
    source = np.ascontiguousarray(result, dtype=np.float32)
    payload = encode_tensor(
        ActivationTensor(
            tensor_id=f"{response.request_id}:expert-result",
            request_id=response.request_id,
            stage_id=response.layer_id,
            token_position=int(getattr(response, "token_position", 0)),
            sequence_length=int(source.shape[0]),
            array=source,
            logical_dtype="float32",
            model_revision=response.model_revision,
            partition_hash=response.integrity.model_fingerprint,
            route_generation=int(getattr(response, "route_generation", 0)),
        )
    )
    semantic = response.model_copy(
        update={
            "result": _tensor_metadata(
                name="result", source=source, payload=payload, payload_index=0
            ).model_dump(mode="json")
        }
    ).model_dump(mode="json")
    return (
        encode_packet(ExpertPacket(kind="response", semantic=semantic, blobs=(payload,))),
        time.perf_counter_ns() - started,
    )


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
            or tensor.token_position != response.token_position
            or tensor.model_revision != response.model_revision
            or tensor.partition_hash != response.integrity.model_fingerprint
            or tensor.route_generation != response.route_generation
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
    return bytes(await reader.readexactly(length))


def _parse_endpoint(endpoint: str) -> tuple[str, int]:
    host, separator, raw_port = endpoint.rpartition(":")
    if not separator or not host:
        raise ValueError(f"invalid expert endpoint {endpoint!r}")
    port = int(raw_port)
    if not 1 <= port <= 65535:
        raise ValueError(f"invalid expert endpoint {endpoint!r}")
    return host, port


def _recv_exact(connection: socket.socket, byte_count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = byte_count
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("expert socket closed before the frame completed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_frame(connection: socket.socket) -> bytes:
    length = _LENGTH.unpack(_recv_exact(connection, _LENGTH.size))[0]
    if length > MAX_FRAME_BYTES:
        raise ValueError("expert response frame is too large")
    return _recv_exact(connection, length)


@dataclass(slots=True)
class ExpertTransportMetrics:
    data_plane: str
    messages_sent: int = 0
    messages_received: int = 0
    request_bytes: int = 0
    response_bytes: int = 0
    payload_bytes: int = 0
    serialisation_ns: int = 0
    socket_ns: int = 0
    queue_ns: int = 0
    total_request_ns: int = 0

    def snapshot(self) -> dict[str, int | str]:
        return asdict(self)


class ExpertTransportClient:
    """Bounded direct expert client used inside a canonical stage."""

    def __init__(
        self,
        endpoint: str,
        *,
        data_plane: DataPlane | str = DataPlane.DIRECT_TCP,
        timeout_s: float = 30.0,
        tls: TlsClientConfig | None = None,
        allow_plaintext_loopback: bool = True,
    ) -> None:
        self.endpoint = endpoint
        self.tls = tls
        require_tls_for_endpoint(
            endpoint,
            tls_configured=tls is not None,
            allow_plaintext_loopback=allow_plaintext_loopback,
            transport_name="expert data plane",
        )
        self.data_plane = DataPlane(data_plane)
        if self.data_plane not in {DataPlane.DIRECT_TCP, DataPlane.RELAYED_TCP}:
            raise ValueError("product expert client requires a TCP data plane")
        if timeout_s <= 0:
            raise ValueError("expert transport timeout must be positive")
        self.timeout_s = timeout_s
        self.metrics = ExpertTransportMetrics(data_plane=self.data_plane.value)

    def _round_trip(self, payload: bytes, *, timeout_s: float | None = None) -> bytes:
        host, port = _parse_endpoint(self.endpoint)
        framed = frame_with_length(payload)
        timeout = self.timeout_s if timeout_s is None else min(self.timeout_s, timeout_s)
        if timeout <= 0:
            raise TimeoutError("expert request deadline elapsed before socket transport")
        started = time.perf_counter_ns()
        socket_started = time.perf_counter_ns()
        raw_connection = socket.create_connection((host, port), timeout=timeout)
        connection: socket.socket = raw_connection
        try:
            if self.tls is not None:
                secure_connection = self.tls.ssl_context().wrap_socket(
                    raw_connection,
                    server_hostname=self.tls.expected_server_name,
                )
                self.tls.validate_peer_der(secure_connection.getpeercert(binary_form=True))
                connection = secure_connection
            connection.settimeout(timeout)
            connection.sendall(framed)
            response = _recv_frame(connection)
        finally:
            connection.close()
            if connection is not raw_connection:
                raw_connection.close()
        self.metrics.socket_ns += time.perf_counter_ns() - socket_started
        self.metrics.messages_sent += 1
        self.metrics.messages_received += 1
        self.metrics.request_bytes += len(framed)
        self.metrics.response_bytes += len(response) + _LENGTH.size
        self.metrics.total_request_ns += time.perf_counter_ns() - started
        return response

    def execute(
        self,
        request: ExpertExecutionRequest,
        activation: np.ndarray,
        down_accumulators: np.ndarray | None = None,
    ) -> tuple[ExpertExecutionResponse, np.ndarray, dict[str, Any]]:
        if time.time_ns() >= request.deadline_ns:
            raise TimeoutError("expert request deadline elapsed before transport")
        started = time.perf_counter_ns()
        before = self.metrics.snapshot()
        encoded, encode_ns = encode_request(request, activation, down_accumulators)
        self.metrics.serialisation_ns += encode_ns
        self.metrics.payload_bytes += int(np.asarray(activation).nbytes) + (
            int(np.asarray(down_accumulators).nbytes) if down_accumulators is not None else 0
        )
        remaining_s = (request.deadline_ns - time.time_ns()) / 1_000_000_000
        payload = self._round_trip(encoded, timeout_s=remaining_s)
        if time.time_ns() >= request.deadline_ns:
            raise TimeoutError("expert request deadline elapsed during transport")
        packet = decode_packet(payload)
        if packet.kind == "control":
            raise RuntimeError(str(packet.semantic.get("error", "worker request failed")))
        response, result, decode_ns = decode_response(payload)
        self.metrics.serialisation_ns += decode_ns
        self.metrics.queue_ns += response.execution_metadata.queue_ns
        if response.request_id != request.request_id or response.session_id != request.session_id:
            raise ValueError("expert response request or session identity mismatch")
        if response.token_position != request.token_position:
            raise ValueError("expert response token identity mismatch")
        if response.route_generation != request.route_generation:
            raise ValueError("expert response route generation mismatch")
        if response.model_revision != request.model_revision:
            raise ValueError("expert response model revision mismatch")
        if response.quantization_fingerprint != request.quantization_fingerprint and (
            request.topology_id != "legacy" or response.quantization_fingerprint
        ):
            raise ValueError("expert response quantisation identity mismatch")
        after = self.metrics.snapshot()
        delta: dict[str, Any] = {}
        for key, value in after.items():
            previous = before.get(key)
            delta[key] = (
                value - previous if isinstance(value, int) and isinstance(previous, int) else value
            )
        delta["request_elapsed_ns"] = time.perf_counter_ns() - started
        return response, result, delta

    def control(self, command: str, **payload: Any) -> dict[str, Any]:
        encoded = encode_packet(
            ExpertPacket(kind="control", semantic={"command": command, **payload}, blobs=())
        )
        response = decode_packet(self._round_trip(encoded))
        if response.kind != "control":
            raise RuntimeError("worker returned a non-control response")
        if not response.semantic.get("ok"):
            raise RuntimeError(str(response.semantic.get("error", "worker control failed")))
        return response.semantic


__all__ = [
    "MAGIC",
    "MAX_FRAME_BYTES",
    "DecodedTensor",
    "EncodedTensor",
    "ExpertPacket",
    "ExpertTransportClient",
    "ExpertTransportMetrics",
    "codec_break_even",
    "decode_array",
    "decode_packet",
    "decode_request",
    "decode_response",
    "encode_array",
    "encode_packet",
    "encode_request",
    "encode_response",
    "frame_with_length",
    "numerical_error",
    "read_length_frame",
]
