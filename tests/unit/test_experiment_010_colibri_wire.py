from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from swarm_inference.experiments.experiment_010.schemas import (
    ExpertExecutionMetadata,
    ExpertExecutionMode,
    ExpertExecutionRequest,
    ExpertExecutionResponse,
    ExpertResponseMode,
    ResultIntegrity,
    TensorWireMetadata,
)
from swarm_inference.experiments.experiment_010.wire import (
    decode_request,
    decode_response,
    encode_request,
    encode_response,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ADAPTER = REPOSITORY_ROOT / "integrations" / "colibri" / "adapter"


class BlobView(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_uint8)), ("length", ctypes.c_uint64)]


class OwnedBytes(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_uint8)), ("length", ctypes.c_size_t)]


class TensorView(ctypes.Structure):
    _fields_ = [
        ("tensor_id", ctypes.c_char * 256),
        ("request_id", ctypes.c_char * 256),
        ("model_revision", ctypes.c_char * 256),
        ("partition_hash", ctypes.c_char * 256),
        ("dtype", ctypes.c_char * 32),
        ("stage_id", ctypes.c_int),
        ("token_position", ctypes.c_int),
        ("sequence_length", ctypes.c_int),
        ("route_generation", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("shape", ctypes.c_int64 * 4),
        ("values", ctypes.POINTER(ctypes.c_float)),
        ("value_count", ctypes.c_size_t),
        ("raw", ctypes.POINTER(ctypes.c_uint8)),
        ("raw_length", ctypes.c_size_t),
    ]


class RouteRequest(ctypes.Structure):
    _fields_ = [
        ("request_id", ctypes.c_char * 256),
        ("model_id", ctypes.c_char * 256),
        ("model_revision", ctypes.c_char * 256),
        ("quantization_fingerprint", ctypes.c_char * 256),
        ("layer_id", ctypes.c_int),
        ("batch_rows", ctypes.c_int),
        ("latent_dimension", ctypes.c_int),
        ("top_k", ctypes.c_int),
        ("expert_ids_by_row", ctypes.POINTER(ctypes.c_int)),
        ("routing_weights_by_row", ctypes.POINTER(ctypes.c_float)),
        ("selected_rank_by_row", ctypes.POINTER(ctypes.c_int)),
        ("response_mode", ctypes.c_int),
        ("execution_mode", ctypes.c_int),
        ("hidden_start", ctypes.c_int),
        ("hidden_end", ctypes.c_int),
        ("microshard_final", ctypes.c_int),
        ("activation", TensorView),
        ("down_accumulators", TensorView),
        ("challenge", ctypes.c_int),
    ]


class RouteResponse(ctypes.Structure):
    _fields_ = [
        ("request_id", ctypes.c_char * 256),
        ("worker_id", ctypes.c_char * 256),
        ("model_revision", ctypes.c_char * 256),
        ("model_fingerprint", ctypes.c_char * 256),
        ("result_hash", ctypes.c_char * 80),
        ("status", ctypes.c_char * 16),
        ("error", ctypes.c_char * 512),
        ("layer_id", ctypes.c_int),
        ("bytes_read", ctypes.c_uint64),
        ("bytes_received", ctypes.c_uint64),
        ("bytes_sent", ctypes.c_uint64),
        ("compute_ns", ctypes.c_uint64),
        ("queue_ns", ctypes.c_uint64),
        ("transfer_ns", ctypes.c_uint64),
        ("result", TensorView),
    ]


def _compiler() -> str | None:
    explicit = os.environ.get("COLIBRI_CC")
    candidates = [
        explicit,
        shutil.which("gcc"),
        shutil.which("cc"),
        r"C:\tmp\winlibs-gcc16-r3\mingw64\bin\gcc.exe",
    ]
    return next((str(value) for value in candidates if value and Path(value).is_file()), None)


@pytest.fixture(scope="module")
def c_wire(tmp_path_factory: pytest.TempPathFactory) -> ctypes.CDLL:
    compiler = _compiler()
    if compiler is None:
        pytest.skip("a C compiler is required for the Colibri cross-language wire ABI test")
    output = tmp_path_factory.mktemp("colibri-c-wire") / (
        "swarm_expert_wire.dll" if os.name == "nt" else "libswarm_expert_wire.so"
    )
    command = [
        compiler,
        "-std=c11",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-shared",
        str(ADAPTER / "swarm_expert_wire.c"),
        "-I",
        str(ADAPTER),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    library = ctypes.CDLL(str(output))
    library.swarm_expert_wire_decode_route_request.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.POINTER(RouteRequest),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.swarm_expert_wire_decode_route_request.restype = ctypes.c_int
    library.swarm_expert_wire_encode_route_request.argtypes = [
        ctypes.POINTER(RouteRequest),
        ctypes.c_uint64,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(OwnedBytes),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.swarm_expert_wire_encode_route_request.restype = ctypes.c_int
    library.swarm_expert_wire_free_route_request.argtypes = [ctypes.POINTER(RouteRequest)]
    library.swarm_expert_wire_decode_route_response.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.POINTER(RouteResponse),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.swarm_expert_wire_decode_route_response.restype = ctypes.c_int
    library.swarm_expert_wire_encode_tensor_f32.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(OwnedBytes),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.swarm_expert_wire_encode_tensor_f32.restype = ctypes.c_int
    library.swarm_expert_wire_encode_packet.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(BlobView),
        ctypes.c_size_t,
        ctypes.POINTER(OwnedBytes),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.swarm_expert_wire_encode_packet.restype = ctypes.c_int
    library.swarm_expert_wire_free_bytes.argtypes = [ctypes.POINTER(OwnedBytes)]
    return library


def _per_row_request() -> tuple[ExpertExecutionRequest, np.ndarray]:
    activation = np.arange(8, dtype=np.float32).reshape(2, 4) / np.float32(8)
    return (
        ExpertExecutionRequest(
            request_id="c-wire-request",
            model_id="olmoe-fixture",
            model_revision="revision-1",
            quantization_fingerprint="quantization-1",
            layer_id=3,
            batch_rows=2,
            latent_dimension=4,
            top_k=2,
            expert_ids_by_row=[[7, 2], [9, 1]],
            routing_weights_by_row=[[0.75, 0.25], [0.625, 0.375]],
            selected_rank_by_row=[[0, 1], [0, 1]],
            response_mode=ExpertResponseMode.PER_EXPERT_EXACT,
            activations={},
            deadline_ns=9_999_999_999,
        ),
        activation,
    )


def _decode_with_c(
    library: ctypes.CDLL, payload: bytes
) -> tuple[int, RouteRequest, str, ctypes.Array[ctypes.c_uint8]]:
    source = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
    decoded = RouteRequest()
    error = ctypes.create_string_buffer(512)
    rc = library.swarm_expert_wire_decode_route_request(
        source, len(payload), ctypes.byref(decoded), error, len(error)
    )
    return rc, decoded, error.value.decode("utf-8", errors="replace"), source


def test_colibri_c_wire_decodes_python_request(c_wire: ctypes.CDLL) -> None:
    request, activation = _per_row_request()
    payload, _ = encode_request(request, activation)
    rc, decoded, error, source = _decode_with_c(c_wire, payload)
    try:
        assert rc == 0, error
        assert decoded.request_id.rstrip(b"\0") == b"c-wire-request"
        assert (decoded.layer_id, decoded.batch_rows, decoded.latent_dimension, decoded.top_k) == (
            3,
            2,
            4,
            2,
        )
        assert list(decoded.expert_ids_by_row[:4]) == [7, 2, 9, 1]
        np.testing.assert_array_equal(
            np.ctypeslib.as_array(decoded.routing_weights_by_row, shape=(4,)),
            np.asarray([0.75, 0.25, 0.625, 0.375], dtype=np.float32),
        )
        assert list(decoded.selected_rank_by_row[:4]) == [0, 1, 0, 1]
        assert decoded.response_mode == 1
        assert list(decoded.activation.shape[: decoded.activation.ndim]) == [2, 4]
        assert (
            ctypes.string_at(decoded.activation.raw, decoded.activation.raw_length)
            == activation.tobytes()
        )
        assert source  # Owns the backing frame for the C tensor view until this point.
    finally:
        c_wire.swarm_expert_wire_free_route_request(ctypes.byref(decoded))


def test_python_wire_decodes_full_colibri_c_request(c_wire: ctypes.CDLL) -> None:
    activation = np.arange(8, dtype=np.float32).reshape(2, 4) / np.float32(8)
    expert_ids = (ctypes.c_int * 4)(7, 2, 9, 1)
    weights = (ctypes.c_float * 4)(0.75, 0.25, 0.625, 0.375)
    ranks = (ctypes.c_int * 4)(0, 1, 0, 1)
    request = RouteRequest()
    request.request_id = b"c-encoded-request"
    request.model_id = b"olmoe-fixture"
    request.model_revision = b"revision-1"
    request.quantization_fingerprint = b"quantization-1"
    request.layer_id = 3
    request.batch_rows = 2
    request.latent_dimension = 4
    request.top_k = 2
    request.expert_ids_by_row = expert_ids
    request.routing_weights_by_row = weights
    request.selected_rank_by_row = ranks
    request.response_mode = 1
    request.execution_mode = 1
    request.activation.token_position = 11
    request.activation.route_generation = 0
    request.activation.values = activation.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    request.activation.value_count = activation.size
    packet = OwnedBytes()
    error = ctypes.create_string_buffer(512)
    rc = c_wire.swarm_expert_wire_encode_route_request(
        ctypes.byref(request),
        9_999_999_999,
        b"MEASURED_LEVEL_A",
        1,
        ctypes.byref(packet),
        error,
        len(error),
    )
    assert rc == 0, error.value.decode()
    try:
        decoded, decoded_activation, _ = decode_request(_c_owned_bytes(packet))
    finally:
        c_wire.swarm_expert_wire_free_bytes(ctypes.byref(packet))
    assert decoded.request_id == "c-encoded-request"
    assert decoded.expert_ids_by_row == [[7, 2], [9, 1]]
    assert decoded.selected_rank_by_row == [[0, 1], [0, 1]]
    assert decoded.metadata["exact_contribution_representation"] == "unweighted_expert_output"
    np.testing.assert_array_equal(decoded_activation, activation)


def test_colibri_c_wire_decodes_python_microshard_accumulator(
    c_wire: ctypes.CDLL,
) -> None:
    activation = np.arange(8, dtype=np.float32).reshape(2, 4) / np.float32(8)
    accumulator = np.arange(16, dtype=np.float32).reshape(2, 2, 4) / np.float32(16)
    request = ExpertExecutionRequest(
        request_id="python-microshard-request",
        model_id="olmoe-fixture",
        model_revision="revision-1",
        quantization_fingerprint="quantization-1",
        layer_id=3,
        batch_rows=2,
        latent_dimension=4,
        top_k=2,
        expert_ids_by_row=[[7, 2], [9, 1]],
        routing_weights_by_row=[[0.75, 0.25], [0.625, 0.375]],
        selected_rank_by_row=[[0, 1], [0, 1]],
        response_mode=ExpertResponseMode.PER_EXPERT_EXACT,
        execution_mode=ExpertExecutionMode.MICROSHARD,
        hidden_start=16,
        hidden_end=32,
        microshard_final=True,
        down_accumulators={},
        activations={},
        deadline_ns=9_999_999_999,
    )
    payload, _ = encode_request(request, activation, accumulator)
    rc, decoded, error, source = _decode_with_c(c_wire, payload)
    try:
        assert rc == 0, error
        assert decoded.execution_mode == 2
        assert (decoded.hidden_start, decoded.hidden_end, decoded.microshard_final) == (16, 32, 1)
        assert list(decoded.down_accumulators.shape[: decoded.down_accumulators.ndim]) == [
            2,
            2,
            4,
        ]
        assert (
            ctypes.string_at(decoded.down_accumulators.raw, decoded.down_accumulators.raw_length)
            == accumulator.tobytes()
        )
        assert source
    finally:
        c_wire.swarm_expert_wire_free_route_request(ctypes.byref(decoded))


def test_python_wire_decodes_colibri_c_microshard_accumulator(
    c_wire: ctypes.CDLL,
) -> None:
    activation = np.arange(8, dtype=np.float32).reshape(2, 4) / np.float32(8)
    accumulator = np.arange(16, dtype=np.float32).reshape(2, 2, 4) / np.float32(16)
    expert_ids = (ctypes.c_int * 4)(7, 2, 9, 1)
    weights = (ctypes.c_float * 4)(0.75, 0.25, 0.625, 0.375)
    ranks = (ctypes.c_int * 4)(0, 1, 0, 1)
    request = RouteRequest()
    request.request_id = b"c-microshard-request"
    request.model_id = b"olmoe-fixture"
    request.model_revision = b"revision-1"
    request.quantization_fingerprint = b"quantization-1"
    request.layer_id = 3
    request.batch_rows = 2
    request.latent_dimension = 4
    request.top_k = 2
    request.expert_ids_by_row = expert_ids
    request.routing_weights_by_row = weights
    request.selected_rank_by_row = ranks
    request.response_mode = 1
    request.execution_mode = 2
    request.hidden_start = 16
    request.hidden_end = 32
    request.microshard_final = 1
    request.activation.token_position = 11
    request.activation.values = activation.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    request.activation.value_count = activation.size
    request.down_accumulators.values = accumulator.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    request.down_accumulators.value_count = accumulator.size
    packet = OwnedBytes()
    error = ctypes.create_string_buffer(512)
    rc = c_wire.swarm_expert_wire_encode_route_request(
        ctypes.byref(request),
        9_999_999_999,
        b"MEASURED_LEVEL_A",
        1,
        ctypes.byref(packet),
        error,
        len(error),
    )
    assert rc == 0, error.value.decode()
    try:
        decoded, decoded_activation, decoded_accumulator, _ = decode_request(
            _c_owned_bytes(packet), include_down_accumulators=True
        )
    finally:
        c_wire.swarm_expert_wire_free_bytes(ctypes.byref(packet))
    assert decoded.execution_mode == ExpertExecutionMode.MICROSHARD
    assert decoded.microshard_final is True
    np.testing.assert_array_equal(decoded_activation, activation)
    np.testing.assert_array_equal(decoded_accumulator, accumulator)


def _c_owned_bytes(value: OwnedBytes) -> bytes:
    return ctypes.string_at(value.data, value.length)


def test_python_wire_decodes_colibri_c_response(c_wire: ctypes.CDLL) -> None:
    result = np.arange(16, dtype=np.float32).reshape(2, 2, 4) / np.float32(16)
    shape = (ctypes.c_int64 * 3)(2, 2, 4)
    tensor = OwnedBytes()
    error = ctypes.create_string_buffer(512)
    rc = c_wire.swarm_expert_wire_encode_tensor_f32(
        b"c-wire-response:expert-result",
        b"c-wire-response",
        3,
        0,
        2,
        b"revision-1",
        b"model-fingerprint-1",
        0,
        shape,
        3,
        result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(tensor),
        error,
        len(error),
    )
    assert rc == 0, error.value.decode()
    try:
        tensor_bytes = _c_owned_bytes(tensor)
        metadata = TensorWireMetadata(
            name="result",
            envelope="SWARMT01",
            dtype="float32",
            shape=[2, 2, 4],
            payload_index=0,
            raw_bytes=result.nbytes,
            encoded_bytes=len(tensor_bytes),
            checksum=hashlib.sha256(tensor_bytes).hexdigest(),
        )
        response = ExpertExecutionResponse(
            request_id="c-wire-response",
            worker_id="native-worker-0",
            model_revision="revision-1",
            layer_id=3,
            result=metadata.model_dump(mode="json"),
            execution_metadata=ExpertExecutionMetadata(
                experts_executed=[7, 2],
                bytes_read=0,
                bytes_received=0,
                bytes_sent=0,
                cache_hits=0,
                cache_misses=0,
                compute_ns=1,
                queue_ns=0,
                transfer_ns=0,
            ),
            integrity=ResultIntegrity(
                result_hash=hashlib.sha256(result.tobytes()).hexdigest(),
                model_fingerprint="model-fingerprint-1",
                worker_signature="fixture-signature",
            ),
        )
        semantic = json.dumps(
            response.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()
        semantic_buffer = ctypes.create_string_buffer(semantic)
        tensor_buffer = (ctypes.c_uint8 * len(tensor_bytes)).from_buffer_copy(tensor_bytes)
        blob = BlobView(tensor_buffer, len(tensor_bytes))
        packet = OwnedBytes()
        rc = c_wire.swarm_expert_wire_encode_packet(
            b"response",
            ctypes.cast(semantic_buffer, ctypes.c_char_p),
            ctypes.byref(blob),
            1,
            ctypes.byref(packet),
            error,
            len(error),
        )
        assert rc == 0, error.value.decode()
        try:
            decoded_response, decoded_result, _ = decode_response(_c_owned_bytes(packet))
        finally:
            c_wire.swarm_expert_wire_free_bytes(ctypes.byref(packet))
        assert decoded_response.worker_id == "native-worker-0"
        np.testing.assert_array_equal(decoded_result, result)
    finally:
        c_wire.swarm_expert_wire_free_bytes(ctypes.byref(tensor))


def test_colibri_c_wire_decodes_full_python_response(c_wire: ctypes.CDLL) -> None:
    result = np.arange(16, dtype=np.float32).reshape(2, 2, 4) / np.float32(16)
    response = ExpertExecutionResponse(
        request_id="python-encoded-response",
        worker_id="python-worker-0",
        model_revision="revision-1",
        layer_id=3,
        result={"codec": "raw_fp32"},
        execution_metadata=ExpertExecutionMetadata(
            experts_executed=[7, 2],
            bytes_read=64,
            bytes_received=32,
            bytes_sent=128,
            cache_hits=1,
            cache_misses=1,
            compute_ns=100,
            queue_ns=20,
            transfer_ns=30,
        ),
        integrity=ResultIntegrity(
            result_hash=f"sha256:{hashlib.sha256(result.tobytes()).hexdigest()}",
            model_fingerprint="model-fingerprint-1",
            worker_signature="fixture-signature",
        ),
    )
    payload, _ = encode_response(response, result)
    source = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
    decoded = RouteResponse()
    error = ctypes.create_string_buffer(512)
    rc = c_wire.swarm_expert_wire_decode_route_response(
        source, len(payload), ctypes.byref(decoded), error, len(error)
    )
    assert rc == 0, error.value.decode()
    assert decoded.request_id.rstrip(b"\0") == b"python-encoded-response"
    assert decoded.worker_id.rstrip(b"\0") == b"python-worker-0"
    assert decoded.status.rstrip(b"\0") == b"ok"
    assert decoded.layer_id == 3
    assert list(decoded.result.shape[: decoded.result.ndim]) == [2, 2, 4]
    np.testing.assert_array_equal(
        np.ctypeslib.as_array(decoded.result.values, shape=(result.size,)).reshape(result.shape),
        result,
    )


@pytest.mark.parametrize("mutation", ["header_length", "trailing", "checksum"])
def test_colibri_c_wire_rejects_malformed_frames(c_wire: ctypes.CDLL, mutation: str) -> None:
    request, activation = _per_row_request()
    payload = bytearray(encode_request(request, activation)[0])
    if mutation == "header_length":
        payload[8:12] = (len(payload)).to_bytes(4, "big")
    elif mutation == "trailing":
        payload.extend(b"x")
    else:
        payload[-1] ^= 1
    rc, decoded, _, source = _decode_with_c(c_wire, bytes(payload))
    try:
        assert rc != 0
        assert source
    finally:
        c_wire.swarm_expert_wire_free_route_request(ctypes.byref(decoded))
