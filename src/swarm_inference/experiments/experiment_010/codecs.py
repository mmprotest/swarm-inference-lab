"""Bounded activation/result codecs with measured encode/decode costs."""

from __future__ import annotations

import time
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from swarm_inference.experiments.experiment_010.schemas import (
    TensorWireMetadata,
    TransportCodec,
)
from swarm_inference.protocol.checksums import sha256_bytes


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
    """Encode a tensor without silently changing the selected codec."""

    selected = TransportCodec(codec)
    started = time.perf_counter_ns()
    source = np.ascontiguousarray(array, dtype=np.float32)
    raw = source.tobytes(order="C")
    scales: float | list[float] | None = None
    dtype: str
    if selected == TransportCodec.RAW_FP32:
        payload = raw
        dtype = "float32"
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
    else:  # pragma: no cover - exhaustive StrEnum guard
        raise ValueError(f"unsupported transport codec {selected}")
    elapsed = time.perf_counter_ns() - started
    return EncodedTensor(
        metadata=TensorWireMetadata(
            name=name,
            dtype=dtype,  # type: ignore[arg-type]
            shape=[int(value) for value in source.shape],
            codec=selected,
            payload_index=payload_index,
            raw_bytes=len(raw),
            encoded_bytes=len(payload),
            scale=scales,
            checksum=sha256_bytes(payload),
        ),
        payload=payload,
        encode_ns=elapsed,
    )


def decode_array(metadata: TensorWireMetadata, payload: bytes) -> DecodedTensor:
    started = time.perf_counter_ns()
    if len(payload) != metadata.encoded_bytes:
        raise ValueError(
            f"encoded tensor length mismatch: expected {metadata.encoded_bytes}, got {len(payload)}"
        )
    if sha256_bytes(payload) != metadata.checksum:
        raise ValueError("encoded tensor checksum mismatch")
    shape = tuple(metadata.shape)
    selected = metadata.codec
    if selected == TransportCodec.RAW_FP32:
        array = np.frombuffer(payload, dtype=np.float32).reshape(shape).copy()
    elif selected == TransportCodec.RAW_FP16:
        array = np.frombuffer(payload, dtype=np.float16).reshape(shape).astype(np.float32)
    elif selected == TransportCodec.INT8_PER_VECTOR:
        if not isinstance(metadata.scale, list):
            raise ValueError("int8_per_vector tensor requires one scale per vector")
        quantized = np.frombuffer(payload, dtype=np.int8).reshape(shape)
        vectors = _vectors(quantized)
        scales = np.asarray(metadata.scale, dtype=np.float32)
        if len(scales) != vectors.shape[0]:
            raise ValueError("int8_per_vector scale count does not match tensor vectors")
        array = (vectors.astype(np.float32) * scales[:, None]).reshape(shape)
    elif selected == TransportCodec.LOSSLESS_GENERAL:
        raw = zlib.decompress(payload)
        if len(raw) != metadata.raw_bytes:
            raise ValueError("lossless tensor decompressed length mismatch")
        array = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    else:  # pragma: no cover
        raise ValueError(f"unsupported transport codec {selected}")
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
