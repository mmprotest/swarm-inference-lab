"""Real-weight CUDA certification fixtures for Experiment 014."""

from __future__ import annotations

import ctypes
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

from swarm_inference.model.mxfp4 import MXFP4Tensor

SCHEMA_VERSION = "experiment-014-k3-cuda-real-expert-v1"
MXFP4_FORMAT = 7
MXFP4_GROUP_SIZE = 32
GROUPED_INT4_FORMAT = 4
GROUPED_INT4_GROUP_SIZE = 64
BF16_EMBEDDING_FORMAT = 8


class KimiCudaError(RuntimeError):
    """Raised when the direct Kimi CUDA path cannot be certified."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_fingerprint(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(json.dumps(list(values.shape), separators=(",", ":")).encode("ascii"))
    digest.update(values.tobytes())
    return "sha256:" + digest.hexdigest()


def _array_fingerprint_streaming(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(json.dumps(list(values.shape), separators=(",", ":")).encode("ascii"))
    raw = memoryview(values).cast("B")
    for offset in range(0, len(raw), 16 * 1024 * 1024):
        digest.update(raw[offset : offset + 16 * 1024 * 1024])
    return "sha256:" + digest.hexdigest()


def _load_safetensor_f32(
    checkpoint: Path, index: dict[str, Any], name: str
) -> tuple[np.ndarray, dict[str, Any]]:
    weight_map = index.get("weight_map", {})
    shard_name = weight_map.get(name)
    if not isinstance(shard_name, str):
        raise KimiCudaError(f"checkpoint index has no tensor {name}")
    shard = checkpoint / shard_name
    with shard.open("rb") as handle:
        header_size_raw = handle.read(8)
        if len(header_size_raw) != 8:
            raise KimiCudaError(f"truncated safetensor header: {shard}")
        header_size = struct.unpack("<Q", header_size_raw)[0]
        header = json.loads(handle.read(header_size).decode("utf-8"))
    metadata = header.get(name)
    if not isinstance(metadata, dict):
        raise KimiCudaError(f"tensor {name} absent from indexed shard {shard_name}")
    offsets = metadata.get("data_offsets")
    shape = metadata.get("shape")
    dtype = metadata.get("dtype")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not isinstance(shape, list)
        or any(not isinstance(value, int) or value < 0 for value in shape)
    ):
        raise KimiCudaError(f"invalid safetensor metadata for {name}")
    start, end = int(offsets[0]), int(offsets[1])
    count = int(np.prod(shape, dtype=np.int64))
    data_offset = 8 + header_size + start
    if dtype == "BF16":
        if end - start != count * 2:
            raise KimiCudaError(f"BF16 byte count mismatch for {name}")
        raw = np.memmap(shard, mode="r", dtype="<u2", offset=data_offset, shape=(count,))
        raw_bits = np.asarray(raw, dtype=np.uint16).reshape(shape)
        values = np.ascontiguousarray(
            (raw_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
        )
    elif dtype == "F32":
        if end - start != count * 4:
            raise KimiCudaError(f"F32 byte count mismatch for {name}")
        raw = np.memmap(shard, mode="r", dtype="<f4", offset=data_offset, shape=tuple(shape))
        raw_bits = np.asarray(raw, dtype=np.float32)
        values = np.ascontiguousarray(raw_bits, dtype=np.float32)
    else:
        raise KimiCudaError(f"unsupported production fixture dtype {dtype!r} for {name}")
    raw_bytes = np.memmap(
        shard, mode="r", dtype=np.uint8, offset=data_offset, shape=(end - start,)
    )
    raw_digest = hashlib.sha256(np.asarray(raw_bytes).tobytes()).hexdigest()
    return values, {
        "name": name,
        "source_shard": shard_name,
        "source_dtype": dtype,
        "shape": shape,
        "source_bytes": end - start,
        "source_raw_sha256": raw_digest,
        "runtime_f32_fingerprint": _array_fingerprint(values),
    }


def _load_safetensor_raw_bf16(
    checkpoint: Path, index: dict[str, Any], name: str
) -> tuple[np.memmap, dict[str, Any]]:
    """Memory-map one BF16 tensor and hash its exact source range without expanding it."""

    shard_name = index.get("weight_map", {}).get(name)
    if not isinstance(shard_name, str):
        raise KimiCudaError(f"checkpoint index has no tensor {name}")
    shard = checkpoint / shard_name
    with shard.open("rb") as handle:
        header_size_raw = handle.read(8)
        if len(header_size_raw) != 8:
            raise KimiCudaError(f"truncated safetensor header: {shard}")
        header_size = struct.unpack("<Q", header_size_raw)[0]
        header = json.loads(handle.read(header_size).decode("utf-8"))
    metadata = header.get(name)
    if not isinstance(metadata, dict) or metadata.get("dtype") != "BF16":
        raise KimiCudaError(f"tensor {name} is not a BF16 production tensor")
    shape = metadata.get("shape")
    offsets = metadata.get("data_offsets")
    if (
        not isinstance(shape, list)
        or any(not isinstance(value, int) or value < 0 for value in shape)
        or not isinstance(offsets, list)
        or len(offsets) != 2
    ):
        raise KimiCudaError(f"invalid safetensor metadata for {name}")
    start, end = (int(offsets[0]), int(offsets[1]))
    count = int(np.prod(shape, dtype=np.int64))
    if end - start != count * 2:
        raise KimiCudaError(f"BF16 byte count mismatch for {name}")
    data_offset = 8 + header_size + start
    raw_digest = hashlib.sha256()
    fingerprint = hashlib.sha256()
    fingerprint.update(b"uint16-bf16")
    fingerprint.update(json.dumps(shape, separators=(",", ":")).encode("ascii"))
    with shard.open("rb") as handle:
        handle.seek(data_offset)
        remaining = end - start
        while remaining:
            chunk = handle.read(min(16 * 1024 * 1024, remaining))
            if not chunk:
                raise KimiCudaError(f"truncated tensor payload for {name}")
            raw_digest.update(chunk)
            fingerprint.update(chunk)
            remaining -= len(chunk)
    values = np.memmap(
        shard, mode="r", dtype="<u2", offset=data_offset, shape=tuple(shape)
    )
    return values, {
        "name": name,
        "source_shard": shard_name,
        "source_dtype": "BF16",
        "shape": shape,
        "source_range": [data_offset, data_offset + end - start],
        "source_bytes": end - start,
        "source_raw_sha256": raw_digest.hexdigest(),
        "runtime_bf16_fingerprint": "sha256:" + fingerprint.hexdigest(),
    }


def _numerical_metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    expected = np.asarray(reference, dtype=np.float64)
    observed = np.asarray(actual, dtype=np.float64)
    if expected.shape != observed.shape:
        raise KimiCudaError(f"output shape mismatch: {observed.shape} != {expected.shape}")
    delta = observed - expected
    reference_norm = float(np.linalg.norm(expected))
    actual_norm = float(np.linalg.norm(observed))
    delta_norm = float(np.linalg.norm(delta))
    denominator = reference_norm if reference_norm else 1.0
    cosine_denominator = reference_norm * actual_norm
    cosine = (
        float(np.dot(expected.ravel(), observed.ravel()) / cosine_denominator)
        if cosine_denominator
        else 1.0
    )
    return {
        "maximum_absolute_error": float(np.max(np.abs(delta))),
        "mean_absolute_error": float(np.mean(np.abs(delta))),
        "relative_l2_error": delta_norm / denominator,
        "cosine_similarity": cosine,
        "reference_l2": reference_norm,
        "actual_l2": actual_norm,
        "reference_finite": bool(np.isfinite(expected).all()),
        "actual_finite": bool(np.isfinite(observed).all()),
    }


def _percentiles(milliseconds: list[float]) -> dict[str, float]:
    values = np.asarray(milliseconds, dtype=np.float64)
    return {
        "minimum_ms": float(np.min(values)),
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "maximum_ms": float(np.max(values)),
        "standard_deviation_ms": float(np.std(values)),
    }


@dataclass(frozen=True, slots=True)
class _RealExpert:
    gate: MXFP4Tensor
    up: MXFP4Tensor
    down: MXFP4Tensor
    names: dict[str, dict[str, str]]
    source_shards: list[str]


@dataclass(frozen=True, slots=True)
class _GroupedInt4Tensor:
    packed: np.ndarray
    scales: np.ndarray
    source: np.ndarray
    input_dimension: int
    output_dimension: int


@dataclass(frozen=True, slots=True)
class _QuantizedInt8Tensor:
    weights: np.ndarray
    scales: np.ndarray
    input_dimension: int
    output_dimension: int


def _quantize_bf16_rows_int8(
    source: np.ndarray, *, chunk_rows: int = 1024
) -> _QuantizedInt8Tensor:
    if source.ndim != 2 or source.dtype != np.dtype("<u2"):
        raise KimiCudaError(f"int8 quantization requires BF16 uint16 [O,I], got {source}")
    output_dimension, input_dimension = (int(source.shape[0]), int(source.shape[1]))
    quantized = np.empty((output_dimension, input_dimension), dtype=np.int8)
    scales = np.empty(output_dimension, dtype=np.float32)
    for start in range(0, output_dimension, chunk_rows):
        stop = min(output_dimension, start + chunk_rows)
        bits = np.asarray(source[start:stop], dtype=np.uint16)
        values = (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
        absolute_maximum = np.max(np.abs(values), axis=1).astype(np.float32)
        chunk_scales = np.maximum(
            absolute_maximum / np.float32(127.0), np.float32(1e-20)
        ).astype(np.float32)
        inverse = np.float32(1.0) / chunk_scales
        chunk = np.rint(values * inverse[:, None])
        quantized[start:stop] = np.clip(chunk, -127, 127).astype(np.int8)
        scales[start:stop] = chunk_scales
    return _QuantizedInt8Tensor(
        weights=quantized,
        scales=scales,
        input_dimension=input_dimension,
        output_dimension=output_dimension,
    )


def _quantized_int8_matvec(
    tensor: _QuantizedInt8Tensor, activation: np.ndarray, *, chunk_rows: int = 1024
) -> np.ndarray:
    """Reference matvec over the exact per-row int8 runtime representation."""

    source = np.ascontiguousarray(activation, dtype=np.float32)
    if source.shape != (tensor.input_dimension,):
        raise KimiCudaError(
            f"int8 matvec input {source.shape} != {(tensor.input_dimension,)}"
        )
    output = np.empty(tensor.output_dimension, dtype=np.float32)
    for start in range(0, tensor.output_dimension, chunk_rows):
        stop = min(tensor.output_dimension, start + chunk_rows)
        rows = np.asarray(tensor.weights[start:stop], dtype=np.float32)
        dots = np.asarray(rows @ source, dtype=np.float32)
        output[start:stop] = np.asarray(
            dots * tensor.scales[start:stop], dtype=np.float32
        )
    return output


def _rmsnorm_reference(
    source: np.ndarray, weight: np.ndarray, epsilon: float
) -> np.ndarray:
    values = np.ascontiguousarray(source, dtype=np.float32)
    scale = np.float32(1.0) / np.sqrt(
        np.float32(np.sum(values.astype(np.float64) ** 2, dtype=np.float64) / values.size)
        + np.float32(epsilon)
    )
    return np.ascontiguousarray(values * scale * weight, dtype=np.float32)


def _mla_stage_reference(
    inputs: np.ndarray,
    tensors: dict[str, _QuantizedInt8Tensor],
    query_norm: np.ndarray,
    kv_norm: np.ndarray,
    *,
    heads: int,
    query_nope: int,
    query_rope: int,
    value_dimension: int,
    kv_lora: int,
    attention_scale: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent sequential decode oracle for the production int8 Gated MLA."""

    steps = int(inputs.shape[0])
    latent_cache = np.zeros((steps, kv_lora), dtype=np.float32)
    rope_cache = np.zeros((steps, query_rope), dtype=np.float32)
    outputs = np.empty((steps, tensors["output"].output_dimension), dtype=np.float32)
    kvb = tensors["kv_b"]
    kvb_weights = kvb.weights.reshape(heads, query_nope + value_dimension, kv_lora)
    kvb_scales = kvb.scales.reshape(heads, query_nope + value_dimension)
    for step, activation in enumerate(inputs):
        query_low_rank = _quantized_int8_matvec(tensors["query_a"], activation)
        query_low_rank = _rmsnorm_reference(query_low_rank, query_norm, epsilon)
        query = _quantized_int8_matvec(tensors["query_b"], query_low_rank).reshape(
            heads, query_nope + query_rope
        )
        compressed_kv = _quantized_int8_matvec(tensors["kv_a"], activation)
        latent_cache[step] = _rmsnorm_reference(
            compressed_kv[:kv_lora], kv_norm, epsilon
        )
        rope_cache[step] = compressed_kv[kv_lora:]
        gate = _quantized_int8_matvec(tensors["gate"], activation).reshape(
            heads, value_dimension
        )
        context = np.empty((heads, value_dimension), dtype=np.float32)
        for head in range(heads):
            q_nope = query[head, :query_nope]
            q_rope = query[head, query_nope:]
            query_rows = np.asarray(kvb_weights[head, :query_nope], dtype=np.float32)
            query_coefficients = np.asarray(
                q_nope * kvb_scales[head, :query_nope], dtype=np.float32
            )
            absorbed_query = np.asarray(
                query_coefficients @ query_rows, dtype=np.float32
            )
            scores = np.asarray(
                latent_cache[: step + 1] @ absorbed_query
                + rope_cache[: step + 1] @ q_rope,
                dtype=np.float32,
            )
            scores = np.asarray(scores * np.float32(attention_scale), dtype=np.float32)
            probabilities = np.exp(
                np.asarray(scores - np.max(scores), dtype=np.float32)
            ).astype(np.float32)
            probabilities = np.asarray(
                probabilities / np.sum(probabilities, dtype=np.float32), dtype=np.float32
            )
            latent_context = np.asarray(
                probabilities @ latent_cache[: step + 1], dtype=np.float32
            )
            value_rows = np.asarray(kvb_weights[head, query_nope:], dtype=np.float32)
            values = np.asarray(value_rows @ latent_context, dtype=np.float32)
            values = np.asarray(
                values * kvb_scales[head, query_nope:], dtype=np.float32
            )
            sigmoid_gate = np.asarray(
                np.float32(1.0) / (np.float32(1.0) + np.exp(-gate[head])),
                dtype=np.float32,
            )
            context[head] = np.asarray(values * sigmoid_gate, dtype=np.float32)
        outputs[step] = _quantized_int8_matvec(tensors["output"], context.reshape(-1))
    return outputs, latent_cache, rope_cache


def _quantize_grouped_int4(
    source: np.ndarray, *, retain_source: bool = True
) -> _GroupedInt4Tensor:
    """Mirror Kimi's production BF16-to-int4-g64 load-time representation."""

    weights = np.ascontiguousarray(source, dtype=np.float32)
    if weights.ndim != 2 or weights.shape[1] % GROUPED_INT4_GROUP_SIZE:
        raise KimiCudaError(f"grouped int4 requires [O,I] with I%64=0, got {weights.shape}")
    output_dimension, input_dimension = weights.shape
    grouped = weights.reshape(output_dimension, -1, GROUPED_INT4_GROUP_SIZE)
    absolute_maximum = np.max(np.abs(grouped), axis=2).astype(np.float32)
    scales = np.maximum(
        absolute_maximum / np.float32(7.0), np.float32(1e-20)
    ).astype(np.float32)
    quantized = np.rint(grouped / scales[:, :, None])
    quantized = np.clip(quantized, -8, 7).astype(np.int8)
    encoded = (quantized.astype(np.int16) + 8).astype(np.uint8)
    packed = np.ascontiguousarray(
        (encoded[:, :, 0::2] | (encoded[:, :, 1::2] << np.uint8(4))).reshape(
            output_dimension, input_dimension // 2
        ),
        dtype=np.uint8,
    )
    return _GroupedInt4Tensor(
        packed=packed,
        scales=np.ascontiguousarray(scales, dtype=np.float32),
        source=weights if retain_source else np.empty((0,), dtype=np.float32),
        input_dimension=input_dimension,
        output_dimension=output_dimension,
    )


def _dequantize_grouped_int4(tensor: _GroupedInt4Tensor) -> np.ndarray:
    encoded = tensor.packed.reshape(tensor.output_dimension, -1)
    signed = np.empty(
        (tensor.output_dimension, tensor.input_dimension), dtype=np.int8
    )
    signed[:, 0::2] = (encoded & np.uint8(0x0F)).astype(np.int8) - 8
    signed[:, 1::2] = (encoded >> np.uint8(4)).astype(np.int8) - 8
    grouped = signed.reshape(
        tensor.output_dimension, -1, GROUPED_INT4_GROUP_SIZE
    ).astype(np.float32)
    return np.ascontiguousarray(
        (grouped * tensor.scales[:, :, None]).reshape(
            tensor.output_dimension, tensor.input_dimension
        ),
        dtype=np.float32,
    )


def _grouped_int4_matvec(
    tensor: _GroupedInt4Tensor, activation: np.ndarray, *, chunk_rows: int = 1024
) -> np.ndarray:
    if activation.shape != (tensor.input_dimension,):
        raise KimiCudaError(
            f"int4 matvec input {activation.shape} != {(tensor.input_dimension,)}"
        )
    output = np.empty(tensor.output_dimension, dtype=np.float32)
    for start in range(0, tensor.output_dimension, chunk_rows):
        stop = min(tensor.output_dimension, start + chunk_rows)
        packed = tensor.packed[start:stop]
        signed = np.empty((stop - start, tensor.input_dimension), dtype=np.int8)
        signed[:, 0::2] = (packed & np.uint8(0x0F)).astype(np.int8) - 8
        signed[:, 1::2] = (packed >> np.uint8(4)).astype(np.int8) - 8
        grouped = signed.reshape(
            stop - start, -1, GROUPED_INT4_GROUP_SIZE
        ).astype(np.float32)
        dequantized = (grouped * tensor.scales[start:stop, :, None]).reshape(
            stop - start, tensor.input_dimension
        )
        output[start:stop] = np.asarray(dequantized @ activation, dtype=np.float32)
    return output


def _kda_core_reference(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    gate: np.ndarray,
    decay: np.ndarray,
    beta_raw: np.ndarray,
    conv_q: np.ndarray,
    conv_k: np.ndarray,
    conv_v: np.ndarray,
    dt: np.ndarray,
    a: np.ndarray,
    output_norm: np.ndarray,
    *,
    gate_lower_bound: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent sequential FP32 KDA core matching ``kimi_k3.c``."""

    steps, projection = q.shape
    heads, head_dimension, conv_width = 96, 128, 4
    if (
        k.shape != q.shape
        or v.shape != q.shape
        or gate.shape != q.shape
        or decay.shape != q.shape
        or beta_raw.shape != (steps, heads)
        or projection != heads * head_dimension
    ):
        raise KimiCudaError("invalid projected KDA reference fixture")
    taps = np.stack(
        [
            conv_q.reshape(projection, conv_width),
            conv_k.reshape(projection, conv_width),
            conv_v.reshape(projection, conv_width),
        ]
    ).astype(np.float32)
    windows = np.zeros((3, projection, conv_width), dtype=np.float32)
    state = np.zeros((heads, head_dimension, head_dimension), dtype=np.float32)
    output = np.empty((steps, projection), dtype=np.float32)
    q_scale = np.float32(1.0 / np.sqrt(np.float32(head_dimension)))
    sources = (q, k, v)
    for step in range(steps):
        convolved: list[np.ndarray] = []
        for kind in range(3):
            windows[kind, :, :-1] = windows[kind, :, 1:]
            windows[kind, :, -1] = sources[kind][step]
            accumulator = np.zeros(projection, dtype=np.float32)
            for tap in range(conv_width):
                accumulator = np.asarray(
                    accumulator + taps[kind, :, tap] * windows[kind, :, tap],
                    dtype=np.float32,
                )
            convolved.append(
                np.asarray(
                    accumulator
                    / (np.float32(1.0) + np.exp(-accumulator)),
                    dtype=np.float32,
                )
            )
        q_heads = convolved[0].reshape(heads, head_dimension)
        k_heads = convolved[1].reshape(heads, head_dimension)
        v_heads = convolved[2].reshape(heads, head_dimension)
        for head in range(heads):
            square_q = np.float32(0.0)
            square_k = np.float32(0.0)
            for item in range(head_dimension):
                square_q = np.float32(
                    square_q + q_heads[head, item] * q_heads[head, item]
                )
                square_k = np.float32(
                    square_k + k_heads[head, item] * k_heads[head, item]
                )
            qn = np.asarray(
                q_heads[head]
                * (np.float32(1.0) / np.sqrt(square_q + np.float32(1e-6)))
                * q_scale,
                dtype=np.float32,
            )
            kn = np.asarray(
                k_heads[head]
                * (np.float32(1.0) / np.sqrt(square_k + np.float32(1e-6))),
                dtype=np.float32,
            )
            start = head * head_dimension
            z = np.asarray(
                decay[step, start : start + head_dimension]
                + dt[start : start + head_dimension],
                dtype=np.float32,
            )
            alpha = np.asarray(
                np.exp(
                    np.float32(gate_lower_bound)
                    / (
                        np.float32(1.0)
                        + np.exp(-np.float32(a[head]) * z)
                    )
                ),
                dtype=np.float32,
            )
            beta = np.float32(
                np.float32(1.0)
                / (
                    np.float32(1.0)
                    + np.exp(-np.float32(beta_raw[step, head]))
                )
            )
            k_state = np.zeros(head_dimension, dtype=np.float32)
            for key in range(head_dimension):
                state[head, key] = np.asarray(
                    state[head, key] * alpha[key], dtype=np.float32
                )
                k_state = np.asarray(
                    k_state + kn[key] * state[head, key], dtype=np.float32
                )
            value_delta = np.asarray(
                (v_heads[head] - k_state) * beta, dtype=np.float32
            )
            head_output = np.zeros(head_dimension, dtype=np.float32)
            for key in range(head_dimension):
                state[head, key] = np.asarray(
                    state[head, key] + kn[key] * value_delta, dtype=np.float32
                )
                head_output = np.asarray(
                    head_output + qn[key] * state[head, key], dtype=np.float32
                )
            mean_square = np.sum(
                head_output.astype(np.float64) ** 2, dtype=np.float64
            ) / head_dimension
            inverse_rms = np.float32(1.0) / np.sqrt(
                np.float32(mean_square) + np.float32(epsilon)
            )
            gate_values = gate[step, start : start + head_dimension]
            sigmoid_gate = np.asarray(
                np.float32(1.0)
                / (np.float32(1.0) + np.exp(-gate_values)),
                dtype=np.float32,
            )
            output[step, start : start + head_dimension] = np.asarray(
                head_output * inverse_rms * output_norm * sigmoid_gate,
                dtype=np.float32,
            )
    return output, state, windows


def _load_real_expert(checkpoint: Path, layer: int, expert: int) -> _RealExpert:
    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.is_file():
        raise KimiCudaError(f"missing checkpoint index: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise KimiCudaError("checkpoint index has no weight_map")
    prefix = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}"
    names = {
        "gate": {
            "packed": f"{prefix}.w1.weight_packed",
            "scale": f"{prefix}.w1.weight_scale",
        },
        "down": {
            "packed": f"{prefix}.w2.weight_packed",
            "scale": f"{prefix}.w2.weight_scale",
        },
        "up": {
            "packed": f"{prefix}.w3.weight_packed",
            "scale": f"{prefix}.w3.weight_scale",
        },
    }
    missing = [name for pair in names.values() for name in pair.values() if name not in weight_map]
    if missing:
        raise KimiCudaError(f"missing routed-expert tensors: {missing}")
    tensors: dict[str, np.ndarray] = {}
    shards = sorted({str(weight_map[name]) for pair in names.values() for name in pair.values()})
    for shard in shards:
        path = checkpoint / shard
        if not path.is_file():
            raise KimiCudaError(f"missing routed-expert shard: {path}")
        wanted = {
            name
            for pair in names.values()
            for name in pair.values()
            if weight_map[name] == shard
        }
        with safe_open(str(path), framework="numpy") as handle:
            for name in wanted:
                tensors[name] = np.ascontiguousarray(handle.get_tensor(name), dtype=np.uint8)

    def matrix(role: str, input_dimension: int, output_dimension: int) -> MXFP4Tensor:
        return MXFP4Tensor(
            packed=tensors[names[role]["packed"]],
            scales=tensors[names[role]["scale"]],
            input_dimension=input_dimension,
            output_dimension=output_dimension,
        )

    latent = 3584
    intermediate = 3072
    return _RealExpert(
        gate=matrix("gate", latent, intermediate),
        up=matrix("up", latent, intermediate),
        down=matrix("down", intermediate, latent),
        names=names,
        source_shards=shards,
    )


class _CudaRuntime:
    """Strict direct binding: every failed CUDA call aborts instead of falling back."""

    def __init__(self, library_path: Path, device: int) -> None:
        self.path = library_path.resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.sha256 = _sha256_file(self.path)
        self._library = ctypes.CDLL(str(self.path))
        pointer = ctypes.c_void_p
        self._library.coli_cuda_init.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self._library.coli_cuda_init.restype = ctypes.c_int
        self._library.coli_cuda_shutdown.argtypes = []
        self._library.coli_cuda_shutdown.restype = None
        self._library.coli_cuda_binary_min_compute_capability.argtypes = []
        self._library.coli_cuda_binary_min_compute_capability.restype = ctypes.c_int
        self._library.coli_cuda_binary_has_forward_ptx.argtypes = []
        self._library.coli_cuda_binary_has_forward_ptx.restype = ctypes.c_int
        self._library.coli_cuda_binary_accepts_compute_capability.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._library.coli_cuda_binary_accepts_compute_capability.restype = ctypes.c_int
        self._library.coli_cuda_tensor_upload_g.argtypes = [
            ctypes.POINTER(pointer),
            pointer,
            pointer,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._library.coli_cuda_tensor_upload_g.restype = ctypes.c_int
        self._library.coli_cuda_tensor_free.argtypes = [pointer]
        self._library.coli_cuda_tensor_free.restype = None
        self._library.coli_cuda_tensor_bytes.argtypes = [pointer]
        self._library.coli_cuda_tensor_bytes.restype = ctypes.c_size_t
        self._library.coli_cuda_mem_info.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self._library.coli_cuda_mem_info.restype = ctypes.c_int
        try:
            shared_memory_query = (
                self._library.coli_cuda_device_shared_memory_limits
            )
        except AttributeError:
            shared_memory_query = None
        if shared_memory_query is not None:
            shared_memory_query.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_size_t),
            ]
            shared_memory_query.restype = ctypes.c_int
        self._library.coli_cuda_profile_begin.argtypes = [ctypes.c_int]
        self._library.coli_cuda_profile_begin.restype = ctypes.c_int
        self._library.coli_cuda_profile_end.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
        ]
        self._library.coli_cuda_profile_end.restype = ctypes.c_int
        self._library.coli_cuda_kimi_expert_mlp.argtypes = [
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
        ]
        self._library.coli_cuda_kimi_expert_mlp.restype = ctypes.c_int
        self._library.coli_cuda_kimi_expert_mlp_dev.argtypes = [
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
        ]
        self._library.coli_cuda_kimi_expert_mlp_dev.restype = ctypes.c_int
        try:
            expert_batch_query = (
                self._library.coli_cuda_kimi_expert_max_certified_batch
            )
        except AttributeError:
            expert_batch_query = None
        if expert_batch_query is not None:
            expert_batch_query.argtypes = []
            expert_batch_query.restype = ctypes.c_int
        try:
            expert_batch_support_query = (
                self._library.coli_cuda_kimi_expert_supports_batch
            )
        except AttributeError:
            expert_batch_support_query = None
        if expert_batch_support_query is not None:
            expert_batch_support_query.argtypes = [ctypes.c_int]
            expert_batch_support_query.restype = ctypes.c_int
        self._library.coli_cuda_kimi_embedding_bf16_dev.argtypes = [
            pointer,
            pointer,
            pointer,
            ctypes.c_int,
        ]
        self._library.coli_cuda_kimi_embedding_bf16_dev.restype = ctypes.c_int
        self._library.coli_cuda_kimi_moe_reduce_dev.argtypes = [
            ctypes.c_int,
            pointer,
            pointer,
            pointer,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._library.coli_cuda_kimi_moe_reduce_dev.restype = ctypes.c_int
        try:
            indexed_copy_rows = self._library.coli_cuda_kimi_indexed_copy_rows_dev
        except AttributeError:
            indexed_copy_rows = None
        if indexed_copy_rows is not None:
            indexed_copy_rows.argtypes = [
                ctypes.c_int,
                pointer,
                pointer,
                pointer,
                ctypes.c_int,
                ctypes.c_int,
            ]
            indexed_copy_rows.restype = ctypes.c_int
        self.indexed_copy_rows_function = indexed_copy_rows
        self._library.coli_cuda_kimi_attnres_mix_dev.argtypes = [
            ctypes.c_int,
            pointer,
            pointer,
            pointer,
            ctypes.c_int,
            pointer,
            ctypes.c_int,
            ctypes.c_float,
        ]
        self._library.coli_cuda_kimi_attnres_mix_dev.restype = ctypes.c_int
        self._library.coli_cuda_kimi_kda_core_dev.argtypes = [
            ctypes.c_int,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
        ]
        self._library.coli_cuda_kimi_kda_core_dev.restype = ctypes.c_int
        self._library.coli_cuda_kimi_mla_cache_append_dev.argtypes = [
            ctypes.c_int,
            pointer,
            pointer,
            pointer,
            pointer,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        self._library.coli_cuda_kimi_mla_cache_append_dev.restype = ctypes.c_int
        self._library.coli_cuda_kimi_mla_absorb_dev.argtypes = [
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        self._library.coli_cuda_kimi_mla_absorb_dev.restype = ctypes.c_int
        try:
            mla_absorb_triangular = (
                self._library.coli_cuda_kimi_mla_absorb_triangular_dev
            )
        except AttributeError:
            mla_absorb_triangular = None
        if mla_absorb_triangular is not None:
            mla_absorb_triangular.argtypes = [
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
            ]
            mla_absorb_triangular.restype = ctypes.c_int
        self.mla_absorb_triangular_function = mla_absorb_triangular
        try:
            mla_absorb_dcp = self._library.coli_cuda_kimi_mla_absorb_dcp_dev
        except AttributeError:
            mla_absorb_dcp = None
        if mla_absorb_dcp is not None:
            mla_absorb_dcp.argtypes = [
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
            ]
            mla_absorb_dcp.restype = ctypes.c_int
        self.mla_absorb_dcp_function = mla_absorb_dcp
        self._library.coli_cuda_kimi_mla_sigmoid_gate_dev.argtypes = [
            ctypes.c_int,
            pointer,
            pointer,
            ctypes.c_size_t,
        ]
        self._library.coli_cuda_kimi_mla_sigmoid_gate_dev.restype = ctypes.c_int
        self._library.coli_cuda_pipe_alloc.argtypes = [ctypes.c_int, ctypes.c_size_t]
        self._library.coli_cuda_pipe_alloc.restype = pointer
        self._library.coli_cuda_pipe_free.argtypes = [ctypes.c_int, pointer]
        self._library.coli_cuda_pipe_free.restype = None
        self._library.coli_cuda_pipe_upload.argtypes = [
            ctypes.c_int,
            pointer,
            pointer,
            ctypes.c_size_t,
        ]
        self._library.coli_cuda_pipe_upload.restype = ctypes.c_int
        self._library.coli_cuda_pipe_download.argtypes = [
            ctypes.c_int,
            pointer,
            pointer,
            ctypes.c_size_t,
        ]
        self._library.coli_cuda_pipe_download.restype = ctypes.c_int
        self._library.coli_cuda_pipe_sync.argtypes = [ctypes.c_int]
        self._library.coli_cuda_pipe_sync.restype = ctypes.c_int
        try:
            error_state_query = self._library.coli_cuda_error_state_ok
        except AttributeError:
            error_state_query = None
        if error_state_query is not None:
            error_state_query.argtypes = [ctypes.c_int]
            error_state_query.restype = ctypes.c_int
        self._library.coli_cuda_pipe_router.argtypes = [
            ctypes.c_int,
            pointer,
            pointer,
            pointer,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._library.coli_cuda_pipe_router.restype = ctypes.c_int
        try:
            router_batch = self._library.coli_cuda_pipe_router_batch
        except AttributeError:
            router_batch = None
        if router_batch is not None:
            router_batch.argtypes = [
                ctypes.c_int,
                pointer,
                pointer,
                pointer,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int),
            ]
            router_batch.restype = ctypes.c_int
        self.router_batch_function = router_batch
        self._library.coli_cuda_pipe_gemm.argtypes = [
            pointer,
            pointer,
            pointer,
            ctypes.c_int,
        ]
        self._library.coli_cuda_pipe_gemm.restype = ctypes.c_int
        try:
            dense_rows_reuse = self._library.coli_cuda_pipe_gemm_rows_reuse
        except AttributeError:
            dense_rows_reuse = None
        if dense_rows_reuse is not None:
            dense_rows_reuse.argtypes = [pointer, pointer, pointer, ctypes.c_int]
            dense_rows_reuse.restype = ctypes.c_int
        self.dense_rows_reuse_function = dense_rows_reuse
        self._library.coli_cuda_pipe_add.argtypes = [
            ctypes.c_int,
            pointer,
            pointer,
            ctypes.c_size_t,
        ]
        self._library.coli_cuda_pipe_add.restype = ctypes.c_int
        self._library.coli_cuda_pipe_copy2d.argtypes = [
            ctypes.c_int,
            pointer,
            ctypes.c_int,
            pointer,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._library.coli_cuda_pipe_copy2d.restype = ctypes.c_int
        self._library.coli_cuda_pipe_rmsnorm_s.argtypes = [
            ctypes.c_int,
            pointer,
            pointer,
            pointer,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._library.coli_cuda_pipe_rmsnorm_s.restype = ctypes.c_int
        self._library.coli_cuda_kimi_router_stats.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        self._library.coli_cuda_kimi_router_stats.restype = None
        self._library.coli_cuda_kimi_router_stats_reset.argtypes = []
        self._library.coli_cuda_kimi_router_stats_reset.restype = None
        self._library.coli_cuda_kimi_dense_stats.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_double),
        ]
        self._library.coli_cuda_kimi_dense_stats.restype = None
        self._library.coli_cuda_kimi_dense_stats_reset.argtypes = []
        self._library.coli_cuda_kimi_dense_stats_reset.restype = None
        self._library.coli_cuda_kimi_expert_stats.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        self._library.coli_cuda_kimi_expert_stats.restype = None
        self._library.coli_cuda_kimi_expert_phase_stats.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        self._library.coli_cuda_kimi_expert_phase_stats.restype = None
        self._library.coli_cuda_kimi_expert_pair_stats.argtypes = [
            ctypes.POINTER(ctypes.c_double)
        ]
        self._library.coli_cuda_kimi_expert_pair_stats.restype = None
        self._library.coli_cuda_kimi_expert_stats_reset.argtypes = []
        self._library.coli_cuda_kimi_expert_stats_reset.restype = None
        self._library.coli_cuda_kimi_set_telemetry.argtypes = [ctypes.c_int]
        self._library.coli_cuda_kimi_set_telemetry.restype = ctypes.c_int
        self._library.coli_cuda_kimi_set_up_first.argtypes = [ctypes.c_int]
        self._library.coli_cuda_kimi_set_up_first.restype = ctypes.c_int
        self._library.coli_cuda_kimi_set_fused_gate_up.argtypes = [ctypes.c_int]
        self._library.coli_cuda_kimi_set_fused_gate_up.restype = ctypes.c_int
        devices = (ctypes.c_int * 1)(device)
        if self._library.coli_cuda_init(devices, 1) != 1:
            raise KimiCudaError("coli_cuda_init rejected the requested device")
        self.device = device
        self.binary_min_compute_capability = int(
            self._library.coli_cuda_binary_min_compute_capability()
        )
        self.binary_has_forward_ptx = bool(self._library.coli_cuda_binary_has_forward_ptx())
        self.capability_negotiation = {
            f"sm_{major}{minor}": bool(
                self._library.coli_cuda_binary_accepts_compute_capability(major, minor)
            )
            for major, minor in ((7, 5), (8, 6), (8, 9), (12, 0))
        }
        self.shared_memory_limits: dict[str, int] | None = None
        if shared_memory_query is not None:
            default_shared = ctypes.c_size_t()
            optin_shared = ctypes.c_size_t()
            if (
                shared_memory_query(
                    device,
                    ctypes.byref(default_shared),
                    ctypes.byref(optin_shared),
                )
                != 1
            ):
                self._library.coli_cuda_shutdown()
                raise KimiCudaError("CUDA shared-memory capability query failed")
            self.shared_memory_limits = {
                "default_per_block_bytes": int(default_shared.value),
                "optin_per_block_bytes": int(optin_shared.value),
            }
        self.expert_max_certified_batch = (
            int(expert_batch_query()) if expert_batch_query is not None else 2
        )
        self.expert_batch_capability_source = (
            "native_export"
            if expert_batch_query is not None
            else "legacy_python_safety_ceiling"
        )
        if self.expert_max_certified_batch < 1:
            self._library.coli_cuda_shutdown()
            raise KimiCudaError("invalid Kimi expert batch capability")
        self.expert_supported_batches = tuple(
            batch
            for batch in range(1, self.expert_max_certified_batch + 1)
            if (
                bool(expert_batch_support_query(batch))
                if expert_batch_support_query is not None
                else True
            )
        )
        self.expert_batch_set_capability_source = (
            "native_exact_size_export"
            if expert_batch_support_query is not None
            else "contiguous_maximum_contract"
        )
        if 1 not in self.expert_supported_batches:
            self._library.coli_cuda_shutdown()
            raise KimiCudaError("Kimi expert capability omitted batch 1")
        self.cuda_error_state_query = error_state_query
        self._tensors: list[ctypes.c_void_p] = []

    @staticmethod
    def _pointer(array: np.ndarray) -> ctypes.c_void_p:
        return ctypes.c_void_p(int(array.ctypes.data))

    def mem_info(self) -> dict[str, int]:
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        if self._library.coli_cuda_mem_info(self.device, ctypes.byref(free), ctypes.byref(total)) != 1:
            raise KimiCudaError("CUDA memory query failed")
        return {"free_bytes": int(free.value), "total_bytes": int(total.value)}

    def profile_begin(self) -> None:
        if self._library.coli_cuda_profile_begin(self.device) != 1:
            raise KimiCudaError("CUDA event profiler rejected begin")

    def profile_end(self) -> float:
        elapsed = ctypes.c_double()
        if self._library.coli_cuda_profile_end(self.device, ctypes.byref(elapsed)) != 1:
            raise KimiCudaError("CUDA event profiler rejected end")
        return float(elapsed.value)

    def upload(self, tensor: MXFP4Tensor) -> ctypes.c_void_p:
        handle = ctypes.c_void_p()
        packed = np.ascontiguousarray(tensor.packed, dtype=np.uint8)
        scales = np.ascontiguousarray(tensor.scales, dtype=np.uint8)
        status = self._library.coli_cuda_tensor_upload_g(
            ctypes.byref(handle),
            self._pointer(packed),
            self._pointer(scales),
            MXFP4_FORMAT,
            tensor.input_dimension,
            tensor.output_dimension,
            self.device,
            MXFP4_GROUP_SIZE,
        )
        if status != 1 or not handle.value:
            raise KimiCudaError("native MXFP4 CUDA tensor upload failed")
        self._tensors.append(handle)
        return handle

    def upload_grouped_int4(self, tensor: _GroupedInt4Tensor) -> ctypes.c_void_p:
        handle = ctypes.c_void_p()
        status = self._library.coli_cuda_tensor_upload_g(
            ctypes.byref(handle),
            self._pointer(tensor.packed),
            self._pointer(tensor.scales),
            GROUPED_INT4_FORMAT,
            tensor.input_dimension,
            tensor.output_dimension,
            self.device,
            GROUPED_INT4_GROUP_SIZE,
        )
        if status != 1 or not handle.value:
            raise KimiCudaError("grouped-int4 CUDA tensor upload failed")
        self._tensors.append(handle)
        return handle

    def upload_float32(self, matrix: np.ndarray) -> ctypes.c_void_p:
        values = np.ascontiguousarray(matrix, dtype=np.float32)
        if values.ndim != 2:
            raise KimiCudaError("float32 CUDA tensor upload requires [output,input]")
        handle = ctypes.c_void_p()
        status = self._library.coli_cuda_tensor_upload_g(
            ctypes.byref(handle),
            self._pointer(values),
            ctypes.c_void_p(),
            0,
            int(values.shape[1]),
            int(values.shape[0]),
            self.device,
            0,
        )
        if status != 1 or not handle.value:
            raise KimiCudaError("float32 CUDA tensor upload failed")
        self._tensors.append(handle)
        return handle

    def upload_bf16_embedding(self, table: np.ndarray) -> ctypes.c_void_p:
        if table.ndim != 2 or table.dtype != np.dtype("<u2") or not table.flags.c_contiguous:
            raise KimiCudaError("BF16 embedding upload requires contiguous uint16 [vocab,hidden]")
        handle = ctypes.c_void_p()
        status = self._library.coli_cuda_tensor_upload_g(
            ctypes.byref(handle),
            self._pointer(table),
            ctypes.c_void_p(),
            BF16_EMBEDDING_FORMAT,
            int(table.shape[1]),
            int(table.shape[0]),
            self.device,
            0,
        )
        if status != 1 or not handle.value:
            raise KimiCudaError("resident BF16 embedding upload failed")
        self._tensors.append(handle)
        return handle

    def upload_int8(self, tensor: _QuantizedInt8Tensor) -> ctypes.c_void_p:
        handle = ctypes.c_void_p()
        status = self._library.coli_cuda_tensor_upload_g(
            ctypes.byref(handle),
            self._pointer(tensor.weights),
            self._pointer(tensor.scales),
            1,
            tensor.input_dimension,
            tensor.output_dimension,
            self.device,
            0,
        )
        if status != 1 or not handle.value:
            raise KimiCudaError("resident int8 CUDA tensor upload failed")
        self._tensors.append(handle)
        return handle

    def tensor_bytes(self, handle: ctypes.c_void_p) -> int:
        return int(self._library.coli_cuda_tensor_bytes(handle))

    def release_tensor(self, handle: ctypes.c_void_p) -> None:
        value = int(handle.value or 0)
        if not value:
            return
        self._library.coli_cuda_tensor_free(handle)
        self._tensors = [
            item for item in self._tensors if int(item.value or 0) != value
        ]

    def execute(
        self,
        handles: tuple[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
        activation: np.ndarray,
    ) -> np.ndarray:
        source = np.ascontiguousarray(activation, dtype=np.float32)
        self._validate_expert_batch(int(source.shape[0]))
        output = np.empty_like(source)
        status = self._library.coli_cuda_kimi_expert_mlp(
            handles[0],
            handles[1],
            handles[2],
            self._pointer(output),
            self._pointer(source),
            source.shape[0],
            ctypes.c_float(4.0),
            ctypes.c_float(25.0),
        )
        if status != 1:
            raise KimiCudaError("native Kimi CUDA expert rejected execution; CPU fallback forbidden")
        return output

    def allocate(self, bytes_: int) -> ctypes.c_void_p:
        pointer = ctypes.c_void_p(self._library.coli_cuda_pipe_alloc(self.device, bytes_))
        if not pointer.value:
            raise KimiCudaError(f"CUDA resident allocation failed for {bytes_} bytes")
        return pointer

    def free(self, pointer: ctypes.c_void_p) -> None:
        self._library.coli_cuda_pipe_free(self.device, pointer)

    def upload_activation(self, destination: ctypes.c_void_p, activation: np.ndarray) -> None:
        source = np.ascontiguousarray(activation, dtype=np.float32)
        if self._library.coli_cuda_pipe_upload(
            self.device, destination, self._pointer(source), source.nbytes
        ) != 1:
            raise KimiCudaError("CUDA resident activation upload failed")

    def upload_bytes(self, destination: ctypes.c_void_p, source_array: np.ndarray) -> None:
        source = np.ascontiguousarray(source_array)
        if self._library.coli_cuda_pipe_upload(
            self.device, destination, self._pointer(source), source.nbytes
        ) != 1:
            raise KimiCudaError("CUDA resident byte-preserving upload failed")

    def download_activation(self, source: ctypes.c_void_p, shape: tuple[int, ...]) -> np.ndarray:
        output = np.empty(shape, dtype=np.float32)
        if self._library.coli_cuda_pipe_download(
            self.device, source, self._pointer(output), output.nbytes
        ) != 1:
            raise KimiCudaError("CUDA resident activation download failed")
        return output

    def execute_resident(
        self,
        handles: tuple[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
        output: ctypes.c_void_p,
        source: ctypes.c_void_p,
        batch: int,
    ) -> None:
        self._validate_expert_batch(batch)
        status = self._library.coli_cuda_kimi_expert_mlp_dev(
            handles[0],
            handles[1],
            handles[2],
            output,
            source,
            batch,
            ctypes.c_float(4.0),
            ctypes.c_float(25.0),
        )
        if status != 1:
            raise KimiCudaError("resident Kimi CUDA expert rejected execution")

    def _validate_expert_batch(self, batch: int) -> None:
        supported = getattr(
            self,
            "expert_supported_batches",
            tuple(range(1, int(self.expert_max_certified_batch) + 1)),
        )
        if batch not in supported:
            exact_sizes = (
                f", certified_sizes={supported}"
                if getattr(self, "expert_batch_set_capability_source", None)
                == "native_exact_size_export"
                else ""
            )
            raise KimiCudaError(
                "Kimi CUDA expert batch rejected before launch: "
                f"requested={batch}, certified_max={self.expert_max_certified_batch}"
                f"{exact_sizes}"
            )

    def synchronize(self) -> None:
        if self._library.coli_cuda_pipe_sync(self.device) != 1:
            raise KimiCudaError("CUDA resident synchronization failed")

    def error_state_ok(self) -> bool:
        """Inspect sticky CUDA state when the exact binary exports the query."""
        if self.cuda_error_state_query is None:
            return True
        return bool(self.cuda_error_state_query(self.device))

    def route(
        self,
        activation: ctypes.c_void_p,
        weight: ctypes.c_void_p,
        bias: ctypes.c_void_p,
        *,
        hidden: int,
        experts: int,
        topk: int,
        accumulate_stats: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        indices = np.empty(topk, dtype=np.int32)
        weights = np.empty(topk, dtype=np.float32)
        effective = ctypes.c_int()
        if not accumulate_stats:
            self._library.coli_cuda_kimi_router_stats_reset()
        status = self._library.coli_cuda_pipe_router(
            self.device,
            activation,
            weight,
            bias,
            hidden,
            experts,
            topk,
            ctypes.c_float(0.0),
            1,
            ctypes.c_float(1.0),
            indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(effective),
        )
        if status != 1:
            raise KimiCudaError("resident Kimi CUDA router rejected execution")
        return indices, weights, int(effective.value)

    def route_batch(
        self,
        activation: ctypes.c_void_p,
        weight: ctypes.c_void_p,
        bias: ctypes.c_void_p,
        *,
        batch: int,
        hidden: int,
        experts: int,
        topk: int,
        accumulate_stats: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Route exact certified rows with one launch pair and one D2H transfer."""
        self._validate_expert_batch(batch)
        if self.router_batch_function is None:
            raise KimiCudaError("Kimi CUDA binary has no batched-router export")
        indices = np.empty((batch, topk), dtype=np.int32)
        weights = np.empty((batch, topk), dtype=np.float32)
        effective = np.empty(batch, dtype=np.int32)
        if not accumulate_stats:
            self._library.coli_cuda_kimi_router_stats_reset()
        status = self.router_batch_function(
            self.device,
            activation,
            weight,
            bias,
            batch,
            hidden,
            experts,
            topk,
            ctypes.c_float(0.0),
            1,
            ctypes.c_float(1.0),
            indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            effective.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
        if status != 1:
            raise KimiCudaError("resident batched Kimi CUDA router rejected execution")
        return indices, weights, effective

    def execute_dense(
        self,
        tensor: ctypes.c_void_p,
        output: ctypes.c_void_p,
        source: ctypes.c_void_p,
        batch: int,
    ) -> None:
        if self._library.coli_cuda_pipe_gemm(tensor, output, source, batch) != 1:
            raise KimiCudaError("resident grouped-int4 CUDA GEMV rejected execution")

    def execute_dense_rows_reuse(
        self,
        tensor: ctypes.c_void_p,
        output: ctypes.c_void_p,
        source: ctypes.c_void_p,
        batch: int,
    ) -> None:
        if batch not in (1, 2, 4, 8):
            raise KimiCudaError(
                "row-reuse dense preflight rejected uncertified batch "
                f"requested={batch}; supported=(1, 2, 4, 8)"
            )
        if self.dense_rows_reuse_function is None:
            raise KimiCudaError("CUDA binary does not expose row-reuse dense execution")
        if self.dense_rows_reuse_function(tensor, output, source, batch) != 1:
            raise KimiCudaError("resident row-reuse dense CUDA execution rejected")

    def execute_add(
        self,
        destination: ctypes.c_void_p,
        source: ctypes.c_void_p,
        count: int,
    ) -> None:
        if self._library.coli_cuda_pipe_add(
            self.device, destination, source, count
        ) != 1:
            raise KimiCudaError("resident CUDA residual addition rejected execution")

    def execute_copy(
        self,
        destination: ctypes.c_void_p,
        source: ctypes.c_void_p,
        count: int,
    ) -> None:
        if self._library.coli_cuda_pipe_copy2d(
            self.device,
            destination,
            count,
            source,
            count,
            count,
            1,
        ) != 1:
            raise KimiCudaError("resident CUDA device copy rejected execution")

    def execute_rmsnorm(
        self,
        output: ctypes.c_void_p,
        source: ctypes.c_void_p,
        weight: ctypes.c_void_p,
        *,
        batch: int,
        dimension: int,
        epsilon: float,
    ) -> None:
        status = self._library.coli_cuda_pipe_rmsnorm_s(
            self.device,
            output,
            source,
            weight,
            batch,
            dimension,
            ctypes.c_float(epsilon),
            dimension,
            dimension,
        )
        if status != 1:
            raise KimiCudaError("resident Kimi CUDA RMSNorm rejected execution")

    def execute_embedding(
        self,
        embedding: ctypes.c_void_p,
        output: ctypes.c_void_p,
        token_ids: ctypes.c_void_p,
        count: int,
    ) -> None:
        if (
            self._library.coli_cuda_kimi_embedding_bf16_dev(
                embedding, output, token_ids, count
            )
            != 1
        ):
            raise KimiCudaError("resident Kimi CUDA embedding gather rejected execution")

    def execute_moe_reduction(
        self,
        output: ctypes.c_void_p,
        expert_rows: ctypes.c_void_p,
        weights: ctypes.c_void_p,
        *,
        count: int,
        dimension: int,
    ) -> None:
        if (
            self._library.coli_cuda_kimi_moe_reduce_dev(
                self.device,
                output,
                expert_rows,
                weights,
                count,
                dimension,
            )
            != 1
        ):
            raise KimiCudaError("resident deterministic Kimi MoE reduction failed")

    def execute_indexed_copy_rows(
        self,
        output: ctypes.c_void_p,
        source: ctypes.c_void_p,
        source_rows: ctypes.c_void_p,
        *,
        rows: int,
        dimension: int,
    ) -> None:
        """Gather source rows into contiguous output with one device launch."""

        if self.indexed_copy_rows_function is None:
            raise KimiCudaError("Kimi CUDA binary has no indexed row-copy export")
        if (
            self.indexed_copy_rows_function(
                self.device,
                output,
                source,
                source_rows,
                rows,
                dimension,
            )
            != 1
        ):
            raise KimiCudaError("resident Kimi indexed row copy failed")

    def execute_attnres_mix(
        self,
        output: ctypes.c_void_p,
        prefix: ctypes.c_void_p,
        block_residuals: ctypes.c_void_p,
        score_weight: ctypes.c_void_p,
        *,
        block_count: int,
        dimension: int,
        epsilon: float,
    ) -> None:
        if (
            self._library.coli_cuda_kimi_attnres_mix_dev(
                self.device,
                output,
                prefix,
                block_residuals,
                block_count,
                score_weight,
                dimension,
                ctypes.c_float(epsilon),
            )
            != 1
        ):
            raise KimiCudaError("resident Kimi AttnRes mix rejected execution")

    def execute_kda_core(
        self,
        output: ctypes.c_void_p,
        q: ctypes.c_void_p,
        k: ctypes.c_void_p,
        v: ctypes.c_void_p,
        gate: ctypes.c_void_p,
        decay: ctypes.c_void_p,
        beta: ctypes.c_void_p,
        conv_q: ctypes.c_void_p,
        conv_k: ctypes.c_void_p,
        conv_v: ctypes.c_void_p,
        window_q: ctypes.c_void_p,
        window_k: ctypes.c_void_p,
        window_v: ctypes.c_void_p,
        state: ctypes.c_void_p,
        dt: ctypes.c_void_p,
        a: ctypes.c_void_p,
        output_norm: ctypes.c_void_p,
        *,
        heads: int = 96,
        head_dimension: int = 128,
        convolution_width: int = 4,
        gate_lower_bound: float = -5.0,
        epsilon: float = 1e-5,
    ) -> None:
        status = self._library.coli_cuda_kimi_kda_core_dev(
            self.device,
            output,
            q,
            k,
            v,
            gate,
            decay,
            beta,
            conv_q,
            conv_k,
            conv_v,
            window_q,
            window_k,
            window_v,
            state,
            dt,
            a,
            output_norm,
            heads,
            head_dimension,
            convolution_width,
            ctypes.c_float(gate_lower_bound),
            ctypes.c_float(epsilon),
        )
        if status != 1:
            raise KimiCudaError("resident Kimi KDA core rejected execution")

    def execute_mla_cache_append(
        self,
        latent_row: ctypes.c_void_p,
        rope_row: ctypes.c_void_p,
        compressed_kv: ctypes.c_void_p,
        norm: ctypes.c_void_p,
        *,
        kv_lora: int,
        rope_dimension: int,
        epsilon: float,
    ) -> None:
        status = self._library.coli_cuda_kimi_mla_cache_append_dev(
            self.device,
            latent_row,
            rope_row,
            compressed_kv,
            norm,
            kv_lora,
            rope_dimension,
            ctypes.c_float(epsilon),
        )
        if status != 1:
            raise KimiCudaError("resident Kimi MLA cache append rejected execution")

    def execute_mla_absorb(
        self,
        kv_b: ctypes.c_void_p,
        context: ctypes.c_void_p,
        query: ctypes.c_void_p,
        latent_cache: ctypes.c_void_p,
        rope_cache: ctypes.c_void_p,
        *,
        heads: int,
        query_nope: int,
        query_rope: int,
        value_dimension: int,
        kv_lora: int,
        context_length: int,
        attention_scale: float,
    ) -> None:
        status = self._library.coli_cuda_kimi_mla_absorb_dev(
            kv_b,
            context,
            query,
            latent_cache,
            rope_cache,
            heads,
            query_nope,
            query_rope,
            value_dimension,
            kv_lora,
            context_length,
            ctypes.c_float(attention_scale),
        )
        if status != 1:
            raise KimiCudaError("resident Kimi MLA absorb attention rejected execution")

    def execute_mla_absorb_triangular(
        self,
        kv_b: ctypes.c_void_p,
        context: ctypes.c_void_p,
        query: ctypes.c_void_p,
        latent_cache: ctypes.c_void_p,
        rope_cache: ctypes.c_void_p,
        *,
        batch: int,
        heads: int,
        query_nope: int,
        query_rope: int,
        value_dimension: int,
        kv_lora: int,
        final_context_length: int,
        attention_scale: float,
    ) -> None:
        """Run exact contiguous-query MLA over one shared resident cache."""

        function = self.mla_absorb_triangular_function
        if function is None:
            raise KimiCudaError("Kimi CUDA binary has no triangular MLA export")
        status = function(
            kv_b,
            context,
            query,
            latent_cache,
            rope_cache,
            batch,
            heads,
            query_nope,
            query_rope,
            value_dimension,
            kv_lora,
            final_context_length,
            ctypes.c_float(attention_scale),
        )
        if status != 1:
            raise KimiCudaError("resident Kimi triangular MLA attention rejected execution")

    def execute_mla_absorb_dcp(
        self,
        kv_b: ctypes.c_void_p,
        context: ctypes.c_void_p,
        query: ctypes.c_void_p,
        latent_cache: ctypes.c_void_p,
        rope_cache: ctypes.c_void_p,
        *,
        batch: int,
        degree: int,
        heads: int,
        query_nope: int,
        query_rope: int,
        value_dimension: int,
        kv_lora: int,
        final_context_length: int,
        attention_scale: float,
    ) -> None:
        """Run exact context-sharded MLA and deterministic device reduction."""

        function = self.mla_absorb_dcp_function
        if function is None:
            raise KimiCudaError("Kimi CUDA binary has no DCP MLA export")
        status = function(
            kv_b,
            context,
            query,
            latent_cache,
            rope_cache,
            batch,
            degree,
            heads,
            query_nope,
            query_rope,
            value_dimension,
            kv_lora,
            final_context_length,
            ctypes.c_float(attention_scale),
        )
        if status != 1:
            raise KimiCudaError("resident Kimi DCP MLA attention rejected execution")

    def execute_mla_gate(
        self,
        context: ctypes.c_void_p,
        gate: ctypes.c_void_p,
        count: int,
    ) -> None:
        if (
            self._library.coli_cuda_kimi_mla_sigmoid_gate_dev(
                self.device, context, gate, count
            )
            != 1
        ):
            raise KimiCudaError("resident Kimi MLA sigmoid gate rejected execution")

    def reset_dense_stats(self) -> None:
        self._library.coli_cuda_kimi_dense_stats_reset()

    def dense_stats(self) -> dict[str, float | int]:
        calls = ctypes.c_uint64()
        kernel = ctypes.c_double()
        self._library.coli_cuda_kimi_dense_stats(
            ctypes.byref(calls), ctypes.byref(kernel)
        )
        count = int(calls.value)
        return {
            "calls": count,
            "kernel_ms_per_call": float(kernel.value) / count if count else 0.0,
        }

    def reset_router_stats(self) -> None:
        self._library.coli_cuda_kimi_router_stats_reset()

    def router_stats(self) -> dict[str, float | int]:
        calls=ctypes.c_uint64()
        logits=ctypes.c_double()
        selection=ctypes.c_double()
        d2h=ctypes.c_double()
        self._library.coli_cuda_kimi_router_stats(
            ctypes.byref(calls),ctypes.byref(logits),ctypes.byref(selection),ctypes.byref(d2h)
        )
        count=int(calls.value)
        return {
            "calls":count,
            "logits_ms_per_call":float(logits.value)/count if count else 0.0,
            "selection_ms_per_call":float(selection.value)/count if count else 0.0,
            "d2h_ms_per_call":float(d2h.value)/count if count else 0.0,
        }

    def reset_stats(self) -> None:
        self._library.coli_cuda_kimi_expert_stats_reset()

    def set_telemetry(self, mode: str) -> None:
        modes = {"minimal": 0, "production": 1, "detailed": 2}
        if mode not in modes or self._library.coli_cuda_kimi_set_telemetry(modes[mode]) != 1:
            raise KimiCudaError(f"CUDA runtime rejected telemetry mode {mode!r}")

    def set_up_first(self, up_first: bool) -> None:
        if self._library.coli_cuda_kimi_set_up_first(int(up_first)) != 1:
            raise KimiCudaError("CUDA runtime rejected the projection-order control")

    def set_fused_gate_up(self, fused: bool) -> None:
        if self._library.coli_cuda_kimi_set_fused_gate_up(int(fused)) != 1:
            raise KimiCudaError("CUDA runtime rejected the gate/up fusion control")

    def stats(self) -> dict[str, float | int]:
        calls = ctypes.c_uint64()
        rows = ctypes.c_uint64()
        h2d_bytes = ctypes.c_uint64()
        d2h_bytes = ctypes.c_uint64()
        h2d_ms = ctypes.c_double()
        kernel_ms = ctypes.c_double()
        d2h_ms = ctypes.c_double()
        self._library.coli_cuda_kimi_expert_stats(
            ctypes.byref(calls),
            ctypes.byref(rows),
            ctypes.byref(h2d_bytes),
            ctypes.byref(d2h_bytes),
            ctypes.byref(h2d_ms),
            ctypes.byref(kernel_ms),
            ctypes.byref(d2h_ms),
        )
        gate_ms = ctypes.c_double()
        up_ms = ctypes.c_double()
        situ_ms = ctypes.c_double()
        down_ms = ctypes.c_double()
        self._library.coli_cuda_kimi_expert_phase_stats(
            ctypes.byref(gate_ms),
            ctypes.byref(up_ms),
            ctypes.byref(situ_ms),
            ctypes.byref(down_ms),
        )
        pair_ms = ctypes.c_double()
        self._library.coli_cuda_kimi_expert_pair_stats(ctypes.byref(pair_ms))
        return {
            "calls": int(calls.value),
            "rows": int(rows.value),
            "h2d_bytes": int(h2d_bytes.value),
            "d2h_bytes": int(d2h_bytes.value),
            "h2d_ms": float(h2d_ms.value),
            "kernel_ms": float(kernel_ms.value),
            "d2h_ms": float(d2h_ms.value),
            "gate_ms": float(gate_ms.value),
            "up_ms": float(up_ms.value),
            "situ_ms": float(situ_ms.value),
            "down_ms": float(down_ms.value),
            "gate_up_pair_ms": float(pair_ms.value),
        }

    def close(self) -> None:
        for handle in reversed(self._tensors):
            self._library.coli_cuda_tensor_free(handle)
        self._tensors.clear()
        self._library.coli_cuda_shutdown()


