"""Native correctness gate for the Swarm-owned generic Colibri MoE ABI."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest


def _runtime() -> ctypes.CDLL:
    configured = os.environ.get("SWARM_COLIBRI_MOE_LIBRARY")
    if not configured:
        pytest.skip("SWARM_COLIBRI_MOE_LIBRARY is not configured")
    path = Path(configured).expanduser().resolve()
    if not path.is_file():
        pytest.fail(f"configured generic Colibri library is missing: {path}")
    return ctypes.CDLL(str(path))


def _f32_pointer(value: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return value.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _i32_pointer(value: np.ndarray) -> ctypes.POINTER(ctypes.c_int32):
    return value.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def _u16_pointer(value: np.ndarray) -> ctypes.POINTER(ctypes.c_uint16):
    return value.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))


def _as_bf16(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=np.float32)
    return np.ascontiguousarray((contiguous.view(np.uint32) >> 16).astype(np.uint16))


def _from_bf16(value: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(value, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


def _swiglu_reference(
    activation: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
) -> np.ndarray:
    gate_value = activation @ gate.T
    up_value = activation @ up.T
    hidden = (gate_value / (np.float32(1.0) + np.exp(-gate_value))) * up_value
    return np.ascontiguousarray(hidden @ down.T, dtype=np.float32)


def _pack_signed_int4(value: np.ndarray) -> np.ndarray:
    source = np.ascontiguousarray(value, dtype=np.int32)
    if source.ndim != 2 or source.shape[1] % 8 or np.any(source < -8) or np.any(source > 7):
        raise ValueError("INT4 fixture must be rank two, width-aligned, and within [-8, 7]")
    encoded = (source + 8).astype(np.uint32)
    packed = np.zeros((source.shape[0], source.shape[1] // 8), dtype=np.uint32)
    for nibble in range(8):
        packed |= encoded[:, nibble::8] << np.uint32(4 * nibble)
    return np.ascontiguousarray(packed.view(np.int32))


def test_native_generic_colibri_f32_and_bf16_match_reference() -> None:
    runtime = _runtime()
    runtime.swarm_moe_abi_version.restype = ctypes.c_char_p
    assert runtime.swarm_moe_abi_version() == b"swarm-colibri-moe-v2"

    activation = np.array(
        [[0.25, -0.5, 0.75, 1.0], [-0.125, 0.375, -0.625, 0.875]],
        dtype=np.float32,
    )
    gate = np.array(
        [[0.5, -0.25, 0.125, 0.75], [-0.5, 0.625, 0.25, -0.125], [0.75, 0.5, -0.5, 0.25]],
        dtype=np.float32,
    )
    up = np.array(
        [[-0.25, 0.5, 0.75, -0.5], [0.125, -0.375, 0.625, 0.25], [0.5, 0.25, -0.125, 0.375]],
        dtype=np.float32,
    )
    down = np.array(
        [[0.25, 0.5, -0.75], [-0.5, 0.125, 0.625], [0.75, -0.25, 0.5], [0.375, 0.625, -0.125]],
        dtype=np.float32,
    )
    output = np.empty_like(activation)
    f32 = runtime.swarm_moe_swiglu_f32
    f32.restype = ctypes.c_int
    status = f32(
        _f32_pointer(activation),
        ctypes.c_size_t(activation.shape[0]),
        ctypes.c_size_t(activation.shape[1]),
        _f32_pointer(gate),
        _f32_pointer(up),
        ctypes.c_size_t(gate.shape[0]),
        _f32_pointer(down),
        _f32_pointer(output),
    )
    assert status == 0
    np.testing.assert_allclose(
        output, _swiglu_reference(activation, gate, up, down), rtol=2e-6, atol=2e-6
    )

    gate_bf16, up_bf16, down_bf16 = map(_as_bf16, (gate, up, down))
    output.fill(np.nan)
    bf16 = runtime.swarm_moe_swiglu_bf16
    bf16.restype = ctypes.c_int
    status = bf16(
        _f32_pointer(activation),
        ctypes.c_size_t(activation.shape[0]),
        ctypes.c_size_t(activation.shape[1]),
        _u16_pointer(gate_bf16),
        _u16_pointer(up_bf16),
        ctypes.c_size_t(gate.shape[0]),
        _u16_pointer(down_bf16),
        _f32_pointer(output),
    )
    assert status == 0
    np.testing.assert_allclose(
        output,
        _swiglu_reference(
            activation,
            _from_bf16(gate_bf16),
            _from_bf16(up_bf16),
            _from_bf16(down_bf16),
        ),
        rtol=2e-6,
        atol=2e-6,
    )


def test_native_generic_colibri_int4_g32_matches_dequantized_reference() -> None:
    runtime = _runtime()
    random = np.random.default_rng(36)
    activation = np.ascontiguousarray(random.normal(0, 0.2, size=(2, 32)), dtype=np.float32)
    gate_q = random.integers(-8, 8, size=(32, 32), dtype=np.int32)
    up_q = random.integers(-8, 8, size=(32, 32), dtype=np.int32)
    down_q = random.integers(-8, 8, size=(32, 32), dtype=np.int32)
    gate_scale = _as_bf16(np.full((32, 1), 0.03125, dtype=np.float32))
    up_scale = _as_bf16(np.full((32, 1), 0.046875, dtype=np.float32))
    down_scale = _as_bf16(np.full((32, 1), 0.0234375, dtype=np.float32))
    output = np.empty_like(activation)

    int4 = runtime.swarm_moe_swiglu_int4_g32
    int4.restype = ctypes.c_int
    status = int4(
        _f32_pointer(activation),
        ctypes.c_size_t(activation.shape[0]),
        ctypes.c_size_t(activation.shape[1]),
        _i32_pointer(_pack_signed_int4(gate_q)),
        _u16_pointer(gate_scale),
        _i32_pointer(_pack_signed_int4(up_q)),
        _u16_pointer(up_scale),
        ctypes.c_size_t(gate_q.shape[0]),
        _i32_pointer(_pack_signed_int4(down_q)),
        _u16_pointer(down_scale),
        ctypes.c_size_t(32),
        _f32_pointer(output),
    )
    assert status == 0
    reference = _swiglu_reference(
        activation,
        gate_q.astype(np.float32) * _from_bf16(gate_scale),
        up_q.astype(np.float32) * _from_bf16(up_scale),
        down_q.astype(np.float32) * _from_bf16(down_scale),
    )
    np.testing.assert_allclose(output, reference, rtol=2e-5, atol=2e-5)
