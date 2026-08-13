"""Exact GPU startup conversion for directly loaded BF16 worker shards."""

from __future__ import annotations

import ctypes
import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.kimi_cuda_runtime import (
    _QuantizedInt8Tensor,
    _quantize_bf16_rows_int8,
)
from swarm_inference.execution.kimi_k3_graph_runtime import (
    _GroupedInt4Tensor,
    _quantize_bf16_grouped_int4,
)


class GpuShardQuantizer:
    """Convert one already-owned BF16 shard; never accepts a layer abstraction."""

    def __init__(self, library: Path, device: int = 0) -> None:
        self.path = library.resolve()
        self.device = device
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self._library = ctypes.CDLL(str(self.path))
        pointer = ctypes.c_void_p
        self._int4 = self._library.exp019_quantize_grouped_int4_host
        self._int4.argtypes = [
            ctypes.c_int,
            pointer,
            ctypes.c_int,
            ctypes.c_int,
            pointer,
            pointer,
        ]
        self._int4.restype = ctypes.c_int
        self._int8 = self._library.exp019_quantize_row_int8_host
        self._int8.argtypes = list(self._int4.argtypes)
        self._int8.restype = ctypes.c_int
        self._row_max = self._library.exp019_bf16_row_max_host
        self._row_max.argtypes = [
            ctypes.c_int,
            pointer,
            ctypes.c_int,
            ctypes.c_int,
            pointer,
        ]
        self._row_max.restype = ctypes.c_int
        self._int8_with_scales = (
            self._library.exp019_quantize_row_int8_with_scales_host
        )
        self._int8_with_scales.argtypes = [
            ctypes.c_int,
            pointer,
            pointer,
            ctypes.c_int,
            ctypes.c_int,
            pointer,
        ]
        self._int8_with_scales.restype = ctypes.c_int
        self._rmsnorm = self._library.exp019_rmsnorm_host
        self._rmsnorm.argtypes = [
            ctypes.c_int,
            pointer,
            pointer,
            pointer,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        self._rmsnorm.restype = ctypes.c_int
        self.audit: list[dict[str, Any]] = []

    @staticmethod
    def _source(source: np.ndarray) -> np.ndarray:
        values = np.ascontiguousarray(source)
        if values.ndim != 2 or values.dtype != np.dtype("<u2"):
            raise ValueError("GPU shard quantizer requires a contiguous BF16 matrix")
        return values

    def grouped_int4(self, source: np.ndarray, *, owner: str) -> _GroupedInt4Tensor:
        values = self._source(source)
        rows, columns = (int(value) for value in values.shape)
        if columns % 64:
            raise ValueError("grouped-int4 input dimension must be divisible by 64")
        packed = np.empty((rows, columns // 2), dtype=np.uint8)
        scales = np.empty((rows, columns // 64), dtype=np.float32)
        started = time.perf_counter_ns()
        status = self._int4(
            self.device,
            values.ctypes.data_as(ctypes.c_void_p),
            rows,
            columns,
            packed.ctypes.data_as(ctypes.c_void_p),
            scales.ctypes.data_as(ctypes.c_void_p),
        )
        if status != 1:
            raise RuntimeError("GPU grouped-int4 shard quantization failed")
        self.audit.append(
            {
                "worker_id": owner,
                "quantization": "grouped_int4",
                "source_bytes": values.nbytes,
                "output_bytes": packed.nbytes + scales.nbytes,
                "wall_ms": (time.perf_counter_ns() - started) / 1e6,
            }
        )
        return _GroupedInt4Tensor(
            packed=packed,
            scales=scales,
            source=np.empty((0,), dtype=np.float32),
            input_dimension=columns,
            output_dimension=rows,
        )

    def row_int8(self, source: np.ndarray, *, owner: str) -> _QuantizedInt8Tensor:
        values = self._source(source)
        rows, columns = (int(value) for value in values.shape)
        quantized = np.empty((rows, columns), dtype=np.int8)
        scales = np.empty(rows, dtype=np.float32)
        started = time.perf_counter_ns()
        status = self._int8(
            self.device,
            values.ctypes.data_as(ctypes.c_void_p),
            rows,
            columns,
            quantized.ctypes.data_as(ctypes.c_void_p),
            scales.ctypes.data_as(ctypes.c_void_p),
        )
        if status != 1:
            raise RuntimeError("GPU row-int8 shard quantization failed")
        self.audit.append(
            {
                "worker_id": owner,
                "quantization": "row_int8",
                "source_bytes": values.nbytes,
                "output_bytes": quantized.nbytes + scales.nbytes,
                "wall_ms": (time.perf_counter_ns() - started) / 1e6,
            }
        )
        return _QuantizedInt8Tensor(
            weights=quantized,
            scales=scales,
            input_dimension=columns,
            output_dimension=rows,
        )

    def row_maximum(self, source: np.ndarray, *, owner: str) -> np.ndarray:
        values = self._source(source)
        rows, columns = (int(value) for value in values.shape)
        maxima = np.empty(rows, dtype=np.float32)
        started = time.perf_counter_ns()
        status = self._row_max(
            self.device,
            values.ctypes.data_as(ctypes.c_void_p),
            rows,
            columns,
            maxima.ctypes.data_as(ctypes.c_void_p),
        )
        if status != 1:
            raise RuntimeError("GPU BF16 row-maximum calculation failed")
        self.audit.append(
            {
                "worker_id": owner,
                "quantization": "row_maximum_partial",
                "source_bytes": values.nbytes,
                "output_bytes": maxima.nbytes,
                "wall_ms": (time.perf_counter_ns() - started) / 1e6,
            }
        )
        return maxima

    def row_int8_with_scales(
        self,
        source: np.ndarray,
        scales: np.ndarray,
        *,
        owner: str,
    ) -> _QuantizedInt8Tensor:
        values = self._source(source)
        rows, columns = (int(value) for value in values.shape)
        scale_values = np.ascontiguousarray(scales, dtype=np.float32)
        if scale_values.shape != (rows,) or np.any(scale_values <= 0):
            raise ValueError("external int8 scales do not match shard output rows")
        quantized = np.empty((rows, columns), dtype=np.int8)
        started = time.perf_counter_ns()
        status = self._int8_with_scales(
            self.device,
            values.ctypes.data_as(ctypes.c_void_p),
            scale_values.ctypes.data_as(ctypes.c_void_p),
            rows,
            columns,
            quantized.ctypes.data_as(ctypes.c_void_p),
        )
        if status != 1:
            raise RuntimeError("GPU row-int8 quantization with global scales failed")
        self.audit.append(
            {
                "worker_id": owner,
                "quantization": "row_int8_with_global_scales",
                "source_bytes": values.nbytes,
                "output_bytes": quantized.nbytes + scale_values.nbytes,
                "wall_ms": (time.perf_counter_ns() - started) / 1e6,
            }
        )
        return _QuantizedInt8Tensor(
            weights=quantized,
            scales=scale_values,
            input_dimension=columns,
            output_dimension=rows,
        )

    def validate_exact(self, source: np.ndarray) -> dict[str, Any]:
        values = self._source(source)
        gpu_int4 = self.grouped_int4(values, owner="quantizer-validation")
        cpu_int4 = _quantize_bf16_grouped_int4(values)
        gpu_int8 = self.row_int8(values, owner="quantizer-validation")
        cpu_int8 = _quantize_bf16_rows_int8(values)
        gpu_maxima = self.row_maximum(values, owner="quantizer-validation")
        float_values = (
            values.astype(np.uint32) << np.uint32(16)
        ).view(np.float32)
        cpu_maxima = np.max(np.abs(float_values), axis=1).astype(np.float32)
        gpu_external = self.row_int8_with_scales(
            values, cpu_int8.scales, owner="quantizer-validation"
        )
        checks = {
            "int4_packed_bit_exact": bool(np.array_equal(gpu_int4.packed, cpu_int4.packed)),
            "int4_scales_bit_exact": bool(np.array_equal(gpu_int4.scales, cpu_int4.scales)),
            "int8_weights_bit_exact": bool(np.array_equal(gpu_int8.weights, cpu_int8.weights)),
            "int8_scales_bit_exact": bool(np.array_equal(gpu_int8.scales, cpu_int8.scales)),
            "row_maxima_bit_exact": bool(np.array_equal(gpu_maxima, cpu_maxima)),
            "external_scale_int8_weights_bit_exact": bool(
                np.array_equal(gpu_external.weights, cpu_int8.weights)
            ),
        }
        return {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "shape": list(values.shape),
            "library": str(self.path),
            "library_sha256": self.sha256,
        }

    def rmsnorm(
        self,
        values: np.ndarray,
        weight: np.ndarray,
        *,
        owner: str,
        epsilon: float = 1e-5,
    ) -> np.ndarray:
        source = np.ascontiguousarray(values, dtype=np.float32)
        if source.ndim == 1:
            source = source[None, :]
        scale = np.ascontiguousarray(weight, dtype=np.float32).reshape(-1)
        if source.ndim != 2 or scale.shape != (source.shape[1],):
            raise ValueError("GPU RMSNorm source and weight geometry differ")
        output = np.empty_like(source)
        started = time.perf_counter_ns()
        status = self._rmsnorm(
            self.device,
            source.ctypes.data_as(ctypes.c_void_p),
            scale.ctypes.data_as(ctypes.c_void_p),
            output.ctypes.data_as(ctypes.c_void_p),
            source.shape[0],
            source.shape[1],
            ctypes.c_float(epsilon),
        )
        if status != 1:
            raise RuntimeError("GPU RMSNorm worker primitive failed")
        self.audit.append(
            {
                "worker_id": owner,
                "quantization": "none_rmsnorm_compute",
                "source_bytes": source.nbytes + scale.nbytes,
                "output_bytes": output.nbytes,
                "wall_ms": (time.perf_counter_ns() - started) / 1e6,
            }
        )
        return output


__all__ = ["GpuShardQuantizer"]
