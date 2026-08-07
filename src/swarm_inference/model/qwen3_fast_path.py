"""Experiment-004-derived exact Qwen3 CUDA candidate ladder."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from swarm_inference.engines.interfaces import ExecutionDevice
from swarm_inference.execution.fast_path import (
    FastPathMeasurement,
    FastPathSupportReport,
    FastPathSupportStatus,
    PreparedFastPath,
    TensorResult,
)
from swarm_inference.model.descriptor import ResolvedModelDescriptor
from swarm_inference.model.qwen3_runtime import (
    Qwen3CacheBackend,
    Qwen3CompileMode,
    Qwen3EngineOptions,
    Qwen3ExecutionProfile,
)

BenchmarkRunner = Callable[[str, Qwen3EngineOptions], FastPathMeasurement | dict[str, Any]]
PrepareRunner = Callable[[str, Qwen3EngineOptions], Any]


class Qwen3CudaFastPath:
    fast_path_id = "qwen3_cuda"
    candidate_modes = (
        "eager",
        "gpu_native_dynamic_cache",
        "static_cache",
        "torch_compile_default",
        "torch_compile_reduce_overhead",
        "torch_compile_max_autotune",
        "manual_cuda_graph",
    )

    def probe(
        self,
        model: ResolvedModelDescriptor,
        device: ExecutionDevice,
    ) -> FastPathSupportReport:
        architecture = (model.architecture or "").lower()
        if model.format != "safetensors" or "qwen3" not in architecture or "moe" in architecture:
            return FastPathSupportReport(
                fast_path_id=self.fast_path_id,
                status=FastPathSupportStatus.UNSUPPORTED_MODEL,
                reason="fast path supports dense Qwen3 safetensors checkpoints only",
            )
        if device.device_type != "cuda":
            return FastPathSupportReport(
                fast_path_id=self.fast_path_id,
                status=FastPathSupportStatus.UNSUPPORTED_DEVICE,
                reason="manual CUDA and GPU-native candidates require a CUDA device",
            )
        if device.usable_memory_bytes and model.weight_bytes > device.usable_memory_bytes:
            return FastPathSupportReport(
                fast_path_id=self.fast_path_id,
                status=FastPathSupportStatus.INSUFFICIENT_MEMORY,
                reason="model weights exceed usable device memory before cache admission",
            )
        return FastPathSupportReport(
            fast_path_id=self.fast_path_id,
            status=FastPathSupportStatus.SUPPORTED,
            reason="CUDA device and dense Qwen3 identity support measured candidate admission",
            candidate_modes=self.candidate_modes,
        )

    @staticmethod
    def options(candidate: str) -> Qwen3EngineOptions:
        cache = (
            Qwen3CacheBackend.DYNAMIC_REFERENCE
            if candidate in {"eager", "gpu_native_dynamic_cache"}
            else Qwen3CacheBackend.STATIC
        )
        compile_mode = {
            "torch_compile_default": Qwen3CompileMode.DEFAULT,
            "torch_compile_reduce_overhead": Qwen3CompileMode.REDUCE_OVERHEAD,
            "torch_compile_max_autotune": Qwen3CompileMode.MAX_AUTOTUNE,
            "manual_cuda_graph": Qwen3CompileMode.MANUAL_CUDA_GRAPH,
        }.get(candidate, Qwen3CompileMode.EAGER)
        return Qwen3EngineOptions(
            profile=Qwen3ExecutionProfile.FAST,
            cache_backend=cache,
            compile_mode=compile_mode,
            final_worker_sampling=True,
            boundary_diagnostics=False,
        )

    def benchmark_candidates(
        self,
        model: ResolvedModelDescriptor,
        device: ExecutionDevice,
        **kwargs: Any,
    ) -> list[FastPathMeasurement]:
        runner = kwargs.get("runner")
        if not callable(runner):
            raise ValueError("Qwen3 fast-path admission requires a real exactness benchmark runner")
        measurements: list[FastPathMeasurement] = []
        for candidate in self.candidate_modes:
            try:
                measured = runner(candidate, self.options(candidate))
                measurement = (
                    measured
                    if isinstance(measured, FastPathMeasurement)
                    else FastPathMeasurement.model_validate(measured)
                )
                if measurement.candidate_mode != candidate:
                    raise ValueError("benchmark result identifies a different candidate")
                measurements.append(measurement)
            except Exception as exc:
                measurements.append(
                    FastPathMeasurement(
                        fast_path_id=self.fast_path_id,
                        candidate_mode=candidate,
                        exactness_passed=False,
                        prefill_tokens_s=0,
                        decode_tokens_s=0,
                        ttft_ms=0,
                        memory_bytes=0,
                        prepare_seconds=0,
                        batch_bucket=int(kwargs.get("batch_bucket", 1)),
                        context_bucket=int(kwargs.get("context_bucket", 1)),
                        failure_reason=f"{type(exc).__name__}: {exc}",
                    )
                )
        return measurements

    def prepare(
        self,
        model: ResolvedModelDescriptor,
        device: ExecutionDevice,
        measurement: FastPathMeasurement,
        **kwargs: Any,
    ) -> PreparedFastPath:
        if not measurement.eligible:
            raise ValueError("cannot prepare a failed or unmeasured candidate")
        runner = kwargs.get("runner")
        if not callable(runner):
            raise ValueError("Qwen3 fast-path preparation requires a stage-aware prepare runner")
        implementation = runner(measurement.candidate_mode, self.options(measurement.candidate_mode))
        return PreparedFastPath(
            fast_path_id=self.fast_path_id,
            candidate_mode=measurement.candidate_mode,
            profile_fingerprint=str(kwargs.get("profile_fingerprint", "unpersisted")),
            implementation=implementation,
            diagnostics={"measurement": measurement.model_dump(mode="json")},
        )

    def execute_prefill(self, prepared: PreparedFastPath, **kwargs: Any) -> TensorResult:
        started = time.perf_counter_ns()
        implementation = prepared.implementation
        if hasattr(implementation, "execute_prefill"):
            value = implementation.execute_prefill(**kwargs)
        elif hasattr(implementation, "prefill_cuda"):
            value = implementation.prefill_cuda(**kwargs)
        else:
            raise TypeError("prepared Qwen3 implementation has no prefill entry point")
        return TensorResult(
            value=value,
            compute_ns=time.perf_counter_ns() - started,
            telemetry={"fast_path": prepared.candidate_mode},
        )

    def execute_decode(self, prepared: PreparedFastPath, **kwargs: Any) -> TensorResult:
        started = time.perf_counter_ns()
        implementation = prepared.implementation
        if hasattr(implementation, "execute_decode"):
            value = implementation.execute_decode(**kwargs)
        elif hasattr(implementation, "decode_cuda"):
            value = implementation.decode_cuda(**kwargs)
        else:
            raise TypeError("prepared Qwen3 implementation has no decode entry point")
        return TensorResult(
            value=value,
            compute_ns=time.perf_counter_ns() - started,
            telemetry={"fast_path": prepared.candidate_mode},
        )


__all__ = ["Qwen3CudaFastPath"]
