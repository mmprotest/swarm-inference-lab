"""Configuration and low-overhead runtime helpers for the Qwen3 stage engine."""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from swarm_inference.exceptions import UnsupportedCacheFormatError


class Qwen3ExecutionProfile(StrEnum):
    CORRECTNESS = "qwen3_correctness"
    FAST = "qwen3_fast"


class AttentionBackend(StrEnum):
    AUTO = "auto"
    SDPA = "sdpa"
    FLASH_ATTENTION_2 = "flash_attention_2"
    FLASHINFER = "flashinfer"
    EAGER = "eager"


class Qwen3CacheBackend(StrEnum):
    DYNAMIC_REFERENCE = "dynamic_reference"
    STATIC = "static"


class Qwen3CompileMode(StrEnum):
    EAGER = "eager"
    DEFAULT = "torch_compile_default"
    REDUCE_OVERHEAD = "torch_compile_reduce_overhead"
    MAX_AUTOTUNE = "torch_compile_max_autotune"
    MANUAL_CUDA_GRAPH = "manual_cuda_graph"


class CacheDType(StrEnum):
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FP8 = "fp8"


@dataclass(frozen=True, slots=True)
class Qwen3EngineOptions:
    profile: Qwen3ExecutionProfile = Qwen3ExecutionProfile.CORRECTNESS
    attention_backend: AttentionBackend = AttentionBackend.AUTO
    cache_backend: Qwen3CacheBackend = Qwen3CacheBackend.DYNAMIC_REFERENCE
    cache_dtype: CacheDType = CacheDType.BFLOAT16
    max_sequence_length: int = 4096
    max_batch_size: int = 64
    compile_mode: Qwen3CompileMode = Qwen3CompileMode.EAGER
    cuda_graph_batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    final_worker_sampling: bool = False
    diagnostic_full_logits: bool = False
    boundary_diagnostics: bool = True
    nvtx_enabled: bool = False
    static_cache_fixed_shape: bool = False

    def __post_init__(self) -> None:
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if not self.cuda_graph_batch_sizes:
            raise ValueError("at least one CUDA graph batch-size bucket is required")
        if any(value <= 0 for value in self.cuda_graph_batch_sizes):
            raise ValueError("CUDA graph batch-size buckets must be positive")
        if tuple(sorted(set(self.cuda_graph_batch_sizes))) != self.cuda_graph_batch_sizes:
            raise ValueError("CUDA graph batch-size buckets must be sorted and unique")
        if (
            self.profile == Qwen3ExecutionProfile.FAST
            and self.cache_backend == Qwen3CacheBackend.DYNAMIC_REFERENCE
        ):
            # Dynamic cache remains a measured ladder option, so it is valid,
            # but callers must select it explicitly rather than get it by
            # default from from_values().
            return

    @classmethod
    def from_values(
        cls,
        *,
        profile: str | Qwen3ExecutionProfile = Qwen3ExecutionProfile.CORRECTNESS,
        attention_backend: str | AttentionBackend | None = None,
        cache_backend: str | Qwen3CacheBackend | None = None,
        cache_dtype: str | CacheDType = CacheDType.BFLOAT16,
        max_sequence_length: int = 4096,
        max_batch_size: int = 64,
        compile_mode: str | Qwen3CompileMode = Qwen3CompileMode.EAGER,
        cuda_graph_batch_sizes: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
        final_worker_sampling: bool | None = None,
        diagnostic_full_logits: bool = False,
        boundary_diagnostics: bool | None = None,
        nvtx_enabled: bool = False,
        static_cache_fixed_shape: bool = False,
    ) -> Qwen3EngineOptions:
        selected_profile = Qwen3ExecutionProfile(profile)
        selected_attention = AttentionBackend(
            attention_backend
            if attention_backend is not None
            else (
                AttentionBackend.EAGER
                if selected_profile == Qwen3ExecutionProfile.CORRECTNESS
                else AttentionBackend.AUTO
            )
        )
        selected_cache = Qwen3CacheBackend(
            cache_backend
            if cache_backend is not None
            else (
                Qwen3CacheBackend.DYNAMIC_REFERENCE
                if selected_profile == Qwen3ExecutionProfile.CORRECTNESS
                else Qwen3CacheBackend.STATIC
            )
        )
        return cls(
            profile=selected_profile,
            attention_backend=selected_attention,
            cache_backend=selected_cache,
            cache_dtype=CacheDType(cache_dtype),
            max_sequence_length=max_sequence_length,
            max_batch_size=max_batch_size,
            compile_mode=Qwen3CompileMode(compile_mode),
            cuda_graph_batch_sizes=tuple(int(value) for value in cuda_graph_batch_sizes),
            final_worker_sampling=(
                selected_profile == Qwen3ExecutionProfile.FAST
                if final_worker_sampling is None
                else final_worker_sampling
            ),
            diagnostic_full_logits=diagnostic_full_logits,
            boundary_diagnostics=(
                selected_profile == Qwen3ExecutionProfile.CORRECTNESS
                if boundary_diagnostics is None
                else boundary_diagnostics
            ),
            nvtx_enabled=nvtx_enabled,
            static_cache_fixed_shape=static_cache_fixed_shape,
        )


@dataclass(slots=True)
class AttentionBackendEvidence:
    requested: str
    selected: str | None = None
    available: dict[str, bool] = field(default_factory=dict)
    startup_correct: dict[str, bool] = field(default_factory=dict)
    median_cuda_ms: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, str] = field(default_factory=dict)
    full_decode_probe_tokens: int = 0
    full_decode_correct: dict[str, bool] = field(default_factory=dict)
    full_decode_tokens_per_second: dict[str, float] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CompileDiagnostics:
    requested_mode: str
    prefill_compiled: bool = False
    decode_compiled: bool = False
    compile_seconds: float = 0.0
    graph_break_count: int = 0
    fallback_used: bool = False
    fallback_reason: str | None = None
    verified_execution: bool = False

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CudaGraphDiagnostics:
    requested: bool
    bucket_size: int | None = None
    captured: bool = False
    replay_verified: bool = False
    capture_seconds: float = 0.0
    fallback_reason: str | None = None

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def attention_backend_availability(torch_module: Any) -> dict[str, bool]:
    """Return startup availability without importing unsupported CUDA packages."""

    cuda_available = bool(torch_module.cuda.is_available())
    return {
        AttentionBackend.EAGER.value: True,
        AttentionBackend.SDPA.value: hasattr(
            torch_module.nn.functional,
            "scaled_dot_product_attention",
        ),
        AttentionBackend.FLASH_ATTENTION_2.value: (
            cuda_available and importlib.util.find_spec("flash_attn") is not None
        ),
        AttentionBackend.FLASHINFER.value: (
            cuda_available and importlib.util.find_spec("flashinfer") is not None
        ),
    }


def validate_attention_backend(
    requested: AttentionBackend,
    *,
    availability: dict[str, bool],
) -> None:
    if requested == AttentionBackend.AUTO:
        return
    if not availability.get(requested.value, False):
        details = ", ".join(
            f"{name}={'available' if value else 'unavailable'}"
            for name, value in sorted(availability.items())
        )
        raise RuntimeError(
            f"Qwen3 attention backend {requested.value!r} is unsupported in this "
            f"environment ({details})"
        )


def auto_attention_candidates(availability: dict[str, bool]) -> tuple[AttentionBackend, ...]:
    ordered = (
        AttentionBackend.FLASH_ATTENTION_2,
        AttentionBackend.SDPA,
        AttentionBackend.EAGER,
    )
    return tuple(item for item in ordered if availability.get(item.value, False))


def select_cuda_graph_bucket(batch_size: int, buckets: Sequence[int]) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for bucket in buckets:
        if bucket >= batch_size:
            return int(bucket)
    raise ValueError(
        f"batch size {batch_size} exceeds largest CUDA graph bucket {max(buckets, default=0)}"
    )


def resolve_cache_torch_dtype(
    torch_module: Any,
    cache_dtype: CacheDType,
    *,
    model_dtype: Any,
    device: Any,
) -> Any:
    if cache_dtype == CacheDType.BFLOAT16:
        return torch_module.bfloat16
    if cache_dtype == CacheDType.FLOAT16:
        return torch_module.float16
    fp8_dtype = getattr(torch_module, "float8_e4m3fn", None)
    if fp8_dtype is None:
        raise UnsupportedCacheFormatError("this PyTorch build does not expose float8_e4m3fn")
    capability = (
        torch_module.cuda.get_device_capability(device)
        if getattr(device, "type", None) == "cuda"
        else (0, 0)
    )
    if capability < (8, 9):
        raise UnsupportedCacheFormatError(
            f"FP8 cache requires compatible CUDA hardware; capability={capability}"
        )
    if model_dtype != fp8_dtype:
        raise UnsupportedCacheFormatError(
            "FP8 cache is exposed only when the attention query dtype is FP8; "
            "implicit per-token BF16/FP8 conversion is not a production cache path"
        )
    return fp8_dtype


@contextmanager
def nvtx_range(torch_module: Any, name: str, *, enabled: bool) -> Iterator[None]:
    """Emit NVTX only for explicitly profiled runs."""

    if not enabled or not torch_module.cuda.is_available():
        yield
        return
    torch_module.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch_module.cuda.nvtx.range_pop()
