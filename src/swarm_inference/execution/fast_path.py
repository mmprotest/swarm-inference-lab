"""Exactness-gated native compute fast-path contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import ConfigDict, Field, NonNegativeInt

from swarm_inference.config.models import StrictModel
from swarm_inference.engines.interfaces import ExecutionDevice
from swarm_inference.model.descriptor import ResolvedModelDescriptor


class FastPathSupportStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED_MODEL = "UNSUPPORTED_MODEL"
    UNSUPPORTED_DEVICE = "UNSUPPORTED_DEVICE"
    INSUFFICIENT_MEMORY = "INSUFFICIENT_MEMORY"
    MISSING_RUNTIME = "MISSING_RUNTIME"
    BROKEN_RUNTIME = "BROKEN_RUNTIME"


class FastPathSupportReport(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fast_path_id: str
    status: FastPathSupportStatus
    reason: str
    candidate_modes: tuple[str, ...] = ()

    @property
    def supported(self) -> bool:
        return self.status == FastPathSupportStatus.SUPPORTED


class FastPathMeasurement(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fast_path_id: str
    candidate_mode: str
    exactness_passed: bool
    prefill_tokens_s: float = Field(ge=0)
    decode_tokens_s: float = Field(ge=0)
    ttft_ms: float = Field(ge=0)
    memory_bytes: NonNegativeInt
    prepare_seconds: float = Field(ge=0)
    supported_batch_sizes: tuple[int, ...] = ()
    batch_bucket: int
    context_bucket: int
    repeat_count: int = Field(default=1, ge=1)
    coefficient_of_variation: float | None = Field(default=None, ge=0)
    failure_reason: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @property
    def eligible(self) -> bool:
        return self.exactness_passed and self.decode_tokens_s > 0 and self.failure_reason is None


class PreparedFastPath(StrictModel):
    fast_path_id: str
    candidate_mode: str
    profile_fingerprint: str
    implementation: Any = Field(exclude=True)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class TensorResult(StrictModel):
    value: Any = Field(exclude=True)
    compute_ns: NonNegativeInt = 0
    telemetry: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class NativeFastPath(Protocol):
    fast_path_id: str

    def probe(
        self,
        model: ResolvedModelDescriptor,
        device: ExecutionDevice,
    ) -> FastPathSupportReport: ...

    def benchmark_candidates(
        self,
        model: ResolvedModelDescriptor,
        device: ExecutionDevice,
        **kwargs: Any,
    ) -> list[FastPathMeasurement]: ...

    def prepare(
        self,
        model: ResolvedModelDescriptor,
        device: ExecutionDevice,
        measurement: FastPathMeasurement,
        **kwargs: Any,
    ) -> PreparedFastPath: ...

    def execute_prefill(self, prepared: PreparedFastPath, **kwargs: Any) -> TensorResult: ...

    def execute_decode(self, prepared: PreparedFastPath, **kwargs: Any) -> TensorResult: ...


__all__ = [
    "FastPathMeasurement",
    "FastPathSupportReport",
    "FastPathSupportStatus",
    "NativeFastPath",
    "PreparedFastPath",
    "TensorResult",
]
