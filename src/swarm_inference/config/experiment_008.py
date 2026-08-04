"""Strict configuration for Experiment 008 single-host adaptive MoE saturation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from swarm_inference.config.loader import load_yaml
from swarm_inference.config.models import ExecutionMode, StrictModel
from swarm_inference.exceptions import ConfigurationError


class Experiment008ModelCandidate(StrictModel):
    model_id: str
    artifact_repository: str
    filename: str
    quantization: str
    revision: str
    architecture: str
    minimum_tensor_bytes: int = Field(default=32 * 1024**3, gt=0)


class Experiment008Models(StrictModel):
    preferred: Experiment008ModelCandidate
    fallback: Experiment008ModelCandidate


class Experiment008BackendConfig(StrictModel):
    backend: Literal["llamacpp"] = "llamacpp"
    server_path: str | None = None
    server_download: bool = True
    release_tag: str = "b9637"
    windows_cuda_version: str = "13.3"
    host: str = "127.0.0.1"
    port_start: int = Field(default=18_080, ge=1024, le=65_000)
    startup_timeout_seconds: float = Field(default=900.0, gt=0)
    request_timeout_seconds: float = Field(default=7_200.0, gt=0)
    keep_servers: bool = False
    flash_attention: bool = True
    memory_map: bool = True
    pinned_memory: bool = True


class Experiment008ProfilingConfig(StrictModel):
    warmup_iterations: int = Field(default=5, ge=1)
    measurement_iterations: int = Field(default=20, ge=3)
    payload_bytes: list[int] = Field(default_factory=lambda: [4096, 1 << 20, 16 << 20, 128 << 20])
    decode_shapes: list[list[int]] = Field(
        default_factory=lambda: [[1, 2048, 512], [1, 2048, 1024]]
    )
    prefill_shapes: list[list[int]] = Field(
        default_factory=lambda: [[128, 2048, 512], [512, 2048, 512]]
    )
    cpu_thread_counts: list[int] = Field(default_factory=lambda: [4, 8, 12, 16, 20])
    resource_sample_interval_seconds: float = Field(default=0.2, gt=0)
    storage_sample_bytes: int = Field(default=512 << 20, gt=0)

    @model_validator(mode="after")
    def validate_profile_matrix(self) -> Experiment008ProfilingConfig:
        if any(value <= 0 for value in self.payload_bytes):
            raise ValueError("profiling payload sizes must be positive")
        if any(
            len(shape) != 3 or any(value <= 0 for value in shape) for shape in self.decode_shapes
        ):
            raise ValueError("decode shapes must contain [batch, input, output]")
        if any(
            len(shape) != 3 or any(value <= 0 for value in shape) for shape in self.prefill_shapes
        ):
            raise ValueError("prefill shapes must contain [batch, input, output]")
        if not self.cpu_thread_counts or any(value <= 0 for value in self.cpu_thread_counts):
            raise ValueError("CPU thread counts must be positive")
        return self


def _default_gpu_layers() -> list[int | Literal["auto", "all"]]:
    return ["auto", 24, 32, 40, 48]


class Experiment008BaselineSearchConfig(StrictModel):
    gpu_layers: list[int | Literal["auto", "all"]] = Field(default_factory=_default_gpu_layers)
    cpu_threads: list[int] = Field(default_factory=lambda: [8, 12, 16, 20])
    batch_sizes: list[int] = Field(default_factory=lambda: [512, 1024, 2048])
    microbatch_sizes: list[int] = Field(default_factory=lambda: [128, 256, 512])
    memory_map: list[bool] = Field(default_factory=lambda: [True, False])
    flash_attention: list[bool] = Field(default_factory=lambda: [True, False])
    cpu_moe_layers: list[int] = Field(default_factory=lambda: [0, 8, 16, 24, 32, 40, 48])
    maximum_candidates: int = Field(default=32, ge=1)
    repeats: int = Field(default=3, ge=1)
    warmup_requests: int = Field(default=1, ge=0)


class Experiment008WorkloadConfig(StrictModel):
    decode_prompt_count: int = Field(default=20, ge=20)
    decode_input_tokens_min: int = Field(default=256, ge=1)
    decode_input_tokens_max: int = Field(default=512, ge=1)
    decode_output_tokens: int = Field(default=512, ge=1)
    long_context_tokens: list[int] = Field(default_factory=lambda: [8_000, 32_000])
    long_prompt_count: int = Field(default=5, ge=5)
    long_output_tokens: int = Field(default=64, ge=1)
    mixed_interactive_output_tokens: int = Field(default=128, ge=1)
    mixed_background_output_tokens: int = Field(default=512, ge=1)
    mixed_measurement_seconds: float = Field(default=120.0, gt=0)
    seed: int = 8008

    @model_validator(mode="after")
    def validate_workloads(self) -> Experiment008WorkloadConfig:
        if self.decode_input_tokens_max < self.decode_input_tokens_min:
            raise ValueError("decode input maximum must not be below its minimum")
        if (
            len(self.long_context_tokens) < 2
            or 8_000 not in self.long_context_tokens
            or 32_000 not in self.long_context_tokens
        ):
            raise ValueError("long-context workloads must include 8,000 and 32,000 tokens")
        return self


class Experiment008AcceptanceConfig(StrictModel):
    minimum_decode_gain_fraction: float = Field(default=0.25, ge=0)
    minimum_ttft_32k_reduction_fraction: float = Field(default=0.25, ge=0)
    minimum_mixed_gain_fraction: float = Field(default=0.20, ge=0)
    maximum_interactive_p95_increase_fraction: float = Field(default=0.05, ge=0)
    maximum_other_workload_regression_fraction: float = Field(default=0.10, ge=0)
    maximum_planner_regret_fraction: float = Field(default=0.05, ge=0)
    minimum_correctness_executions: int = Field(default=50, ge=1)


class Experiment008Config(StrictModel):
    schema_version: str = "experiment-008-v1"
    name: Literal["experiment-008-single-host-adaptive-moe-saturation"]
    execution_mode: Literal[ExecutionMode.SINGLE_HOST_ADAPTIVE_MOE_SATURATION]
    models: Experiment008Models
    backend: Experiment008BackendConfig = Field(default_factory=Experiment008BackendConfig)
    profiling: Experiment008ProfilingConfig = Field(default_factory=Experiment008ProfilingConfig)
    baseline_search: Experiment008BaselineSearchConfig = Field(
        default_factory=Experiment008BaselineSearchConfig
    )
    workloads: Experiment008WorkloadConfig = Field(default_factory=Experiment008WorkloadConfig)
    acceptance: Experiment008AcceptanceConfig = Field(default_factory=Experiment008AcceptanceConfig)
    output_root: str = "artifacts/runs"

    @property
    def seed(self) -> int:
        """Expose the shared experiment-config seed contract without duplicate state."""

        return self.workloads.seed


def load_experiment_008_config(path: Path) -> Experiment008Config:
    try:
        return Experiment008Config.model_validate(load_yaml(path))
    except ValueError as exc:
        raise ConfigurationError(f"invalid Experiment 008 configuration {path}: {exc}") from exc
