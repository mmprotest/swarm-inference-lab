"""Strict configuration for Experiment 004 production-engine benchmarking."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from swarm_inference.config.models import ExecutionMode, StrictModel

PRIMARY_MODEL_ID = "Qwen/Qwen3-0.6B"
PRIMARY_MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
SECONDARY_MODEL_ID = "Qwen/Qwen3-4B"


class EngineExecutionProfiles(StrictModel):
    correctness: Literal["qwen3_correctness"] = "qwen3_correctness"
    performance: Literal["qwen3_fast"] = "qwen3_fast"


class EngineModelSettings(StrictModel):
    model_id: str
    revision: str | None
    dtype: Literal["bfloat16"] = "bfloat16"


class EngineSets(StrictModel):
    required: list[
        Literal[
            "custom_correctness",
            "custom_fast",
            "huggingface_eager",
            "huggingface_optimised",
            "sglang",
        ]
    ]
    optional: list[Literal["vllm", "tensorrt_llm"]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_engine_sets(self) -> EngineSets:
        required = {
            "custom_correctness",
            "custom_fast",
            "huggingface_eager",
            "huggingface_optimised",
            "sglang",
        }
        if set(self.required) != required or len(self.required) != len(required):
            raise ValueError(
                "required engines must contain each Experiment 004 baseline exactly once"
            )
        if len(set(self.optional)) != len(self.optional):
            raise ValueError("optional engines must be unique")
        return self


class EngineWorkload(StrictModel):
    name: str
    input_tokens: int = Field(ge=1)
    output_tokens: int = Field(ge=1)
    concurrency: list[int]

    @model_validator(mode="after")
    def validate_concurrency(self) -> EngineWorkload:
        if not self.concurrency or any(value <= 0 for value in self.concurrency):
            raise ValueError("workload concurrency values must be positive")
        if sorted(set(self.concurrency)) != self.concurrency:
            raise ValueError("workload concurrency must be sorted and unique")
        return self


class PrefixReuseSettings(StrictModel):
    enabled: bool = True
    request_count: Literal[4] = 4
    common_prefix_tokens: int = Field(default=1024, ge=1)
    unique_suffix_tokens: int = Field(default=32, ge=1)
    output_tokens: int = Field(default=128, ge=1)


class Qwen3EngineSettings(StrictModel):
    attention_backend: Literal[
        "auto",
        "sdpa",
        "flash_attention_2",
        "flashinfer",
        "eager",
    ] = "auto"


class CustomEngineSettings(StrictModel):
    attention_backends: list[Literal["auto", "sdpa", "flash_attention_2", "flashinfer", "eager"]]
    cache_backends: list[Literal["dynamic_reference", "static"]]
    cache_dtype: Literal["bfloat16", "float16", "fp8"] = "bfloat16"
    compile_modes: list[
        Literal[
            "eager",
            "default",
            "reduce-overhead",
            "max-autotune",
            "manual-cuda-graph",
        ]
    ]
    cuda_graph_batch_sizes: list[int]
    max_sequence_length: int = Field(default=4096, ge=1)
    final_worker_sampling: Literal[True] = True
    continuous_batching: Literal[True] = True
    gpu_resident_local_path: Literal[True] = True
    prefill_chunk_size: int = Field(default=512, ge=1)

    @model_validator(mode="after")
    def validate_ladder(self) -> CustomEngineSettings:
        if not self.attention_backends:
            raise ValueError("at least one attention backend is required")
        if not self.cache_backends:
            raise ValueError("at least one cache backend is required")
        if not self.compile_modes:
            raise ValueError("at least one compile mode is required")
        if (
            not self.cuda_graph_batch_sizes
            or sorted(set(self.cuda_graph_batch_sizes)) != self.cuda_graph_batch_sizes
            or any(value <= 0 for value in self.cuda_graph_batch_sizes)
        ):
            raise ValueError("cuda_graph_batch_sizes must be a sorted unique positive list")
        return self


class EngineAcceptanceSettings(StrictModel):
    require_exact_greedy_token_identity: Literal[True] = True
    minimum_speedup_over_current_baseline: float = Field(default=4.0, ge=4.0)
    minimum_fraction_of_fastest_production_engine: float = Field(default=0.50, ge=0.50, le=1)
    target_fraction_of_fastest_production_engine: float = Field(default=0.80, ge=0.80, le=1)
    maximum_result_cv: float = Field(default=0.10, gt=0, le=0.10)


class ExternalEnvironmentSettings(StrictModel):
    root: str = "artifacts/engine-environments"
    huggingface_transformers: str = "4.57.6"
    huggingface_torch: str = "2.13.0"
    sglang: str = "0.5.16"
    vllm: str = "0.25.1"
    tensorrt_llm: str = "1.3.0rc15"


class MeasurementSettings(StrictModel):
    gpu_sample_interval_ms: int = Field(default=100, ge=20)
    profile_decode_tokens: int = Field(default=32, ge=4)
    use_nsight_systems_when_available: bool = True
    use_pytorch_profiler: bool = True


class EnginePerformanceConfig(StrictModel):
    name: str
    seed: int = 4
    execution_mode: Literal[ExecutionMode.SINGLE_HOST_ENGINE_BENCHMARK]
    profiles: EngineExecutionProfiles = Field(default_factory=EngineExecutionProfiles)
    models: list[EngineModelSettings]
    engines: EngineSets
    workloads: list[EngineWorkload]
    prefix_reuse: PrefixReuseSettings = Field(default_factory=PrefixReuseSettings)
    repeats: int = Field(default=5, ge=5)
    warmup_requests: int = Field(default=3, ge=3)
    qwen3_engine: Qwen3EngineSettings = Field(default_factory=Qwen3EngineSettings)
    custom_engine: CustomEngineSettings
    acceptance: EngineAcceptanceSettings
    external_environments: ExternalEnvironmentSettings = Field(
        default_factory=ExternalEnvironmentSettings
    )
    measurement: MeasurementSettings = Field(default_factory=MeasurementSettings)
    output_root: str = "artifacts/runs"

    @model_validator(mode="after")
    def validate_experiment_shape(self) -> EnginePerformanceConfig:
        if len(self.models) != 2:
            raise ValueError("Experiment 004 requires primary Qwen3-0.6B and secondary Qwen3-4B")
        primary, secondary = self.models
        if primary.model_id != PRIMARY_MODEL_ID or primary.revision != PRIMARY_MODEL_REVISION:
            raise ValueError(
                "primary model must use the immutable Experiment 002 Qwen3-0.6B revision"
            )
        if secondary.model_id != SECONDARY_MODEL_ID:
            raise ValueError("secondary model must be Qwen/Qwen3-4B")
        required_workloads = {
            "decode-focused": (1, 512, [1, 4, 16, 64]),
            "realistic": (128, 256, [1, 4, 16, 64]),
            "medium-prefill": (2048, 128, [1, 4, 16]),
        }
        observed = {
            item.name: (item.input_tokens, item.output_tokens, item.concurrency)
            for item in self.workloads
        }
        if observed != required_workloads:
            raise ValueError("workloads must exactly match the required Experiment 004 matrix")
        return self


def load_engine_performance_config(path: str | Path) -> EnginePerformanceConfig:
    resolved = Path(path).expanduser().resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"engine performance config must contain a mapping: {resolved}")
    return EnginePerformanceConfig.model_validate(payload)
