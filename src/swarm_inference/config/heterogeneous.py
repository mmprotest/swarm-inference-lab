"""Strict Experiment 007 configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from swarm_inference.config.models import ExecutionMode, StrictModel
from swarm_inference.planner import NodeRole, PlannerObjective


class ExperimentReferences(StrictModel):
    experiment_004_run: str | None = None
    experiment_006_run: str | None = None


class ModelBackendConfig(StrictModel):
    backend: str
    model_id: str
    revision: str | None = None
    dtype: str | None = None


class CpuDraftConfig(ModelBackendConfig):
    formats: tuple[str, ...]
    draft_lengths: tuple[int, ...]


class MixedPipelineConfig(StrictModel):
    model_id: str
    revision: str
    pipeline_stages: int = Field(gt=0)
    backends: tuple[str, ...]
    generated_tokens: int = Field(default=32, gt=0)

    @model_validator(mode="after")
    def validate_route(self) -> MixedPipelineConfig:
        if len(self.backends) != self.pipeline_stages:
            raise ValueError("mixed pipeline backend count must equal pipeline stage count")
        return self


class MoeExperimentConfig(StrictModel):
    source_experiment: int = 6
    cpu_expert_counts: tuple[int, ...]
    placement_policies: tuple[str, ...]
    weight_formats: tuple[str, ...] = ("BF16", "INT8", "Q4")
    sequence_length: int = Field(default=64, gt=0)


class BackgroundExperimentConfig(StrictModel):
    gpu_concurrency: tuple[int, ...]
    cpu_concurrency: tuple[int, ...]
    interactive_output_tokens: int = Field(default=128, gt=0)
    background_output_tokens: int = Field(default=128, gt=0)


class PlannerExperimentConfig(StrictModel):
    objectives: tuple[PlannerObjective, ...]
    roles: tuple[NodeRole, ...]
    maximum_regret_fraction: float = Field(default=0.10, ge=0)


class NonDegradationConfig(StrictModel):
    maximum_interactive_p95_increase_fraction: float = Field(default=0.05, ge=0)
    maximum_interactive_throughput_decrease_fraction: float = Field(default=0.05, ge=0)


class PositiveContributionConfig(StrictModel):
    minimum_speculative_speedup_fraction: float = Field(default=0.10, ge=0)
    minimum_background_throughput_gain_fraction: float = Field(default=0.10, ge=0)
    minimum_expert_retained_throughput_fraction: float = Field(default=0.70, ge=0, le=1)


class Arm64Config(StrictModel):
    cross_compile: bool = True
    qemu_protocol_test: bool = True


class NetworkProjectionConfig(StrictModel):
    profiles: tuple[str, ...]


class WorkloadConfig007(StrictModel):
    short_input_tokens: int = Field(default=128, gt=0)
    short_output_tokens: int = Field(default=256, gt=0)
    short_concurrency: tuple[int, ...] = (1, 4, 16, 64)
    long_input_tokens: int = Field(default=2048, gt=0)
    long_output_tokens: int = Field(default=128, gt=0)
    long_concurrency: tuple[int, ...] = (1, 4, 16)
    speculative_prompt_count: int = Field(default=100, ge=100)
    speculative_output_tokens: int = Field(default=128, gt=0)
    audit_rates: tuple[float, ...] = (0.0, 0.01, 0.05, 0.10)
    lease_durations_seconds: tuple[int, ...] = (30, 60, 300, 900, 3600, 14400, 86400)


class BackendEnvironmentConfig(StrictModel):
    root: str = "artifacts/backend-environments"
    sglang_image: str = "lmsysorg/sglang:v0.5.16"
    sglang_version: str = "0.5.16"
    llamacpp_repository: str = "https://github.com/ggml-org/llama.cpp.git"
    llamacpp_commit: str | None = None
    torch_cpu_python: str | None = None


class HeterogeneousExperimentConfig(StrictModel):
    name: str
    execution_mode: Literal[ExecutionMode.HETEROGENEOUS_SINGLE_HOST_REAL_MODEL]
    seed: int = Field(default=7007, ge=1)
    references: ExperimentReferences
    gpu_target: ModelBackendConfig
    cpu_draft: CpuDraftConfig
    mixed_pipeline: MixedPipelineConfig
    moe: MoeExperimentConfig
    background: BackgroundExperimentConfig
    planner: PlannerExperimentConfig
    non_degradation: NonDegradationConfig
    positive_contribution: PositiveContributionConfig
    arm64: Arm64Config
    network_projection: NetworkProjectionConfig
    workloads: WorkloadConfig007 = Field(default_factory=WorkloadConfig007)
    backend_environments: BackendEnvironmentConfig = Field(default_factory=BackendEnvironmentConfig)
    output_root: str = "artifacts/runs"
    profile: bool = False

    @model_validator(mode="after")
    def validate_required_experiment(self) -> HeterogeneousExperimentConfig:
        if self.gpu_target.backend != "sglang" or self.gpu_target.model_id != "Qwen/Qwen3-4B":
            raise ValueError("Experiment 007 GPU target must be Qwen/Qwen3-4B on SGLang")
        if self.cpu_draft.backend != "llamacpp" or self.cpu_draft.model_id != "Qwen/Qwen3-0.6B":
            raise ValueError("Experiment 007 CPU draft must be Qwen/Qwen3-0.6B on llama.cpp")
        if set(self.cpu_draft.formats) != {"Q8_0", "Q4_K_M"}:
            raise ValueError("CPU draft formats must contain Q8_0 and Q4_K_M")
        if set(self.cpu_draft.draft_lengths) != {1, 2, 4, 8}:
            raise ValueError("draft lengths must contain 1, 2, 4, and 8")
        if self.mixed_pipeline.pipeline_stages != 4:
            raise ValueError("mixed critical-path experiment requires four stages")
        if self.mixed_pipeline.backends != (
            "torch-cuda",
            "torch-cpu",
            "torch-cuda",
            "torch-cpu",
        ):
            raise ValueError("mixed pipeline must alternate CUDA, CPU, CUDA, CPU")
        required_roles = set(NodeRole)
        missing_roles = required_roles - set(self.planner.roles)
        if missing_roles:
            raise ValueError(f"planner does not evaluate roles: {sorted(missing_roles)}")
        return self


def load_heterogeneous_config(path: Path) -> HeterogeneousExperimentConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Experiment 007 configuration must be a YAML mapping")
    return HeterogeneousExperimentConfig.model_validate(payload)
