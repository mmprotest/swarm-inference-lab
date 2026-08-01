"""Strict configuration for the Experiment 007 benchmark corrections."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from swarm_inference.config.models import ExecutionMode, StrictModel


class OriginalRunConfig(StrictModel):
    run_id: str


class ExpertCorrectnessConfig(StrictModel):
    atol: float = Field(default=0.02, ge=0)
    rtol: float = Field(default=0.02, ge=0)
    minimum_cosine_similarity: float = Field(default=0.999, ge=-1, le=1)


class CpuExpertCorrectionConfig(StrictModel):
    model_source: Literal["experiment_006"] = "experiment_006"
    routing_tokens: int = Field(default=10_000, ge=10_000)
    repeats: int = Field(default=5, ge=5)
    warmup_iterations: int = Field(default=10, ge=10)
    maximum_repeats: int = Field(default=30, ge=5)
    maximum_variability_epochs: int = Field(default=3, ge=1)
    maximum_coefficient_of_variation: float = Field(default=0.10, gt=0)
    cpu_expert_counts: tuple[int, ...] = (1, 2, 4, 8, 16)
    placement_policies: tuple[str, ...]
    minimum_cpu_dispatch_fraction: float = Field(default=0.01, ge=0, le=1)
    formats: tuple[str, ...] = ("bfloat16", "int8", "four_bit")
    correctness: ExpertCorrectnessConfig = Field(default_factory=ExpertCorrectnessConfig)

    @model_validator(mode="after")
    def validate_matrix(self) -> CpuExpertCorrectionConfig:
        required_policies = {
            "coldest_experts_on_cpu",
            "hottest_experts_on_cpu",
            "random_experts_on_cpu",
            "load_balanced_experts_on_cpu",
            "frequency_band_experts_on_cpu",
        }
        if set(self.placement_policies) != required_policies:
            raise ValueError("CPU expert correction requires the complete placement-policy matrix")
        if set(self.cpu_expert_counts) != {1, 2, 4, 8, 16}:
            raise ValueError("CPU expert counts must contain 1, 2, 4, 8, and 16")
        if set(self.formats) != {"bfloat16", "int8", "four_bit"}:
            raise ValueError("expert formats must contain bfloat16, int8, and four_bit")
        if self.maximum_repeats < self.repeats:
            raise ValueError("maximum repeats cannot be below the initial repeat count")
        return self


class BackgroundCorrectionConfig(StrictModel):
    warmup_seconds: float = Field(default=30, gt=0)
    measurement_seconds: float = Field(default=120, gt=0)
    drain_timeout_seconds: float = Field(default=60, gt=0)
    repeats: int = Field(default=3, ge=3)
    gpu_concurrency: tuple[int, ...] = (1, 4, 16)
    cpu_concurrency: tuple[int, ...] = (1, 2, 4)
    traffic_modes: tuple[str, ...] = ("closed_loop", "open_loop")
    input_token_lengths: tuple[int, ...] = (64, 128, 512)
    output_token_lengths: tuple[int, ...] = (64, 128, 256)
    open_loop_concurrency: int = Field(default=4, gt=0)
    open_loop_cpu_concurrency: int = Field(default=1, gt=0)
    open_loop_load_fractions: tuple[float, ...] = (0.60, 0.95)
    workload_seed: int = Field(default=7007, ge=1)

    @model_validator(mode="after")
    def validate_matrix(self) -> BackgroundCorrectionConfig:
        if set(self.gpu_concurrency) != {1, 4, 16}:
            raise ValueError("GPU concurrency must contain 1, 4, and 16")
        if set(self.cpu_concurrency) != {1, 2, 4}:
            raise ValueError("CPU concurrency must contain 1, 2, and 4")
        if set(self.traffic_modes) != {"closed_loop", "open_loop"}:
            raise ValueError("background traffic modes must contain closed_loop and open_loop")
        if any(not 0 < value <= 1 for value in self.open_loop_load_fractions):
            raise ValueError("open-loop load fractions must be in (0, 1]")
        return self


class CorrectedPositiveContributionConfig(StrictModel):
    minimum_combined_gain_fraction: float = Field(default=0.10, ge=0)
    maximum_gpu_p95_increase_fraction: float = Field(default=0.05, ge=0)
    maximum_gpu_throughput_decrease_fraction: float = Field(default=0.05, ge=0)
    minimum_expert_retained_throughput_fraction: float = Field(default=0.70, ge=0, le=1)


class CorrectedPlannerConfig(StrictModel):
    held_out_evaluation: bool = True
    maximum_regret_fraction: float = Field(default=0.10, ge=0)


class Experiment007CorrectionsConfig(StrictModel):
    name: Literal["experiment-007-corrections"]
    execution_mode: ExecutionMode = ExecutionMode.HETEROGENEOUS_SINGLE_HOST_REAL_MODEL
    seed: int = Field(default=7007, ge=1)
    original_run: OriginalRunConfig
    cpu_expert: CpuExpertCorrectionConfig
    background: BackgroundCorrectionConfig
    positive_contribution: CorrectedPositiveContributionConfig
    planner: CorrectedPlannerConfig
    output_root: str = "artifacts/runs"


def load_experiment_007_corrections_config(path: Path) -> Experiment007CorrectionsConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Experiment 007 correction configuration must be a YAML mapping")
    return Experiment007CorrectionsConfig.model_validate(payload)
