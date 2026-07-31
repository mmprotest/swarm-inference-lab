"""Strict configuration for Experiment 006 microsharding."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from swarm_inference.config.models import ExecutionMode, StrictModel


class MicroshardingModelConfig(StrictModel):
    model_id: str
    revision: str | None
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"


class DenseModelsConfig(StrictModel):
    primary: MicroshardingModelConfig
    secondary: MicroshardingModelConfig


class DenseParallelismConfig(StrictModel):
    pipeline_stage_counts: list[int]
    tensor_parallel_degrees: list[int]
    vocabulary_parallel: bool = True
    sequence_parallel_prefill: str = "optional"

    @model_validator(mode="after")
    def validate_degrees(self) -> DenseParallelismConfig:
        if not self.pipeline_stage_counts or any(item <= 0 for item in self.pipeline_stage_counts):
            raise ValueError("pipeline stage counts must be positive")
        if not self.tensor_parallel_degrees or any(
            item <= 0 for item in self.tensor_parallel_degrees
        ):
            raise ValueError("tensor-parallel degrees must be positive")
        return self


class MicroshardingCollectivesConfig(StrictModel):
    exact_format: str = "bfloat16"
    optional_formats: list[str] = Field(default_factory=list)
    algorithms: list[str]


class MicroshardingCorrectnessConfig(StrictModel):
    require_exact_greedy_tokens: bool = True
    boundary_atol: float = Field(default=0.02, ge=0)
    boundary_rtol: float = Field(default=0.02, ge=0)
    minimum_cosine_similarity: float = Field(default=0.999, ge=-1, le=1)
    prompts: int = Field(default=8, ge=8)
    max_new_tokens: int = Field(default=32, ge=32)


class MicroshardingMeasurementConfig(StrictModel):
    isolated_rank_warmup_iterations: int = Field(default=25, ge=0)
    isolated_rank_measurement_iterations: int = Field(default=100, ge=100)
    repeats: int = Field(default=5, ge=1)


class MicroshardingProjectionConfig(StrictModel):
    batch_sizes: list[int]
    prefill_lengths: list[int]
    decode_tokens: int = Field(gt=0)
    network_profiles: list[str]


class DeterministicMoEConfig(StrictModel):
    enabled: bool = True
    expert_counts: list[int]
    top_k_values: list[int]
    expert_parallel_degrees: list[int]
    expert_tensor_degrees: list[int]


class RealMoELayerConfig(StrictModel):
    enabled: bool = True
    model_id: str
    revision: str | None = None
    maximum_download_gib: float = Field(default=25, gt=0)
    expert_parallel_degrees: list[int]
    expert_tensor_degrees: list[int]
    selected_layer: int = Field(default=24, ge=0)


class MoEExperimentConfig(StrictModel):
    deterministic_fixture: DeterministicMoEConfig
    real_layer: RealMoELayerConfig


class K3ProjectionConfig(StrictModel):
    enabled: bool = True
    metadata_only: bool = True
    model_id: str = "moonshotai/Kimi-K3"
    revision: str = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
    node_memory_gib: list[int]
    target_single_stream_tps: float = Field(default=20, gt=0)


class MicroshardingAcceptanceConfig(StrictModel):
    dense_token_identity_required: bool = True
    dense_shard_memory_tolerance: float = Field(default=0.05, ge=0)
    kv_partition_required: bool = True
    collective_projector_required: bool = True
    deterministic_moe_required: bool = True
    real_moe_layer_required_when_downloadable: bool = True
    maximum_result_cv: float = Field(default=0.10, ge=0)


class MicroshardingExperimentConfig(StrictModel):
    name: str
    execution_mode: Literal[ExecutionMode.LOGICAL_SINGLE_GPU_MICROSHARDING]
    seed: int = 6006
    backend: Literal["torch-cuda"]
    experiment_004_run: str | None = None
    dense_models: DenseModelsConfig
    dense_parallelism: DenseParallelismConfig
    collectives: MicroshardingCollectivesConfig
    correctness: MicroshardingCorrectnessConfig
    measurement: MicroshardingMeasurementConfig
    projection: MicroshardingProjectionConfig
    moe: MoEExperimentConfig
    k3_projection: K3ProjectionConfig
    acceptance: MicroshardingAcceptanceConfig


def load_microsharding_config(path: Path) -> MicroshardingExperimentConfig:
    payload = yaml.safe_load(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Experiment 006 config must be a YAML mapping")
    return MicroshardingExperimentConfig.model_validate(payload)
