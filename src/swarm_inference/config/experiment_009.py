"""Strict configuration for Experiment 009's Colibri adaptive expert runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from swarm_inference.config.loader import load_yaml
from swarm_inference.config.models import ExecutionMode, StrictModel
from swarm_inference.exceptions import ConfigurationError


class Experiment009DependencyConfig(StrictModel):
    repository: Literal["JustVugg/colibri"] = "JustVugg/colibri"
    release: Literal["v1.4.0"] = "v1.4.0"
    commit: Literal["b085b48888a88d9a1c00b151a9979774b72cdbfd"] = (
        "b085b48888a88d9a1c00b151a9979774b72cdbfd"
    )
    license: Literal["Apache-2.0"] = "Apache-2.0"
    checkout: str = "third_party/colibri"
    build_directory: str = "build/colibri"


class Experiment009BackendConfig(StrictModel):
    backend: Literal["colibri"] = "colibri"
    mode: Literal["stock", "bridge"] = "bridge"
    telemetry_level: Literal["off", "summary", "detailed", "trace"] = "summary"
    # The generated GLM fixture has a deliberately tiny cache.  The practical
    # OLMoE baseline uses Colibri's native default of 16 slots per layer.
    cap: int = Field(default=1, ge=1)
    practical_baseline_cap: int = Field(default=16, ge=1)
    quant_bits: int = Field(default=8, ge=2, le=8)
    startup_timeout_seconds: float = Field(default=600, gt=0)
    request_timeout_seconds: float = Field(default=1800, gt=0)
    ram_safety_reserve_bytes: int = Field(default=8 * 1024**3, ge=0)


class Experiment009FixtureConfig(StrictModel):
    path: str = "build/fixtures/glm_tiny"
    model_id: str = "experiment-009-glm-tiny"
    model_revision: str = "colibri-upstream-seed-1234"
    model_family: Literal["glm-5.2"] = "glm-5.2"
    prompt: str = "?"
    expected_input_token_ids: list[int] = Field(default_factory=lambda: [63])
    # The tiny GLM oracle has toolchain-sensitive near-tied logits. Correctness
    # is direct-versus-adapter identity plus the teacher-forced oracle floor,
    # not a greedy continuation pinned to one compiler build.
    expected_output_token_ids: list[int] = Field(default_factory=list)


class Experiment009PracticalModelConfig(StrictModel):
    repository: str = "allenai/OLMoE-1B-7B-0125-Instruct"
    revision: str = "b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e"
    model_id: str = "allenai/OLMoE-1B-7B-0125-Instruct"
    model_family: Literal["olmoe"] = "olmoe"
    converted_path: str = "artifacts/models/colibri/olmoe-1b-7b-0125-instruct-merged"


class Experiment009TuningCandidateConfig(StrictModel):
    candidate_id: str
    settings: dict[str, Any] = Field(default_factory=dict)


class Experiment009TuningConfig(StrictModel):
    repeats: int = Field(default=3, ge=3)
    minimum_gain_fraction: float = Field(default=0.03, ge=0)
    maximum_p95_regression_fraction: float = Field(default=0.05, ge=0)
    replay_completion_tokens: int = Field(default=8, ge=2)
    candidates: list[Experiment009TuningCandidateConfig] = Field(
        default_factory=lambda: [
            Experiment009TuningCandidateConfig(candidate_id="baseline"),
            Experiment009TuningCandidateConfig(candidate_id="prefetch_l1", settings={"PILOT": "1"}),
            Experiment009TuningCandidateConfig(candidate_id="prefetch_l2", settings={"PILOT": "2"}),
        ]
    )

    @model_validator(mode="after")
    def baseline_first(self) -> Experiment009TuningConfig:
        if not self.candidates or self.candidates[0].candidate_id != "baseline":
            raise ValueError("the baseline tuning candidate must be first")
        if len({item.candidate_id for item in self.candidates}) != len(self.candidates):
            raise ValueError("tuning candidate IDs must be unique")
        return self


class Experiment009PromptConfig(StrictModel):
    prompt_id: str
    workload_group: Literal[
        "general_chat", "coding", "mathematics_reasoning", "multilingual_long_form"
    ]
    partition: Literal["calibration", "heldout"]
    text: str


RoutingPolicyName = Literal[
    "plain_lru",
    "frequency_hot_experts",
    "recent_token_reuse",
    "transition_history",
    "colibri_recommended",
    "swarm_planner",
]


def _default_routing_policies() -> list[RoutingPolicyName]:
    return [
        "plain_lru",
        "frequency_hot_experts",
        "recent_token_reuse",
        "colibri_recommended",
        "swarm_planner",
    ]


class Experiment009RoutingConfig(StrictModel):
    prompts: list[Experiment009PromptConfig]
    policies: list[RoutingPolicyName] = Field(default_factory=_default_routing_policies)
    cache_slots_per_layer: int = Field(default=1, ge=1)
    hot_slots_per_layer: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def validate_split(self) -> Experiment009RoutingConfig:
        from swarm_inference.backends.colibri.placement import (
            PromptPartition,
            validate_prompt_partitions,
        )

        validate_prompt_partitions(
            PromptPartition(
                prompt_id=item.prompt_id,
                workload_group=item.workload_group,
                partition=item.partition,
            )
            for item in self.prompts
        )
        return self


class Experiment009AcceptanceConfig(StrictModel):
    minimum_correctness_requests: int = Field(default=50, ge=50)
    maximum_decode_regression_fraction: float = Field(default=0.03, ge=0)
    maximum_ttft_regression_fraction: float = Field(default=0.05, ge=0)
    maximum_p95_regression_fraction: float = Field(default=0.05, ge=0)


class Experiment009Config(StrictModel):
    schema_version: Literal["experiment-009-v1"] = "experiment-009-v1"
    name: Literal["experiment-009-colibri-adaptive-expert-runtime"]
    execution_mode: Literal[ExecutionMode.COLIBRI_ADAPTIVE_EXPERT_RUNTIME]
    seed: int = 9009
    dependency: Experiment009DependencyConfig = Field(default_factory=Experiment009DependencyConfig)
    backend: Experiment009BackendConfig = Field(default_factory=Experiment009BackendConfig)
    fixture: Experiment009FixtureConfig = Field(default_factory=Experiment009FixtureConfig)
    practical_model: Experiment009PracticalModelConfig = Field(
        default_factory=Experiment009PracticalModelConfig
    )
    tuning: Experiment009TuningConfig = Field(default_factory=Experiment009TuningConfig)
    routing: Experiment009RoutingConfig
    acceptance: Experiment009AcceptanceConfig = Field(default_factory=Experiment009AcceptanceConfig)
    output_root: str = "artifacts/runs"


def load_experiment_009_config(path: Path) -> Experiment009Config:
    try:
        return Experiment009Config.model_validate(load_yaml(path))
    except ValueError as exc:
        raise ConfigurationError(f"invalid Experiment 009 configuration {path}: {exc}") from exc
