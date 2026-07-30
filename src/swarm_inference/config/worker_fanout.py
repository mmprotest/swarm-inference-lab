"""Strict Experiment 003 worker-fanout configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from swarm_inference.config.models import (
    Backend,
    DataPlaneMode,
    ExecutionMode,
    ExperimentConfig,
    ExperimentWorkerConfig,
    NetworkProfile,
    NodeProfile,
    QueueConfig,
    SchedulerMode,
    StrictModel,
    SyntheticModelConfig,
    TransportConfig,
)

IMMUTABLE_QWEN3_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def _default_unprovisioned_stage_counts() -> list[int | str]:
    return [4, 14, "maximum_stable"]


def _default_boundary_validation_counts() -> list[int | str]:
    return [1, 4, 14, "maximum_runnable"]


class FanoutModelSettings(StrictModel):
    model_id: str = "Qwen/Qwen3-0.6B"
    revision: str = IMMUTABLE_QWEN3_REVISION
    dtype: Literal["bfloat16"] = "bfloat16"
    device: Literal["cuda"] = "cuda"


class FanoutSweepSettings(StrictModel):
    initial_worker_counts: list[int] = Field(default_factory=lambda: [1, 2, 4, 7, 14, 21, 28])
    adaptive_search: bool = True
    maximum_worker_count: int = Field(default=28, ge=1, le=28)
    repeats: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def validate_counts(self) -> FanoutSweepSettings:
        if not self.initial_worker_counts:
            raise ValueError("initial_worker_counts cannot be empty")
        if any(
            count < 1 or count > self.maximum_worker_count for count in self.initial_worker_counts
        ):
            raise ValueError("initial worker counts must be within the configured maximum")
        if len(set(self.initial_worker_counts)) != len(self.initial_worker_counts):
            raise ValueError("initial worker counts must be unique")
        return self


class FanoutWorkload(StrictModel):
    input_tokens_approx: int = Field(default=128, ge=1)
    max_new_tokens: int = Field(ge=1)
    concurrency_levels: list[int] | None = None


class FanoutWorkloads(StrictModel):
    cold: FanoutWorkload = Field(default_factory=lambda: FanoutWorkload(max_new_tokens=32))
    warm: FanoutWorkload = Field(
        default_factory=lambda: FanoutWorkload(
            max_new_tokens=128,
            concurrency_levels=[1, 4],
        )
    )


class FanoutWarmupSettings(StrictModel):
    cuda_context: Literal[True] = True
    stage_local: bool = True
    full_pipeline_tokens: int = Field(default=4, ge=4)


class AcquisitionProfile(StrictModel):
    bandwidth_mbps: float | None = Field(default=None, gt=0)
    latency_ms: float = Field(default=0, ge=0)


class UnprovisionedSettings(StrictModel):
    enabled: bool = True
    representative_stage_counts: list[int | str] = Field(
        default_factory=_default_unprovisioned_stage_counts
    )
    acquisition_profiles: list[str] = Field(
        default_factory=lambda: [
            "local_disk",
            "gigabit_lan",
            "residential_fast",
            "residential_slow",
        ]
    )


class NodeStateSettings(StrictModel):
    cached_cold: bool = True
    hot_standby: bool = True
    unprovisioned: UnprovisionedSettings = Field(default_factory=UnprovisionedSettings)


class ResourceLimits(StrictModel):
    max_gpu_memory_fraction_for_stable: float = Field(default=0.95, gt=0, le=1)
    max_system_memory_fraction_for_stable: float = Field(default=0.90, gt=0, le=1)
    max_worker_start_seconds: float = Field(default=180, gt=0)
    max_pipeline_ready_seconds: float = Field(default=300, gt=0)
    max_request_seconds: float = Field(default=600, gt=0)


class FanoutCorrectness(StrictModel):
    require_exact_token_identity: Literal[True] = True
    require_direct_data_plane: Literal[True] = True
    require_stage_isolation: Literal[True] = True
    boundary_validation_counts: list[int | str] = Field(
        default_factory=_default_boundary_validation_counts
    )
    boundary_atol: float = Field(default=0.02, ge=0)
    boundary_rtol: float = Field(default=0.02, ge=0)
    minimum_cosine_similarity: float = Field(default=0.999, ge=-1, le=1)


class RejoinSettings(StrictModel):
    enabled: bool = True
    committed_tokens_before_failure: int = Field(default=4, ge=1)
    failure_stage: Literal["middle"] | int = "middle"


class EconomicsSettings(StrictModel):
    availability_seconds: list[float] = Field(
        default_factory=lambda: [
            30.0,
            60.0,
            300.0,
            900.0,
            3600.0,
            14400.0,
            86400.0,
        ]
    )
    productive_fraction_targets: list[float] = Field(
        default_factory=lambda: [0.50, 0.75, 0.90, 0.95]
    )

    @model_validator(mode="after")
    def validate_economics(self) -> EconomicsSettings:
        if any(value <= 0 for value in self.availability_seconds):
            raise ValueError("availability durations must be positive")
        if any(value < 0 or value >= 1 for value in self.productive_fraction_targets):
            raise ValueError("productive-fraction targets must be in [0, 1)")
        return self


class FanoutExperimentConfig(StrictModel):
    name: str = "experiment-003-worker-fanout"
    seed: int = Field(default=20260730, gt=0)
    execution_mode: Literal[ExecutionMode.SINGLE_HOST_LOOPBACK_REAL_MODEL_FANOUT]
    backend: Literal["torch-cuda"]
    data_plane: Literal["direct"]
    model: FanoutModelSettings
    sweep: FanoutSweepSettings
    workloads: FanoutWorkloads
    warmup: FanoutWarmupSettings
    node_states: NodeStateSettings
    shard_acquisition_profiles: dict[str, AcquisitionProfile] = Field(
        default_factory=lambda: {
            "local_disk": AcquisitionProfile(bandwidth_mbps=None, latency_ms=0),
            "gigabit_lan": AcquisitionProfile(bandwidth_mbps=1000, latency_ms=2),
            "residential_fast": AcquisitionProfile(bandwidth_mbps=100, latency_ms=20),
            "residential_slow": AcquisitionProfile(bandwidth_mbps=20, latency_ms=50),
        }
    )
    hot_standby_idle_seconds: list[float] = Field(default_factory=lambda: [0.0, 10.0, 60.0])
    resource_limits: ResourceLimits
    correctness: FanoutCorrectness
    rejoin: RejoinSettings
    economics: EconomicsSettings
    file_cache_control: dict[str, object] = Field(
        default_factory=lambda: {
            "controlled": False,
            "method": (
                "Windows operating-system file cache was not flushed; measurements are "
                "local-shard-read-with-uncontrolled-os-cache."
            ),
        }
    )

    @model_validator(mode="after")
    def validate_non_negotiable_settings(self) -> FanoutExperimentConfig:
        if self.model.revision != IMMUTABLE_QWEN3_REVISION:
            raise ValueError(
                f"Experiment 003 requires the immutable Qwen3 revision {IMMUTABLE_QWEN3_REVISION}"
            )
        required_profiles = {
            "local_disk",
            "gigabit_lan",
            "residential_fast",
            "residential_slow",
        }
        if not required_profiles <= set(self.shard_acquisition_profiles):
            raise ValueError("all required shard-acquisition profiles must be configured")
        if self.workloads.warm.concurrency_levels != [1, 4]:
            raise ValueError("the required warm concurrency levels are exactly [1, 4]")
        return self

    def runtime_config(
        self,
        *,
        worker_count: int,
        model_layer_count: int,
        model_hidden_size: int,
        logical_weight_limit_bytes: int,
    ) -> ExperimentConfig:
        return ExperimentConfig(
            name=f"{self.name}-{worker_count}-workers",
            execution_mode=ExecutionMode.SINGLE_HOST_LOOPBACK_REAL_MODEL_FANOUT,
            seed=self.seed,
            scheduler=SchedulerMode.STATIC,
            backend="cuda",
            data_plane=DataPlaneMode.DIRECT,
            model=SyntheticModelConfig(
                layer_count=model_layer_count,
                hidden_size=model_hidden_size,
                activation_dtype="float16",
                stage_count=worker_count,
                bytes_per_layer=1,
                cache_bytes_per_token_per_layer=1,
            ),
            worker=ExperimentWorkerConfig(
                logical_memory_limit_bytes=logical_weight_limit_bytes,
                outbound_queue_capacity=64,
                inbound_queue_capacity=64,
                max_inflight_operations=4,
                route_lease_seconds=self.resource_limits.max_request_seconds,
            ),
            transport=TransportConfig(
                persistent_streams=True,
                coordinator_relay_fallback=False,
                checksum=True,
            ),
            workload={
                "concurrent_requests": 4,
                "prompt_tokens": self.workloads.warm.input_tokens_approx,
                "output_tokens": self.workloads.warm.max_new_tokens,
            },
            queue=QueueConfig(
                capacity=64,
                max_microbatch_size=1,
                max_microbatch_wait_ms=0,
                request_deadline_ms=self.resource_limits.max_request_seconds * 1000,
            ),
            network=NetworkProfile(
                name="localhost-real-model-fanout",
                base_latency_ms=0,
                jitter_ms=0,
                upload_bandwidth_bytes_s=10_000_000_000,
                download_bandwidth_bytes_s=10_000_000_000,
                measured=True,
            ),
            nodes=[
                NodeProfile(
                    name="rtx5090-stage-worker",
                    count=worker_count,
                    memory_bytes=logical_weight_limit_bytes,
                    compute_rate_layers_s=1,
                    supported_backends=[Backend.TORCH_CUDA],
                    network_profile="localhost-real-model-fanout",
                    measured=True,
                )
            ],
            node_counts=[worker_count],
            concurrent_request_counts=[1, 4],
            model_id=self.model.model_id,
            model_revision=self.model.revision,
            notes=[
                "Every process shares one RTX 5090; worker count does not add physical compute.",
                "Intermediate activations use the direct worker-to-worker data plane.",
            ],
        )


def load_fanout_experiment_config(path: str | Path) -> FanoutExperimentConfig:
    resolved = Path(path).expanduser().resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"worker-fanout config must contain a mapping: {resolved}")
    return FanoutExperimentConfig.model_validate(payload)
