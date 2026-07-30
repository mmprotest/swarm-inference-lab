"""Strict configuration for Experiment 002 real-model execution."""

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


class RealModelSettings(StrictModel):
    model_id: str = "Qwen/Qwen3-0.6B"
    revision: str | None = None
    shard_directory: str
    stage_count: int = Field(default=4, ge=1)
    dtype: Literal["bfloat16"] = "bfloat16"
    device: Literal["cuda"] = "cuda"


class RealGenerationSettings(StrictModel):
    max_new_tokens: int = Field(default=16, gt=0)
    greedy: Literal[True] = True
    temperature: Literal[0] = 0
    top_p: None = None
    top_k: None = None
    thinking_enabled: Literal[False] = False
    batch_size: Literal[1] = 1


class RealWorkerSettings(StrictModel):
    count: int = Field(default=4, ge=1)
    one_stage_per_worker: Literal[True] = True
    logical_weight_limit_bytes: int | None = Field(default=None, gt=0)
    logical_total_memory_limit_bytes: int | None = Field(default=None, gt=0)
    spawn_method: Literal["spawn"] = "spawn"
    cuda_device: int = Field(default=0, ge=0)


class RealTransportSettings(StrictModel):
    persistent_streams: Literal[True] = True
    coordinator_relay_fallback: Literal[False] = False
    checksum: Literal[True] = True
    activation_transfer_dtype: Literal["bfloat16"] = "bfloat16"


class RealQueueSettings(StrictModel):
    capacity: int = Field(default=32, gt=0)
    max_microbatch_size: Literal[1] = 1
    max_microbatch_wait_ms: Literal[0] = 0


class RealCorrectnessSettings(StrictModel):
    require_exact_token_identity: Literal[True] = True
    boundary_atol: float = Field(default=0.02, ge=0)
    boundary_rtol: float = Field(default=0.02, ge=0)
    minimum_cosine_similarity: float = Field(default=0.999, ge=-1, le=1)
    require_cache_replay: Literal[True] = True
    require_stage_isolation: Literal[True] = True
    require_direct_data_plane: Literal[True] = True


class RealTimeoutSettings(StrictModel):
    worker_start_seconds: float = Field(default=120, gt=0)
    model_load_seconds: float = Field(default=300, gt=0)
    request_seconds: float = Field(default=300, gt=0)
    shutdown_seconds: float = Field(default=30, gt=0)


class RealExperimentConfig(StrictModel):
    name: str
    seed: int = 1
    execution_mode: Literal[ExecutionMode.SINGLE_HOST_LOOPBACK_REAL_MODEL]
    backend: Literal["torch-cuda"]
    data_plane: Literal["direct"]
    scheduler: Literal["static-stage-route"]
    model: RealModelSettings
    generation: RealGenerationSettings
    workers: RealWorkerSettings
    transport: RealTransportSettings
    queues: RealQueueSettings
    correctness: RealCorrectnessSettings
    timeouts: RealTimeoutSettings

    @model_validator(mode="after")
    def validate_initial_proof_shape(self) -> RealExperimentConfig:
        if self.model.stage_count != 4:
            raise ValueError("Experiment 002 requires exactly four model stages")
        if self.workers.count != 4:
            raise ValueError("Experiment 002 requires exactly four worker processes")
        return self

    def runtime_config(
        self,
        *,
        model_layer_count: int,
        model_hidden_size: int,
        logical_weight_limit_bytes: int,
    ) -> ExperimentConfig:
        """Translate real-only settings to the established coordinator schema."""

        return ExperimentConfig(
            name=self.name,
            execution_mode=ExecutionMode.SINGLE_HOST_LOOPBACK_REAL_MODEL,
            seed=0,
            scheduler=SchedulerMode.STATIC,
            backend="cuda",
            data_plane=DataPlaneMode.DIRECT,
            model=SyntheticModelConfig(
                layer_count=model_layer_count,
                hidden_size=model_hidden_size,
                activation_dtype="float16",
                stage_count=4,
                bytes_per_layer=1,
                cache_bytes_per_token_per_layer=1,
            ),
            worker=ExperimentWorkerConfig(
                logical_memory_limit_bytes=logical_weight_limit_bytes,
                outbound_queue_capacity=self.queues.capacity,
                inbound_queue_capacity=self.queues.capacity,
                max_inflight_operations=1,
            ),
            transport=TransportConfig(
                persistent_streams=True,
                coordinator_relay_fallback=False,
                checksum=True,
            ),
            workload={
                "concurrent_requests": 1,
                "prompt_tokens": 1,
                "output_tokens": self.generation.max_new_tokens,
            },
            queue=QueueConfig(
                capacity=self.queues.capacity,
                max_microbatch_size=1,
                max_microbatch_wait_ms=0,
                request_deadline_ms=self.timeouts.request_seconds * 1000,
            ),
            network=NetworkProfile(
                name="localhost-real-model",
                base_latency_ms=0,
                jitter_ms=0,
                upload_bandwidth_bytes_s=10_000_000_000,
                download_bandwidth_bytes_s=10_000_000_000,
                measured=True,
            ),
            nodes=[
                NodeProfile(
                    name="rtx5090-stage-worker",
                    count=4,
                    memory_bytes=logical_weight_limit_bytes,
                    compute_rate_layers_s=1,
                    supported_backends=[Backend.TORCH_CUDA],
                    network_profile="localhost-real-model",
                    measured=True,
                )
            ],
            node_counts=[4],
            concurrent_request_counts=[1, 2],
            model_id=self.model.model_id,
            model_revision=self.model.revision or "unresolved",
            notes=[
                "Process-isolated real Qwen3 execution on one physical RTX 5090.",
                "This configuration does not claim single-request speedup or multi-host scaling.",
            ],
        )


def load_real_experiment_config(path: str | Path) -> RealExperimentConfig:
    resolved = Path(path).expanduser().resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"real experiment config must contain a mapping: {resolved}")
    return RealExperimentConfig.model_validate(payload)
