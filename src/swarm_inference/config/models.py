"""Typed domain and experiment models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
Probability = Annotated[float, Field(ge=0, le=1)]


class StrictModel(BaseModel):
    """Base model that rejects misspelled and unknown fields."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ExecutionMode(StrEnum):
    SIMULATION = "simulation"
    SINGLE_HOST_LOOPBACK = "single-host-loopback"
    PHYSICAL_LAN = "physical-lan"
    PHYSICAL_WAN = "physical-wan"


class Backend(StrEnum):
    SYNTHETIC = "synthetic"
    TORCH_CPU = "torch-cpu"
    TORCH_CUDA = "torch-cuda"
    TORCH_MPS = "torch-mps"


class WorkloadClass(StrEnum):
    INTERACTIVE = "interactive"
    STANDARD = "standard"
    BACKGROUND = "background"


class SchedulerMode(StrEnum):
    STATIC = "static"
    FASTEST_ROUTE = "fastest-route"
    REPLICATED = "replicated-stage"
    WORKLOAD_TIER = "workload-tier"
    ADVERSARIAL = "adversarial"


class BackpressurePolicy(StrEnum):
    REJECT = "reject"
    WAIT = "wait"


class HealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    QUARANTINED = "quarantined"


class RequestStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VerificationState(StrEnum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    DISAGREEMENT = "disagreement"
    REJECTED = "rejected"


class OperationKind(StrEnum):
    PREFILL = "prefill"
    DECODE = "decode"
    REPLAY = "replay"
    CANARY = "canary"


class JitterDistribution(StrEnum):
    NONE = "none"
    UNIFORM = "uniform"
    NORMAL = "normal"


class TensorSpec(StrictModel):
    dtype: str
    shape: list[int | str]
    byte_order: Literal["little", "big"] = "little"


class CacheSpec(StrictModel):
    format: str = "dynamic-kv"
    bytes_per_token: NonNegativeInt
    reconstructable_by_replay: bool = True


class StageDefinition(StrictModel):
    stage_id: NonNegativeInt
    layer_start: NonNegativeInt
    layer_end: NonNegativeInt
    owns_embeddings: bool = False
    owns_final_norm: bool = False
    owns_output_head: bool = False
    required_memory_bytes: PositiveInt
    estimated_execution_ms: dict[str, float] = Field(default_factory=dict)
    input_spec: TensorSpec
    output_spec: TensorSpec
    cache_spec: CacheSpec
    tensor_names: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_layer_range(self) -> StageDefinition:
        if self.layer_end <= self.layer_start:
            raise ValueError("layer_end must be greater than layer_start")
        return self


class StageReplica(StrictModel):
    stage_id: NonNegativeInt
    worker_id: str
    shard_hash: str
    load_status: Literal["unassigned", "loading", "loaded", "failed"] = "unassigned"
    warm: bool = False
    measured_service_rate: NonNegativeFloat = 0.0
    current_requests: list[str] = Field(default_factory=list)
    queue_depth: NonNegativeInt = 0
    health: HealthStatus = HealthStatus.HEALTHY
    reputation: float = Field(default=1.0, ge=0, le=1)
    last_successful_output: datetime | None = None
    failure_count: NonNegativeInt = 0
    endpoint: str | None = None


class StageBenchmark(StrictModel):
    stage_id: int | None = None
    worker_class: str
    operation: OperationKind
    sequence_length: PositiveInt
    batch_size: PositiveInt
    mean_ms: NonNegativeFloat
    p95_ms: NonNegativeFloat
    samples: PositiveInt
    measured: bool = True


class WorkerCapability(StrictModel):
    worker_id: str
    public_key: str
    hostname: str
    operating_system: str
    architecture: str
    backend: Backend
    cpu_model: str
    logical_cpu_count: PositiveInt
    physical_cpu_count: PositiveInt | None = None
    total_ram_bytes: PositiveInt
    available_ram_bytes: NonNegativeInt
    gpu_model: str | None = None
    total_vram_bytes: NonNegativeInt = 0
    available_vram_bytes: NonNegativeInt = 0
    supported_dtypes: list[str] = Field(default_factory=list)
    supported_quantisation_formats: list[str] = Field(default_factory=list)
    measured_memory_bandwidth_bytes_s: NonNegativeFloat | None = None
    stage_benchmarks: list[StageBenchmark] = Field(default_factory=list)
    upload_bandwidth_bytes_s: NonNegativeFloat
    download_bandwidth_bytes_s: NonNegativeFloat
    coordinator_latency_ms: NonNegativeFloat
    reliability_score: Probability = 1.0
    current_shard_assignments: list[int] = Field(default_factory=list)
    current_queue_depth: NonNegativeInt = 0
    last_heartbeat: datetime = Field(default_factory=lambda: datetime.now(UTC))
    memory_limit_bytes: PositiveInt | None = None
    max_concurrent_stage_operations: PositiveInt = 1
    endpoint: str | None = None
    profile_source: Literal["measured", "assumed", "mixed"] = "measured"

    @property
    def effective_memory_bytes(self) -> int:
        available = (
            self.available_vram_bytes
            if self.backend in {Backend.TORCH_CUDA, Backend.TORCH_MPS}
            else self.available_ram_bytes
        )
        if self.memory_limit_bytes is None:
            return available
        return min(available, self.memory_limit_bytes)


class AttentionConfig(StrictModel):
    head_count: PositiveInt
    key_value_head_count: PositiveInt
    head_dimension: PositiveInt
    rope_theta: float | None = None
    sliding_window: int | None = None


class ModelManifest(StrictModel):
    schema_version: str = "1"
    model_id: str
    model_revision: str
    architecture: str
    tokenizer_id: str
    layer_count: PositiveInt
    hidden_size: PositiveInt
    attention: AttentionConfig
    vocabulary_size: PositiveInt
    weight_dtype: str
    quantisation_format: str | None = None
    total_weight_bytes: PositiveInt
    embedding_bytes: NonNegativeInt
    output_head_bytes: NonNegativeInt
    per_layer_weight_bytes: list[NonNegativeInt]
    estimated_cache_bytes_per_token_per_layer: NonNegativeInt
    activation_bytes_per_stage_boundary: PositiveInt
    stages: list[StageDefinition]
    shard_hashes: dict[str, str]
    compatible_worker_backends: list[Backend]
    source_tensor_hashes: dict[str, str] = Field(default_factory=dict)
    shared_tensors: dict[str, list[int]] = Field(default_factory=dict)
    source_files: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_manifest(self) -> ModelManifest:
        if len(self.per_layer_weight_bytes) != self.layer_count:
            raise ValueError("per_layer_weight_bytes must contain one entry per layer")
        ordered = sorted(self.stages, key=lambda stage: stage.stage_id)
        if [stage.stage_id for stage in ordered] != list(range(len(ordered))):
            raise ValueError("stage IDs must be contiguous from zero")
        cursor = 0
        for stage in ordered:
            if stage.layer_start != cursor:
                raise ValueError("stages must cover layers contiguously without gaps")
            cursor = stage.layer_end
        if cursor != self.layer_count:
            raise ValueError("stages must cover every model layer")
        return self


class SamplingConfig(StrictModel):
    temperature: NonNegativeFloat = 0.0
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: NonNegativeInt = 0
    max_new_tokens: PositiveInt = 16

    @property
    def greedy(self) -> bool:
        return self.temperature == 0


class RequestState(StrictModel):
    request_id: str
    workload_class: WorkloadClass = WorkloadClass.STANDARD
    prompt_token_ids: list[int]
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    random_seed: int
    current_token_position: NonNegativeInt = 0
    committed_output_tokens: list[int] = Field(default_factory=list)
    stage_route: list[str] = Field(default_factory=list)
    stage_local_cache_ownership: dict[int, str] = Field(default_factory=dict)
    retry_count: NonNegativeInt = 0
    deadline_monotonic_s: NonNegativeFloat | None = None
    priority: int = 0
    status: RequestStatus = RequestStatus.PENDING
    verification_state: VerificationState = VerificationState.UNVERIFIED


class AvailabilityWindow(StrictModel):
    start_s: NonNegativeFloat
    end_s: NonNegativeFloat

    @model_validator(mode="after")
    def validate_window(self) -> AvailabilityWindow:
        if self.end_s <= self.start_s:
            raise ValueError("availability window end must be after start")
        return self


class NodeProfile(StrictModel):
    name: str
    count: PositiveInt = 1
    memory_bytes: PositiveInt
    compute_rate_layers_s: float = Field(gt=0)
    supported_backends: list[Backend]
    network_profile: str
    reliability: Probability = 1.0
    availability_pattern: list[AvailabilityWindow] = Field(default_factory=list)
    energy_watts: NonNegativeFloat | None = None
    max_concurrent_stage_operations: PositiveInt = 1
    measured: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class OutageWindow(StrictModel):
    start_s: NonNegativeFloat
    end_s: NonNegativeFloat

    @model_validator(mode="after")
    def validate_window(self) -> OutageWindow:
        if self.end_s <= self.start_s:
            raise ValueError("outage end must be after start")
        return self


class NetworkProfile(StrictModel):
    name: str
    base_latency_ms: NonNegativeFloat
    jitter_ms: NonNegativeFloat = 0.0
    jitter_distribution: JitterDistribution = JitterDistribution.UNIFORM
    upload_bandwidth_bytes_s: float = Field(gt=0)
    download_bandwidth_bytes_s: float = Field(gt=0)
    packet_loss: Probability = 0.0
    duplication_probability: Probability = 0.0
    reordering_probability: Probability = 0.0
    outage_windows: list[OutageWindow] = Field(default_factory=list)
    permanent_failure_at_s: NonNegativeFloat | None = None
    max_in_flight_bytes: PositiveInt = 64 * 1024 * 1024
    measured: bool = False


class SyntheticModelConfig(StrictModel):
    layer_count: PositiveInt = 16
    bytes_per_layer: PositiveInt = 64 * 1024 * 1024
    hidden_size: PositiveInt = 1024
    activation_dtype: Literal["float16", "float32"] = "float16"
    cache_bytes_per_token_per_layer: NonNegativeInt = 4096
    compute_work_per_layer: float = Field(default=1.0, gt=0)
    routing: Literal["dense", "moe"] = "dense"
    expert_count: PositiveInt = 1
    active_experts: PositiveInt = 1
    model_seed: int = 1
    stage_count: PositiveInt = 4

    @model_validator(mode="after")
    def validate_experts_and_stages(self) -> SyntheticModelConfig:
        if self.active_experts > self.expert_count:
            raise ValueError("active_experts cannot exceed expert_count")
        if self.stage_count > self.layer_count:
            raise ValueError("stage_count cannot exceed layer_count")
        return self

    @property
    def activation_bytes(self) -> int:
        width = 2 if self.activation_dtype == "float16" else 4
        return self.hidden_size * width


class WorkloadConfig(StrictModel):
    concurrent_requests: PositiveInt = 16
    prompt_tokens: PositiveInt = 16
    output_tokens: PositiveInt = 32
    workload_class: WorkloadClass = WorkloadClass.STANDARD
    arrival_interval_ms: NonNegativeFloat = 0
    duration_s: float | None = Field(default=None, gt=0)


class QueueConfig(StrictModel):
    capacity: PositiveInt = 256
    max_microbatch_size: PositiveInt = 1
    max_microbatch_wait_ms: NonNegativeFloat = 0
    request_deadline_ms: float = Field(default=120_000, gt=0)
    backpressure_policy: BackpressurePolicy = BackpressurePolicy.REJECT


class FaultConfig(StrictModel):
    churn_rate_per_hour: Probability = 0.0
    burst_failure_fraction: Probability = 0.0
    corrupt_worker_fraction: Probability = 0.0
    audit_fraction: Probability = 0.0
    slow_worker_fraction: Probability = 0.0
    slow_worker_multiplier: float = Field(default=1.0, ge=1.0)
    coordinator_restart_at_s: NonNegativeFloat | None = None
    join_events_s: list[NonNegativeFloat] = Field(default_factory=list)


class IntegrityConfig(StrictModel):
    enabled: bool = True
    audit_fraction: Probability = 0.05
    disagreement_penalty: Probability = 0.25
    agreement_reward: Probability = 0.01
    quarantine_threshold: Probability = 0.5
    real_model_atol: NonNegativeFloat = 1e-5
    real_model_rtol: NonNegativeFloat = 1e-4


class AcceptanceConfig(StrictModel):
    minimum_aggregate_verified_tokens_s: NonNegativeFloat = 20.0
    minimum_duration_s: NonNegativeFloat = 300.0
    minimum_stage_utilisation: Probability = 0.70
    minimum_doubling_gain: NonNegativeFloat = 1.6
    maximum_capacity_imbalance: Probability = 0.20
    minimum_completion_fraction: Probability = 0.99
    maximum_churn_throughput_degradation: Probability = 0.20


class ExperimentConfig(StrictModel):
    schema_version: str = "1"
    name: str
    execution_mode: ExecutionMode
    seed: int
    scheduler: SchedulerMode
    model: SyntheticModelConfig = Field(default_factory=SyntheticModelConfig)
    workload: WorkloadConfig = Field(default_factory=WorkloadConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    faults: FaultConfig = Field(default_factory=FaultConfig)
    integrity: IntegrityConfig = Field(default_factory=IntegrityConfig)
    acceptance: AcceptanceConfig = Field(default_factory=AcceptanceConfig)
    network: NetworkProfile
    nodes: list[NodeProfile]
    node_counts: list[PositiveInt] = Field(default_factory=list)
    concurrent_request_counts: list[PositiveInt] = Field(default_factory=list)
    warmup_s: NonNegativeFloat = 0.0
    steady_state_s: float = Field(default=30.0, gt=0)
    output_root: str = "artifacts/runs"
    model_id: str = "synthetic"
    model_revision: str = "synthetic-v1"
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_experiment(self) -> ExperimentConfig:
        if not self.nodes:
            raise ValueError("at least one node profile is required")
        return self
