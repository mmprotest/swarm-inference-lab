"""Strict semantic contracts for Experiment 010.

The wire encoding is deliberately separate from these models. All transports
must preserve this schema even when tensor bytes travel outside the JSON frame.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from swarm_inference.config.models import StrictModel


class EvidenceCategory(StrEnum):
    MEASURED_PHYSICAL = "MEASURED_PHYSICAL"
    MEASURED_SINGLE_HOST = "MEASURED_SINGLE_HOST"
    MEASURED_NETWORK_EMULATION = "MEASURED_NETWORK_EMULATION"
    SYNTHETIC_FIXTURE = "SYNTHETIC_FIXTURE"
    SIMULATED_CALIBRATED = "SIMULATED_CALIBRATED"
    SIMULATED_UNCALIBRATED = "SIMULATED_UNCALIBRATED"
    PROJECTED = "PROJECTED"


class Experiment010Mode(StrEnum):
    QUICK = "quick"
    DEVELOPMENT = "development"
    FULL = "full"
    FRONTIER = "frontier"


class Experiment010Verdict(StrEnum):
    PASS_STRONG = "PASS_STRONG"
    PASS_CLOSURE = "PASS_CLOSURE"
    PASS_CAPACITY = "PASS_CAPACITY"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"


class GateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_EVALUATED = "NOT_EVALUATED"


class ExecutionStrategy(StrEnum):
    LOCAL_WHOLE_EXPERT = "local_whole_expert"
    REMOTE_WHOLE_EXPERT = "remote_whole_expert"
    EQUAL_MICROSHARDS = "equal_microshards"
    ASYMMETRIC_MICROSHARDS = "asymmetric_microshards"
    COALESCED_MICROSHARDS = "coalesced_microshards"
    CACHE_ONLY = "cache_only"
    STORAGE_ONLY = "storage_only"
    BACKGROUND_INFERENCE = "background_inference"
    VERIFICATION = "verification"
    IDLE = "idle"


class DataPlane(StrEnum):
    IN_PROCESS = "in_process"
    SHARED_MEMORY = "shared_memory"
    DIRECT_TCP = "direct_tcp"
    RELAYED_TCP = "relayed_tcp"


class ExpertExecutionMode(StrEnum):
    WHOLE_EXPERT = "whole_expert"
    MICROSHARD = "microshard"


class DeterminismMode(StrEnum):
    EXACT = "exact"
    QUALITY_BOUNDED = "quality_bounded"


class TransportCodec(StrEnum):
    RAW_FP32 = "raw_fp32"
    RAW_FP16 = "raw_fp16"
    INT8_PER_VECTOR = "int8_per_vector"
    LOSSLESS_GENERAL = "lossless_general"


class ReductionMode(StrEnum):
    FIXED_ORDER_FP32 = "fixed_order_fp32"
    TREE_FP32 = "tree_fp32"
    FAST_BACKEND_NATIVE = "fast_backend_native"


class PlannerObjective(StrEnum):
    MAX_DECODE_THROUGHPUT = "max_decode_throughput"
    MIN_TTFT = "min_ttft"
    MAX_VERIFIED_AGGREGATE_THROUGHPUT = "max_verified_aggregate_throughput"
    MIN_NETWORK_BYTES = "min_network_bytes"
    MIN_ENERGY_PER_VERIFIED_TOKEN = "min_energy_per_verified_token"
    MAX_CAPACITY_SUBJECT_TO_LATENCY = "max_capacity_subject_to_latency"


class ServicePhase(StrEnum):
    PREFILL = "prefill"
    DECODE = "decode"
    MIXED_SERVICE = "mixed_service"


class FailureType(StrEnum):
    WORKER_TERMINATION = "worker_termination"
    WORKER_PAUSE = "worker_pause"
    FIXED_DELAY = "fixed_delay"
    RANDOM_DELAY = "random_delay"
    CACHE_DROP = "cache_drop"
    STORAGE_SLOWDOWN = "storage_slowdown"
    NETWORK_OUTAGE = "network_outage"
    NETWORK_PARTITION = "network_partition"
    STALE_RESULT = "stale_result"
    DUPLICATE_RESULT = "duplicate_result"
    MALFORMED_RESULT = "malformed_result"
    WRONG_MODEL_REVISION = "wrong_model_revision"
    WRONG_EXPERT = "wrong_expert"
    BIT_FLIP = "bit_flip"
    ZERO_RESULT = "zero_result"
    LOWER_PRECISION_RESULT = "lower_precision_result"


class RecoveryStrategy(StrEnum):
    WAIT_ALL = "wait_all"
    TIMEOUT_LOCAL_FALLBACK = "timeout_local_fallback"
    TIMEOUT_ALTERNATE_WORKER = "timeout_alternate_worker"
    HEDGED_DUPLICATE = "hedged_duplicate"
    SAMPLED_REPLICATION = "sampled_replication"
    SMALL_TILE_WORK_STEALING = "small_tile_work_stealing"


class TensorWireMetadata(StrictModel):
    name: str
    envelope: Literal["raw", "SWARMT01"] = "raw"
    dtype: Literal["float32", "float16", "int8", "uint8"]
    shape: list[int] = Field(min_length=1)
    codec: TransportCodec = TransportCodec.RAW_FP32
    payload_index: int = Field(ge=0)
    raw_bytes: int = Field(ge=0)
    encoded_bytes: int = Field(ge=0)
    scale: float | list[float] | None = None
    checksum: str

    @field_validator("shape")
    @classmethod
    def positive_shape(cls, value: list[int]) -> list[int]:
        if any(item <= 0 for item in value):
            raise ValueError("tensor dimensions must be positive")
        return value


class ExpertExecutionRequest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    request_id: str
    model_id: str
    model_revision: str
    quantization_fingerprint: str
    layer_id: int = Field(ge=0)
    batch_rows: int = Field(gt=0)
    latent_dimension: int = Field(gt=0)
    expert_ids: list[int] = Field(min_length=1)
    routing_weights: list[float] = Field(min_length=1)
    activations: dict[str, Any]
    deadline_ns: int = Field(gt=0)
    execution_mode: ExpertExecutionMode = ExpertExecutionMode.WHOLE_EXPERT
    determinism_mode: DeterminismMode = DeterminismMode.EXACT
    compression: TransportCodec = TransportCodec.RAW_FP32
    hidden_start: int | None = Field(default=None, ge=0)
    hidden_end: int | None = Field(default=None, gt=0)
    reduction_mode: ReductionMode = ReductionMode.FIXED_ORDER_FP32
    challenge: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_request(self) -> ExpertExecutionRequest:
        if len(self.routing_weights) != len(self.expert_ids):
            raise ValueError("expert IDs and routing weights must have equal lengths")
        if any(expert < 0 for expert in self.expert_ids):
            raise ValueError("expert IDs must be non-negative")
        if self.determinism_mode == DeterminismMode.EXACT:
            if self.compression != TransportCodec.RAW_FP32:
                raise ValueError("exact mode requires raw_fp32 transport")
            if self.reduction_mode != ReductionMode.FIXED_ORDER_FP32:
                raise ValueError("exact mode requires fixed_order_fp32 reduction")
        if self.execution_mode == ExpertExecutionMode.MICROSHARD:
            if self.hidden_start is None or self.hidden_end is None:
                raise ValueError("microshard execution requires a hidden range")
            if self.hidden_end <= self.hidden_start:
                raise ValueError("microshard hidden range must be non-empty")
        elif self.hidden_start is not None or self.hidden_end is not None:
            raise ValueError("whole-expert execution cannot carry a hidden range")
        return self


class ExpertExecutionMetadata(StrictModel):
    experts_executed: list[int]
    bytes_read: int = Field(ge=0)
    bytes_received: int = Field(ge=0)
    bytes_sent: int = Field(ge=0)
    cache_hits: int = Field(ge=0)
    cache_misses: int = Field(ge=0)
    compute_ns: int = Field(ge=0)
    queue_ns: int = Field(ge=0)
    transfer_ns: int = Field(ge=0)
    serialisation_ns: int = Field(default=0, ge=0)
    copy_ns: int = Field(default=0, ge=0)
    kernel_transition_ns: int = Field(default=0, ge=0)
    backend: str = "cpu"
    device: str = "cpu"
    resident_tensor_bytes: int = Field(default=0, ge=0)
    expert_resident_bytes: int = Field(default=0, ge=0)
    fallback_events: list[dict[str, Any]] = Field(default_factory=list)


class ResultIntegrity(StrictModel):
    result_hash: str
    model_fingerprint: str
    worker_signature: str


class ExpertExecutionResponse(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    request_id: str
    worker_id: str
    model_revision: str
    layer_id: int = Field(ge=0)
    result: dict[str, Any]
    execution_metadata: ExpertExecutionMetadata
    integrity: ResultIntegrity
    status: Literal["ok", "error"] = "ok"
    error: str | None = None

    @model_validator(mode="after")
    def error_contract(self) -> ExpertExecutionResponse:
        if self.status == "error" and not self.error:
            raise ValueError("error responses require a reason")
        if self.status == "ok" and self.error is not None:
            raise ValueError("successful responses cannot carry an error")
        return self


class WorkerBudget(StrictModel):
    worker_id: str
    memory_budget_bytes: int = Field(gt=0)
    expert_residency_budget_bytes: int = Field(gt=0)
    cache_budget_bytes: int = Field(ge=0)
    thread_count: int = Field(gt=0)
    cpu_affinity: list[int] = Field(min_length=1)
    storage_directory: str
    device: str
    backend: str
    physical_memory_limit: bool = False


class WorkerManifest(StrictModel):
    worker_id: str
    process_id: int = Field(gt=0)
    endpoint: str
    control_endpoint: str | None = None
    universal_worker_abi: dict[str, Any] = Field(default_factory=dict)
    model_id: str
    model_revision: str
    quantization_fingerprint: str
    model_fingerprint: str
    bridge_version: str
    owned_experts: dict[str, list[int]] = Field(default_factory=dict)
    owned_microshards: list[dict[str, Any]] = Field(default_factory=list)
    tensor_hashes: dict[str, str] = Field(default_factory=dict)
    resident_tensor_bytes: int = Field(ge=0)
    expert_bytes: int = Field(ge=0)
    cache_bytes: int = Field(ge=0)
    peak_rss_bytes: int = Field(ge=0)
    roles: list[str] = Field(default_factory=list)


class NetworkShapeProfile(StrictModel):
    name: str
    bandwidth_bps: float | None = Field(default=None, gt=0)
    one_way_latency_ms: float = Field(ge=0)
    jitter_ms: float = Field(default=0, ge=0)
    message_loss_probability: float = Field(default=0, ge=0, le=1)
    duplication_probability: float = Field(default=0, ge=0, le=1)
    reordering_probability: float = Field(default=0, ge=0, le=1)
    outage_intervals_ms: list[tuple[float, float]] = Field(default_factory=list)
    queue_depth: int = Field(default=64, gt=0)
    concurrent_flow_limit: int = Field(default=1, gt=0)
    seed: int = 1010

    @model_validator(mode="after")
    def outage_ranges(self) -> NetworkShapeProfile:
        if any(start < 0 or end <= start for start, end in self.outage_intervals_ms):
            raise ValueError("outage intervals must be positive non-empty ranges")
        return self


class PlannerCandidate(StrictModel):
    candidate_id: str
    phase: ServicePhase
    strategy: ExecutionStrategy
    workers: list[str]
    objective: PlannerObjective
    predicted_utility: float
    lower_confidence_bound: float | None = None
    measured_utility: float | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    throughput: float | None = Field(default=None, ge=0)
    network_bytes: int | None = Field(default=None, ge=0)
    energy_joules_per_token: float | None = Field(default=None, ge=0)
    reliability_gate: bool = True
    correctness_gate: bool = True
    slo_gate: bool = True
    capacity_required: bool = False
    explanation: list[str] = Field(default_factory=list)


class PhasePlan(StrictModel):
    phase: ServicePhase
    objective: PlannerObjective
    selected_candidate_id: str
    selected_strategy: ExecutionStrategy
    selected_workers: list[str]
    rejected: list[dict[str, Any]]
    capacity_exception: bool = False
    codec: TransportCodec = TransportCodec.RAW_FP32
    explanation: list[str]


class GateResult(StrictModel):
    gate_id: int = Field(ge=1, le=16)
    name: str
    status: GateStatus
    evidence_category: EvidenceCategory | None = None
    reasons: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


def classify_verdict(
    gates: list[GateResult],
    *,
    mode: Experiment010Mode,
    real_distributed_expert_execution: bool,
    positive_measured_utility: bool,
    genuine_capacity_result: bool,
) -> Experiment010Verdict:
    """Compute the fail-closed Experiment 010 verdict from saved gate rows."""

    if not real_distributed_expert_execution:
        return Experiment010Verdict.FAIL
    if mode != Experiment010Mode.FULL:
        return Experiment010Verdict.PARTIAL
    passed = {gate.gate_id for gate in gates if gate.status == GateStatus.PASS}
    foundational = set(range(1, 17))
    if foundational <= passed:
        if positive_measured_utility or genuine_capacity_result:
            return Experiment010Verdict.PASS_STRONG
        return Experiment010Verdict.PASS_CLOSURE
    capacity_gates = {3, 4, 6, 7, 8}
    if genuine_capacity_result and capacity_gates <= passed:
        return Experiment010Verdict.PASS_CAPACITY
    return Experiment010Verdict.PARTIAL
