"""Experiment 010 evidence contracts over the canonical expert protocol.

Product wire, tensor, worker-manifest, and execution schemas are owned by
``swarm_inference.protocol.expert``.  This module retains only experiment
workloads, fault profiles, planning evidence, gates, and historical verdicts.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from swarm_inference.config.models import StrictModel
from swarm_inference.protocol.expert import (
    DataPlane,
    DeterminismMode,
    ExpertExecutionMetadata,
    ExpertExecutionMode,
    ExpertExecutionRequest,
    ExpertExecutionResponse,
    ExpertResponseMode,
    ReductionMode,
    ResultIntegrity,
    TensorWireMetadata,
    TransportCodec,
    WorkerBudget,
    WorkerManifest,
)


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


class RunCompleteness(StrEnum):
    FULL_COMPLETE = "FULL_COMPLETE"
    INCOMPLETE_FULL_RUN = "INCOMPLETE_FULL_RUN"
    DEVELOPMENT_COMPLETE = "DEVELOPMENT_COMPLETE"
    QUICK_COMPLETE = "QUICK_COMPLETE"


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
    CONCURRENT_DECODE = "concurrent_decode"
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


__all__ = [
    "DataPlane",
    "DeterminismMode",
    "EvidenceCategory",
    "ExecutionStrategy",
    "Experiment010Mode",
    "Experiment010Verdict",
    "ExpertExecutionMetadata",
    "ExpertExecutionMode",
    "ExpertExecutionRequest",
    "ExpertExecutionResponse",
    "ExpertResponseMode",
    "FailureType",
    "GateResult",
    "GateStatus",
    "NetworkShapeProfile",
    "PhasePlan",
    "PlannerCandidate",
    "PlannerObjective",
    "RecoveryStrategy",
    "ReductionMode",
    "ResultIntegrity",
    "RunCompleteness",
    "ServicePhase",
    "TensorWireMetadata",
    "TransportCodec",
    "WorkerBudget",
    "WorkerManifest",
    "classify_verdict",
]
