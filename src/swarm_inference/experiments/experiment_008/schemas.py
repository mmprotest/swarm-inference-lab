"""Evidence, tensor-tile, plan, and verdict schemas for Experiment 008."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from swarm_inference.config.models import StrictModel


class EvidenceClass(StrEnum):
    MEASURED = "MEASURED"
    EMULATED = "EMULATED"
    PROJECTED = "PROJECTED"


class ExecutionStatus(StrEnum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    UNSUPPORTED = "UNSUPPORTED"
    INCOMPLETE = "INCOMPLETE"
    NOT_RUN = "NOT_RUN"


class Experiment008Verdict(StrEnum):
    PASS_STRONG = "PASS_STRONG"
    PASS_CAPACITY_AND_ARCHITECTURE = "PASS_CAPACITY_AND_ARCHITECTURE"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"


class GateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_EVALUATED = "NOT_EVALUATED"


class TensorTile(StrictModel):
    """Logical tensor or tensor slice with backend-neutral placement metadata."""

    model_id: str
    model_revision: str
    layer_id: int = Field(ge=-1)
    tensor_name: str
    tensor_role: str
    expert_id: int | None = Field(default=None, ge=0)
    logical_shape: list[int]
    logical_slice: dict[str, Any]
    physical_layout: str
    dtype: str
    quantization: str
    quantization_metadata: dict[str, Any]
    accumulator_dtype: str
    byte_size: int = Field(ge=0)
    content_hash: str
    allowed_backends: list[str]
    current_residency: str
    planned_execution_device: str

    @field_validator("logical_shape")
    @classmethod
    def validate_shape(cls, value: list[int]) -> list[int]:
        if not value or any(item <= 0 for item in value):
            raise ValueError("logical tensor shapes must be non-empty and positive")
        return value

    @field_validator("content_hash")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not value:
            raise ValueError("content_hash cannot be empty")
        return value


class ExpertMicroshard(StrictModel):
    """Matched expert projection ranges; the three slices are one atomic shard."""

    layer_id: int = Field(ge=0)
    expert_id: int = Field(ge=0)
    hidden_start: int = Field(ge=0)
    hidden_end: int = Field(gt=0)
    up: TensorTile
    gate: TensorTile
    down: TensorTile

    @model_validator(mode="after")
    def validate_matching_projection_ranges(self) -> ExpertMicroshard:
        if self.hidden_end <= self.hidden_start:
            raise ValueError("expert microshard range must be non-empty")
        expected_roles = {"up": self.up, "gate": self.gate, "down": self.down}
        for role, tile in expected_roles.items():
            if tile.tensor_role != f"routed_expert_{role}_projection":
                raise ValueError(f"{role} tile has the wrong tensor role")
            if tile.layer_id != self.layer_id or tile.expert_id != self.expert_id:
                raise ValueError("expert microshard tile identity does not match its group")
            projection = tile.logical_slice.get("projection_range")
            if projection != {
                "hidden_start": self.hidden_start,
                "hidden_end": self.hidden_end,
            }:
                raise ValueError("up, gate, and down must preserve the same projection range")
        return self


class ModelPreflight(StrictModel):
    model_id: str
    model_revision: str
    model_architecture: str
    quantization_format: str
    total_tensor_bytes: int = Field(ge=0)
    total_expert_bytes: int = Field(ge=0)
    layer_count: int = Field(ge=0)
    routed_expert_count: int = Field(ge=0)
    experts_selected_per_token: int = Field(ge=0)
    shared_expert_count: int = Field(ge=0)
    system_ram_required_bytes: int = Field(ge=0)
    system_ram_available_bytes: int = Field(ge=0)
    physical_vram_bytes: int = Field(ge=0)
    backend_selected: str
    backend_limitations: list[str]
    genuinely_exceeds_32gb: bool
    genuinely_exceeds_physical_vram: bool
    eligible: bool
    rejection_reasons: list[str] = Field(default_factory=list)


class TechniqueDecision(StrictModel):
    technique: str
    enabled: bool
    execution_status: ExecutionStatus
    evidence_class: EvidenceClass | None = None
    predicted_utility: float | None = None
    measured_utility: float | None = None
    reason: str


class TensorPlacement(StrictModel):
    tensor_pattern: str
    tensor_role: str
    residency: Literal["GPU", "CPU", "MAPPED", "ON_DEMAND", "BACKEND_MANAGED"]
    execution_device: Literal["GPU", "CPU", "GPU_AFTER_PREFETCH", "BACKEND_SELECTED"]
    byte_size: int = Field(ge=0)
    reason: str


class PhasePlan(StrictModel):
    schema_version: str = "experiment-008-plan-v1"
    plan_id: str
    configuration: Literal["A", "B", "C", "D", "E", "F", "G"]
    phase: Literal["prefill", "decode", "mixed"]
    objective: Literal[
        "maximum_decode_throughput",
        "minimum_time_to_first_token",
        "maximum_mixed_verified_throughput",
        "minimum_peak_vram_subject_to_latency",
    ]
    placements: list[TensorPlacement]
    techniques: list[TechniqueDecision]
    backend_arguments: list[str]
    predicted_metrics: dict[str, float | int | None]
    constraints: dict[str, float | int | str]
    explanation: list[str]


class CostBreakdown(StrictModel):
    transfer_ms: float = Field(ge=0)
    dequantization_ms: float = Field(ge=0)
    compute_ms: float = Field(ge=0)
    synchronization_ms: float = Field(ge=0)
    reduction_ms: float = Field(ge=0)
    cache_miss_ms: float = Field(ge=0)
    contention_ms: float = Field(ge=0)
    completion_ms: float = Field(ge=0)
    critical_path: list[str]


class BenchmarkObservation(StrictModel):
    configuration: Literal["A", "B", "C", "D", "E", "F", "G"]
    workload: Literal["decode", "prefill_8k", "prefill_32k", "mixed"]
    plan_id: str
    status: ExecutionStatus
    evidence_class: EvidenceClass | None = None
    metrics: dict[str, Any]
    unavailable_reason: str | None = None
    exit_code: int | None = None

    @model_validator(mode="after")
    def require_reason_for_missing_execution(self) -> BenchmarkObservation:
        if self.status != ExecutionStatus.COMPLETED and not self.unavailable_reason:
            raise ValueError("non-completed observations require an unavailable_reason")
        if self.status == ExecutionStatus.COMPLETED and self.evidence_class is None:
            raise ValueError("completed observations require an evidence classification")
        return self


class GateResult(StrictModel):
    gate_id: int = Field(ge=1, le=6)
    name: str
    status: GateStatus
    evidence_class: EvidenceClass | None
    reasons: list[str]
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)


def overall_verdict(
    gates: list[GateResult], *, real_model_generation_succeeded: bool, official_full_run: bool
) -> Experiment008Verdict:
    """Classify the experiment without promoting fixture or incomplete evidence."""

    by_id = {gate.gate_id: gate for gate in gates}
    if not official_full_run:
        return (
            Experiment008Verdict.PARTIAL
            if real_model_generation_succeeded
            else Experiment008Verdict.FAIL
        )
    if not real_model_generation_succeeded:
        return Experiment008Verdict.FAIL
    foundational = all(
        by_id.get(gate_id) is not None and by_id[gate_id].status == GateStatus.PASS
        for gate_id in (1, 2, 4, 6)
    )
    if not foundational:
        return Experiment008Verdict.PARTIAL
    if by_id.get(3) is not None and by_id[3].status == GateStatus.PASS:
        return Experiment008Verdict.PASS_STRONG
    return Experiment008Verdict.PASS_CAPACITY_AND_ARCHITECTURE
