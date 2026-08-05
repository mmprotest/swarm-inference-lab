"""Typed control records for product planning, deployment, and publication."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, PositiveInt, model_validator

from swarm_inference.config.models import StrictModel, WorkerCapability
from swarm_inference.model.partition import StageAssignment
from swarm_inference.model.product import (
    ProductModelMetadata,
    ProductModelReference,
    ProductModelSpec,
)
from swarm_inference.protocol.stage_worker import LoadedStageStatus


class WorkerModelProbeRequest(StrictModel):
    worker_id: str
    request_id: str
    reference: ProductModelReference
    deadline_unix_ns: PositiveInt | None = None


class WorkerModelProbeResponse(StrictModel):
    worker_id: str
    request_id: str
    available: bool
    detail: str = ""
    spec: ProductModelSpec | None = None
    metadata: ProductModelMetadata | None = None
    resolved_from_local_cache: bool = True
    worker_download_permitted: bool = False

    @model_validator(mode="after")
    def validate_result(self) -> WorkerModelProbeResponse:
        if self.available and (self.spec is None or self.metadata is None):
            raise ValueError("an available worker model probe requires spec and metadata")
        if not self.available and (self.spec is not None or self.metadata is not None):
            raise ValueError("an unavailable worker model probe cannot include resolved metadata")
        return self


class WorkerEligibilityReport(StrictModel):
    worker_id: str
    eligible: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    effective_memory_bytes: int = Field(ge=0)
    active_session_count: int = Field(ge=0)
    exact_model_identity: bool = False
    measured_profile: bool = False


class PlanWorkerAssignment(StrictModel):
    stage_id: int = Field(ge=0)
    worker_id: str
    control_endpoint: str
    data_endpoint: str
    device: str
    effective_memory_bytes: int = Field(gt=0)
    required_memory_bytes: int = Field(gt=0)
    assignment: StageAssignment


class PlanCandidateReport(StrictModel):
    name: str
    topology: str
    stage_count: PositiveInt
    partition_method: Literal["equal", "balanced"]
    feasible: bool
    selected: bool = False
    rejection_reasons: list[str] = Field(default_factory=list)
    worker_ids: list[str] = Field(default_factory=list)
    memory_estimates_bytes: dict[str, int] = Field(default_factory=dict)
    compute_estimates_ms: dict[str, float] = Field(default_factory=dict)
    network_estimates_ms: dict[str, float] = Field(default_factory=dict)
    expected_critical_path_waits_ms: dict[str, float] = Field(default_factory=dict)
    expected_critical_path_ms: float | None = Field(default=None, ge=0)
    expected_utility_tokens_s: float | None = Field(default=None, ge=0)


class StagePlanReport(StrictModel):
    selected_topology: str | None
    rejected_candidates: list[str] = Field(default_factory=list)
    worker_assignments: list[PlanWorkerAssignment] = Field(default_factory=list)
    memory_estimates_bytes: dict[str, int] = Field(default_factory=dict)
    compute_estimates_ms: dict[str, float] = Field(default_factory=dict)
    network_estimates_ms: dict[str, float] = Field(default_factory=dict)
    expected_critical_path_waits_ms: dict[str, float] = Field(default_factory=dict)
    reason_for_selection: str
    candidates: list[PlanCandidateReport]
    worker_eligibility: list[WorkerEligibilityReport]


class ProductExpertPlacement(StrictModel):
    layer_id: int = Field(ge=0)
    expert_id: int = Field(ge=0)
    strategy: Literal["local", "whole-remote", "microshard-remote"]
    worker_ids: list[str] = Field(default_factory=list)
    worker_endpoints: dict[str, str] = Field(default_factory=dict)
    expert_hashes: dict[str, str] = Field(default_factory=dict)
    microshards: list[dict[str, Any]] = Field(default_factory=list)
    measured_utility_ms: float = 0.0
    capacity_required: bool = False
    local_fallback_permitted: bool = False
    forced_remote: bool = False
    explanation: list[str] = Field(default_factory=list)
    rejected: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_placement(self) -> ProductExpertPlacement:
        if len(self.worker_ids) != len(set(self.worker_ids)) or any(
            not worker_id for worker_id in self.worker_ids
        ):
            raise ValueError("expert placement worker identities must be unique and non-empty")
        if self.forced_remote and self.local_fallback_permitted:
            raise ValueError("forced-remote expert placement cannot permit local fallback")
        if self.strategy == "local":
            if self.worker_ids or self.worker_endpoints or self.expert_hashes or self.microshards:
                raise ValueError("local expert placement cannot contain remote ownership")
            if self.forced_remote or self.local_fallback_permitted:
                raise ValueError("local expert placement cannot be forced or a fallback target")
            return self
        workers = set(self.worker_ids)
        if set(self.worker_endpoints) != workers or any(
            not endpoint for endpoint in self.worker_endpoints.values()
        ):
            raise ValueError("remote expert endpoints must exactly match placement workers")
        if self.strategy == "whole-remote":
            if len(self.worker_ids) != 1 or self.microshards:
                raise ValueError("whole-expert placement requires one unsliced remote owner")
            worker_id = self.worker_ids[0]
            if not self.expert_hashes.get(worker_id, "").startswith("sha256:"):
                raise ValueError("whole-expert placement requires a content hash")
            return self
        if len(workers) < 2 or not self.microshards:
            raise ValueError("native microsharding requires at least two physical workers")
        ordered = sorted(
            self.microshards,
            key=lambda item: (
                int(item.get("hidden_start", -1)),
                int(item.get("hidden_end", -1)),
                str(item.get("worker_id", "")),
            ),
        )
        cursor = 0
        logical_width: int | None = None
        shard_workers: set[str] = set()
        for shard in ordered:
            try:
                worker_id = str(shard["worker_id"])
                layer_id = int(shard["layer_id"])
                expert_id = int(shard["expert_id"])
                start = int(shard["hidden_start"])
                end = int(shard["hidden_end"])
                logical = int(shard["logical_intermediate_dimension"])
                content_hash = str(shard["content_hash"])
                raw_group = shard.get("quantization_group_size")
                group = int(raw_group) if raw_group is not None else None
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("microshard placement descriptor is incomplete") from exc
            if worker_id not in workers or layer_id != self.layer_id or expert_id != self.expert_id:
                raise ValueError("microshard identity does not match its placement")
            if logical_width is None:
                logical_width = logical
            if logical != logical_width or start != cursor or end <= start or end > logical:
                raise ValueError("microshard placement is not a gap-free matched union")
            if start == 0 and end == logical:
                raise ValueError("a microshard worker cannot own the full expert")
            if not content_hash.startswith("sha256:"):
                raise ValueError("microshard placement requires content hashes")
            if group is not None and (
                group <= 0 or start % group or (end != logical and end % group)
            ):
                raise ValueError("microshard placement splits a quantisation group")
            cursor = end
            shard_workers.add(worker_id)
        if logical_width is None or cursor != logical_width or shard_workers != workers:
            raise ValueError("microshard ownership does not exactly cover the planned workers")
        return self


class ProductStageExpertPlan(StrictModel):
    stage_id: int = Field(ge=0)
    policy: Literal["auto", "local", "whole-remote", "microshard-remote", "hybrid"]
    require_remote_experts: bool = False
    placements: list[ProductExpertPlacement] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_stage_experts(self) -> ProductStageExpertPlan:
        identities = [(item.layer_id, item.expert_id) for item in self.placements]
        if len(identities) != len(set(identities)):
            raise ValueError("a stage expert may have only one placement")
        if self.require_remote_experts and any(
            item.strategy == "local" or item.local_fallback_permitted for item in self.placements
        ):
            raise ValueError("forced-remote stage plans cannot use or fall back to local experts")
        return self


class ProductStagePlan(StrictModel):
    plan_id: str
    topology_id: str
    generation: PositiveInt = 1
    created_monotonic_ns: PositiveInt
    model: ProductModelSpec
    stage_count: PositiveInt
    partition_method: Literal["equal", "balanced"]
    max_sequence_tokens: PositiveInt
    assignments: list[PlanWorkerAssignment]
    expert_plans: list[ProductStageExpertPlan] = Field(default_factory=list)
    expert_model_fingerprint: str = ""
    expert_quantization_fingerprint: str = ""
    report: StagePlanReport

    @model_validator(mode="after")
    def validate_topology(self) -> ProductStagePlan:
        ordered = sorted(self.assignments, key=lambda item: item.stage_id)
        if self.assignments != ordered:
            raise ValueError("product plan assignments must be ordered by stage ID")
        if [item.stage_id for item in ordered] != list(range(self.stage_count)):
            raise ValueError("product plan stages must be contiguous from zero")
        if len({item.worker_id for item in ordered}) != self.stage_count:
            raise ValueError("one product worker may own only one stage in a topology")
        layers = [layer for item in ordered for layer in item.assignment.layer_ids]
        if layers != list(range(self.model.layer_count)):
            raise ValueError("product plan must own every model layer exactly once")
        if self.expert_plans:
            expert_stage_ids = [item.stage_id for item in self.expert_plans]
            if expert_stage_ids != list(range(self.stage_count)):
                raise ValueError("product expert plans must follow the contiguous stage topology")
            for expert_plan, stage in zip(self.expert_plans, ordered, strict=True):
                allowed_layers = set(stage.assignment.layer_ids)
                if any(item.layer_id not in allowed_layers for item in expert_plan.placements):
                    raise ValueError("expert placement lies outside its owning contiguous stage")
            if any(
                placement.strategy != "local"
                for expert_plan in self.expert_plans
                for placement in expert_plan.placements
            ) and (not self.expert_model_fingerprint or not self.expert_quantization_fingerprint):
                raise ValueError(
                    "remote expert plans require exact model and quantisation identity"
                )
        return self


class ModelInspectRequest(StrictModel):
    reference: ProductModelReference


class ModelInspectResponse(StrictModel):
    spec: ProductModelSpec
    metadata: ProductModelMetadata
    worker_eligibility: list[WorkerEligibilityReport]


class ModelPlanRequest(StrictModel):
    reference: ProductModelReference
    stage_count: int | None = Field(default=None, ge=1, le=2)
    partition_method: Literal["auto", "equal", "balanced"] = "auto"
    require_distributed: bool = False
    max_sequence_tokens: PositiveInt = 2048
    expert_policy: Literal["auto", "local", "whole-remote", "microshard-remote", "hybrid"] = "auto"
    require_remote_experts: bool = False
    allow_expert_local_fallback: bool = False


class ModelPlanResponse(StrictModel):
    plan: ProductStagePlan


class ModelDeployRequest(StrictModel):
    plan: ProductStagePlan


class DeploymentPhase(StrEnum):
    RESERVING = "reserving"
    LOADING = "loading"
    VERIFYING_LOADS = "verifying-loads"
    INSTALLING_ROUTES = "installing-routes"
    VERIFYING_PEERS = "verifying-peers"
    READY = "ready"
    RECOVERING = "recovering"
    ROLLING_BACK = "rolling-back"
    FAILED = "failed"
    UNLOADING = "unloading"
    UNLOADED = "unloaded"


class DeploymentWorkerStatus(StrictModel):
    worker_id: str
    stage_id: int = Field(ge=0)
    control_endpoint: str
    data_endpoint: str
    process_id: int | None = Field(default=None, gt=0)
    reserved: bool = False
    loaded: bool = False
    ownership_verified: bool = False
    route_installed: bool = False
    peer_verified: bool = False
    load_count: int = Field(default=0, ge=0)
    detail: str = ""


class DeploymentStatus(StrictModel):
    deployment_id: str
    plan_id: str
    topology_id: str
    generation: PositiveInt
    model: ProductModelSpec
    phase: DeploymentPhase
    ready: bool
    idempotent: bool = False
    workers: list[DeploymentWorkerStatus]
    created_monotonic_ns: PositiveInt
    updated_monotonic_ns: PositiveInt
    detail: str = ""


class ModelDeployResponse(StrictModel):
    deployment: DeploymentStatus


class ModelUnloadRequest(StrictModel):
    topology_id: str | None = None
    force: bool = False


class ModelUnloadResponse(StrictModel):
    deployment: DeploymentStatus | None
    detail: str


class TopologyStatusRequest(StrictModel):
    topology_id: str | None = None


class TopologyStatusResponse(StrictModel):
    deployments: list[DeploymentStatus]


class WorkersRequest(StrictModel):
    include_unhealthy: bool = True


class WorkerProductStatus(StrictModel):
    capability: WorkerCapability
    healthy_registration: bool
    heartbeat_age_s: float = Field(ge=0)
    last_heartbeat_unix_ns: int | None = Field(default=None, gt=0)
    control_endpoint: str | None = None
    data_endpoint: str | None = None
    loaded_stages: list[LoadedStageStatus] = Field(default_factory=list)
    active_sessions: int = Field(default=0, ge=0)
    queue_depths: dict[str, int] = Field(default_factory=dict)
    memory_bytes: dict[str, int] = Field(default_factory=dict)
    expert_status: dict[str, Any] = Field(default_factory=dict)
    last_error: str | None = None
    detail: str = ""


class WorkersResponse(StrictModel):
    workers: list[WorkerProductStatus]


class ProductTokenPublication(StrictModel):
    worker_id: str
    request_id: str
    session_id: str
    topology_id: str
    route_generation: PositiveInt
    model_revision: str
    token_position: int = Field(ge=0)
    token_id: int = Field(ge=0)
    decoded_text_fragment: str = ""
    published_monotonic_ns: PositiveInt
    request_generation: PositiveInt = 1
    replay_only: bool = False
    expert_trace: list[dict[str, Any]] = Field(default_factory=list)
    expert_metrics: dict[str, Any] = Field(default_factory=dict)
    signature: str = ""


class ProductRequestPhase(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    RECOVERING = "recovering"
    RECOVERABLE = "recoverable"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProductRequestRecoveryState(StrictModel):
    """Durable coordinator-owned state sufficient for exact full replay."""

    request_id: str
    request_generation: PositiveInt = 1
    session_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    topology_id: str
    route_generation: PositiveInt
    prompt_token_ids: list[int]
    accepted_generated_token_ids: list[int] = Field(default_factory=list)
    next_token_position: int = Field(ge=0)
    sampling_policy: Literal["greedy"] = "greedy"
    active_workers: list[str]
    stage_assignments: list[PlanWorkerAssignment]
    recovery_count: int = Field(default=0, ge=0)
    last_healthy_checkpoint: int = Field(default=0, ge=0)
    status: ProductRequestPhase
    last_error: str | None = None
    started_unix_ns: PositiveInt
    updated_unix_ns: PositiveInt

    @model_validator(mode="after")
    def validate_replay_position(self) -> ProductRequestRecoveryState:
        if self.next_token_position != len(self.accepted_generated_token_ids):
            raise ValueError("next token position must equal the accepted token count")
        if self.last_healthy_checkpoint > self.next_token_position:
            raise ValueError("last healthy checkpoint is beyond accepted history")
        if self.active_workers != [item.worker_id for item in self.stage_assignments]:
            raise ValueError("active worker list must match ordered stage assignments")
        return self


class CoordinatorStatusRequest(StrictModel):
    pass


class CoordinatorStatusResponse(StrictModel):
    coordinator_identity: str
    coordinator_public_key_fingerprint: str
    uptime_s: float = Field(ge=0)
    state_directory: str
    registered_worker_count: int = Field(ge=0)
    healthy_worker_count: int = Field(ge=0)
    known_deployments: int = Field(ge=0)
    active_topology_id: str | None = None
    route_generation: int | None = Field(default=None, ge=1)
    active_session_count: int = Field(ge=0)
    generated_tokens: int = Field(ge=0)
    throughput_tokens_s: float = Field(ge=0)
    time_to_first_token_s: float | None = Field(default=None, ge=0)
    inter_token_latency_s: float | None = Field(default=None, ge=0)
    recovery_count: int = Field(ge=0)
    recovering_requests: int = Field(ge=0)
    queue_depths: dict[str, int] = Field(default_factory=dict)
    memory_bytes: dict[str, int] = Field(default_factory=dict)
    reservations: dict[str, Any] = Field(default_factory=dict)
    expert_worker_count: int = Field(default=0, ge=0)
    owned_experts: int = Field(default=0, ge=0)
    owned_microshards: int = Field(default=0, ge=0)
    expert_cache_resident_bytes: int = Field(default=0, ge=0)
    expert_cache_hits: int = Field(default=0, ge=0)
    expert_cache_misses: int = Field(default=0, ge=0)
    remote_expert_calls: int = Field(default=0, ge=0)
    remote_microshard_calls: int = Field(default=0, ge=0)
    expert_fallbacks: int = Field(default=0, ge=0)
    expert_bytes_transferred: int = Field(default=0, ge=0)
    expert_critical_path_ns: int = Field(default=0, ge=0)
    expert_reduction_modes: list[str] = Field(default_factory=list)
    last_error: str | None = None


class SessionsRequest(StrictModel):
    include_terminal: bool = False


class ProductSessionStatus(StrictModel):
    request_id: str
    request_generation: PositiveInt
    session_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    topology_id: str
    route_generation: PositiveInt
    status: ProductRequestPhase
    token_position: int = Field(ge=0)
    accepted_token_ids: list[int] = Field(default_factory=list)
    active_workers: list[str] = Field(default_factory=list)
    kv_cache_bytes: int = Field(default=0, ge=0)
    queue_depth: int = Field(default=0, ge=0)
    recovery_count: int = Field(default=0, ge=0)
    last_healthy_checkpoint: int = Field(default=0, ge=0)
    time_to_first_token_s: float | None = Field(default=None, ge=0)
    inter_token_latency_s: float | None = Field(default=None, ge=0)
    last_error: str | None = None


class SessionsResponse(StrictModel):
    sessions: list[ProductSessionStatus]


class CancelProductRequest(StrictModel):
    request_id: str


class CancelProductResponse(StrictModel):
    request_id: str
    accepted: bool
    idempotent: bool = False
    status: ProductRequestPhase
    released_kv_bytes: int = Field(default=0, ge=0)
    detail: str = ""


__all__ = [
    "CancelProductRequest",
    "CancelProductResponse",
    "CoordinatorStatusRequest",
    "CoordinatorStatusResponse",
    "DeploymentPhase",
    "DeploymentStatus",
    "DeploymentWorkerStatus",
    "ModelDeployRequest",
    "ModelDeployResponse",
    "ModelInspectRequest",
    "ModelInspectResponse",
    "ModelPlanRequest",
    "ModelPlanResponse",
    "ModelUnloadRequest",
    "ModelUnloadResponse",
    "PlanCandidateReport",
    "PlanWorkerAssignment",
    "ProductExpertPlacement",
    "ProductRequestPhase",
    "ProductRequestRecoveryState",
    "ProductSessionStatus",
    "ProductStageExpertPlan",
    "ProductStagePlan",
    "ProductTokenPublication",
    "SessionsRequest",
    "SessionsResponse",
    "StagePlanReport",
    "TopologyStatusRequest",
    "TopologyStatusResponse",
    "WorkerEligibilityReport",
    "WorkerModelProbeRequest",
    "WorkerModelProbeResponse",
    "WorkerProductStatus",
    "WorkersRequest",
    "WorkersResponse",
]
