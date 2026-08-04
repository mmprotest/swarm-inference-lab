"""Typed control-plane messages for the persistent stage worker.

The runtime transports these records in protobuf ``Any`` envelopes through the
existing generic gRPC service.  Tensor payloads never use these messages; they
remain on the direct binary stage-ring data plane.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, NonNegativeInt, PositiveInt, model_validator

from swarm_inference.config.models import StrictModel, WorkerCapability
from swarm_inference.model.partition import StageAssignment


class _StageControlModel(StrictModel):
    @model_validator(mode="after")
    def _reject_empty_strings(self) -> _StageControlModel:
        for name, value in self.__dict__.items():
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"stage control field {name!r} cannot be empty")
        return self


class StageRouteEndpoint(_StageControlModel):
    """Identity and advertised address of one adjacent stage."""

    worker_id: str
    stage_id: NonNegativeInt
    data_endpoint: str
    assignment: StageAssignment | None = None


class GetStageCapabilitiesRequest(_StageControlModel):
    worker_id: str
    request_id: str
    deadline_unix_ns: PositiveInt | None = None


class GetStageCapabilitiesResponse(_StageControlModel):
    worker_id: str
    request_id: str
    capability: WorkerCapability


class LoadStageRequest(_StageControlModel):
    worker_id: str
    request_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    topology_id: str
    route_generation: NonNegativeInt = 0
    stage_count: PositiveInt = 1
    assignment: StageAssignment
    device: str
    dtype: str
    model_path: str | None = None
    allow_download: bool = False
    lease_expiry_unix_ns: PositiveInt | None = None
    deadline_unix_ns: PositiveInt | None = None


class UnloadStageRequest(_StageControlModel):
    worker_id: str
    request_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    topology_id: str
    route_generation: NonNegativeInt
    stage_count: PositiveInt = 1
    assignment: StageAssignment
    device: str
    dtype: str
    force: bool = False
    deadline_unix_ns: PositiveInt | None = None


class InstallStageRouteRequest(_StageControlModel):
    worker_id: str
    request_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    topology_id: str
    route_generation: PositiveInt
    assignment: StageAssignment
    device: str
    dtype: str
    previous_stage: StageRouteEndpoint | None
    next_stage: StageRouteEndpoint | None
    stage_count: PositiveInt
    stage_zero_publication_destination: str | None = None
    lease_expiry_unix_ns: PositiveInt
    deadline_unix_ns: PositiveInt | None = None
    replace: bool = False


class RemoveStageRouteRequest(_StageControlModel):
    worker_id: str
    request_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    topology_id: str
    route_generation: PositiveInt
    stage_id: NonNegativeInt
    device: str
    dtype: str
    deadline_unix_ns: PositiveInt | None = None


class VerifyStageRouteRequest(_StageControlModel):
    worker_id: str
    request_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    topology_id: str
    route_generation: PositiveInt
    stage_id: NonNegativeInt
    device: str
    dtype: str
    deadline_unix_ns: PositiveInt | None = None


class OpenStageSessionRequest(_StageControlModel):
    worker_id: str
    request_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    topology_id: str
    route_generation: PositiveInt
    stage_id: NonNegativeInt
    device: str
    dtype: str
    session_id: str
    lease_expiry_unix_ns: PositiveInt | None = None
    deadline_unix_ns: PositiveInt | None = None


class CloseStageSessionRequest(OpenStageSessionRequest):
    pass


class CancelStageSessionRequest(OpenStageSessionRequest):
    pass


class GetStageStatusRequest(_StageControlModel):
    worker_id: str
    request_id: str
    topology_id: str | None = None
    deadline_unix_ns: PositiveInt | None = None


class TokenizeStageRequest(_StageControlModel):
    worker_id: str
    request_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    topology_id: str
    route_generation: PositiveInt
    stage_id: NonNegativeInt = 0
    device: str
    dtype: str
    text: str
    add_special_tokens: bool = True
    deadline_unix_ns: PositiveInt | None = None


class TokenizeStageResponse(_StageControlModel):
    worker_id: str
    request_id: str
    token_ids: list[NonNegativeInt]


class DrainWorkerRequest(_StageControlModel):
    worker_id: str
    request_id: str
    deadline_unix_ns: PositiveInt | None = None
    cancel_active_sessions: bool = False


class StageActionResponse(_StageControlModel):
    worker_id: str
    request_id: str
    accepted: bool
    detail: str
    idempotent: bool = False
    released_kv_bytes: NonNegativeInt = 0


class StageSessionStatus(_StageControlModel):
    topology_id: str
    session_id: str
    model_revision: str
    route_generation: PositiveInt
    stage_id: NonNegativeInt
    cache_position: NonNegativeInt
    kv_cache_bytes: NonNegativeInt
    opened_monotonic_ns: PositiveInt
    last_operation_monotonic_ns: PositiveInt
    cancelled: bool = False


class LoadedStageStatus(_StageControlModel):
    model_id: str
    model_revision: str
    tokenizer_revision: str
    topology_id: str
    assignment: StageAssignment
    device: str
    dtype: str
    model_path: str
    ownership: dict[str, Any]
    loaded_monotonic_ns: PositiveInt
    load_count: PositiveInt
    process_rss_before_bytes: NonNegativeInt
    process_rss_after_bytes: NonNegativeInt
    cuda_allocated_before_bytes: NonNegativeInt
    cuda_allocated_after_bytes: NonNegativeInt
    cuda_reserved_before_bytes: NonNegativeInt
    cuda_reserved_after_bytes: NonNegativeInt


class InstalledStageRouteStatus(_StageControlModel):
    topology_id: str
    route_generation: PositiveInt
    previous_stage: StageRouteEndpoint | None
    next_stage: StageRouteEndpoint | None
    stage_count: PositiveInt
    stage_zero_publication_destination: str | None
    lease_expiry_unix_ns: PositiveInt


class StageStatusResponse(_StageControlModel):
    worker_id: str
    request_id: str
    process_id: PositiveInt
    draining: bool
    loaded_stage: LoadedStageStatus | None
    installed_route: InstalledStageRouteStatus | None
    sessions: list[StageSessionStatus] = Field(default_factory=list)
    execution_queue_depth: NonNegativeInt
    execution_queue_capacity: PositiveInt
    token_queue_depth: NonNegativeInt
    token_queue_capacity: PositiveInt
    dropped_token_publications: NonNegativeInt


# Concise aliases mirror the operation names used by the documented RPC API.
GetStageCapabilities = GetStageCapabilitiesRequest
LoadStage = LoadStageRequest
UnloadStage = UnloadStageRequest
InstallStageRoute = InstallStageRouteRequest
RemoveStageRoute = RemoveStageRouteRequest
VerifyStageRoute = VerifyStageRouteRequest
OpenStageSession = OpenStageSessionRequest
CloseStageSession = CloseStageSessionRequest
CancelStageSession = CancelStageSessionRequest
GetStageStatus = GetStageStatusRequest
TokenizeStage = TokenizeStageRequest
DrainWorker = DrainWorkerRequest


__all__ = [
    "CancelStageSession",
    "CancelStageSessionRequest",
    "CloseStageSession",
    "CloseStageSessionRequest",
    "DrainWorker",
    "DrainWorkerRequest",
    "GetStageCapabilities",
    "GetStageCapabilitiesRequest",
    "GetStageCapabilitiesResponse",
    "GetStageStatus",
    "GetStageStatusRequest",
    "InstallStageRoute",
    "InstallStageRouteRequest",
    "InstalledStageRouteStatus",
    "LoadStage",
    "LoadStageRequest",
    "LoadedStageStatus",
    "OpenStageSession",
    "OpenStageSessionRequest",
    "RemoveStageRoute",
    "RemoveStageRouteRequest",
    "StageActionResponse",
    "StageRouteEndpoint",
    "StageSessionStatus",
    "StageStatusResponse",
    "TokenizeStage",
    "TokenizeStageRequest",
    "TokenizeStageResponse",
    "UnloadStage",
    "UnloadStageRequest",
    "VerifyStageRoute",
    "VerifyStageRouteRequest",
]
