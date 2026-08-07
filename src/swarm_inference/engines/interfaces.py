"""Backend-neutral execution-engine contracts and planning records."""

from __future__ import annotations

from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import ConfigDict, Field, NonNegativeInt, PositiveInt

from swarm_inference.config.models import StrictModel
from swarm_inference.model.descriptor import ResolvedModelDescriptor


class EngineSupportStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    UNSUPPORTED_ARCHITECTURE = "UNSUPPORTED_ARCHITECTURE"
    MISSING_RUNTIME = "MISSING_RUNTIME"
    MISSING_DEVICE_CAPABILITY = "MISSING_DEVICE_CAPABILITY"
    INSUFFICIENT_MEMORY = "INSUFFICIENT_MEMORY"
    CONVERSION_AVAILABLE = "CONVERSION_AVAILABLE"
    BROKEN_RUNTIME = "BROKEN_RUNTIME"


class ExecutionDevice(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    device_id: str
    device_type: Literal["cpu", "cuda", "metal", "mps", "rocm", "vulkan"]
    name: str
    uuid: str | None = None
    total_memory_bytes: NonNegativeInt = 0
    usable_memory_bytes: NonNegativeInt = 0
    runtime_version: str | None = None
    driver_version: str | None = None
    measured_prefill_tokens_s: float | None = Field(default=None, ge=0)
    measured_decode_tokens_s: float | None = Field(default=None, ge=0)
    features: tuple[str, ...] = ()


class AdapterFastPathCapability(StrictModel):
    """Fast-path modes exposed by one adapter on one worker runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    adapter_id: str
    fast_path_id: str
    candidate_modes: tuple[str, ...] = ()


class ExecutionProfileCapability(StrictModel):
    """Content-addressed, measured engine profile safe to expose to planning."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str
    mechanism: str
    adapter_id: str | None = None
    model_fingerprint: str
    content_fingerprint: str
    exactness_passed: bool
    measured_utility: float
    evidence_fingerprint: str


class ExecutionEngineCapability(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    engine_id: str
    enabled: bool
    runtime_revision: str | None = None
    binary_hashes: dict[str, str] = Field(default_factory=dict)
    formats: tuple[str, ...] = ()
    devices: tuple[ExecutionDevice, ...] = ()
    adapters: tuple[str, ...] = ()
    fast_paths: tuple[str, ...] = ()
    adapter_fast_paths: tuple[AdapterFastPathCapability, ...] = ()
    execution_profiles: tuple[ExecutionProfileCapability, ...] = ()
    roles: tuple[str, ...] = ()
    detail: str = ""


class WorkerExecutionCapability(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    worker_id: str
    node_id: str
    engines: tuple[ExecutionEngineCapability, ...]
    queue_depth: NonNegativeInt = 0
    reliability: float = Field(default=1.0, ge=0, le=1)
    network_latency_ms: dict[str, float] = Field(default_factory=dict)
    network_bandwidth_bytes_s: dict[str, float] = Field(default_factory=dict)
    resident_model_fingerprints: tuple[str, ...] = ()
    storage_available_bytes: NonNegativeInt = 0

    def engine(self, engine_id: str) -> ExecutionEngineCapability | None:
        return next((item for item in self.engines if item.engine_id == engine_id), None)


class ClusterCapabilities(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    workers: tuple[WorkerExecutionCapability, ...]

    @property
    def aggregate_usable_memory_bytes(self) -> int:
        # The same physical CPU/GPU is commonly exposed by several engines.
        # Count it once, and also avoid multiplying shared RAM when a node runs
        # more than one logical worker process.
        physical: dict[tuple[str, str], int] = {}
        for worker in self.workers:
            for engine in worker.engines:
                if not engine.enabled:
                    continue
                for device in engine.devices:
                    identity = device.uuid or f"{device.device_type}:{device.device_id}"
                    key = (worker.node_id, identity)
                    physical[key] = max(
                        physical.get(key, 0),
                        int(device.usable_memory_bytes),
                    )
        return sum(physical.values())

    def workers_for_engine(self, engine_id: str) -> tuple[WorkerExecutionCapability, ...]:
        return tuple(
            worker
            for worker in self.workers
            if (capability := worker.engine(engine_id)) is not None and capability.enabled
        )


class EngineSupportReport(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    engine_id: str
    status: EngineSupportStatus
    reason: str
    supported_worker_ids: tuple[str, ...] = ()
    adapter_id: str | None = None
    conversion: dict[str, Any] | None = None
    runtime_identity: dict[str, Any] = Field(default_factory=dict)

    @property
    def supported(self) -> bool:
        return self.status == EngineSupportStatus.SUPPORTED


class ExecutionRequest(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    objective: Literal["speed", "throughput", "capacity", "balanced"] = "balanced"
    require_distributed: bool = False
    concurrency: PositiveInt = 1
    priority: int = 0
    max_context_tokens: PositiveInt = 2048
    max_new_tokens: PositiveInt = 16
    requested_engine: str | None = None
    requested_nodes: tuple[str, ...] = ()
    excluded_nodes: tuple[str, ...] = ()
    forced_mechanisms: tuple[str, ...] = ()
    quality_preference: float = Field(default=0.6, ge=0, le=1)


class PhasePlan(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: Literal["prefill", "decode"]
    worker_roles: dict[str, str]
    parameters: dict[str, Any] = Field(default_factory=dict)


class MechanismEvidence(StrictModel):
    """Exact, runtime-matched utility evidence for one optional mechanism."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mechanism: str
    exactness_passed: bool
    measured_utility: float
    evidence_fingerprint: str = Field(min_length=1)
    runtime_fingerprint: str = Field(min_length=1)


class ExecutionPlan(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str
    engine_id: str
    model_fingerprint: str
    execution_identity: str
    objective: Literal["speed", "throughput", "capacity", "balanced"]
    topology: str
    worker_roles: dict[str, str]
    idle_workers: dict[str, str] = Field(default_factory=dict)
    stage_assignments: tuple[dict[str, Any], ...] = ()
    fast_paths: dict[str, str] = Field(default_factory=dict)
    expert_strategy: dict[str, str] = Field(default_factory=dict)
    optional_mechanisms: dict[str, bool] = Field(default_factory=dict)
    mechanism_evidence: tuple[MechanismEvidence, ...] = ()
    engine_parameters: dict[str, Any] = Field(default_factory=dict)
    prefill_plan: PhasePlan
    decode_plan: PhasePlan
    predicted_ttft_ms: float = Field(ge=0)
    predicted_decode_tokens_s: float = Field(ge=0)
    predicted_aggregate_tokens_s: float = Field(ge=0)
    predicted_network_bytes: NonNegativeInt = 0
    predicted_messages_per_token: float = Field(default=0, ge=0)
    predicted_bytes_per_token: float = Field(default=0, ge=0)
    predicted_serial_waits_per_token: float = Field(default=0, ge=0)
    startup_cost_ms: float = Field(default=0, ge=0)
    required_memory_bytes: NonNegativeInt = 0
    score: float
    explanation: tuple[str, ...] = ()


class Deployment(StrictModel):
    deployment_id: str
    engine_id: str
    execution_identity: str
    plan: ExecutionPlan
    ready: bool = False
    endpoints: dict[str, str] = Field(default_factory=dict)
    process_ids: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InferenceRequest(StrictModel):
    request_id: str
    prompt: str
    max_new_tokens: PositiveInt = 16
    seed: NonNegativeInt = 1
    temperature: float = Field(default=0.0, ge=0)


class InferenceEvent(StrictModel):
    event_type: Literal["started", "token", "completed", "failed"]
    request_id: str
    sequence_number: NonNegativeInt
    token_id: int | None = None
    text: str = ""
    detail: str = ""
    telemetry: dict[str, Any] = Field(default_factory=dict)


# Public semantic names used by the hierarchical product planner.  They are
# aliases rather than parallel schemas, keeping one canonical wire contract.
ProductExecutionPlan = ExecutionPlan
PrefillPlan = PhasePlan
DecodePlan = PhasePlan


@runtime_checkable
class ExecutionEngine(Protocol):
    engine_id: str

    def probe(
        self,
        model: ResolvedModelDescriptor,
        cluster: ClusterCapabilities,
    ) -> EngineSupportReport: ...

    async def candidate_plans(
        self,
        model: ResolvedModelDescriptor,
        cluster: ClusterCapabilities,
        request: ExecutionRequest,
    ) -> list[ExecutionPlan]: ...

    async def prepare(self, plan: ExecutionPlan) -> Deployment: ...

    def submit(
        self,
        deployment: Deployment,
        request: InferenceRequest,
    ) -> AsyncIterator[InferenceEvent]: ...

    async def unload(self, deployment: Deployment) -> None: ...


__all__ = [
    "AdapterFastPathCapability",
    "ClusterCapabilities",
    "DecodePlan",
    "Deployment",
    "EngineSupportReport",
    "EngineSupportStatus",
    "ExecutionDevice",
    "ExecutionEngine",
    "ExecutionEngineCapability",
    "ExecutionPlan",
    "ExecutionProfileCapability",
    "ExecutionRequest",
    "InferenceEvent",
    "InferenceRequest",
    "MechanismEvidence",
    "PhasePlan",
    "PrefillPlan",
    "ProductExecutionPlan",
    "WorkerExecutionCapability",
]
