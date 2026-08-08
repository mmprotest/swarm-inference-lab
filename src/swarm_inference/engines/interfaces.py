"""Backend-neutral execution-engine contracts and planning records."""

from __future__ import annotations

from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator

from swarm_inference.config.models import StrictModel
from swarm_inference.engines.topology import NetworkLinkProfile
from swarm_inference.model.descriptor import ResolvedModelDescriptor


class EngineSupportStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    UNSUPPORTED_ARCHITECTURE = "UNSUPPORTED_ARCHITECTURE"
    UNSUPPORTED_QUANTIZATION = "UNSUPPORTED_QUANTIZATION"
    MISSING_RUNTIME = "MISSING_RUNTIME"
    MISSING_DEVICE_CAPABILITY = "MISSING_DEVICE_CAPABILITY"
    INSUFFICIENT_MEMORY = "INSUFFICIENT_MEMORY"
    CONVERSION_AVAILABLE = "CONVERSION_AVAILABLE"
    BROKEN_RUNTIME = "BROKEN_RUNTIME"
    COMPONENT_SUPPORTED = "COMPONENT_SUPPORTED"


class CompatibilityStatus(StrEnum):
    """Evidence-bounded status used by the generated compatibility registry."""

    SUPPORTED = "SUPPORTED"
    SUPPORTED_WITH_LIMITATIONS = "SUPPORTED_WITH_LIMITATIONS"
    SOFTWARE_VALIDATED = "SOFTWARE_VALIDATED"
    REAL_MODEL_VALIDATED = "REAL_MODEL_VALIDATED"
    PHYSICAL_DISTRIBUTED_VALIDATED = "PHYSICAL_DISTRIBUTED_VALIDATED"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    UNSUPPORTED_ARCHITECTURE = "UNSUPPORTED_ARCHITECTURE"
    UNSUPPORTED_QUANTIZATION = "UNSUPPORTED_QUANTIZATION"
    UNAVAILABLE_RUNTIME = "UNAVAILABLE_RUNTIME"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NOT_TESTED = "NOT_TESTED"


class ExecutionComponentType(StrEnum):
    """Canonical whole-model component vocabulary.

    A component describes ownership of model work, not a deployment process.  A
    single persistent worker may own several components and one component may be
    implemented by several logical workers.
    """

    TOKENIZATION = "tokenization"
    EMBEDDING = "embedding"
    ATTENTION = "attention"
    KV_CACHE = "kv-cache"
    ROUTER = "router"
    ROUTED_EXPERTS = "routed-experts"
    SHARED_EXPERTS = "shared-experts"
    DENSE_MLP = "dense-mlp"
    NORMALIZATION = "normalization"
    LM_HEAD = "lm-head"
    SAMPLING = "sampling"
    TOKEN_PUBLICATION = "token-publication"


class ComponentBoundaryContract(StrictModel):
    """Semantically complete contract for one component input or output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    boundary_id: str = Field(min_length=1)
    value_kind: Literal[
        "token-ids",
        "hidden-states",
        "router-logits",
        "expert-routes",
        "kv-state",
        "logits",
        "sampled-token-ids",
        "published-token",
    ]
    shape: tuple[int | str, ...]
    dtype: str = Field(min_length=1)
    device: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    batch_dimension: int | None = Field(default=None, ge=0)
    sequence_dimension: int | None = Field(default=None, ge=0)
    sequence_position: Literal[
        "prompt-relative",
        "absolute-cache-position",
        "current-token",
        "not-applicable",
    ]
    token_position: Literal[
        "prompt-token",
        "decode-token",
        "same-as-input",
        "next-token",
        "not-applicable",
    ]
    kv_identity: str | None = None
    route_identity: str | None = None

    @model_validator(mode="after")
    def dimensions_and_identities_are_explicit(self) -> ComponentBoundaryContract:
        for value in self.shape:
            if isinstance(value, bool) or (isinstance(value, int) and value <= 0):
                raise ValueError("component boundary dimensions must be positive or symbolic")
            if isinstance(value, str) and not value.strip():
                raise ValueError("symbolic component boundary dimensions cannot be empty")
        for axis in (self.batch_dimension, self.sequence_dimension):
            if axis is not None and axis >= len(self.shape):
                raise ValueError("component boundary axis lies outside its shape")
        if self.value_kind == "kv-state" and not self.kv_identity:
            raise ValueError("KV-state boundaries require an explicit KV identity")
        if self.value_kind in {"router-logits", "expert-routes"} and not self.route_identity:
            raise ValueError("routing boundaries require an explicit route identity")
        return self

    def semantic_mismatches(
        self,
        consumer: ComponentBoundaryContract,
        *,
        allow_device_transfer: bool,
    ) -> tuple[str, ...]:
        """Return every semantic mismatch; never silently coerce a boundary."""

        fields = (
            "value_kind",
            "shape",
            "dtype",
            "model_revision",
            "batch_dimension",
            "sequence_dimension",
            "sequence_position",
            "token_position",
            "kv_identity",
            "route_identity",
        )
        mismatches = tuple(
            field for field in fields if getattr(self, field) != getattr(consumer, field)
        )
        if self.device != consumer.device and not allow_device_transfer:
            return (*mismatches, "device")
        return mismatches


class ComponentPlacement(StrictModel):
    """Physical/logical placement and direct-data-path policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    worker_ids: tuple[str, ...] = Field(min_length=1)
    device: str = Field(min_length=1)
    memory_tiers: tuple[Literal["vram", "ram", "nvme"], ...] = ()
    persistent: bool = True
    direct_data_path: bool = True
    bounded_queue_depth: PositiveInt = 1
    colocation_group: str | None = None
    require_same_device: bool = False

    @model_validator(mode="after")
    def workers_are_unique(self) -> ComponentPlacement:
        if len(set(self.worker_ids)) != len(self.worker_ids):
            raise ValueError("component placement contains duplicate logical workers")
        if self.require_same_device and not self.colocation_group:
            raise ValueError("same-device placement requires an explicit colocation group")
        if self.colocation_group is not None and not self.colocation_group.strip():
            raise ValueError("component colocation group cannot be empty")
        return self


class ExecutionComponent(StrictModel):
    """One independently placed unit of a complete model execution graph."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    component_id: str = Field(min_length=1)
    component_type: ExecutionComponentType
    engine_id: str = Field(min_length=1)
    architecture_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    placement: ComponentPlacement
    input_contracts: tuple[ComponentBoundaryContract, ...] = ()
    output_contracts: tuple[ComponentBoundaryContract, ...] = ()
    depends_on: tuple[str, ...] = ()
    estimated_compute_ms: float = Field(default=0.0, ge=0)
    estimated_memory_bytes: NonNegativeInt = 0
    estimated_network_bytes: NonNegativeInt = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def contracts_belong_to_placement(self) -> ExecutionComponent:
        contracts = (*self.input_contracts, *self.output_contracts)
        allowed_devices = {self.placement.device, "cpu"}
        if any(item.device not in allowed_devices for item in contracts):
            raise ValueError("component contract device is not owned by its placement")
        if any(item.model_revision != self.model_revision for item in contracts):
            raise ValueError("component boundary carries a different model revision")
        if self.component_id in self.depends_on:
            raise ValueError("component cannot depend on itself")
        return self


class ComponentDataEdge(StrictModel):
    """Explicit data movement between two component boundary contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_component_id: str = Field(min_length=1)
    source_boundary_id: str = Field(min_length=1)
    target_component_id: str = Field(min_length=1)
    target_boundary_id: str = Field(min_length=1)
    transport: Literal["in-process", "direct-worker", "coordinator-control"]
    device_transfer: bool = False
    bounded_asynchronous: bool = True
    estimated_bytes: NonNegativeInt = 0


class ComponentPlanFragment(StrictModel):
    """Provider offer consumed by the canonical composite planner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fragment_id: str = Field(min_length=1)
    provider_engine_id: str = Field(min_length=1)
    controller_engine_id: str = Field(min_length=1)
    model_fingerprint: str = Field(min_length=1)
    execution_identity: str = Field(min_length=1)
    components: tuple[ExecutionComponent, ...] = Field(min_length=1)
    edges: tuple[ComponentDataEdge, ...] = ()
    worker_roles: dict[str, str]
    required_memory_bytes: NonNegativeInt = 0
    predicted_ttft_ms: float = Field(ge=0)
    predicted_decode_tokens_s: float = Field(ge=0)
    predicted_aggregate_tokens_s: float = Field(ge=0)
    predicted_messages_per_token: float | None = Field(default=None, ge=0)
    predicted_bytes_per_token: float | None = Field(default=None, ge=0)
    predicted_serial_waits_per_token: float | None = Field(default=None, ge=0)
    score: float
    engine_parameters: dict[str, Any] = Field(default_factory=dict)
    optional_mechanisms: dict[str, bool] = Field(default_factory=dict)
    explanation: tuple[str, ...] = ()


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
    quantizations: tuple[str, ...] = ()
    devices: tuple[ExecutionDevice, ...] = ()
    adapters: tuple[str, ...] = ()
    model_architectures: tuple[str, ...] = ()
    required_features: tuple[str, ...] = ()
    unsupported_features: tuple[str, ...] = ()
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
    network_links: dict[str, NetworkLinkProfile] = Field(default_factory=dict)
    resident_model_fingerprints: tuple[str, ...] = ()
    storage_available_bytes: NonNegativeInt = 0

    def engine(self, engine_id: str) -> ExecutionEngineCapability | None:
        return next((item for item in self.engines if item.engine_id == engine_id), None)

    def link_to(self, worker_id: str) -> NetworkLinkProfile:
        profile = self.network_links.get(worker_id)
        if profile is not None:
            return profile
        return NetworkLinkProfile(
            rtt_ms=self.network_latency_ms.get(worker_id),
            bandwidth_bytes_s=self.network_bandwidth_bytes_s.get(worker_id),
            provenance=(
                "legacy-worker-measurements"
                if worker_id in self.network_latency_ms
                or worker_id in self.network_bandwidth_bytes_s
                else "unmeasured"
            ),
        )


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
    compatibility: Literal["supported", "unsupported", "conditionally_supported"] | None = None
    model_architecture: str | None = None
    model_format: str | None = None
    required_runtime: str | None = None
    required_features: tuple[str, ...] = ()
    unsupported_features: tuple[str, ...] = ()
    architecture_supported: bool | None = None
    format_supported: bool | None = None
    quantization_supported: bool | None = None
    hardware_supported: bool | None = None
    capabilities: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    expected_compute_cost: float | None = Field(default=None, ge=0)
    expected_network_cost: float | None = Field(default=None, ge=0)
    expected_memory_cost: NonNegativeInt | None = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    validation_status: CompatibilityStatus = CompatibilityStatus.NOT_TESTED
    support_scope: Literal["complete_model", "hybrid", "component"] = "complete_model"

    @model_validator(mode="after")
    def derive_compatibility(self) -> EngineSupportReport:
        architecture_supported = self.architecture_supported
        format_supported = self.format_supported
        quantization_supported = self.quantization_supported
        hardware_supported = self.hardware_supported
        if architecture_supported is None:
            architecture_supported = self.status != EngineSupportStatus.UNSUPPORTED_ARCHITECTURE
        if format_supported is None:
            format_supported = self.status != EngineSupportStatus.UNSUPPORTED_FORMAT
        if quantization_supported is None:
            quantization_supported = True
        if hardware_supported is None:
            hardware_supported = self.status not in {
                EngineSupportStatus.MISSING_DEVICE_CAPABILITY,
                EngineSupportStatus.INSUFFICIENT_MEMORY,
                EngineSupportStatus.MISSING_RUNTIME,
                EngineSupportStatus.BROKEN_RUNTIME,
            }
        object.__setattr__(self, "architecture_supported", architecture_supported)
        object.__setattr__(self, "format_supported", format_supported)
        object.__setattr__(self, "quantization_supported", quantization_supported)
        object.__setattr__(self, "hardware_supported", hardware_supported)
        if not self.rejection_reasons and self.status != EngineSupportStatus.SUPPORTED:
            object.__setattr__(self, "rejection_reasons", (self.reason,))
        if self.compatibility is not None:
            return self
        value: Literal["supported", "unsupported", "conditionally_supported"]
        if self.status == EngineSupportStatus.SUPPORTED:
            value = "supported"
        elif self.status in {
            EngineSupportStatus.CONVERSION_AVAILABLE,
            EngineSupportStatus.COMPONENT_SUPPORTED,
        }:
            value = "conditionally_supported"
        else:
            value = "unsupported"
        object.__setattr__(self, "compatibility", value)
        return self

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
    components: tuple[ExecutionComponent, ...] = ()
    component_edges: tuple[ComponentDataEdge, ...] = ()
    critical_path: tuple[str, ...] = ()
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
    predicted_network_bytes: NonNegativeInt | None = None
    predicted_messages_per_token: float | None = Field(default=None, ge=0)
    predicted_bytes_per_token: float | None = Field(default=None, ge=0)
    predicted_serial_waits_per_token: float | None = Field(default=None, ge=0)
    number_of_wan_stage_boundaries: NonNegativeInt | None = None
    persistent_connections: bool | None = None
    network_cost_confidence: Literal["measured", "estimated", "unmeasured"] = "unmeasured"
    network_cost_provenance: str = "unmeasured"
    startup_cost_ms: float = Field(default=0, ge=0)
    required_memory_bytes: NonNegativeInt = 0
    score: float
    explanation: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_composite_graph(self) -> ExecutionPlan:
        """Fail closed when a plan advertises a partial or incoherent graph."""

        if not self.components:
            if self.component_edges or self.critical_path:
                raise ValueError("component edges/critical path require execution components")
            return self
        by_id = {item.component_id: item for item in self.components}
        if len(by_id) != len(self.components):
            raise ValueError("composite execution plan contains duplicate component IDs")
        if any(
            item.architecture_id != self.components[0].architecture_id for item in self.components
        ):
            raise ValueError("composite execution components disagree on architecture identity")
        revisions = {item.model_revision for item in self.components}
        if len(revisions) != 1:
            raise ValueError("composite execution components disagree on model revision")
        required = {
            ExecutionComponentType.TOKENIZATION,
            ExecutionComponentType.EMBEDDING,
            ExecutionComponentType.ATTENTION,
            ExecutionComponentType.KV_CACHE,
            ExecutionComponentType.NORMALIZATION,
            ExecutionComponentType.LM_HEAD,
            ExecutionComponentType.SAMPLING,
            ExecutionComponentType.TOKEN_PUBLICATION,
        }
        present = {item.component_type for item in self.components}
        missing = required - present
        if missing:
            values = ", ".join(sorted(item.value for item in missing))
            raise ValueError(f"composite execution plan is incomplete: missing {values}")
        if not present.intersection(
            {ExecutionComponentType.DENSE_MLP, ExecutionComponentType.ROUTED_EXPERTS}
        ):
            raise ValueError("composite execution plan has no dense or routed MLP computation")
        if (
            ExecutionComponentType.ROUTED_EXPERTS in present
            and ExecutionComponentType.ROUTER not in present
        ):
            raise ValueError("routed-expert execution requires a router component")
        for component in self.components:
            unknown = set(component.depends_on) - set(by_id)
            if unknown:
                raise ValueError(
                    f"component {component.component_id!r} has unknown dependencies: "
                    + ", ".join(sorted(unknown))
                )
        if not self.critical_path or set(self.critical_path) != set(by_id):
            raise ValueError("composite critical path must name every component exactly once")
        if len(self.critical_path) != len(set(self.critical_path)):
            raise ValueError("composite critical path contains duplicate components")

        connected_dependencies: set[tuple[str, str]] = set()
        for edge in self.component_edges:
            try:
                source = by_id[edge.source_component_id]
                target = by_id[edge.target_component_id]
            except KeyError as exc:
                raise ValueError("component edge references an unknown component") from exc
            source_contracts = {item.boundary_id: item for item in source.output_contracts}
            target_contracts = {item.boundary_id: item for item in target.input_contracts}
            try:
                producer = source_contracts[edge.source_boundary_id]
                consumer = target_contracts[edge.target_boundary_id]
            except KeyError as exc:
                raise ValueError("component edge references an unknown boundary") from exc
            mismatches = producer.semantic_mismatches(
                consumer,
                allow_device_transfer=edge.device_transfer,
            )
            if mismatches:
                raise ValueError(
                    "component edge silently changes model semantics: " + ", ".join(mismatches)
                )
            source_workers = set(source.placement.worker_ids)
            target_workers = set(target.placement.worker_ids)
            crosses_workers = not source_workers.intersection(target_workers)
            if crosses_workers and edge.transport != "direct-worker":
                raise ValueError("cross-worker model data must use the direct-worker transport")
            if edge.transport == "coordinator-control" and producer.value_kind not in {
                "sampled-token-ids",
                "published-token",
            }:
                raise ValueError("coordinator transport cannot relay model activations")
            connected_dependencies.add((target.component_id, source.component_id))
        for component in self.components:
            for dependency in component.depends_on:
                if (component.component_id, dependency) not in connected_dependencies:
                    raise ValueError("component dependency has no validated data edge")
        return self


class CompositeExecutionPlan(ExecutionPlan):
    """Execution plan whose completeness is proven from composable fragments."""

    @model_validator(mode="after")
    def requires_components(self) -> CompositeExecutionPlan:
        if not self.components:
            raise ValueError("composite execution plans require a component graph")
        if len({item.engine_id for item in self.components}) < 2:
            raise ValueError("composite execution plans require at least two execution engines")
        return self


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

    def probe_model_support(
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


@runtime_checkable
class ExecutionComponentProvider(Protocol):
    """Optional engine capability used by canonical whole-model composition."""

    engine_id: str

    async def candidate_components(
        self,
        model: ResolvedModelDescriptor,
        cluster: ClusterCapabilities,
        request: ExecutionRequest,
    ) -> list[ComponentPlanFragment]: ...


__all__ = [
    "AdapterFastPathCapability",
    "ClusterCapabilities",
    "CompatibilityStatus",
    "ComponentBoundaryContract",
    "ComponentDataEdge",
    "ComponentPlacement",
    "ComponentPlanFragment",
    "CompositeExecutionPlan",
    "DecodePlan",
    "Deployment",
    "EngineSupportReport",
    "EngineSupportStatus",
    "ExecutionComponent",
    "ExecutionComponentProvider",
    "ExecutionComponentType",
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
