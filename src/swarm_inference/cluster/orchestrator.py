"""One-shot product orchestration over the canonical coordinator APIs."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import Counter
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import Field, NonNegativeInt, PositiveInt

from swarm_inference.cluster.artifacts import (
    ArtifactManager,
    ModelArtifactBuilder,
    StageArtifactBuilder,
)
from swarm_inference.cluster.models import ArtifactManifest, node_id_from_fingerprint
from swarm_inference.cluster.pairing import create_cluster_authentication
from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.config.models import StrictModel
from swarm_inference.coordinator.canonical_planner import (
    CanonicalPlanner,
    CanonicalPlanningDecision,
)
from swarm_inference.coordinator.service import CoordinatorClient
from swarm_inference.engines.capabilities import cluster_execution_capabilities
from swarm_inference.engines.interfaces import (
    ClusterCapabilities,
    Deployment,
    EngineSupportReport,
    ExecutionPlan,
    ExecutionRequest,
    InferenceEvent,
    InferenceRequest,
)
from swarm_inference.engines.registry import (
    ExecutionEngineRegistry,
    default_engine_registry,
)
from swarm_inference.engines.worker_control import CoordinatorAuthorizedEngineLifecycle
from swarm_inference.exceptions import IntegrityError
from swarm_inference.model.adapter import (
    NativeModelAdapterRegistry,
    default_native_adapter_registry,
)
from swarm_inference.model.product import ModelResolutionPolicy, ProductModelReference
from swarm_inference.model.resolver import (
    ModelResolution,
    ModelSourceResolver,
    ResolutionResources,
)
from swarm_inference.protocol.cluster import ClusterStatusRequest
from swarm_inference.protocol.messages import (
    StreamEventType,
    SubmitRequest,
    SubmitStreamEvent,
)
from swarm_inference.protocol.product import (
    ModelDeployRequest,
    ModelPlanRequest,
    ProductStagePlan,
    WorkersRequest,
)
from swarm_inference.runtime.telemetry import (
    InferenceTelemetryRecord,
    ProductTelemetry,
    build_inference_telemetry_record,
    execution_runtime_revisions,
)
from swarm_inference.security.identity import WorkerIdentity
from swarm_inference.transport.grpc_transport import GrpcTransport

_IMMUTABLE_REVISION = re.compile(r"^[0-9a-fA-F]{40}([0-9a-fA-F]{24})?$")
_TOKENIZER_HASH = re.compile(r"^sha256:[0-9a-fA-F]{64}$")


class RunProgress(StrictModel):
    schema_version: Literal[1] = 1
    event: str
    stage: str
    timestamp_unix_ns: PositiveInt
    detail: str
    node_id: str | None = None
    category: str | None = None


class ClusterRunSummary(StrictModel):
    schema_version: Literal[1] = 1
    document_version: Literal[1] = 1
    run_id: str
    status: Literal["dry-run", "completed", "failed", "cancelled"]
    model_id: str
    model_revision: str
    tokenizer_revision: str
    mode: Literal["speed", "throughput", "capacity", "balanced"]
    plan: ProductStagePlan | ExecutionPlan
    model_fingerprint: str = ""
    model_format: str = "unknown"
    variant: str | None = None
    quantization: str | None = None
    engine_id: str = "native-stage"
    engine_revision: str | None = None
    engine_runtime_revisions: dict[str, str] = Field(default_factory=dict)
    execution_identity: str = ""
    engine_support: tuple[EngineSupportReport, ...] = ()
    canonical_decision: CanonicalPlanningDecision | None = None
    artifact_ids: list[str] = Field(default_factory=list)
    deployment_id: str | None = None
    topology_id: str | None = None
    output_token_ids: list[NonNegativeInt] = Field(default_factory=list)
    decoded_text: str = ""
    event_count: NonNegativeInt = 0
    started_at_unix_ns: PositiveInt
    completed_at_unix_ns: PositiveInt
    elapsed_seconds: float = Field(ge=0)
    detail: str = ""
    telemetry: InferenceTelemetryRecord | None = None


class SourceResolver(Protocol):
    async def __call__(
        self,
        model_id: str,
        model_revision: str,
        tokenizer_revision: str,
        cache_directory: Path,
        maximum_bytes: int,
    ) -> Path: ...


class WorkerEngineLifecycle(Protocol):
    async def prepare(self, plan: ExecutionPlan) -> Deployment: ...

    def submit(
        self,
        deployment: Deployment,
        request: InferenceRequest,
    ) -> AsyncIterator[InferenceEvent]: ...

    async def unload(self, deployment: Deployment) -> None: ...


ProgressSink = Callable[[RunProgress], None]
StreamSink = Callable[[SubmitStreamEvent], None]


def validate_immutable_reference(model_revision: str, tokenizer_revision: str) -> None:
    if not (
        _IMMUTABLE_REVISION.fullmatch(model_revision)
        or _TOKENIZER_HASH.fullmatch(model_revision)
    ):
        raise ValueError(
            "model revision must be an immutable commit or sha256:<digest> identity"
        )
    if not (
        _IMMUTABLE_REVISION.fullmatch(tokenizer_revision)
        or _TOKENIZER_HASH.fullmatch(tokenizer_revision)
    ):
        raise ValueError("tokenizer revision must be an immutable commit or sha256:<digest>")


async def resolve_upstream_source(
    model_id: str,
    model_revision: str,
    tokenizer_revision: str,
    cache_directory: Path,
    maximum_bytes: int,
) -> Path:
    """Backward-compatible selective acquisition for one immutable native source."""

    if maximum_bytes <= 0:
        raise ValueError("source download byte bound must be positive")
    resolver = ModelSourceResolver(cache_directory=cache_directory)
    resolution = await asyncio.to_thread(
        resolver.inspect,
        model_id,
        revision=model_revision,
        resources=ResolutionResources(aggregate_usable_memory_bytes=maximum_bytes),
    )
    descriptor = resolution.descriptor
    if descriptor.revision.lower() != model_revision.lower():
        raise RuntimeError("model source resolved to a different immutable revision")
    if sum(item.size_bytes for item in descriptor.files) > maximum_bytes:
        raise OSError(
            f"selected immutable source exceeds the {maximum_bytes}-byte acquisition bound"
        )
    if _IMMUTABLE_REVISION.fullmatch(tokenizer_revision) and (
        descriptor.source_type == "huggingface"
        and tokenizer_revision.lower() != model_revision.lower()
    ):
        raise RuntimeError(
            "a tokenizer commit differing from the model commit requires a pre-merged "
            "verified local source; use a tokenizer sha256 identity for upstream runs"
        )
    paths = await resolver.acquire_async(descriptor)
    if descriptor.source_type == "local":
        local = Path(model_id).expanduser().resolve()
        return local if local.is_dir() else local.parent
    common = Path(os.path.commonpath([str(item) for item in paths])).resolve()
    source = common if common.is_dir() else common.parent
    while source.name.lower() != descriptor.revision.lower() and source.parent != source:
        source = source.parent
    if source.name.lower() != descriptor.revision.lower():
        raise RuntimeError("selectively acquired files do not share the immutable snapshot root")
    return source


class ClusterOrchestrator:
    """Validate, plan, artifact, deploy, and stream without a second runtime."""

    def __init__(
        self,
        *,
        state: ClusterStateStore,
        client_factory: Callable[[str], CoordinatorClient] = CoordinatorClient,
        source_resolver: SourceResolver | None = None,
        model_source_resolver: ModelSourceResolver | None = None,
        adapter_registry: NativeModelAdapterRegistry | None = None,
        engine_registry: ExecutionEngineRegistry | None = None,
        canonical_planner: CanonicalPlanner | None = None,
        worker_engine_lifecycle_factory: Callable[..., WorkerEngineLifecycle] | None = None,
        worker_transport_factory: Callable[[], GrpcTransport] = GrpcTransport,
        product_telemetry: ProductTelemetry | None = None,
        progress_sink: ProgressSink | None = None,
        stream_sink: StreamSink | None = None,
        maximum_source_bytes: int = 100 * 1024**3,
        source_timeout_seconds: float = 3600.0,
        network_measurement_ttl_seconds: int = 900,
        network_refresh_wait_seconds: float = 35.0,
        maximum_engine_recovery_attempts: int = 1,
    ) -> None:
        if maximum_source_bytes <= 0:
            raise ValueError("source byte bound must be positive")
        if not 0 < source_timeout_seconds <= 7200:
            raise ValueError("source timeout must be in (0, 7200] seconds")
        if network_measurement_ttl_seconds <= 0:
            raise ValueError("network measurement TTL must be positive")
        if not 0 <= network_refresh_wait_seconds <= 120:
            raise ValueError("network refresh wait must be in [0, 120] seconds")
        if not 0 <= maximum_engine_recovery_attempts <= 10:
            raise ValueError("engine recovery attempts must be in [0, 10]")
        self.state = state
        self.client_factory = client_factory
        self.source_resolver = source_resolver
        self.model_source_resolver = model_source_resolver or ModelSourceResolver(
            cache_directory=self.state.paths.artifacts / "source-cache"
        )
        self.adapter_registry = adapter_registry or default_native_adapter_registry()
        self.engine_registry = engine_registry or default_engine_registry()
        self.canonical_planner = canonical_planner or CanonicalPlanner(self.engine_registry)
        self.worker_engine_lifecycle_factory = (
            worker_engine_lifecycle_factory or CoordinatorAuthorizedEngineLifecycle
        )
        self.worker_transport_factory = worker_transport_factory
        self.product_telemetry = product_telemetry or ProductTelemetry(
            self.state.paths.logs / "product-events.jsonl"
        )
        self.progress_sink = progress_sink
        self.stream_sink = stream_sink
        self.maximum_source_bytes = maximum_source_bytes
        self.source_timeout_seconds = source_timeout_seconds
        self.network_measurement_ttl_seconds = network_measurement_ttl_seconds
        self.network_refresh_wait_seconds = network_refresh_wait_seconds
        self.maximum_engine_recovery_attempts = maximum_engine_recovery_attempts

    def _progress(self, event: str, stage: str, detail: str) -> None:
        if self.progress_sink is not None:
            self.progress_sink(
                RunProgress(
                    event=event,
                    stage=stage,
                    timestamp_unix_ns=time.time_ns(),
                    detail=detail,
                )
            )

    @staticmethod
    def _select_dtype(workers: object) -> str:
        values = getattr(workers, "workers", [])
        healthy = [item for item in values if item.healthy_registration]
        if not healthy:
            raise RuntimeError("no healthy product workers are registered")
        counts: Counter[str] = Counter()
        for item in healthy:
            counts.update(set(item.capability.supported_activation_dtypes))
        priority = {"bfloat16": 3, "float16": 2, "float32": 1}
        candidates = [item for item in counts if item in priority]
        if not candidates:
            raise RuntimeError("healthy workers share no benchmark-approved product dtype")
        return max(candidates, key=lambda item: (counts[item], priority[item], item))

    async def _wait_for_fresh_links(self, client: CoordinatorClient, workers: object) -> None:
        registrations = [
            item for item in getattr(workers, "workers", []) if item.healthy_registration
        ]
        worker_ids = sorted(item.capability.worker_id for item in registrations)
        if len(worker_ids) < 2 or self.network_refresh_wait_seconds == 0:
            return
        expected = {
            (source, destination)
            for source in worker_ids
            for destination in worker_ids
            if source != destination
        }
        cluster = self.state.load_cluster()
        if cluster is None:
            raise RuntimeError("node is not paired with a cluster")
        identity = self.state.load_or_create_node_identity()
        node_id = node_id_from_fingerprint(identity.public_key_fingerprint)
        deadline = time.monotonic() + self.network_refresh_wait_seconds
        announced = False
        while True:
            body = {"include_artifacts": False, "include_network": True}
            status = await client.cluster_status(
                ClusterStatusRequest(
                    authentication=create_cluster_authentication(
                        identity=identity,
                        node_id=node_id,
                        action="cluster-status",
                        body=body,
                    ),
                    include_artifacts=False,
                    include_network=True,
                )
            )
            cutoff = time.time_ns() - self.network_measurement_ttl_seconds * 1_000_000_000
            fresh = {
                (item.source_worker_id, item.destination_worker_id)
                for item in status.network_links
                if item.measured
                and item.authentication_verified
                and item.measured_at_unix_ns >= cutoff
            }
            missing = expected - fresh
            if not missing:
                self._progress(
                    "network-evidence-ready",
                    "network-measurement",
                    f"verified {len(expected)} fresh directed links",
                )
                return
            if time.monotonic() >= deadline:
                self._progress(
                    "network-evidence-incomplete",
                    "network-measurement",
                    f"{len(missing)} directed links remain stale or unmeasured; planner will exclude unsupported paths",
                )
                return
            if not announced:
                self._progress(
                    "network-refresh-wait",
                    "network-measurement",
                    f"waiting for node agents to refresh {len(missing)} directed links",
                )
                announced = True
            await asyncio.sleep(1.0)

    async def _execution_capabilities(
        self,
        client: CoordinatorClient,
        workers: object,
    ) -> ClusterCapabilities:
        """Translate the authenticated worker catalog into engine planning facts."""

        registrations = [
            item for item in getattr(workers, "workers", []) if item.healthy_registration
        ]
        body = {"include_artifacts": False, "include_network": True}
        cluster = self.state.load_cluster()
        if cluster is None:
            raise RuntimeError("node is not paired with a cluster")
        identity = self.state.load_or_create_node_identity()
        node_id = node_id_from_fingerprint(identity.public_key_fingerprint)
        status = await client.cluster_status(
            ClusterStatusRequest(
                authentication=create_cluster_authentication(
                    identity=identity,
                    node_id=node_id,
                    action="cluster-status",
                    body=body,
                ),
                include_artifacts=False,
                include_network=True,
            )
        )
        latency: dict[str, dict[str, float]] = {}
        bandwidth: dict[str, dict[str, float]] = {}
        for link in status.network_links:
            if not (link.measured and link.authentication_verified):
                continue
            latency.setdefault(link.source_worker_id, {})[
                link.destination_worker_id
            ] = float(
                link.one_way_estimate_ms
                if link.one_way_estimate_ms is not None
                else link.round_trip_latency_ms / 2
            )
            bandwidth.setdefault(link.source_worker_id, {})[
                link.destination_worker_id
            ] = float(link.upload_bytes_per_s)
        return cluster_execution_capabilities(
            [item.capability for item in registrations],
            latency_by_worker=latency,
            bandwidth_by_worker=bandwidth,
        )

    async def _inspect_resolution(
        self,
        *,
        model_id: str,
        model_revision: str | None,
        tokenizer_revision: str | None,
        variant: str | None,
        quantization: str | None,
        mode: Literal["speed", "throughput", "capacity", "balanced"],
        aggregate_usable_memory_bytes: int,
        local_fast_memory_bytes: int,
    ) -> tuple[ModelResolution, str, Path | None]:
        """Resolve immutable facts without acquiring selected weights."""

        resources = ResolutionResources(
            aggregate_usable_memory_bytes=max(1, aggregate_usable_memory_bytes),
            local_fast_memory_bytes=max(0, local_fast_memory_bytes),
        )
        acquired_source: Path | None = None
        if self.source_resolver is not None:
            if model_revision is None or tokenizer_revision is None:
                raise ValueError(
                    "a custom legacy source resolver requires explicit immutable model and "
                    "tokenizer revisions"
                )
            validate_immutable_reference(model_revision, tokenizer_revision)
            acquired_source = await asyncio.wait_for(
                self.source_resolver(
                    model_id,
                    model_revision,
                    tokenizer_revision,
                    self.state.paths.artifacts / "source-cache",
                    self.maximum_source_bytes,
                ),
                timeout=self.source_timeout_seconds,
            )
            resolution = await asyncio.to_thread(
                self.model_source_resolver.inspect,
                acquired_source,
                revision=model_revision,
                variant=variant,
                quantization=quantization,
                objective=mode,
                resources=resources,
            )
            descriptor = resolution.descriptor.model_copy(
                update={"model_id": model_id, "revision": model_revision}
            )
            resolution = ModelResolution(
                descriptor=descriptor,
                variants=resolution.variants,
                variant_candidates=resolution.variant_candidates,
                repository_files=resolution.repository_files,
            )
            return resolution, tokenizer_revision, acquired_source

        resolution = await asyncio.wait_for(
            asyncio.to_thread(
                self.model_source_resolver.inspect,
                model_id,
                revision=model_revision,
                variant=variant,
                quantization=quantization,
                objective=mode,
                resources=resources,
            ),
            timeout=self.source_timeout_seconds,
        )
        descriptor = resolution.descriptor
        selected_tokenizer = tokenizer_revision or descriptor.tokenizer_identity or descriptor.revision
        validate_immutable_reference(descriptor.revision, selected_tokenizer)
        selected_bytes = sum(item.size_bytes for item in descriptor.files)
        if selected_bytes > self.maximum_source_bytes:
            raise OSError(
                f"selected immutable model files are {selected_bytes} bytes, above the "
                f"{self.maximum_source_bytes}-byte bound"
            )
        return resolution, selected_tokenizer, None

    async def _acquire_resolution(
        self,
        resolution: ModelResolution,
        *,
        already_acquired: Path | None,
    ) -> tuple[Path, ModelResolution]:
        if already_acquired is not None:
            return already_acquired, resolution
        paths = await asyncio.wait_for(
            self.model_source_resolver.acquire_async(resolution.descriptor),
            timeout=self.source_timeout_seconds,
        )
        descriptor = resolution.descriptor.model_copy(
            update={"local_paths": tuple(str(item.resolve()) for item in paths)}
        )
        acquired = ModelResolution(
            descriptor=descriptor,
            variants=resolution.variants,
            variant_candidates=resolution.variant_candidates,
            repository_files=resolution.repository_files,
        )
        local = Path(descriptor.model_id).expanduser()
        if descriptor.source_type == "local":
            resolved_local = local.resolve()
            source = resolved_local if resolved_local.is_dir() else resolved_local.parent
        else:
            source = self._selected_source_root(paths, descriptor.revision)
        return source, acquired

    async def _build_artifacts(
        self,
        source: Path,
        plan: ProductStagePlan,
    ) -> tuple[ProductStagePlan, list[ArtifactManifest]]:
        configuration = self.state.load_node_configuration()
        storage_limit = configuration.storage_limit_bytes if configuration else 100 * 1024**3
        manager = ArtifactManager(
            state=self.state,
            node_id=plan.assignments[0].worker_id.split("/", 1)[0],
            storage_limit_bytes=storage_limit,
        )
        builder = StageArtifactBuilder(
            artifact_root=self.state.paths.artifacts,
            temporary_root=self.state.paths.downloads,
        )
        manifests: list[ArtifactManifest] = []
        assignments = []
        for item in plan.assignments:
            self._progress(
                "artifact-preparing",
                "preparing-artifacts",
                f"building stage {item.stage_id} owned tensors",
            )
            manifest = await asyncio.to_thread(
                builder.build,
                source,
                model_id=plan.model.model_id,
                model_revision=plan.model.model_revision,
                tokenizer_revision=plan.model.tokenizer_revision,
                assignment=item.assignment,
                stage_count=plan.stage_count,
                dtype=plan.model.dtype,
                quantization=plan.model.quantization or "none",
                model_fingerprint=plan.model.model_fingerprint or None,
                adapter_id=plan.model.adapter_id,
                before_publish=lambda manifest: manager.evict_to_fit(manifest.total_size_bytes),
            )
            manager.register(manifest.artifact_id)
            manifests.append(manifest)
            assignments.append(
                item.model_copy(
                    update={
                        "artifact_id": manifest.artifact_id,
                        "artifact_manifest": manifest,
                    },
                    deep=True,
                )
            )
        return plan.model_copy(update={"assignments": assignments}, deep=True), manifests

    async def _build_model_artifact(
        self,
        descriptor: object,
        *,
        engine_id: str,
        node_id: str,
    ) -> tuple[ArtifactManager, ArtifactManifest]:
        configuration = self.state.load_node_configuration()
        storage_limit = configuration.storage_limit_bytes if configuration else 100 * 1024**3
        manager = ArtifactManager(
            state=self.state,
            node_id=node_id,
            storage_limit_bytes=storage_limit,
        )
        builder = ModelArtifactBuilder(
            artifact_root=self.state.paths.artifacts,
            temporary_root=self.state.paths.downloads,
        )
        manifest = await asyncio.to_thread(
            builder.build,
            descriptor,
            engine_id=engine_id,
            before_publish=lambda item: manager.evict_to_fit(item.total_size_bytes or 0),
        )
        manager.register(manifest.artifact_id)
        return manager, manifest

    @staticmethod
    def _worker_control_endpoints(workers: object) -> dict[str, str]:
        endpoints: dict[str, str] = {}
        for item in getattr(workers, "workers", []):
            if not item.healthy_registration:
                continue
            capability = item.capability
            endpoint = (
                getattr(item, "control_endpoint", None)
                or getattr(capability, "control_endpoint", None)
                or getattr(capability, "endpoint", None)
            )
            if endpoint:
                endpoints[capability.worker_id] = endpoint
        return endpoints

    @staticmethod
    def _selected_source_root(paths: tuple[Path, ...], revision: str) -> Path:
        if not paths:
            raise FileNotFoundError("model acquisition returned no files")
        common = Path(os.path.commonpath([str(item) for item in paths])).resolve()
        source = common if common.is_dir() else common.parent
        while source.name.lower() != revision.lower() and source.parent != source:
            source = source.parent
        if source.name.lower() == revision.lower():
            return source
        config = next((item for item in paths if item.name == "config.json"), None)
        if config is not None:
            return config.parent
        raise RuntimeError("acquired native files do not expose a checkpoint root")

    async def run(
        self,
        *,
        model_id: str,
        prompt: str,
        model_revision: str | None = None,
        tokenizer_revision: str | None = None,
        variant: str | None = None,
        quantization: str | None = None,
        requested_engine: str | None = None,
        require_distributed: bool = False,
        concurrency: int = 1,
        max_context_tokens: int = 2048,
        mode: Literal["speed", "throughput", "capacity", "balanced"] = "speed",
        dry_run: bool = False,
        required_node_ids: list[str] | None = None,
        excluded_node_ids: list[str] | None = None,
        max_new_tokens: int = 16,
        seed: int = 1,
    ) -> ClusterRunSummary:
        if model_revision is not None:
            validate_immutable_reference(
                model_revision,
                tokenizer_revision or model_revision,
            )
        elif tokenizer_revision is not None and not (
            _IMMUTABLE_REVISION.fullmatch(tokenizer_revision)
            or _TOKENIZER_HASH.fullmatch(tokenizer_revision)
        ):
            raise ValueError("tokenizer revision must be an immutable commit or sha256:<digest>")
        if not prompt:
            raise ValueError("run prompt cannot be empty")
        if max_new_tokens <= 0:
            raise ValueError("maximum new tokens must be positive")
        if concurrency <= 0 or max_context_tokens <= 0:
            raise ValueError("concurrency and context capacity must be positive")
        cluster = self.state.load_cluster()
        if cluster is None:
            raise RuntimeError("node is not paired with a cluster")
        run_id = f"run-{uuid4().hex}"
        started_ns = time.time_ns()
        started_monotonic = time.monotonic()
        client = self.client_factory(cluster.coordinator_endpoint)
        worker_transport: GrpcTransport | None = None
        try:
            self._progress("run-started", "validation", "validating immutable model identity")
            workers = await client.workers(WorkersRequest(include_unhealthy=False))
            dtype = self._select_dtype(workers)
            self._progress("backend-selected", "capability-refresh", f"selected dtype {dtype}")
            await self._wait_for_fresh_links(client, workers)
            cluster_capabilities = await self._execution_capabilities(client, workers)
            local_fast_memory = max(
                (
                    device.usable_memory_bytes
                    for worker in cluster_capabilities.workers
                    for engine_capability in worker.engines
                    if engine_capability.enabled
                    for device in engine_capability.devices
                    if device.device_type in {"cuda", "mps", "metal", "rocm", "vulkan"}
                ),
                default=0,
            )
            resolution, selected_tokenizer, acquired_source = await self._inspect_resolution(
                model_id=model_id,
                model_revision=model_revision,
                tokenizer_revision=tokenizer_revision,
                variant=variant,
                quantization=quantization,
                mode=mode,
                aggregate_usable_memory_bytes=(
                    cluster_capabilities.aggregate_usable_memory_bytes
                ),
                local_fast_memory_bytes=local_fast_memory,
            )
            descriptor = resolution.descriptor
            model_id = descriptor.model_id
            model_revision = descriptor.revision
            tokenizer_revision = selected_tokenizer
            self._progress(
                "model-resolved",
                "resolution",
                f"pinned {model_id}@{model_revision}; variant={descriptor.variant or 'native'}; "
                f"quantization={descriptor.quantization or 'none'}",
            )
            planning_request = ExecutionRequest(
                objective=mode,
                require_distributed=require_distributed,
                concurrency=concurrency,
                max_context_tokens=max_context_tokens,
                max_new_tokens=max_new_tokens,
                requested_engine=requested_engine,
                requested_nodes=tuple(sorted(set(required_node_ids or []))),
                excluded_nodes=tuple(sorted(set(excluded_node_ids or []))),
            )
            self._progress("plan-started", "planning", f"planning {mode} objective")
            decision = await self.canonical_planner.plan(
                descriptor,
                cluster_capabilities,
                planning_request,
            )
            selected = decision.selected
            runtime_revisions = execution_runtime_revisions(
                cluster_capabilities,
                selected,
            )
            unique_runtime_revisions = sorted(set(runtime_revisions.values()))
            engine_revision = (
                unique_runtime_revisions[0]
                if len(unique_runtime_revisions) == 1
                else None
            )
            self._progress(
                "plan-completed",
                "planning",
                f"selected engine={selected.engine_id} topology={selected.topology}",
            )

            source: Path | None = acquired_source
            plan: ProductStagePlan | ExecutionPlan = selected
            manifests: list[ArtifactManifest] = []
            if selected.engine_id == "native-stage" and (
                not dry_run or descriptor.source_type == "local"
            ):
                source, resolution = await self._acquire_resolution(
                    resolution,
                    already_acquired=acquired_source,
                )
                descriptor = resolution.descriptor
                config = json.loads((source / "config.json").read_text(encoding="utf-8"))
                adapter = self.adapter_registry.resolve_config(config)
                selected_nodes = {
                    worker.node_id
                    for worker in cluster_capabilities.workers
                    if worker.worker_id in selected.worker_roles
                    and selected.worker_roles[worker.worker_id]
                    not in {"idle", "background_replica", "storage_cache", "verification"}
                }
                selected_stage_count = max(1, len(selected.stage_assignments))
                reference = ProductModelReference(
                    model_id=model_id,
                    model_revision=model_revision,
                    tokenizer_revision=tokenizer_revision,
                    adapter_id=adapter.adapter_id,
                    dtype=dtype,
                    model_fingerprint=descriptor.content_fingerprint,
                    model_format=descriptor.format,
                    quantization=descriptor.quantization,
                    resolution_policy=ModelResolutionPolicy.LOCAL_ONLY,
                )
                planned = await client.plan_model(
                    ModelPlanRequest(
                        reference=reference,
                        stage_count=selected_stage_count,
                        mode=mode,
                        require_distributed=require_distributed,
                        required_node_ids=sorted(
                            set(required_node_ids or []) | selected_nodes
                        ),
                        excluded_node_ids=sorted(set(excluded_node_ids or [])),
                        allow_artifact_provisioning=True,
                        max_sequence_tokens=max_context_tokens,
                    )
                )
                plan = planned.plan.model_copy(
                    update={
                        "engine_id": selected.engine_id,
                        "engine_revision": engine_revision,
                        "optional_mechanisms": selected.optional_mechanisms,
                        "prefill_parameters": selected.prefill_plan.parameters,
                        "decode_parameters": selected.decode_plan.parameters,
                        "idle_workers": selected.idle_workers,
                    }
                )

            if dry_run:
                completed = time.time_ns()
                return ClusterRunSummary(
                    run_id=run_id,
                    status="dry-run",
                    model_id=model_id,
                    model_revision=model_revision,
                    tokenizer_revision=tokenizer_revision,
                    model_fingerprint=descriptor.content_fingerprint,
                    model_format=descriptor.format,
                    variant=descriptor.variant,
                    quantization=descriptor.quantization,
                    engine_id=selected.engine_id,
                    engine_revision=engine_revision,
                    engine_runtime_revisions=runtime_revisions,
                    execution_identity=selected.execution_identity,
                    engine_support=decision.engine_support,
                    canonical_decision=decision,
                    mode=mode,
                    plan=plan,
                    started_at_unix_ns=started_ns,
                    completed_at_unix_ns=completed,
                    elapsed_seconds=time.monotonic() - started_monotonic,
                    detail="resolved immutable metadata and completed planning without acquisition",
                )

            request_id: str
            request_started_monotonic_s: float
            token_monotonic_s: list[float] = []
            terminal_engine_metrics: dict[str, object] = {}
            terminal_timing_metrics: dict[str, float] = {}
            per_token_expert_metrics: list[dict[str, object]] = []
            recoveries = 0
            if selected.engine_id == "native-stage":
                assert source is not None and isinstance(plan, ProductStagePlan)
                plan, manifests = await self._build_artifacts(source, plan)
                self._progress(
                    "deployment-started",
                    "deployment",
                    "entering canonical transactional stage deployment",
                )
                deployed = await client.deploy_model(ModelDeployRequest(plan=plan))
                if not deployed.deployment.ready:
                    raise RuntimeError(
                        deployed.deployment.detail or "deployment did not become ready"
                    )
                self._progress(
                    "deployment-ready",
                    "deployment",
                    f"topology {plan.topology_id} is ready",
                )
                request = SubmitRequest(
                    request_id=f"request-{uuid4().hex}",
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    random_seed=seed,
                    model_id=model_id,
                    model_revision=model_revision,
                )
                output_tokens: list[int] = []
                decoded: list[str] = []
                events = 0
                terminal: SubmitStreamEvent | None = None
                request_id = request.request_id
                request_started_monotonic_s = time.monotonic()
                async for event in client.submit_stream(request):
                    events += 1
                    if self.stream_sink is not None:
                        self.stream_sink(event)
                    if event.event_type == StreamEventType.TOKEN_GENERATED:
                        assert event.token_id is not None
                        output_tokens.append(event.token_id)
                        decoded.append(event.decoded_text_fragment)
                        token_monotonic_s.append(time.monotonic())
                        per_token_expert_metrics.append(dict(event.expert_metrics))
                    if event.event_type == StreamEventType.RECOVERY_COMPLETED:
                        recoveries += 1
                    if event.event_type in {
                        StreamEventType.REQUEST_COMPLETED,
                        StreamEventType.REQUEST_FAILED,
                        StreamEventType.REQUEST_CANCELLED,
                    }:
                        terminal = event
                if terminal is None:
                    raise RuntimeError("submission stream ended without a terminal event")
                status: Literal["completed", "failed", "cancelled"]
                if terminal.event_type == StreamEventType.REQUEST_COMPLETED:
                    status = "completed"
                elif terminal.event_type == StreamEventType.REQUEST_CANCELLED:
                    status = "cancelled"
                else:
                    status = "failed"
                deployment_id = deployed.deployment.deployment_id
                topology_id = plan.topology_id
                detail = terminal.status_detail
                terminal_timing_metrics = dict(terminal.timing_metrics)
            else:
                source, resolution = await self._acquire_resolution(
                    resolution,
                    already_acquired=acquired_source,
                )
                descriptor = resolution.descriptor
                engine = self.engine_registry.get(selected.engine_id)
                binder = getattr(engine, "bind_acquired_model", None)
                if callable(binder):
                    binder(descriptor, tuple(Path(item) for item in descriptor.local_paths))
                acquired_decision = await self.canonical_planner.plan(
                    descriptor,
                    cluster_capabilities,
                    planning_request,
                )
                if acquired_decision.selected.execution_identity != selected.execution_identity:
                    raise RuntimeError("model acquisition changed immutable execution identity")
                if acquired_decision.selected.engine_id != selected.engine_id:
                    raise RuntimeError("model acquisition changed the selected execution engine")
                decision = acquired_decision
                selected = decision.selected
                selected = selected.model_copy(
                    update={
                        "engine_parameters": {
                            **selected.engine_parameters,
                            # Host paths are acquisition facts, not portable worker inputs.
                            # The verified artifact ID is resolved independently on each worker.
                            "model_paths": [],
                        }
                    }
                )
                plan = selected
                identity: WorkerIdentity = self.state.load_or_create_node_identity()
                local_node_id = node_id_from_fingerprint(identity.public_key_fingerprint)
                self._progress(
                    "artifact-preparing",
                    "preparing-artifacts",
                    f"publishing selected {descriptor.format} files as one immutable artifact",
                )
                artifact_manager, artifact_manifest = await self._build_model_artifact(
                    descriptor,
                    engine_id=selected.engine_id,
                    node_id=local_node_id,
                )
                manifests = [artifact_manifest]
                worker_transport = self.worker_transport_factory()
                lifecycle = self.worker_engine_lifecycle_factory(
                    coordinator=client,
                    transport=worker_transport,
                    identity=identity,
                    node_id=local_node_id,
                    worker_endpoints=self._worker_control_endpoints(workers),
                    artifact_manager=artifact_manager,
                    artifact_manifest=artifact_manifest,
                )
                deployment = await lifecycle.prepare(selected)
                if (
                    not deployment.ready
                    or deployment.engine_id != selected.engine_id
                    or deployment.execution_identity != selected.execution_identity
                ):
                    raise IntegrityError(
                        "execution engine prepared a deployment with an invalid identity"
                    )
                self._progress(
                    "deployment-ready",
                    "deployment",
                    f"engine {selected.engine_id} topology {selected.topology} is ready",
                )
                inference_request = InferenceRequest(
                    request_id=f"request-{uuid4().hex}",
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                )
                output_tokens = []
                decoded = []
                events = 0
                status = "failed"
                detail = "engine stream ended without a terminal event"
                request_id = inference_request.request_id
                request_started_monotonic_s = time.monotonic()
                try:
                    while True:
                        replay_prefix = tuple(output_tokens)
                        replay_position = 0
                        terminal_completed = False
                        try:
                            async for event in lifecycle.submit(
                                deployment,
                                inference_request,
                            ):
                                events += 1
                                if event.telemetry:
                                    terminal_engine_metrics.update(event.telemetry)
                                if event.event_type == "token" and event.token_id is not None:
                                    if replay_position < len(replay_prefix):
                                        expected = replay_prefix[replay_position]
                                        if event.token_id != expected:
                                            raise IntegrityError(
                                                "restart-and-replay token divergence at "
                                                f"position {replay_position}: expected {expected}, "
                                                f"received {event.token_id}"
                                            )
                                        replay_position += 1
                                        continue
                                    output_tokens.append(event.token_id)
                                    decoded.append(event.text)
                                    token_monotonic_s.append(time.monotonic())
                                    if self.stream_sink is not None:
                                        self.stream_sink(
                                            SubmitStreamEvent(
                                                event_type=StreamEventType.TOKEN_GENERATED,
                                                request_id=inference_request.request_id,
                                                sequence_number=event.sequence_number,
                                                monotonic_timestamp_ns=time.monotonic_ns(),
                                                topology_id=selected.topology,
                                                model_revision=model_revision,
                                                token_position=len(output_tokens) - 1,
                                                token_id=event.token_id,
                                                decoded_text_fragment=event.text,
                                            )
                                        )
                                elif event.event_type == "completed":
                                    if replay_position != len(replay_prefix):
                                        raise IntegrityError(
                                            "restart-and-replay completed before every accepted "
                                            "token was verified"
                                        )
                                    status = "completed"
                                    detail = event.detail
                                    terminal_completed = True
                                    if event.text and not any(decoded):
                                        decoded.append(event.text)
                                    break
                                elif event.event_type == "failed":
                                    raise RuntimeError(
                                        event.detail or "execution engine reported failure"
                                    )
                            if not terminal_completed:
                                raise RuntimeError(
                                    "engine stream ended without a terminal event"
                                )
                            if recoveries:
                                terminal_engine_metrics["restart_replay"] = {
                                    "recovery_count": recoveries,
                                    "verified_prefix_tokens": len(replay_prefix),
                                    "execution_identity_preserved": True,
                                }
                            break
                        except IntegrityError as exc:
                            status = "failed"
                            detail = f"exact replay failed closed: {exc}"
                            terminal_engine_metrics["restart_replay"] = {
                                "recovery_count": recoveries,
                                "verified_prefix_tokens": replay_position,
                                "execution_identity_preserved": True,
                                "failure": str(exc),
                            }
                            break
                        except Exception as exc:
                            if recoveries >= self.maximum_engine_recovery_attempts:
                                status = "failed"
                                detail = (
                                    "engine recovery exhausted after "
                                    f"{recoveries} attempt(s): {type(exc).__name__}: {exc}"
                                )
                                break
                            recoveries += 1
                            self._progress(
                                "recovery-started",
                                "submission",
                                f"restarting engine deployment and verifying "
                                f"{len(output_tokens)} accepted token(s)",
                            )
                            with suppress(Exception):
                                await lifecycle.unload(deployment)
                            try:
                                replacement = await lifecycle.prepare(selected)
                                if (
                                    not replacement.ready
                                    or replacement.engine_id != selected.engine_id
                                    or replacement.execution_identity
                                    != selected.execution_identity
                                ):
                                    raise IntegrityError(
                                        "recovery deployment changed execution identity"
                                    )
                            except Exception as recovery_exc:
                                status = "failed"
                                detail = (
                                    "engine recovery could not recreate the immutable "
                                    f"deployment: {type(recovery_exc).__name__}: {recovery_exc}"
                                )
                                break
                            deployment = replacement
                            self._progress(
                                "recovery-ready",
                                "submission",
                                "replacement deployment is ready; replay verification started",
                            )
                except asyncio.CancelledError:
                    with suppress(Exception):
                        await lifecycle.unload(deployment)
                    raise
                if status != "completed":
                    with suppress(Exception):
                        await lifecycle.unload(deployment)
                deployment_id = deployment.deployment_id
                topology_id = selected.topology

            completed = time.time_ns()
            completed_monotonic_s = time.monotonic()
            telemetry_record = build_inference_telemetry_record(
                request_id=request_id,
                model=descriptor,
                execution_plan=selected,
                deployed_plan=plan,
                cluster=cluster_capabilities,
                status=status,
                submitted_monotonic_s=request_started_monotonic_s,
                completed_monotonic_s=completed_monotonic_s,
                token_monotonic_s=token_monotonic_s,
                terminal_metrics=terminal_engine_metrics,
                terminal_timing_metrics=terminal_timing_metrics,
                per_token_expert_metrics=per_token_expert_metrics,
                recoveries=recoveries,
            )
            self.product_telemetry.record_inference(telemetry_record)
            summary = ClusterRunSummary(
                run_id=run_id,
                status=status,
                model_id=model_id,
                model_revision=model_revision,
                tokenizer_revision=tokenizer_revision,
                model_fingerprint=descriptor.content_fingerprint,
                model_format=descriptor.format,
                variant=descriptor.variant,
                quantization=descriptor.quantization,
                engine_id=selected.engine_id,
                engine_revision=telemetry_record.engine_revision,
                engine_runtime_revisions=telemetry_record.engine_runtime_revisions,
                execution_identity=selected.execution_identity,
                engine_support=decision.engine_support,
                canonical_decision=decision,
                mode=mode,
                plan=plan,
                artifact_ids=[item.artifact_id for item in manifests],
                deployment_id=deployment_id,
                topology_id=topology_id,
                output_token_ids=output_tokens,
                decoded_text="".join(decoded),
                event_count=events,
                started_at_unix_ns=started_ns,
                completed_at_unix_ns=completed,
                elapsed_seconds=time.monotonic() - started_monotonic,
                detail=detail,
                telemetry=telemetry_record,
            )
            self._progress("run-completed", "submission", f"request {status}")
            return summary
        finally:
            if worker_transport is not None:
                await worker_transport.close()
            await client.close()


__all__ = [
    "ClusterOrchestrator",
    "ClusterRunSummary",
    "RunProgress",
    "resolve_upstream_source",
    "validate_immutable_reference",
]
