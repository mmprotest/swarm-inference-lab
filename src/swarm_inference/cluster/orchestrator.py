"""One-shot product orchestration over the canonical coordinator APIs."""

from __future__ import annotations

import asyncio
import re
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import Field, NonNegativeInt, PositiveInt

from swarm_inference.cluster.artifacts import ArtifactManager, StageArtifactBuilder
from swarm_inference.cluster.models import ArtifactManifest, node_id_from_fingerprint
from swarm_inference.cluster.pairing import create_cluster_authentication
from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.config.models import StrictModel
from swarm_inference.coordinator.service import CoordinatorClient
from swarm_inference.model.product import ModelResolutionPolicy, ProductModelReference
from swarm_inference.model.source_paths import materialized_snapshot_path
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
    mode: Literal["speed", "capacity", "balanced"]
    plan: ProductStagePlan
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


class SourceResolver(Protocol):
    async def __call__(
        self,
        model_id: str,
        model_revision: str,
        tokenizer_revision: str,
        cache_directory: Path,
        maximum_bytes: int,
    ) -> Path: ...


ProgressSink = Callable[[RunProgress], None]
StreamSink = Callable[[SubmitStreamEvent], None]


def validate_immutable_reference(model_revision: str, tokenizer_revision: str) -> None:
    if not _IMMUTABLE_REVISION.fullmatch(model_revision):
        raise ValueError("model revision must be an immutable 40- or 64-character commit hash")
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
    """Resolve one immutable source after a bounded repository-size check."""

    local = Path(model_id).expanduser()
    if local.is_dir():
        return local.resolve()
    if maximum_bytes <= 0:
        raise ValueError("source download byte bound must be positive")

    def download() -> Path:
        from huggingface_hub import HfApi, snapshot_download

        api = HfApi()
        info = api.model_info(model_id, revision=model_revision, files_metadata=True)
        siblings = info.siblings
        if siblings is None:
            raise RuntimeError("upstream repository did not expose a file manifest")
        sizes = [item.size for item in siblings if item.size is not None]
        if len(sizes) != len(siblings):
            raise RuntimeError("upstream repository did not expose complete file-size metadata")
        total = sum(sizes)
        if total > maximum_bytes:
            raise OSError(
                f"immutable source is {total} bytes, above the {maximum_bytes}-byte bound"
            )
        materialized = materialized_snapshot_path(cache_directory, model_id, model_revision)
        resolved = snapshot_download(
            repo_id=model_id,
            revision=model_revision,
            local_dir=str(materialized),
            local_files_only=False,
        )
        source = Path(resolved).resolve()
        if source.name.lower() != model_revision.lower():
            raise RuntimeError("upstream snapshot resolved to a different immutable revision")
        # A commit-based tokenizer identity must refer to this model repository
        # snapshot. SHA-256 identities are verified over tokenizer.json by the
        # artifact builder.
        if _IMMUTABLE_REVISION.fullmatch(tokenizer_revision) and (
            tokenizer_revision.lower() != model_revision.lower()
        ):
            raise RuntimeError(
                "a tokenizer commit differing from the model commit requires a pre-merged "
                "verified local source; use a tokenizer sha256 identity for upstream runs"
            )
        return source

    cache_directory.mkdir(parents=True, exist_ok=True)
    return await asyncio.to_thread(download)


class ClusterOrchestrator:
    """Validate, plan, artifact, deploy, and stream without a second runtime."""

    def __init__(
        self,
        *,
        state: ClusterStateStore,
        client_factory: Callable[[str], CoordinatorClient] = CoordinatorClient,
        source_resolver: SourceResolver = resolve_upstream_source,
        progress_sink: ProgressSink | None = None,
        stream_sink: StreamSink | None = None,
        maximum_source_bytes: int = 100 * 1024**3,
        source_timeout_seconds: float = 3600.0,
        network_measurement_ttl_seconds: int = 900,
        network_refresh_wait_seconds: float = 35.0,
    ) -> None:
        if maximum_source_bytes <= 0:
            raise ValueError("source byte bound must be positive")
        if not 0 < source_timeout_seconds <= 7200:
            raise ValueError("source timeout must be in (0, 7200] seconds")
        if network_measurement_ttl_seconds <= 0:
            raise ValueError("network measurement TTL must be positive")
        if not 0 <= network_refresh_wait_seconds <= 120:
            raise ValueError("network refresh wait must be in [0, 120] seconds")
        self.state = state
        self.client_factory = client_factory
        self.source_resolver = source_resolver
        self.progress_sink = progress_sink
        self.stream_sink = stream_sink
        self.maximum_source_bytes = maximum_source_bytes
        self.source_timeout_seconds = source_timeout_seconds
        self.network_measurement_ttl_seconds = network_measurement_ttl_seconds
        self.network_refresh_wait_seconds = network_refresh_wait_seconds

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

    async def run(
        self,
        *,
        model_id: str,
        model_revision: str,
        tokenizer_revision: str,
        prompt: str,
        mode: Literal["speed", "capacity", "balanced"] = "speed",
        dry_run: bool = False,
        required_node_ids: list[str] | None = None,
        excluded_node_ids: list[str] | None = None,
        max_new_tokens: int = 16,
        seed: int = 1,
    ) -> ClusterRunSummary:
        validate_immutable_reference(model_revision, tokenizer_revision)
        if not prompt:
            raise ValueError("run prompt cannot be empty")
        if max_new_tokens <= 0:
            raise ValueError("maximum new tokens must be positive")
        cluster = self.state.load_cluster()
        if cluster is None:
            raise RuntimeError("node is not paired with a cluster")
        run_id = f"run-{uuid4().hex}"
        started_ns = time.time_ns()
        started_monotonic = time.monotonic()
        client = self.client_factory(cluster.coordinator_endpoint)
        try:
            self._progress("run-started", "validation", "validating immutable model identity")
            workers = await client.workers(WorkersRequest(include_unhealthy=False))
            dtype = self._select_dtype(workers)
            self._progress("backend-selected", "capability-refresh", f"selected dtype {dtype}")
            await self._wait_for_fresh_links(client, workers)
            source = await asyncio.wait_for(
                self.source_resolver(
                    model_id,
                    model_revision,
                    tokenizer_revision,
                    self.state.paths.artifacts / "source-cache",
                    self.maximum_source_bytes,
                ),
                timeout=self.source_timeout_seconds,
            )
            reference = ProductModelReference(
                model_id=model_id,
                model_revision=model_revision,
                tokenizer_revision=tokenizer_revision,
                adapter_id="olmoe",
                dtype=dtype,
                resolution_policy=ModelResolutionPolicy.LOCAL_ONLY,
            )
            self._progress("plan-started", "planning", f"planning {mode} objective")
            planned = await client.plan_model(
                ModelPlanRequest(
                    reference=reference,
                    mode=mode,
                    required_node_ids=sorted(set(required_node_ids or [])),
                    excluded_node_ids=sorted(set(excluded_node_ids or [])),
                    allow_artifact_provisioning=True,
                )
            )
            plan = planned.plan
            self._progress(
                "plan-completed",
                "planning",
                f"selected {plan.stage_count} stage(s) with {plan.report.confidence} confidence",
            )
            if dry_run:
                completed = time.time_ns()
                return ClusterRunSummary(
                    run_id=run_id,
                    status="dry-run",
                    model_id=model_id,
                    model_revision=model_revision,
                    tokenizer_revision=tokenizer_revision,
                    mode=mode,
                    plan=plan,
                    started_at_unix_ns=started_ns,
                    completed_at_unix_ns=completed,
                    elapsed_seconds=time.monotonic() - started_monotonic,
                    detail="validated immutable source and completed planning; no mutation requested",
                )
            plan, manifests = await self._build_artifacts(source, plan)
            self._progress(
                "deployment-started",
                "deployment",
                "entering canonical transactional deployment",
            )
            deployed = await client.deploy_model(ModelDeployRequest(plan=plan))
            if not deployed.deployment.ready:
                raise RuntimeError(deployed.deployment.detail or "deployment did not become ready")
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
            async for event in client.submit_stream(request):
                events += 1
                if self.stream_sink is not None:
                    self.stream_sink(event)
                if event.event_type == StreamEventType.TOKEN_GENERATED:
                    assert event.token_id is not None
                    output_tokens.append(event.token_id)
                    decoded.append(event.decoded_text_fragment)
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
            completed = time.time_ns()
            summary = ClusterRunSummary(
                run_id=run_id,
                status=status,
                model_id=model_id,
                model_revision=model_revision,
                tokenizer_revision=tokenizer_revision,
                mode=mode,
                plan=plan,
                artifact_ids=[item.artifact_id for item in manifests],
                deployment_id=deployed.deployment.deployment_id,
                topology_id=plan.topology_id,
                output_token_ids=output_tokens,
                decoded_text="".join(decoded),
                event_count=events,
                started_at_unix_ns=started_ns,
                completed_at_unix_ns=completed,
                elapsed_seconds=time.monotonic() - started_monotonic,
                detail=terminal.status_detail,
            )
            self._progress("run-completed", "submission", f"request {status}")
            return summary
        finally:
            await client.close()


__all__ = [
    "ClusterOrchestrator",
    "ClusterRunSummary",
    "RunProgress",
    "resolve_upstream_source",
    "validate_immutable_reference",
]
