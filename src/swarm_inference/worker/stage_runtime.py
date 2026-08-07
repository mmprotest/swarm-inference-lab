"""Persistent stage ownership, routing, execution, and session lifecycle."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import os
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar
from uuid import uuid4

import psutil
import torch

from swarm_inference.config.models import WorkerCapability
from swarm_inference.exceptions import (
    BackpressureError,
    IntegrityError,
    MemoryLimitExceededError,
    TransportError,
)
from swarm_inference.execution.interfaces import StageExecutionResult, StageExecutor
from swarm_inference.host import is_wildcard_host, split_endpoint
from swarm_inference.model.adapter import (
    NativeModelAdapter,
    NativeModelAdapterRegistry,
    default_native_adapter_registry,
)
from swarm_inference.model.partition import StageAssignment
from swarm_inference.model.product import (
    ModelResolutionPolicy,
    ProductLayerCost,
    ProductModelMetadata,
    ProductModelSpec,
)
from swarm_inference.protocol.expert import (
    SignedExpertRouteLease,
    verify_expert_route_lease,
)
from swarm_inference.protocol.product import (
    ProductStageExpertPlan,
    WorkerModelProbeRequest,
    WorkerModelProbeResponse,
)
from swarm_inference.protocol.routes import (
    BoundedNonceCache,
    PeerHandshake,
    SignedRouteLease,
    route_lease_hash,
    sign_peer_handshake,
    verify_peer_handshake,
    verify_worker_route_lease,
)
from swarm_inference.protocol.stage_ring import (
    MessageSequenceValidator,
    Operation,
    SequenceAllocator,
    StageMessage,
)
from swarm_inference.protocol.stage_worker import (
    CancelStageSessionRequest,
    CloseStageSessionRequest,
    DrainWorkerRequest,
    GetStageCapabilitiesRequest,
    GetStageCapabilitiesResponse,
    GetStageStatusRequest,
    InstalledStageRouteStatus,
    InstallStageRouteRequest,
    LoadedStageStatus,
    LoadStageRequest,
    OpenStageSessionRequest,
    RemoveStageRouteRequest,
    StageActionResponse,
    StageStatusResponse,
    TokenizeStageRequest,
    TokenizeStageResponse,
    UnloadStageRequest,
    VerifyStageRouteRequest,
)
from swarm_inference.runtime.performance_profiles import FastPathProfileStore
from swarm_inference.security.identity import WorkerIdentity, public_key_fingerprint
from swarm_inference.transport.stage_ring_connection import StageRingConnectionPool
from swarm_inference.transport.stage_tensor import pack_tensor, unpack_tensor
from swarm_inference.worker.stage_sessions import StageSessionRegistry

TokenPublisher = Callable[["TokenPublication"], Awaitable[None]]
AttributeT = TypeVar("AttributeT")


class StageLoader(Protocol):
    def __call__(
        self,
        request: LoadStageRequest,
        resolved_model_path: Path | None,
    ) -> StageExecutor: ...


@dataclass(frozen=True, slots=True)
class ProcessMemorySnapshot:
    rss_bytes: int
    cuda_allocated_bytes: int
    cuda_reserved_bytes: int


@dataclass(frozen=True, slots=True)
class TokenPublication:
    destination: str | None
    message: StageMessage
    enqueued_monotonic_ns: int


@dataclass(slots=True)
class _LoadedStage:
    request: LoadStageRequest
    executor: StageExecutor
    status: LoadedStageStatus


@dataclass(slots=True)
class _QueuedStageExecution:
    message: StageMessage
    future: asyncio.Future[StageMessage]


_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "f16": torch.float16,
    "float32": torch.float32,
    "f32": torch.float32,
}


def _normalise_dtype(value: str) -> str:
    key = value.strip().lower()
    aliases = {
        "bf16": "bfloat16",
        "f16": "float16",
        "f32": "float32",
    }
    key = aliases.get(key, key)
    if key not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"unsupported stage execution dtype {value!r}")
    return key


def _normalise_device(value: str) -> str:
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA stage device requested but CUDA is unavailable")
        return f"cuda:{torch.cuda.current_device()}"
    return str(device)


def _validate_endpoint(endpoint: str, *, name: str) -> None:
    host, port = split_endpoint(endpoint)
    if is_wildcard_host(host):
        raise ValueError(f"{name} cannot advertise a wildcard address")
    if port == 0:
        raise ValueError(f"{name} must use a non-zero port")


def _validate_stage_assignment_values(
    owned: StageAssignment,
    *,
    stage_count: int,
) -> None:
    if isinstance(owned.stage_id, bool) or not 0 <= owned.stage_id < stage_count:
        raise ValueError("stage assignment ID is outside the topology")
    if (
        isinstance(owned.layer_start, bool)
        or isinstance(owned.layer_end, bool)
        or owned.layer_start < 0
        or owned.layer_end <= owned.layer_start
    ):
        raise ValueError("stage assignment has an invalid layer interval")
    if owned.layer_ids != tuple(range(owned.layer_start, owned.layer_end)):
        raise ValueError("stage assignment is not a complete contiguous layer interval")
    nonnegative = (
        owned.weight_bytes,
        owned.estimated_compute_ns,
        owned.kv_cache_bytes_per_token,
        owned.peak_temporary_bytes,
        owned.activation_bytes,
    )
    if any(isinstance(value, bool) or value < 0 for value in nonnegative):
        raise ValueError("stage assignment resource measurements cannot be negative")
    if owned.weight_bytes == 0:
        raise ValueError("stage assignment must own a positive weight byte count")
    if owned.measured_compute_ns is not None and (
        isinstance(owned.measured_compute_ns, bool) or owned.measured_compute_ns < 0
    ):
        raise ValueError("stage measured compute time cannot be negative")
    if not isinstance(owned.device, str) or not owned.device.strip():
        raise ValueError("stage assignment device cannot be empty")
    if owned.owns_embeddings != (owned.stage_id == 0):
        raise ValueError("only stage zero may own model embeddings")
    final = owned.stage_id == stage_count - 1
    if owned.owns_final_norm != final or owned.owns_output_projection != final:
        raise ValueError("only the final stage may own final normalization and projection")


class PersistentStageRuntime:
    """Keep one canonical stage loaded across independently scoped sessions."""

    def __init__(
        self,
        *,
        worker_id: str,
        device: str,
        dtype: str,
        memory_limit_bytes: int,
        maximum_sessions: int,
        execution_queue_capacity: int = 256,
        token_queue_capacity: int = 256,
        model_cache_dir: str | Path | None = None,
        configured_model_path: str | Path | None = None,
        allow_model_download: bool = False,
        capability: WorkerCapability | None = None,
        adapter_registry: NativeModelAdapterRegistry | None = None,
        loader: StageLoader | None = None,
        token_publisher: TokenPublisher | None = None,
        connection_pool: StageRingConnectionPool | None = None,
        identity: WorkerIdentity | None = None,
        trusted_coordinators: dict[str, str] | None = None,
        require_authenticated_routes: bool = False,
        route_future_tolerance_s: float = 30.0,
        nonce_cache_capacity: int = 4096,
        artifact_resolver: Callable[[str], Path] | None = None,
        artifact_lease_acquirer: Callable[[str, str], str] | None = None,
        artifact_lease_releaser: Callable[[str], bool] | None = None,
        fast_path_profile_store: FastPathProfileStore | None = None,
    ) -> None:
        if not worker_id:
            raise ValueError("stage runtime worker ID cannot be empty")
        if not device:
            raise ValueError("stage runtime device cannot be empty")
        if memory_limit_bytes <= 0:
            raise ValueError("stage runtime memory limit must be positive")
        if execution_queue_capacity <= 0 or token_queue_capacity <= 0:
            raise ValueError("stage runtime queues must be bounded by positive capacities")
        self.worker_id = worker_id
        self.device = _normalise_device(device)
        self.dtype = _normalise_dtype(dtype)
        self.memory_limit_bytes = memory_limit_bytes
        self.model_cache_dir = (
            Path(model_cache_dir).expanduser().resolve() if model_cache_dir is not None else None
        )
        self.configured_model_path = (
            Path(configured_model_path).expanduser().resolve()
            if configured_model_path is not None
            else None
        )
        self.allow_model_download = allow_model_download
        self.fast_path_profile_store = fast_path_profile_store
        self._artifact_resolver = artifact_resolver
        self._artifact_lease_acquirer = artifact_lease_acquirer
        self._artifact_lease_releaser = artifact_lease_releaser
        self._loaded_artifact_lease_id: str | None = None
        self.capability = capability
        self._adapters = adapter_registry or default_native_adapter_registry()
        self.identity = identity
        self._trusted_coordinators = dict(trusted_coordinators or {})
        self.require_authenticated_routes = require_authenticated_routes
        self.require_authenticated_peers = require_authenticated_routes
        self.route_future_tolerance_ns = int(route_future_tolerance_s * 1_000_000_000)
        if self.route_future_tolerance_ns < 0:
            raise ValueError("route future tolerance cannot be negative")
        self._route_nonce_cache = BoundedNonceCache(capacity=nonce_cache_capacity)
        self._expert_route_nonce_cache = BoundedNonceCache(capacity=nonce_cache_capacity)
        self._peer_nonce_cache = BoundedNonceCache(capacity=nonce_cache_capacity)
        self._verified_route_lease: SignedRouteLease | None = None
        self._verified_expert_route_lease: SignedExpertRouteLease | None = None
        self.sessions = StageSessionRegistry(maximum_sessions=maximum_sessions)
        self.execution_queue_capacity = execution_queue_capacity
        self.token_queue_capacity = token_queue_capacity
        self._execution_queue: asyncio.Queue[_QueuedStageExecution] = asyncio.Queue(
            maxsize=execution_queue_capacity
        )
        self._token_queue: asyncio.Queue[TokenPublication] = asyncio.Queue(
            maxsize=token_queue_capacity
        )
        self._loader = loader or self._load_native_stage
        self._custom_loader = loader is not None
        self._token_publisher = token_publisher
        self.connection_pool = connection_pool or StageRingConnectionPool(
            queue_capacity=execution_queue_capacity,
            handshake_factory=self._peer_handshake_message,
            handshake_verifier=self._verify_peer_handshake_response,
        )
        self._loaded: _LoadedStage | None = None
        self._route: InstallStageRouteRequest | None = None
        self._last_route_generation: int | None = None
        self._execution_runner: asyncio.Task[None] | None = None
        self._token_runner: asyncio.Task[None] | None = None
        self._executor_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._sequence_validator = MessageSequenceValidator()
        self._sequence_allocator = SequenceAllocator()
        self._draining = False
        self._closed = False
        self._load_count = 0
        self._dropped_token_publications = 0
        self._tokenizer: Any | None = None
        self._sync_capability()

    @property
    def loaded_executor(self) -> StageExecutor | None:
        return self._loaded.executor if self._loaded is not None else None

    @property
    def load_count(self) -> int:
        return self._load_count

    @property
    def installed_route(self) -> InstallStageRouteRequest | None:
        return self._route

    @property
    def draining(self) -> bool:
        return self._draining

    def begin_draining(self) -> None:
        """Synchronously reject new lifecycle work before asynchronous cleanup."""

        self._draining = True

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("stage runtime is closed")
        if self._execution_runner is None:
            self._execution_runner = asyncio.create_task(
                self._execution_loop(), name=f"stage-execution:{self.device}"
            )
        if self._token_runner is None:
            self._token_runner = asyncio.create_task(
                self._token_publication_loop(), name="stage-token-publication"
            )

    def _check_worker(self, worker_id: str) -> None:
        if worker_id != self.worker_id:
            raise ValueError(
                f"stage control request addressed to {worker_id!r}, worker is {self.worker_id!r}"
            )

    @staticmethod
    def _check_deadline(deadline_unix_ns: int | None) -> None:
        if deadline_unix_ns is not None and deadline_unix_ns <= time.time_ns():
            raise TimeoutError("stage control request deadline has expired")

    @staticmethod
    def _check_lease(lease_expiry_unix_ns: int | None) -> None:
        if lease_expiry_unix_ns is not None and lease_expiry_unix_ns <= time.time_ns():
            raise ValueError("stage lease has expired")

    def _validate_assignment(self, request: LoadStageRequest) -> None:
        assignment = request.assignment
        _validate_stage_assignment_values(assignment, stage_count=request.stage_count)
        if assignment.device != request.device:
            raise ValueError("stage assignment device does not match load request")
        if _normalise_device(request.device) != self.device:
            raise ValueError(
                f"stage request device {request.device!r} does not match configured "
                f"device {self.device!r}"
            )
        if _normalise_dtype(request.dtype) != self.dtype:
            raise ValueError("stage request dtype does not match the configured runtime dtype")
        estimated_peak = assignment.weight_bytes + assignment.peak_temporary_bytes
        if estimated_peak > self.memory_limit_bytes:
            raise MemoryLimitExceededError(
                f"stage estimated resident peak {estimated_peak} bytes exceeds configured "
                f"logical limit {self.memory_limit_bytes}"
            )

    def _resolve_exact_model_path(
        self,
        *,
        model_id: str,
        model_revision: str,
        model_path: str | None,
        allow_download: bool,
    ) -> Path:
        candidates: list[Path] = []
        if self.configured_model_path is not None:
            candidates.append(self.configured_model_path)
        if model_path is not None:
            supplied = Path(model_path).expanduser()
            candidates.append(supplied)
            if self.model_cache_dir is not None and not supplied.is_absolute():
                candidates.append(self.model_cache_dir / supplied)
        model_id_path = Path(model_id).expanduser()
        candidates.append(model_id_path)
        for candidate in candidates:
            if candidate.is_dir():
                return candidate.resolve()
        try:
            from huggingface_hub import snapshot_download

            resolved = snapshot_download(
                repo_id=model_id,
                revision=model_revision,
                cache_dir=str(self.model_cache_dir) if self.model_cache_dir is not None else None,
                local_files_only=not allow_download,
            )
        except Exception as exc:
            mode = "download-enabled" if allow_download else "local-only"
            raise FileNotFoundError(
                f"could not resolve exact model {model_id}@{model_revision} in {mode} mode: {exc}"
            ) from exc
        return Path(resolved).resolve()

    def _resolve_model_path(self, request: LoadStageRequest) -> Path:
        if request.artifact_id is not None:
            if self._artifact_resolver is not None:
                return self._artifact_resolver(request.artifact_id).resolve()
            if self.model_cache_dir is None:
                raise FileNotFoundError("artifact loading requires a configured artifact cache")
            from swarm_inference.cluster.artifacts import resolve_verified_artifact

            return resolve_verified_artifact(self.model_cache_dir, request.artifact_id)
        return self._resolve_exact_model_path(
            model_id=request.model_id,
            model_revision=request.model_revision,
            model_path=request.model_path,
            allow_download=request.allow_download and self.allow_model_download,
        )

    @staticmethod
    def _metadata_revision(model_path: Path, filenames: tuple[str, ...]) -> str | None:
        download = model_path / ".cache" / "huggingface" / "download"
        revisions: set[str] = set()
        for filename in filenames:
            metadata = download / f"{filename}.metadata"
            if metadata.is_file():
                first = metadata.read_text(encoding="utf-8").splitlines()
                if first and first[0].strip():
                    revisions.add(first[0].strip())
        if len(revisions) > 1:
            raise IntegrityError("checkpoint files report conflicting source revisions")
        if revisions:
            return next(iter(revisions))
        parts = model_path.parts
        if len(parts) >= 2 and parts[-2] == "snapshots" and parts[-1]:
            return parts[-1]
        return None

    def _verify_model_identity_values(
        self,
        *,
        model_id: str,
        requested_model_revision: str,
        requested_tokenizer_revision: str,
        model_path: Path,
        requested_adapter_id: str | None = None,
        artifact_tokenizer_revision: str | None = None,
    ) -> NativeModelAdapter:
        config_path = model_path / "config.json"
        index_path = model_path / "model.safetensors.index.json"
        tensor_files = tuple(model_path.glob("*.safetensors"))
        if not config_path.is_file() or (not index_path.is_file() and not tensor_files):
            raise IntegrityError("resolved native checkpoint is missing config or tensors")
        config_value = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config_value, dict):
            raise IntegrityError("resolved checkpoint config is not a JSON object")
        try:
            adapter = self._adapters.resolve_config(config_value)
        except LookupError as exc:
            raise IntegrityError("resolved checkpoint has no unambiguous native adapter") from exc
        if requested_adapter_id is not None and adapter.adapter_id != requested_adapter_id:
            raise IntegrityError(
                f"native adapter mismatch: resolved={adapter.adapter_id!r} "
                f"requested={requested_adapter_id!r}"
            )
        revision_files = ["config.json"]
        if index_path.is_file():
            revision_files.append(index_path.name)
        resolved_model_revision = self._metadata_revision(
            model_path, tuple(revision_files)
        )
        config_revision = config_value.get("_commit_hash")
        if resolved_model_revision is None and isinstance(config_revision, str):
            resolved_model_revision = config_revision or None
        elif (
            isinstance(config_revision, str)
            and config_revision
            and config_revision != resolved_model_revision
        ):
            raise IntegrityError("checkpoint config revision conflicts with local metadata")
        if resolved_model_revision != requested_model_revision:
            raise IntegrityError(
                f"model revision mismatch: resolved={resolved_model_revision!r} "
                f"requested={requested_model_revision!r}"
            )
        tokenizer_files = ("tokenizer.json", "tokenizer_config.json")
        if artifact_tokenizer_revision is None and not any(
            (model_path / filename).is_file() for filename in tokenizer_files
        ):
            raise IntegrityError("resolved checkpoint has no tokenizer identity files")
        resolved_tokenizer_revision: str | None
        if artifact_tokenizer_revision is not None:
            resolved_tokenizer_revision = artifact_tokenizer_revision
        elif requested_tokenizer_revision.startswith("sha256:"):
            tokenizer_json = model_path / "tokenizer.json"
            if not tokenizer_json.is_file():
                raise IntegrityError(
                    "sha256 tokenizer identity requires a local tokenizer.json file"
                )
            resolved_tokenizer_revision = (
                "sha256:" + hashlib.sha256(tokenizer_json.read_bytes()).hexdigest()
            )
        else:
            resolved_tokenizer_revision = self._metadata_revision(model_path, tokenizer_files)
            if resolved_tokenizer_revision is None:
                resolved_tokenizer_revision = resolved_model_revision
        if resolved_tokenizer_revision != requested_tokenizer_revision:
            raise IntegrityError(
                f"tokenizer revision mismatch: resolved={resolved_tokenizer_revision!r} "
                f"requested={requested_tokenizer_revision!r}"
            )
        identity_path = model_path / "swarm-model-identity.json"
        if identity_path.is_file():
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            if not isinstance(identity, dict) or identity.get("model_id") != model_id:
                raise IntegrityError("resolved checkpoint model ID does not match the request")
        return adapter

    def _verify_model_identity(
        self, request: LoadStageRequest, model_path: Path
    ) -> NativeModelAdapter:
        artifact_tokenizer_revision: str | None = None
        if request.artifact_id is not None:
            from swarm_inference.cluster.artifacts import verify_artifact_directory

            manifest = verify_artifact_directory(
                model_path, expected_artifact_id=request.artifact_id
            )
            assignment = request.assignment
            if (
                manifest.model_id != request.model_id
                or manifest.model_revision != request.model_revision
                or manifest.tokenizer_revision != request.tokenizer_revision
                or manifest.dtype != request.dtype
                or manifest.layer_start != assignment.layer_start
                or manifest.layer_end != assignment.layer_end
                or manifest.owns_embeddings != assignment.owns_embeddings
                or manifest.owns_final_norm != assignment.owns_final_norm
                or manifest.owns_output_projection != assignment.owns_output_projection
                or (
                    request.adapter_id is not None
                    and manifest.adapter_id != request.adapter_id
                )
            ):
                raise IntegrityError("artifact identity or ownership differs from the load request")
            artifact_tokenizer_revision = manifest.tokenizer_revision
        return self._verify_model_identity_values(
            model_id=request.model_id,
            requested_model_revision=request.model_revision,
            requested_tokenizer_revision=request.tokenizer_revision,
            model_path=model_path,
            requested_adapter_id=request.adapter_id,
            artifact_tokenizer_revision=artifact_tokenizer_revision,
        )

    def _load_native_stage(
        self,
        request: LoadStageRequest,
        resolved_model_path: Path | None,
    ) -> StageExecutor:
        if resolved_model_path is None:
            raise FileNotFoundError("native stage loading requires a resolved local checkpoint")
        config_value = json.loads(
            (resolved_model_path / "config.json").read_text(encoding="utf-8")
        )
        adapter = (
            self._adapters.get(request.adapter_id)
            if request.adapter_id is not None
            else self._adapters.resolve_config(config_value)
        )
        executor = adapter.create_stage_executor(
            request=request,
            resolved_model_path=resolved_model_path,
            fast_path_profile_store=self.fast_path_profile_store,
        )
        if not isinstance(executor, StageExecutor):
            raise TypeError(
                f"native adapter {adapter.adapter_id!r} returned an invalid stage executor"
            )
        return executor

    def _memory_snapshot(self) -> ProcessMemorySnapshot:
        rss = int(psutil.Process().memory_info().rss)
        allocated = 0
        reserved = 0
        device = torch.device(self.device)
        if (
            device.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_initialized()  # type: ignore[no-untyped-call]
        ):
            allocated = int(torch.cuda.memory_allocated(device))
            reserved = int(torch.cuda.memory_reserved(device))
        return ProcessMemorySnapshot(
            rss_bytes=rss,
            cuda_allocated_bytes=allocated,
            cuda_reserved_bytes=reserved,
        )

    @staticmethod
    def _same_load(left: LoadStageRequest, right: LoadStageRequest) -> bool:
        excluded = {"request_id", "deadline_unix_ns", "lease_expiry_unix_ns", "route_generation"}
        return left.model_dump(mode="json", exclude=excluded) == right.model_dump(
            mode="json", exclude=excluded
        )

    async def load_stage(self, request: LoadStageRequest) -> StageActionResponse:
        self._check_worker(request.worker_id)
        self._check_deadline(request.deadline_unix_ns)
        self._check_lease(request.lease_expiry_unix_ns)
        if self._draining:
            raise RuntimeError("worker is draining and cannot load a stage")
        self._validate_assignment(request)
        async with self._executor_lock:
            if self._draining:
                raise RuntimeError("worker is draining and cannot load a stage")
            if self._loaded is not None:
                if self._same_load(self._loaded.request, request):
                    return StageActionResponse(
                        worker_id=self.worker_id,
                        request_id=request.request_id,
                        accepted=True,
                        detail="identical stage is already resident",
                        idempotent=True,
                    )
                raise RuntimeError(
                    f"device {self.device} already owns an incompatible resident stage"
                )
            resolved_path: Path | None
            if request.artifact_id is not None:
                resolved_path = self._resolve_model_path(request)
            elif self._custom_loader:
                candidate = Path(request.model_path).expanduser() if request.model_path else None
                resolved_path = (
                    candidate.resolve() if candidate is not None and candidate.is_dir() else None
                )
            else:
                resolved_path = self._resolve_model_path(request)
            if not self._custom_loader:
                if resolved_path is None:
                    raise FileNotFoundError("product stage has no resolved model source")
                adapter = self._verify_model_identity(request, resolved_path)
                expert_plan = (
                    ProductStageExpertPlan.model_validate(request.expert_plan)
                    if request.expert_plan is not None
                    else None
                )
                validate_assignment = getattr(adapter, "validate_stage_assignment", None)
                if not callable(validate_assignment):
                    raise IntegrityError(
                        f"native adapter {adapter.adapter_id!r} cannot validate stage ownership"
                    )
                validate_assignment(
                    resolved_path,
                    assignment=request.assignment,
                    stage_count=request.stage_count,
                    model_revision=request.model_revision,
                    tokenizer_revision=request.tokenizer_revision,
                    remote_experts=(
                        {
                            (item.layer_id, item.expert_id)
                            for item in expert_plan.placements
                            if item.strategy != "local" and not item.local_fallback_permitted
                        }
                        if expert_plan is not None
                        else None
                    ),
                )
            artifact_lease_id: str | None = None
            if request.artifact_id is not None and self._artifact_lease_acquirer is not None:
                artifact_lease_id = self._artifact_lease_acquirer(
                    request.artifact_id, request.topology_id
                )
            before = self._memory_snapshot()
            executor: StageExecutor | None = None
            try:
                executor = await asyncio.to_thread(self._loader, request, resolved_path)
                ownership = executor.ownership
                assignment = request.assignment
                if (
                    ownership.stage_id != assignment.stage_id
                    or ownership.layer_start != assignment.layer_start
                    or ownership.layer_end != assignment.layer_end
                    or ownership.owns_embeddings != assignment.owns_embeddings
                    or ownership.owns_final_norm != assignment.owns_final_norm
                    or ownership.owns_output_projection != assignment.owns_output_projection
                ):
                    raise IntegrityError("loaded executor ownership does not match its assignment")
                if ownership.parameter_bytes > assignment.weight_bytes:
                    raise IntegrityError(
                        "loaded executor owns more parameter bytes than its exact assignment"
                    )
                after = self._memory_snapshot()
                rss_delta = max(0, after.rss_bytes - before.rss_bytes)
                cuda_delta = max(0, after.cuda_allocated_bytes - before.cuda_allocated_bytes)
                # A custom loader is the deterministic test/integration seam; process RSS
                # can move when its worker thread is created and is not attributable to
                # fake weights. Product loaders enforce both measured deltas.
                actual_resident = max(
                    ownership.parameter_bytes,
                    cuda_delta,
                    0 if self._custom_loader else rss_delta,
                )
                if actual_resident > self.memory_limit_bytes:
                    raise MemoryLimitExceededError(
                        f"loaded stage resident delta {actual_resident} bytes exceeds configured "
                        f"logical limit {self.memory_limit_bytes}"
                    )
            except BaseException:
                if executor is not None:
                    with suppress(Exception):
                        await asyncio.to_thread(executor.close)
                self._release_device_memory()
                if artifact_lease_id is not None and self._artifact_lease_releaser is not None:
                    self._artifact_lease_releaser(artifact_lease_id)
                raise
            self._load_count += 1
            path_text = (
                str(resolved_path)
                if resolved_path is not None
                else (request.model_path or "<custom-loader>")
            )
            status = LoadedStageStatus(
                model_id=request.model_id,
                model_revision=request.model_revision,
                tokenizer_revision=request.tokenizer_revision,
                topology_id=request.topology_id,
                assignment=request.assignment,
                device=self.device,
                dtype=self.dtype,
                model_path=path_text,
                artifact_id=request.artifact_id,
                ownership=executor.ownership.to_dict(),
                loaded_monotonic_ns=time.monotonic_ns(),
                load_count=self._load_count,
                process_rss_before_bytes=before.rss_bytes,
                process_rss_after_bytes=after.rss_bytes,
                cuda_allocated_before_bytes=before.cuda_allocated_bytes,
                cuda_allocated_after_bytes=after.cuda_allocated_bytes,
                cuda_reserved_before_bytes=before.cuda_reserved_bytes,
                cuda_reserved_after_bytes=after.cuda_reserved_bytes,
            )
            self._loaded = _LoadedStage(
                request=request.model_copy(deep=True), executor=executor, status=status
            )
            self._loaded_artifact_lease_id = artifact_lease_id
            self._sync_capability()
        await self.start()
        return StageActionResponse(
            worker_id=self.worker_id,
            request_id=request.request_id,
            accepted=True,
            detail="stage loaded and resident",
        )

    def _require_loaded(self) -> _LoadedStage:
        if self._loaded is None:
            raise RuntimeError("worker has no resident stage")
        return self._loaded

    def _validate_loaded_identity(
        self,
        *,
        model_id: str,
        model_revision: str,
        tokenizer_revision: str,
        topology_id: str,
        stage_id: int,
        device: str,
        dtype: str,
    ) -> _LoadedStage:
        loaded = self._require_loaded()
        request = loaded.request
        if (
            request.model_id != model_id
            or request.model_revision != model_revision
            or request.tokenizer_revision != tokenizer_revision
            or request.topology_id != topology_id
            or request.assignment.stage_id != stage_id
        ):
            raise ValueError("stage control identity does not match the resident stage")
        if _normalise_device(device) != self.device or _normalise_dtype(dtype) != self.dtype:
            raise ValueError("stage control device or dtype does not match the resident stage")
        return loaded

    async def unload_stage(self, request: UnloadStageRequest) -> StageActionResponse:
        self._check_worker(request.worker_id)
        self._check_deadline(request.deadline_unix_ns)
        async with self._executor_lock:
            if self._loaded is None:
                return StageActionResponse(
                    worker_id=self.worker_id,
                    request_id=request.request_id,
                    accepted=True,
                    detail="stage is already unloaded",
                    idempotent=True,
                )
            loaded = self._validate_loaded_identity(
                model_id=request.model_id,
                model_revision=request.model_revision,
                tokenizer_revision=request.tokenizer_revision,
                topology_id=request.topology_id,
                stage_id=request.assignment.stage_id,
                device=request.device,
                dtype=request.dtype,
            )
            if request.assignment != loaded.request.assignment:
                raise ValueError("unload assignment does not match the resident stage")
            if request.stage_count != loaded.request.stage_count:
                raise ValueError("unload stage count does not match the resident topology")
            expected_generation = (
                self._route.route_generation
                if self._route is not None
                else (
                    self._last_route_generation
                    if self._last_route_generation is not None
                    else loaded.request.route_generation
                )
            )
            if request.route_generation != expected_generation:
                raise ValueError("unload route generation does not match the resident stage")
            if self.sessions.active_count and not request.force:
                raise RuntimeError("cannot unload a stage with active sessions")
            released = await self._unload_locked(force=request.force)
        return StageActionResponse(
            worker_id=self.worker_id,
            request_id=request.request_id,
            accepted=True,
            detail="resident stage unloaded",
            released_kv_bytes=released,
        )

    async def _unload_locked(self, *, force: bool) -> int:
        loaded = self._require_loaded()
        if self.sessions.active_count and not force:
            raise RuntimeError("resident stage still has active sessions")
        old_endpoint = (
            self._route.next_stage.data_endpoint if self._route and self._route.next_stage else None
        )
        released = await asyncio.to_thread(self.sessions.cancel_all, loaded.executor)
        await asyncio.to_thread(loaded.executor.close)
        artifact_lease_id = self._loaded_artifact_lease_id
        self._loaded = None
        self._loaded_artifact_lease_id = None
        self._route = None
        self._verified_route_lease = None
        self._verified_expert_route_lease = None
        self._last_route_generation = None
        self._tokenizer = None
        self._sequence_validator = MessageSequenceValidator()
        self._sequence_allocator = SequenceAllocator()
        self._sync_capability()
        if old_endpoint is not None:
            await self.connection_pool.remove(old_endpoint)
        self._release_device_memory()
        if artifact_lease_id is not None and self._artifact_lease_releaser is not None:
            self._artifact_lease_releaser(artifact_lease_id)
        return released

    def _release_device_memory(self) -> None:
        gc.collect()
        device = torch.device(self.device)
        if (
            device.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_initialized()  # type: ignore[no-untyped-call]
        ):
            with suppress(RuntimeError):
                torch.cuda.empty_cache()

    async def install_route(self, request: InstallStageRouteRequest) -> StageActionResponse:
        self._check_worker(request.worker_id)
        self._check_deadline(request.deadline_unix_ns)
        self._check_lease(request.lease_expiry_unix_ns)
        lease = request.route_lease
        authentication_configured = bool(self._trusted_coordinators)
        if self.require_authenticated_routes and lease is None:
            raise IntegrityError("coordinator-signed route lease is required")
        if lease is not None and (self.require_authenticated_routes or authentication_configured):
            capability = self.capability
            if capability is None:
                raise IntegrityError("worker capability identity is unavailable")
            control_endpoint = capability.control_endpoint or capability.endpoint
            data_endpoint = capability.data_plane_endpoint
            if control_endpoint is None or data_endpoint is None:
                raise IntegrityError("worker control and data endpoints must be configured")
            verify_worker_route_lease(
                lease,
                self._trusted_coordinators,
                worker_id=self.worker_id,
                worker_public_key=capability.public_key,
                control_endpoint=control_endpoint,
                data_endpoint=data_endpoint,
                topology_id=request.topology_id,
                route_generation=request.route_generation,
                model_id=request.model_id,
                model_revision=request.model_revision,
                tokenizer_revision=request.tokenizer_revision,
                assignment=request.assignment,
                device=request.device,
                dtype=request.dtype,
                last_route_generation=self._last_route_generation,
                future_tolerance_ns=self.route_future_tolerance_ns,
                nonce_cache=self._route_nonce_cache,
            )
        expert_lease = request.expert_route_lease
        loaded = self._validate_loaded_identity(
            model_id=request.model_id,
            model_revision=request.model_revision,
            tokenizer_revision=request.tokenizer_revision,
            topology_id=request.topology_id,
            stage_id=request.assignment.stage_id,
            device=request.device,
            dtype=request.dtype,
        )
        expert_plan = (
            ProductStageExpertPlan.model_validate(loaded.request.expert_plan)
            if loaded.request.expert_plan is not None
            else None
        )
        has_remote_experts = expert_plan is not None and any(
            item.strategy != "local" for item in expert_plan.placements
        )
        if has_remote_experts and expert_lease is None:
            raise IntegrityError("remote expert execution requires a signed expert route lease")
        if expert_lease is not None:
            if self.identity is None or self.capability is None:
                raise IntegrityError("stage identity is unavailable for expert route installation")
            verify_expert_route_lease(
                expert_lease,
                self._trusted_coordinators,
                last_route_generation=self._last_route_generation,
                nonce_cache=self._expert_route_nonce_cache,
            )
            participant = next(
                (item for item in expert_lease.participants if item.worker_id == self.worker_id),
                None,
            )
            if (
                participant is None
                or "contiguous-stage" not in participant.roles
                or participant.worker_public_key != self.identity.public_key_b64
                or participant.worker_public_key_fingerprint != self.identity.public_key_fingerprint
            ):
                raise IntegrityError("stage identity is absent from the expert route lease")
            if (
                expert_lease.topology_id != request.topology_id
                or expert_lease.route_generation != request.route_generation
                or expert_lease.model_id != request.model_id
                or expert_lease.model_revision != request.model_revision
                or expert_lease.model_fingerprint != (loaded.request.expert_model_fingerprint or "")
                or expert_lease.quantization_fingerprint
                != (loaded.request.expert_quantization_fingerprint or "")
            ):
                raise IntegrityError("expert route lease model or topology identity mismatch")
            install_expert_route = getattr(loaded.executor, "install_expert_route", None)
            if not callable(install_expert_route):
                raise IntegrityError(
                    "selected native stage executor does not support remote expert routes"
                )
            install_expert_route(
                expert_lease,
                identity=self.identity,
                worker_id=self.worker_id,
            )
        if self._draining:
            raise RuntimeError("worker is draining and cannot install a route")
        async with self._executor_lock:
            if self._draining:
                raise RuntimeError("worker is draining and cannot install a route")
            endpoint_to_remove, response = self._install_route_locked(request)
            self._verified_route_lease = lease
            self._verified_expert_route_lease = expert_lease
        if endpoint_to_remove is not None:
            await self.connection_pool.remove(endpoint_to_remove)
        return response

    def _install_route_locked(
        self,
        request: InstallStageRouteRequest,
    ) -> tuple[str | None, StageActionResponse]:
        loaded = self._validate_loaded_identity(
            model_id=request.model_id,
            model_revision=request.model_revision,
            tokenizer_revision=request.tokenizer_revision,
            topology_id=request.topology_id,
            stage_id=request.assignment.stage_id,
            device=request.device,
            dtype=request.dtype,
        )
        if request.assignment != loaded.request.assignment:
            raise ValueError("route assignment does not match the resident stage")
        if request.stage_count != loaded.request.stage_count:
            raise ValueError("route stage count does not match the loaded topology")
        stage_id = request.assignment.stage_id
        if not stage_id < request.stage_count:
            raise ValueError("route stage ID is outside the declared stage count")
        if stage_id == 0:
            if request.previous_stage is not None:
                raise ValueError("stage zero cannot declare a previous stage")
        elif request.previous_stage is None or request.previous_stage.stage_id != stage_id - 1:
            raise ValueError("route previous stage is not contiguous")
        if stage_id == request.stage_count - 1:
            if request.next_stage is not None:
                raise ValueError("final stage cannot declare a next stage")
        elif request.next_stage is None or request.next_stage.stage_id != stage_id + 1:
            raise ValueError("route next stage is not contiguous")
        for name, peer in (
            ("previous stage", request.previous_stage),
            ("next stage", request.next_stage),
        ):
            if peer is not None:
                _validate_endpoint(peer.data_endpoint, name=name)
                if peer.worker_id == self.worker_id:
                    raise ValueError(f"{name} cannot refer to this worker")
                if peer.assignment is not None:
                    _validate_stage_assignment_values(
                        peer.assignment,
                        stage_count=request.stage_count,
                    )
                    if peer.assignment.stage_id != peer.stage_id:
                        raise ValueError(f"{name} assignment identity is inconsistent")
        if (
            request.previous_stage is not None
            and request.previous_stage.assignment is not None
            and request.previous_stage.assignment.layer_end != request.assignment.layer_start
        ):
            raise ValueError("previous-stage assignment is not layer-contiguous")
        if (
            request.next_stage is not None
            and request.next_stage.assignment is not None
            and request.next_stage.assignment.layer_start != request.assignment.layer_end
        ):
            raise ValueError("next-stage assignment is not layer-contiguous")
        if request.stage_zero_publication_destination is not None:
            _validate_endpoint(
                request.stage_zero_publication_destination,
                name="stage-zero publication destination",
            )
        previous_route = self._route
        if request.route_generation < loaded.request.route_generation:
            raise ValueError("stale route generation")
        if (
            previous_route is None
            and self._last_route_generation is not None
            and request.route_generation <= self._last_route_generation
        ):
            raise ValueError("stale route generation")
        if previous_route is not None:
            if request.route_generation < previous_route.route_generation:
                raise ValueError("stale route generation")
            if request.route_generation == previous_route.route_generation:
                excluded = {"request_id", "deadline_unix_ns", "replace"}
                if request.model_dump(mode="json", exclude=excluded) == previous_route.model_dump(
                    mode="json", exclude=excluded
                ):
                    return (
                        None,
                        StageActionResponse(
                            worker_id=self.worker_id,
                            request_id=request.request_id,
                            accepted=True,
                            detail="identical route is already installed",
                            idempotent=True,
                        ),
                    )
                raise ValueError("route generation is already installed with different contents")
            if not request.replace:
                raise ValueError("route replacement must be explicit")
            if self.sessions.active_count:
                raise RuntimeError("cannot replace a route while sessions are active")
        old_endpoint = (
            previous_route.next_stage.data_endpoint
            if previous_route is not None and previous_route.next_stage is not None
            else None
        )
        self._route = request.model_copy(deep=True)
        self._last_route_generation = request.route_generation
        self._sync_capability()
        endpoint_to_remove = (
            old_endpoint
            if old_endpoint is not None
            and (
                self.require_authenticated_peers
                or request.next_stage is None
                or request.next_stage.data_endpoint != old_endpoint
            )
            else None
        )
        return (
            endpoint_to_remove,
            StageActionResponse(
                worker_id=self.worker_id,
                request_id=request.request_id,
                accepted=True,
                detail="stage route installed",
            ),
        )

    async def remove_route(self, request: RemoveStageRouteRequest) -> StageActionResponse:
        self._check_worker(request.worker_id)
        self._check_deadline(request.deadline_unix_ns)
        async with self._executor_lock:
            endpoint_to_remove, response = self._remove_route_locked(request)
        if endpoint_to_remove is not None:
            await self.connection_pool.remove(endpoint_to_remove)
        return response

    def _remove_route_locked(
        self,
        request: RemoveStageRouteRequest,
    ) -> tuple[str | None, StageActionResponse]:
        self._validate_loaded_identity(
            model_id=request.model_id,
            model_revision=request.model_revision,
            tokenizer_revision=request.tokenizer_revision,
            topology_id=request.topology_id,
            stage_id=request.stage_id,
            device=request.device,
            dtype=request.dtype,
        )
        route = self._route
        if route is None:
            if (
                self._last_route_generation is not None
                and request.route_generation != self._last_route_generation
            ):
                raise ValueError("stale route generation")
            return (
                None,
                StageActionResponse(
                    worker_id=self.worker_id,
                    request_id=request.request_id,
                    accepted=True,
                    detail="stage route is already absent",
                    idempotent=True,
                ),
            )
        if request.route_generation != route.route_generation:
            raise ValueError("route removal generation does not match the installed route")
        if self.sessions.active_count:
            raise RuntimeError("cannot remove a route while sessions are active")
        endpoint = route.next_stage.data_endpoint if route.next_stage is not None else None
        self._route = None
        self._verified_route_lease = None
        self._verified_expert_route_lease = None
        self._sync_capability()
        return (
            endpoint,
            StageActionResponse(
                worker_id=self.worker_id,
                request_id=request.request_id,
                accepted=True,
                detail="stage route removed",
            ),
        )

    async def verify_route(self, request: VerifyStageRouteRequest) -> StageActionResponse:
        """Prove that the installed next-stage connection is usable and persistent."""

        self._check_worker(request.worker_id)
        self._check_deadline(request.deadline_unix_ns)
        loaded = self._validate_loaded_identity(
            model_id=request.model_id,
            model_revision=request.model_revision,
            tokenizer_revision=request.tokenizer_revision,
            topology_id=request.topology_id,
            stage_id=request.stage_id,
            device=request.device,
            dtype=request.dtype,
        )
        route = self._route
        if route is None or route.route_generation != request.route_generation:
            raise ValueError("stage route verification generation mismatch")
        self._check_lease(route.lease_expiry_unix_ns)
        next_stage = route.next_stage
        if next_stage is None:
            return StageActionResponse(
                worker_id=self.worker_id,
                request_id=request.request_id,
                accepted=True,
                detail="final stage has no downstream peer",
                idempotent=True,
            )
        next_assignment = next_stage.assignment
        if next_assignment is None:
            raise ValueError("next-stage route is missing its exact assignment")
        probe_session = f"__route_probe__:{request.topology_id}:{request.route_generation}"
        current_stage = loaded.request.assignment.stage_id
        probe_attributes: dict[str, object] = {
            "model_id": request.model_id,
            "route_generation": request.route_generation,
            "source_worker_id": self.worker_id,
            "destination_worker_id": next_stage.worker_id,
        }
        if self.require_authenticated_peers:
            probe_attributes["peer_handshake"] = self._signed_peer_handshake(
                peer_worker_id=next_stage.worker_id,
                peer_stage_id=next_stage.stage_id,
            ).model_dump(mode="json")
        probe = StageMessage(
            operation=Operation.HELLO,
            model_revision=request.model_revision,
            tokenizer_revision=request.tokenizer_revision,
            topology_id=request.topology_id,
            stage_id=next_stage.stage_id,
            layer_start=next_assignment.layer_start,
            layer_end=next_assignment.layer_end,
            session_id=probe_session,
            request_id=request.request_id,
            sequence_number=self._sequence_allocator.next(
                probe_session, current_stage, next_stage.stage_id
            ),
            token_position=-1,
            source_stage=current_stage,
            destination_stage=next_stage.stage_id,
            attributes=probe_attributes,
        )
        response = await self.connection_pool.send(next_stage.data_endpoint, probe)
        if (
            response.operation != Operation.HELLO
            or response.status != "OK"
            or response.source_stage != next_stage.stage_id
            or response.destination_stage != current_stage
        ):
            raise TransportError("next-stage peer returned an invalid route probe response")
        if self.require_authenticated_peers:
            self._verify_peer_handshake_response(probe, response)
        self._sequence_validator.validate(response)
        return StageActionResponse(
            worker_id=self.worker_id,
            request_id=request.request_id,
            accepted=True,
            detail="persistent next-stage connection verified",
        )

    async def tokenize(self, request: TokenizeStageRequest) -> TokenizeStageResponse:
        """Tokenize on stage zero from its independently resolved exact snapshot."""

        self._check_worker(request.worker_id)
        self._check_deadline(request.deadline_unix_ns)
        async with self._executor_lock:
            loaded = self._validate_loaded_identity(
                model_id=request.model_id,
                model_revision=request.model_revision,
                tokenizer_revision=request.tokenizer_revision,
                topology_id=request.topology_id,
                stage_id=request.stage_id,
                device=request.device,
                dtype=request.dtype,
            )
            if request.stage_id != 0 or not loaded.request.assignment.owns_embeddings:
                raise ValueError("prompt tokenization is owned by stage zero")
            route = self._route
            if route is None or route.route_generation != request.route_generation:
                raise ValueError("tokenization route generation mismatch")
            if self._tokenizer is None:
                model_path = Path(loaded.status.model_path)
                if not model_path.is_dir():
                    raise RuntimeError("resident stage has no tokenizer-capable local snapshot")

                def load_tokenizer() -> Any:
                    from transformers import AutoTokenizer

                    return AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
                        model_path,
                        local_files_only=True,
                    )

                self._tokenizer = await asyncio.to_thread(load_tokenizer)
            tokenizer = self._tokenizer
        assert tokenizer is not None
        encoded = await asyncio.to_thread(
            tokenizer,
            request.text,
            add_special_tokens=request.add_special_tokens,
            return_tensors=None,
        )
        token_ids = [int(value) for value in encoded["input_ids"]]
        if not token_ids:
            raise ValueError("prompt tokenization produced no token IDs")
        return TokenizeStageResponse(
            worker_id=self.worker_id,
            request_id=request.request_id,
            token_ids=token_ids,
        )

    def decode_token_id(self, token_id: int) -> str:
        tokenizer = self._tokenizer
        if tokenizer is None:
            return ""
        return str(tokenizer.decode([token_id], skip_special_tokens=False))

    def _validate_session_request(
        self,
        request: OpenStageSessionRequest,
        *,
        require_live_lease: bool = True,
    ) -> tuple[_LoadedStage, InstallStageRouteRequest]:
        self._check_worker(request.worker_id)
        self._check_deadline(request.deadline_unix_ns)
        if require_live_lease:
            self._check_lease(request.lease_expiry_unix_ns)
        loaded = self._validate_loaded_identity(
            model_id=request.model_id,
            model_revision=request.model_revision,
            tokenizer_revision=request.tokenizer_revision,
            topology_id=request.topology_id,
            stage_id=request.stage_id,
            device=request.device,
            dtype=request.dtype,
        )
        route = self._route
        if route is None or route.topology_id != request.topology_id:
            raise ValueError("unknown stage topology route")
        if route.route_generation != request.route_generation:
            raise ValueError("stale stage route generation")
        if require_live_lease:
            self._check_lease(route.lease_expiry_unix_ns)
        return loaded, route

    async def open_session(self, request: OpenStageSessionRequest) -> StageActionResponse:
        async with self._executor_lock:
            if self._draining:
                raise RuntimeError("worker is draining and cannot open a session")
            loaded, _ = self._validate_session_request(request)
            await asyncio.to_thread(
                self.sessions.open,
                loaded.executor,
                topology_id=request.topology_id,
                session_id=request.session_id,
                model_revision=request.model_revision,
                route_generation=request.route_generation,
                request_generation=request.request_generation,
                stage_id=request.stage_id,
            )
            self._sync_capability()
        return StageActionResponse(
            worker_id=self.worker_id,
            request_id=request.request_id,
            accepted=True,
            detail="stage session opened",
        )

    async def close_session(
        self,
        request: CloseStageSessionRequest,
        *,
        reset_sequences: bool = True,
    ) -> StageActionResponse:
        async with self._executor_lock:
            loaded, _ = self._validate_session_request(request, require_live_lease=False)
            self.sessions.require(
                topology_id=request.topology_id,
                session_id=request.session_id,
                model_revision=request.model_revision,
                route_generation=request.route_generation,
                request_generation=request.request_generation,
                stage_id=request.stage_id,
            )
            released = await asyncio.to_thread(
                self.sessions.close,
                loaded.executor,
                topology_id=request.topology_id,
                session_id=request.session_id,
            )
            if reset_sequences:
                self._sequence_validator.reset_session(request.session_id)
                self._sequence_allocator.reset_session(request.session_id)
            self._sync_capability()
        return StageActionResponse(
            worker_id=self.worker_id,
            request_id=request.request_id,
            accepted=True,
            detail="stage session closed",
            released_kv_bytes=released,
        )

    async def cancel_session(
        self,
        request: CancelStageSessionRequest,
        *,
        reset_sequences: bool = True,
    ) -> StageActionResponse:
        async with self._executor_lock:
            loaded, _ = self._validate_session_request(request, require_live_lease=False)
            self.sessions.require(
                topology_id=request.topology_id,
                session_id=request.session_id,
                model_revision=request.model_revision,
                route_generation=request.route_generation,
                request_generation=request.request_generation,
                stage_id=request.stage_id,
            )
            released = await asyncio.to_thread(
                self.sessions.cancel,
                loaded.executor,
                topology_id=request.topology_id,
                session_id=request.session_id,
            )
            if reset_sequences:
                self._sequence_validator.reset_session(request.session_id)
                self._sequence_allocator.reset_session(request.session_id)
            self._sync_capability()
        return StageActionResponse(
            worker_id=self.worker_id,
            request_id=request.request_id,
            accepted=True,
            detail="stage session cancelled",
            released_kv_bytes=released,
        )

    async def get_capabilities(
        self, request: GetStageCapabilitiesRequest
    ) -> GetStageCapabilitiesResponse:
        self._check_worker(request.worker_id)
        self._check_deadline(request.deadline_unix_ns)
        if self.capability is None:
            raise RuntimeError("stage runtime has no attached worker capability record")
        self._sync_capability()
        return GetStageCapabilitiesResponse(
            worker_id=self.worker_id,
            request_id=request.request_id,
            capability=self.capability.model_copy(deep=True),
        )

    async def inspect_model(
        self,
        request: WorkerModelProbeRequest,
    ) -> WorkerModelProbeResponse:
        """Resolve and inspect an exact native model identity without loading weights."""

        self._check_worker(request.worker_id)
        self._check_deadline(request.deadline_unix_ns)
        reference = request.reference
        try:
            requested_adapter = self._adapters.get(reference.adapter_id)
        except KeyError:
            return WorkerModelProbeResponse(
                worker_id=self.worker_id,
                request_id=request.request_id,
                available=False,
                detail=f"unsupported model adapter {reference.adapter_id!r}",
                worker_download_permitted=self.allow_model_download,
            )
        if _normalise_dtype(reference.dtype) != self.dtype:
            return WorkerModelProbeResponse(
                worker_id=self.worker_id,
                request_id=request.request_id,
                available=False,
                detail=(
                    f"requested dtype {reference.dtype!r} does not match worker runtime "
                    f"dtype {self.dtype!r}"
                ),
                worker_download_permitted=self.allow_model_download,
            )
        download_requested = reference.resolution_policy == ModelResolutionPolicy.ALLOW_DOWNLOAD
        allow_download = download_requested and self.allow_model_download
        try:
            model_path = await asyncio.to_thread(
                self._resolve_exact_model_path,
                model_id=reference.model_id,
                model_revision=reference.model_revision,
                model_path=None,
                allow_download=allow_download,
            )
            resolved_adapter = await asyncio.to_thread(
                self._verify_model_identity_values,
                model_id=reference.model_id,
                requested_model_revision=reference.model_revision,
                requested_tokenizer_revision=reference.tokenizer_revision,
                model_path=model_path,
                requested_adapter_id=requested_adapter.adapter_id,
            )
            inspect_partition = getattr(resolved_adapter, "inspect_partition_metadata", None)
            if not callable(inspect_partition):
                raise IntegrityError(
                    f"native adapter {resolved_adapter.adapter_id!r} cannot inspect partitions"
                )
            partition = await asyncio.to_thread(
                inspect_partition,
                model_path,
                model_revision=reference.model_revision,
                tokenizer_revision=reference.tokenizer_revision,
            )
            metadata = ProductModelMetadata(
                layer_costs=tuple(
                    ProductLayerCost.model_validate(asdict(cost)) for cost in partition.layer_costs
                ),
                embedding_weight_bytes=partition.embedding_weight_bytes,
                final_weight_bytes=partition.final_weight_bytes,
                dtype_bytes=partition.dtype_bytes,
                hidden_size=partition.hidden_size,
                metadata_hash=partition.metadata_hash,
                expert_count=partition.expert_count,
                experts_per_token=partition.experts_per_token,
                expert_intermediate_size=partition.expert_intermediate_size,
                model_fingerprint=partition.model_fingerprint,
                quantization_fingerprint=partition.quantization_fingerprint,
            )
            spec = ProductModelSpec.resolved(reference, metadata)
            return WorkerModelProbeResponse(
                worker_id=self.worker_id,
                request_id=request.request_id,
                available=True,
                detail="exact model and tokenizer metadata resolved",
                spec=spec,
                metadata=metadata,
                resolved_from_local_cache=not allow_download,
                worker_download_permitted=self.allow_model_download,
            )
        except Exception as exc:
            return WorkerModelProbeResponse(
                worker_id=self.worker_id,
                request_id=request.request_id,
                available=False,
                detail=f"{type(exc).__name__}: {exc}",
                resolved_from_local_cache=not allow_download,
                worker_download_permitted=self.allow_model_download,
            )

    async def status(self, request: GetStageStatusRequest) -> StageStatusResponse:
        self._check_worker(request.worker_id)
        self._check_deadline(request.deadline_unix_ns)
        async with self._executor_lock:
            loaded = self._loaded
            if request.topology_id is not None and (
                loaded is None or loaded.request.topology_id != request.topology_id
            ):
                raise ValueError("unknown stage topology")
            sessions = []
            if loaded is not None:
                sessions = await asyncio.to_thread(self.sessions.statuses, loaded.executor)
        route_status = None
        if self._route is not None:
            route_status = InstalledStageRouteStatus(
                topology_id=self._route.topology_id,
                route_generation=self._route.route_generation,
                previous_stage=self._route.previous_stage,
                next_stage=self._route.next_stage,
                stage_count=self._route.stage_count,
                stage_zero_publication_destination=(self._route.stage_zero_publication_destination),
                lease_expiry_unix_ns=self._route.lease_expiry_unix_ns,
                authenticated=self._verified_route_lease is not None,
                route_lease_hash=(
                    route_lease_hash(self._verified_route_lease)
                    if self._verified_route_lease is not None
                    else None
                ),
            )
        return StageStatusResponse(
            worker_id=self.worker_id,
            request_id=request.request_id,
            process_id=os.getpid(),
            draining=self._draining,
            loaded_stage=loaded.status if loaded is not None else None,
            installed_route=route_status,
            sessions=sessions,
            execution_queue_depth=self._execution_queue.qsize(),
            execution_queue_capacity=self.execution_queue_capacity,
            token_queue_depth=self._token_queue.qsize(),
            token_queue_capacity=self.token_queue_capacity,
            dropped_token_publications=self._dropped_token_publications,
            expert_status=(
                status()
                if loaded is not None
                and callable(status := getattr(loaded.executor, "expert_status", None))
                else {}
            ),
        )

    async def drain(self, request: DrainWorkerRequest) -> StageActionResponse:
        self._check_worker(request.worker_id)
        self._check_deadline(request.deadline_unix_ns)
        already = self._draining
        self._draining = True
        released = 0
        if request.cancel_active_sessions:
            async with self._executor_lock:
                if self._loaded is not None:
                    released = await asyncio.to_thread(
                        self.sessions.cancel_all, self._loaded.executor
                    )
                    self._sync_capability()
        return StageActionResponse(
            worker_id=self.worker_id,
            request_id=request.request_id,
            accepted=True,
            detail="worker is draining",
            idempotent=already,
            released_kv_bytes=released,
        )

    def _sync_capability(self) -> None:
        capability = self.capability
        if capability is None:
            return
        capability.device_identifier = self.device
        capability.active_session_count = self.sessions.active_count
        capability.currently_loaded_model_revisions = (
            [self._loaded.request.model_revision] if self._loaded is not None else []
        )
        capability.currently_loaded_topology_ids = (
            [self._loaded.request.topology_id] if self._loaded is not None else []
        )

    def refresh_capability(self) -> None:
        """Synchronise dynamic stage fields before heartbeat serialization."""

        self._sync_capability()

    def configure_route_trust(
        self,
        *,
        coordinator_identity: str,
        coordinator_public_key: str,
        expected_fingerprint: str | None = None,
    ) -> None:
        """Pin the coordinator key before accepting authenticated routes."""

        fingerprint = public_key_fingerprint(coordinator_public_key)
        if expected_fingerprint is not None and fingerprint != expected_fingerprint:
            raise IntegrityError(
                "coordinator fingerprint does not match worker trust configuration"
            )
        existing = self._trusted_coordinators.get(coordinator_identity)
        if existing is not None and existing != coordinator_public_key:
            raise IntegrityError("coordinator identity is already pinned to another key")
        self._trusted_coordinators[coordinator_identity] = coordinator_public_key
        self.require_authenticated_routes = True
        self.require_authenticated_peers = True

    def _signed_peer_handshake(
        self,
        *,
        peer_worker_id: str,
        peer_stage_id: int,
    ) -> PeerHandshake:
        route = self._route
        lease = self._verified_route_lease
        identity = self.identity
        if route is None or lease is None or identity is None:
            raise IntegrityError("authenticated peer route is not configured")
        handshake = PeerHandshake(
            worker_id=self.worker_id,
            public_key_fingerprint=identity.public_key_fingerprint,
            topology_id=route.topology_id,
            route_generation=route.route_generation,
            stage_id=route.assignment.stage_id,
            peer_stage_id=peer_stage_id,
            model_revision=route.model_revision,
            nonce=uuid4().hex,
            timestamp_unix_ns=time.time_ns(),
            route_lease_hash=route_lease_hash(lease),
        )
        return sign_peer_handshake(handshake, identity)

    def _peer_handshake_message(self, endpoint: str) -> StageMessage | None:
        """Build the connection-opening proof for the installed next-stage peer."""

        if not self.require_authenticated_peers:
            return None
        route, loaded = self._route_and_loaded()
        peer = route.next_stage
        if peer is None or peer.data_endpoint != endpoint or peer.assignment is None:
            raise IntegrityError("outbound endpoint is not the installed next-stage peer")
        handshake = self._signed_peer_handshake(
            peer_worker_id=peer.worker_id,
            peer_stage_id=peer.stage_id,
        )
        session_id = (
            f"__peer_handshake__:{route.topology_id}:{route.route_generation}:"
            f"{loaded.request.assignment.stage_id}:{uuid4().hex}"
        )
        return StageMessage(
            operation=Operation.HELLO,
            model_revision=route.model_revision,
            tokenizer_revision=route.tokenizer_revision,
            topology_id=route.topology_id,
            stage_id=peer.stage_id,
            layer_start=peer.assignment.layer_start,
            layer_end=peer.assignment.layer_end,
            session_id=session_id,
            request_id=f"peer-handshake:{uuid4().hex}",
            sequence_number=self._sequence_allocator.next(
                session_id,
                loaded.request.assignment.stage_id,
                peer.stage_id,
            ),
            token_position=-1,
            source_stage=loaded.request.assignment.stage_id,
            destination_stage=peer.stage_id,
            attributes={
                "model_id": route.model_id,
                "route_generation": route.route_generation,
                "source_worker_id": self.worker_id,
                "destination_worker_id": peer.worker_id,
                "peer_handshake": handshake.model_dump(mode="json"),
            },
        )

    def _verify_handshake(
        self,
        message: StageMessage,
        *,
        expected_worker_id: str,
        expected_stage_id: int,
        expected_peer_stage_id: int,
    ) -> None:
        lease = self._verified_route_lease
        if lease is None:
            raise IntegrityError("installed route has no verified coordinator lease")
        raw = message.attributes.get("peer_handshake")
        if not isinstance(raw, dict):
            raise IntegrityError("peer handshake is missing")
        handshake = PeerHandshake.model_validate(raw)
        verify_peer_handshake(
            handshake,
            lease,
            expected_worker_id=expected_worker_id,
            expected_stage_id=expected_stage_id,
            expected_peer_stage_id=expected_peer_stage_id,
            timestamp_tolerance_ns=self.route_future_tolerance_ns,
            nonce_cache=self._peer_nonce_cache,
        )

    def _verify_peer_handshake_response(
        self,
        request: StageMessage,
        response: StageMessage,
    ) -> None:
        if not self.require_authenticated_peers:
            return
        route = self._route
        if route is None or route.next_stage is None:
            raise IntegrityError("peer handshake response has no installed next stage")
        if (
            response.operation != Operation.HELLO
            or response.status != "OK"
            or response.session_id != request.session_id
            or response.request_id != request.request_id
            or response.source_stage != route.next_stage.stage_id
            or response.destination_stage != route.assignment.stage_id
        ):
            raise IntegrityError("peer handshake response identity mismatch")
        self._verify_handshake(
            response,
            expected_worker_id=route.next_stage.worker_id,
            expected_stage_id=route.next_stage.stage_id,
            expected_peer_stage_id=route.assignment.stage_id,
        )

    def _peer_handshake_response(self, message: StageMessage) -> dict[str, object]:
        route = self._route
        if route is None or route.previous_stage is None:
            raise IntegrityError("inbound peer handshake has no installed previous stage")
        handshake = self._signed_peer_handshake(
            peer_worker_id=route.previous_stage.worker_id,
            peer_stage_id=route.previous_stage.stage_id,
        )
        return {
            "worker_id": self.worker_id,
            "route_generation": route.route_generation,
            "peer_handshake": handshake.model_dump(mode="json"),
        }

    def _route_and_loaded(
        self,
        *,
        require_live_lease: bool = True,
    ) -> tuple[InstallStageRouteRequest, _LoadedStage]:
        loaded = self._require_loaded()
        route = self._route
        if route is None:
            raise ValueError("worker has no installed stage route")
        if require_live_lease:
            self._check_lease(route.lease_expiry_unix_ns)
        return route, loaded

    def _message_attribute(
        self,
        message: StageMessage,
        key: str,
        expected: type[AttributeT],
    ) -> AttributeT:
        value = message.attributes.get(key)
        if not isinstance(value, expected):
            raise ValueError(f"stage message attribute {key!r} is missing or malformed")
        return value

    def _validate_message_base(
        self,
        message: StageMessage,
        *,
        require_live_lease: bool = True,
    ) -> tuple[InstallStageRouteRequest, _LoadedStage, int]:
        route, loaded = self._route_and_loaded(require_live_lease=require_live_lease)
        assignment = loaded.request.assignment
        if message.model_revision != loaded.request.model_revision:
            raise ValueError("wrong model revision")
        if message.tokenizer_revision != loaded.request.tokenizer_revision:
            raise ValueError("wrong tokenizer revision")
        if message.topology_id != loaded.request.topology_id:
            raise ValueError("unknown stage topology")
        if (
            message.stage_id != assignment.stage_id
            or message.destination_stage != assignment.stage_id
            or (message.layer_start, message.layer_end)
            != (assignment.layer_start, assignment.layer_end)
        ):
            raise ValueError("stage message destination ownership mismatch")
        route_generation = self._message_attribute(message, "route_generation", int)
        if isinstance(route_generation, bool) or route_generation != route.route_generation:
            raise ValueError("stale route generation")
        model_id = self._message_attribute(message, "model_id", str)
        if model_id != loaded.request.model_id:
            raise ValueError("wrong model identity")
        if message.operation in {Operation.PREFILL, Operation.DECODE, Operation.TOKEN_RESULT}:
            request_generation = message.attributes.get("request_generation", 1)
            replay_only = message.attributes.get("replay_only", False)
            if (
                isinstance(request_generation, bool)
                or not isinstance(request_generation, int)
                or request_generation <= 0
            ):
                raise ValueError("stage request generation is malformed")
            if not isinstance(replay_only, bool):
                raise ValueError("stage replay marker is malformed")
            if self.require_authenticated_routes and (
                "request_generation" not in message.attributes
                or "replay_only" not in message.attributes
            ):
                raise ValueError("authenticated stage frame lacks request replay identity")
        source_worker = self._message_attribute(message, "source_worker_id", str)
        destination_worker = self._message_attribute(message, "destination_worker_id", str)
        expected_source_stage = route.previous_stage.stage_id if route.previous_stage else -1
        expected_source_worker = (
            route.previous_stage.worker_id if route.previous_stage else "coordinator"
        )
        if (
            message.source_stage != expected_source_stage
            or source_worker != expected_source_worker
            or destination_worker != self.worker_id
        ):
            raise ValueError("stage message peer identity does not match the installed route")
        return route, loaded, int(route_generation)

    @staticmethod
    def _request_generation(message: StageMessage) -> int:
        value = message.attributes.get("request_generation", 1)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("stage request generation is malformed")
        return value

    def _control_message_response(
        self,
        message: StageMessage,
        *,
        operation: Operation,
        attributes: dict[str, object] | None = None,
    ) -> StageMessage:
        loaded = self._require_loaded()
        sequence = self._sequence_allocator.next(
            message.session_id, loaded.request.assignment.stage_id, message.source_stage
        )
        return StageMessage(
            operation=operation,
            model_revision=message.model_revision,
            tokenizer_revision=message.tokenizer_revision,
            topology_id=message.topology_id,
            stage_id=loaded.request.assignment.stage_id,
            layer_start=loaded.request.assignment.layer_start,
            layer_end=loaded.request.assignment.layer_end,
            session_id=message.session_id,
            request_id=message.request_id,
            sequence_number=sequence,
            token_position=message.token_position,
            source_stage=loaded.request.assignment.stage_id,
            destination_stage=message.source_stage,
            status="OK",
            attributes=attributes or {},
        )

    async def handle_message(self, message: StageMessage) -> StageMessage:
        """Validate and execute one authorized direct data-plane operation."""

        await self.start()
        route, loaded, route_generation = self._validate_message_base(
            message,
            require_live_lease=message.operation
            not in {Operation.CLOSE_SESSION, Operation.CANCEL_SESSION},
        )
        request_generation = self._request_generation(message)
        if message.operation == Operation.OPEN_SESSION:
            self._sequence_validator.validate(message)
            await self.open_session(
                OpenStageSessionRequest(
                    worker_id=self.worker_id,
                    request_id=message.request_id,
                    model_id=loaded.request.model_id,
                    model_revision=message.model_revision,
                    tokenizer_revision=message.tokenizer_revision,
                    topology_id=message.topology_id,
                    route_generation=route_generation,
                    stage_id=message.stage_id,
                    device=self.device,
                    dtype=self.dtype,
                    session_id=message.session_id,
                    request_generation=request_generation,
                    lease_expiry_unix_ns=route.lease_expiry_unix_ns,
                )
            )
            return self._control_message_response(message, operation=Operation.OPEN_SESSION)
        if message.operation in {Operation.HELLO, Operation.HEALTH}:
            self._sequence_validator.validate(message)
            response_attributes: dict[str, object] = {
                "worker_id": self.worker_id,
                "route_generation": route.route_generation,
                "active_session_count": self.sessions.active_count,
                "execution_queue_depth": self._execution_queue.qsize(),
            }
            if message.operation == Operation.HELLO and self.require_authenticated_peers:
                previous = route.previous_stage
                if previous is None:
                    raise IntegrityError("stage zero cannot accept a worker peer handshake")
                self._verify_handshake(
                    message,
                    expected_worker_id=previous.worker_id,
                    expected_stage_id=previous.stage_id,
                    expected_peer_stage_id=route.assignment.stage_id,
                )
                response_attributes.update(self._peer_handshake_response(message))
            return self._control_message_response(
                message,
                operation=message.operation,
                attributes=response_attributes,
            )
        if message.operation in {Operation.CLOSE_SESSION, Operation.CANCEL_SESSION}:
            self.sessions.require(
                topology_id=message.topology_id,
                session_id=message.session_id,
                model_revision=message.model_revision,
                route_generation=route_generation,
                request_generation=request_generation,
                stage_id=message.stage_id,
            )
            self._sequence_validator.validate(message)
            if message.operation == Operation.CLOSE_SESSION:
                result = await self.close_session(
                    CloseStageSessionRequest(
                        worker_id=self.worker_id,
                        request_id=message.request_id,
                        model_id=loaded.request.model_id,
                        model_revision=message.model_revision,
                        tokenizer_revision=message.tokenizer_revision,
                        topology_id=message.topology_id,
                        route_generation=route_generation,
                        stage_id=message.stage_id,
                        device=self.device,
                        dtype=self.dtype,
                        session_id=message.session_id,
                        request_generation=request_generation,
                        lease_expiry_unix_ns=route.lease_expiry_unix_ns,
                    ),
                    reset_sequences=False,
                )
            else:
                result = await self.cancel_session(
                    CancelStageSessionRequest(
                        worker_id=self.worker_id,
                        request_id=message.request_id,
                        model_id=loaded.request.model_id,
                        model_revision=message.model_revision,
                        tokenizer_revision=message.tokenizer_revision,
                        topology_id=message.topology_id,
                        route_generation=route_generation,
                        stage_id=message.stage_id,
                        device=self.device,
                        dtype=self.dtype,
                        session_id=message.session_id,
                        request_generation=request_generation,
                        lease_expiry_unix_ns=route.lease_expiry_unix_ns,
                    ),
                    reset_sequences=False,
                )
            response = self._control_message_response(
                message,
                operation=message.operation,
                attributes={"released_kv_bytes": result.released_kv_bytes},
            )
            self._sequence_validator.reset_session(message.session_id)
            self._sequence_allocator.reset_session(message.session_id)
            return response
        if message.operation not in {Operation.PREFILL, Operation.DECODE}:
            raise ValueError(f"unsupported product stage data operation {message.operation.name}")
        cache_position = self._message_attribute(message, "cache_position_start", int)
        if isinstance(cache_position, bool) or cache_position < 0:
            raise ValueError("stage cache position is malformed")
        if message.operation == Operation.PREFILL and cache_position != 0:
            raise ValueError("stage prefill must begin at cache position zero")
        if message.operation == Operation.DECODE and cache_position == 0:
            raise ValueError("stage decode requires an existing prefilled cache")
        tensor_metadata = self._message_attribute(message, "tensor", dict)
        tensor, _ = unpack_tensor(message.payload, dict(tensor_metadata))
        if (
            tuple(tensor.shape) != message.tensor_shape
            or str(tensor_metadata["dtype"]) != message.tensor_dtype
        ):
            raise ValueError("stage tensor metadata does not match message metadata")
        if str(tensor_metadata["compression_mode"]) != message.compression_mode:
            raise ValueError("stage tensor compression metadata mismatch")
        if loaded.request.assignment.owns_embeddings:
            if tensor.dtype != torch.int64 or tensor.ndim != 2:
                raise ValueError("stage zero requires a rank-two int64 token tensor")
        elif tensor.dtype != _DTYPES[self.dtype] or tensor.ndim != 3:
            raise ValueError(
                "nonzero stages require a rank-three hidden-state tensor in the "
                "configured activation dtype"
            )
        self.sessions.require(
            topology_id=message.topology_id,
            session_id=message.session_id,
            model_revision=message.model_revision,
            route_generation=route_generation,
            request_generation=request_generation,
            stage_id=message.stage_id,
        )
        self._sequence_validator.validate(message)
        self.sessions.require(
            topology_id=message.topology_id,
            session_id=message.session_id,
            model_revision=message.model_revision,
            route_generation=route_generation,
            request_generation=request_generation,
            stage_id=message.stage_id,
            cache_position_start=int(cache_position),
        )
        loop = asyncio.get_running_loop()
        future: asyncio.Future[StageMessage] = loop.create_future()
        try:
            self._execution_queue.put_nowait(_QueuedStageExecution(message=message, future=future))
        except asyncio.QueueFull as exc:
            raise BackpressureError("stage execution queue is full") from exc
        return await future

    async def _execution_loop(self) -> None:
        while True:
            item = await self._execution_queue.get()
            try:
                try:
                    response = await self._execute_message(item.message)
                except Exception as exc:
                    if not item.future.done():
                        item.future.set_exception(exc)
                else:
                    if not item.future.done():
                        item.future.set_result(response)
            finally:
                self._execution_queue.task_done()

    async def _execute_message(self, message: StageMessage) -> StageMessage:
        tensor_metadata = dict(message.attributes["tensor"])
        tensor, decode_ns = unpack_tensor(message.payload, tensor_metadata)
        cache_position = int(message.attributes["cache_position_start"])
        deadline_ns = int(message.attributes.get("deadline_ns", time.time_ns() + 30_000_000_000))
        async with self._executor_lock:
            route, loaded, route_generation = self._validate_message_base(message)
            request_generation = self._request_generation(message)
            self.sessions.require(
                topology_id=message.topology_id,
                session_id=message.session_id,
                model_revision=message.model_revision,
                route_generation=route_generation,
                request_generation=request_generation,
                stage_id=message.stage_id,
                cache_position_start=cache_position,
            )
            expert_context: dict[str, Any] = {}
            context_factory = getattr(loaded.executor, "execution_context", None)
            if callable(context_factory):
                expert_context = context_factory(
                    request_id=message.request_id,
                    token_position=message.token_position,
                    deadline_ns=deadline_ns,
                )
                if not isinstance(expert_context, dict):
                    raise TypeError("stage execution context must be a dictionary")
            if loaded.request.assignment.owns_embeddings:
                result = await asyncio.to_thread(
                    loaded.executor.execute_prefill,
                    session_id=message.session_id,
                    token_ids=tensor,
                    cache_position_start=cache_position,
                    **expert_context,
                )
            else:
                result = await asyncio.to_thread(
                    loaded.executor.execute_decode,
                    session_id=message.session_id,
                    hidden_states=tensor,
                    cache_position_start=cache_position,
                    **expert_context,
                )
            self.sessions.update_cache_position(
                topology_id=message.topology_id,
                session_id=message.session_id,
                new_position=result.cache_sequence_length,
            )
        if route.next_stage is not None:
            forwarded = self._forward_message(
                message,
                result=result,
                route=route,
                decode_ns=decode_ns,
            )
            downstream = await self.connection_pool.send(
                route.next_stage.data_endpoint,
                forwarded,
            )
            self.sessions.require(
                topology_id=message.topology_id,
                session_id=message.session_id,
                model_revision=message.model_revision,
                route_generation=route_generation,
                request_generation=request_generation,
                stage_id=message.stage_id,
                cache_position_start=result.cache_sequence_length,
            )
            self._validate_downstream_response(
                incoming=message,
                forwarded=forwarded,
                downstream=downstream,
                route=route,
                cache_sequence_length=result.cache_sequence_length,
            )
            response = self._relay_downstream(message, downstream)
            if route.assignment.stage_id == 0:
                self._enqueue_token_publication(route, response)
            return response
        response = self._token_result(message, result=result, route=route, decode_ns=decode_ns)
        if route.assignment.stage_id == 0:
            self._enqueue_token_publication(route, response)
        return response

    def _forward_message(
        self,
        incoming: StageMessage,
        *,
        result: StageExecutionResult,
        route: InstallStageRouteRequest,
        decode_ns: int,
    ) -> StageMessage:
        assert route.next_stage is not None
        next_assignment = route.next_stage.assignment
        if next_assignment is None:
            raise ValueError("next-stage route identity is missing its exact assignment")
        packed = pack_tensor(result.stage_boundary_hidden_states, requested_mode="none")
        current_stage = self._require_loaded().request.assignment.stage_id
        sequence = self._sequence_allocator.next(
            incoming.session_id, current_stage, route.next_stage.stage_id
        )
        return StageMessage(
            operation=incoming.operation,
            model_revision=incoming.model_revision,
            tokenizer_revision=incoming.tokenizer_revision,
            topology_id=incoming.topology_id,
            stage_id=route.next_stage.stage_id,
            layer_start=next_assignment.layer_start,
            layer_end=next_assignment.layer_end,
            session_id=incoming.session_id,
            request_id=incoming.request_id,
            sequence_number=sequence,
            token_position=incoming.token_position,
            source_stage=current_stage,
            destination_stage=route.next_stage.stage_id,
            tensor_shape=packed.shape,
            tensor_dtype=packed.dtype,
            compression_mode=packed.compression_mode,
            payload=packed.payload,
            attributes={
                "model_id": route.model_id,
                "route_generation": route.route_generation,
                "request_generation": int(incoming.attributes.get("request_generation", 1)),
                "replay_only": bool(incoming.attributes.get("replay_only", False)),
                "source_worker_id": self.worker_id,
                "destination_worker_id": route.next_stage.worker_id,
                "cache_position_start": int(incoming.attributes["cache_position_start"]),
                "cache_sequence_length": result.cache_sequence_length,
                "deadline_ns": int(
                    incoming.attributes.get("deadline_ns", time.time_ns() + 30_000_000_000)
                ),
                "compute_ns": result.compute_ns,
                "decode_ns": decode_ns,
                "expert_trace": self._combined_expert_trace(incoming, result),
                "expert_metrics": self._combined_expert_metrics(incoming, result),
                "tensor": packed.attributes(),
            },
        )

    def _token_result(
        self,
        incoming: StageMessage,
        *,
        result: StageExecutionResult,
        route: InstallStageRouteRequest,
        decode_ns: int,
    ) -> StageMessage:
        token_ids = result.sampled_token_ids
        if token_ids is None:
            raise ValueError("final stage did not produce greedy token IDs")
        packed = pack_tensor(token_ids, requested_mode="none")
        current_stage = self._require_loaded().request.assignment.stage_id
        sequence = self._sequence_allocator.next(
            incoming.session_id, current_stage, incoming.source_stage
        )
        return StageMessage(
            operation=Operation.TOKEN_RESULT,
            model_revision=incoming.model_revision,
            tokenizer_revision=incoming.tokenizer_revision,
            topology_id=incoming.topology_id,
            stage_id=current_stage,
            layer_start=incoming.layer_start,
            layer_end=incoming.layer_end,
            session_id=incoming.session_id,
            request_id=incoming.request_id,
            sequence_number=sequence,
            token_position=incoming.token_position,
            source_stage=current_stage,
            destination_stage=incoming.source_stage,
            tensor_shape=packed.shape,
            tensor_dtype=packed.dtype,
            compression_mode=packed.compression_mode,
            payload=packed.payload,
            attributes={
                "model_id": route.model_id,
                "route_generation": route.route_generation,
                "request_generation": int(incoming.attributes.get("request_generation", 1)),
                "replay_only": bool(incoming.attributes.get("replay_only", False)),
                "source_worker_id": self.worker_id,
                "destination_worker_id": (
                    route.previous_stage.worker_id if route.previous_stage else "coordinator"
                ),
                "cache_sequence_length": result.cache_sequence_length,
                "compute_ns": result.compute_ns,
                "decode_ns": decode_ns,
                "expert_trace": self._combined_expert_trace(incoming, result),
                "expert_metrics": self._combined_expert_metrics(incoming, result),
                "tensor": packed.attributes(),
            },
        )

    @staticmethod
    def _combined_expert_trace(
        incoming: StageMessage,
        result: StageExecutionResult,
    ) -> list[dict[str, Any]]:
        previous = incoming.attributes.get("expert_trace", [])
        if not isinstance(previous, list) or any(not isinstance(item, dict) for item in previous):
            raise ValueError("incoming expert trace is malformed")
        return [*(dict(item) for item in previous), *(dict(item) for item in result.expert_events)]

    @staticmethod
    def _combined_expert_metrics(
        incoming: StageMessage,
        result: StageExecutionResult,
    ) -> dict[str, Any]:
        previous = incoming.attributes.get("expert_metrics", {})
        if not isinstance(previous, dict):
            raise ValueError("incoming expert metrics are malformed")
        combined: dict[str, Any] = dict(previous)
        for key, value in result.expert_metrics.items():
            prior = combined.get(key)
            if isinstance(value, (int, float)) and isinstance(prior, (int, float)):
                combined[key] = prior + value
            else:
                combined[key] = value
        return combined

    def _validate_downstream_response(
        self,
        *,
        incoming: StageMessage,
        forwarded: StageMessage,
        downstream: StageMessage,
        route: InstallStageRouteRequest,
        cache_sequence_length: int,
    ) -> None:
        next_stage = route.next_stage
        if next_stage is None:
            raise RuntimeError("downstream response arrived without an installed next stage")
        current_stage = route.assignment.stage_id
        if (
            downstream.model_revision != incoming.model_revision
            or downstream.tokenizer_revision != incoming.tokenizer_revision
            or downstream.topology_id != incoming.topology_id
            or downstream.session_id != incoming.session_id
            or downstream.request_id != incoming.request_id
            or downstream.token_position != incoming.token_position
        ):
            raise ValueError("downstream response request identity mismatch")
        if (
            downstream.stage_id != next_stage.stage_id
            or downstream.source_stage != next_stage.stage_id
            or downstream.destination_stage != current_stage
            or (downstream.layer_start, downstream.layer_end)
            != (forwarded.layer_start, forwarded.layer_end)
        ):
            raise ValueError("downstream response stage identity mismatch")
        if downstream.operation == Operation.ERROR:
            self._sequence_validator.validate(downstream)
            detail = downstream.attributes.get("error", "downstream stage rejected the frame")
            raise TransportError(
                f"downstream worker {next_stage.worker_id} at {next_stage.data_endpoint}: {detail}"
            )
        if downstream.operation != Operation.TOKEN_RESULT or downstream.status != "OK":
            raise ValueError("downstream response is not a successful token result")
        if self._message_attribute(downstream, "model_id", str) != route.model_id:
            raise ValueError("downstream response model identity mismatch")
        generation = self._message_attribute(downstream, "route_generation", int)
        if isinstance(generation, bool) or generation != route.route_generation:
            raise ValueError("downstream response route generation mismatch")
        if int(downstream.attributes.get("request_generation", 1)) != int(
            incoming.attributes.get("request_generation", 1)
        ) or bool(downstream.attributes.get("replay_only", False)) != bool(
            incoming.attributes.get("replay_only", False)
        ):
            raise ValueError("downstream response request generation mismatch")
        if (
            self._message_attribute(downstream, "source_worker_id", str) != next_stage.worker_id
            or self._message_attribute(downstream, "destination_worker_id", str) != self.worker_id
        ):
            raise ValueError("downstream response worker identity mismatch")
        downstream_cache_position = self._message_attribute(
            downstream, "cache_sequence_length", int
        )
        if (
            isinstance(downstream_cache_position, bool)
            or downstream_cache_position != cache_sequence_length
        ):
            raise ValueError("downstream response cache position mismatch")
        tensor_metadata = self._message_attribute(downstream, "tensor", dict)
        token_ids, _ = unpack_tensor(downstream.payload, dict(tensor_metadata))
        if (
            tuple(token_ids.shape) != downstream.tensor_shape
            or str(tensor_metadata["dtype"]) != downstream.tensor_dtype
            or str(tensor_metadata["compression_mode"]) != downstream.compression_mode
            or token_ids.dtype != torch.int64
        ):
            raise ValueError("downstream token tensor metadata mismatch")
        self._sequence_validator.validate(downstream)

    def _relay_downstream(
        self,
        incoming: StageMessage,
        downstream: StageMessage,
    ) -> StageMessage:
        current = self._require_loaded().request.assignment
        sequence = self._sequence_allocator.next(
            incoming.session_id, current.stage_id, incoming.source_stage
        )
        attributes = dict(downstream.attributes)
        attributes["relay_stage_id"] = current.stage_id
        attributes["source_worker_id"] = self.worker_id
        route = self._route
        attributes["destination_worker_id"] = (
            route.previous_stage.worker_id
            if route is not None and route.previous_stage is not None
            else "coordinator"
        )
        return StageMessage(
            operation=downstream.operation,
            model_revision=downstream.model_revision,
            tokenizer_revision=downstream.tokenizer_revision,
            topology_id=downstream.topology_id,
            stage_id=current.stage_id,
            layer_start=current.layer_start,
            layer_end=current.layer_end,
            session_id=downstream.session_id,
            request_id=downstream.request_id,
            sequence_number=sequence,
            token_position=downstream.token_position,
            source_stage=current.stage_id,
            destination_stage=incoming.source_stage,
            tensor_shape=downstream.tensor_shape,
            tensor_dtype=downstream.tensor_dtype,
            compression_mode=downstream.compression_mode,
            payload=downstream.payload,
            status=downstream.status,
            attributes=attributes,
        )

    def _enqueue_token_publication(
        self,
        route: InstallStageRouteRequest,
        message: StageMessage,
    ) -> None:
        publication = TokenPublication(
            destination=route.stage_zero_publication_destination,
            message=message,
            enqueued_monotonic_ns=time.monotonic_ns(),
        )
        try:
            self._token_queue.put_nowait(publication)
        except asyncio.QueueFull as exc:
            self._dropped_token_publications += 1
            raise BackpressureError("stage-zero token publication queue is full") from exc

    async def _token_publication_loop(self) -> None:
        while True:
            publication = await self._token_queue.get()
            try:
                if self._token_publisher is not None:
                    try:
                        await self._token_publisher(publication)
                    except Exception:
                        self._dropped_token_publications += 1
            finally:
                self._token_queue.task_done()

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                return
            self.begin_draining()
            with suppress(TimeoutError):
                await asyncio.wait_for(self._execution_queue.join(), timeout=30.0)
            if self._execution_runner is not None:
                self._execution_runner.cancel()
                await asyncio.gather(self._execution_runner, return_exceptions=True)
                self._execution_runner = None
            execution_error = TransportError("stage runtime is shutting down")
            while not self._execution_queue.empty():
                item = self._execution_queue.get_nowait()
                if not item.future.done():
                    item.future.set_exception(execution_error)
                self._execution_queue.task_done()
            async with self._executor_lock:
                if self._loaded is not None:
                    await self._unload_locked(force=True)
            if self._token_runner is not None:
                self._token_runner.cancel()
                await asyncio.gather(self._token_runner, return_exceptions=True)
                self._token_runner = None
            while not self._token_queue.empty():
                self._token_queue.get_nowait()
                self._token_queue.task_done()
            await self.connection_pool.close()
            self._closed = True


__all__ = [
    "PersistentStageRuntime",
    "ProcessMemorySnapshot",
    "StageLoader",
    "TokenPublication",
    "TokenPublisher",
]
