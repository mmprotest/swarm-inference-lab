"""Central coordinator core and gRPC control service."""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar, cast
from uuid import uuid4

import grpc
import numpy as np

from swarm_inference.config.models import (
    Backend,
    DataPlaneMode,
    ExecutionMode,
    ExperimentConfig,
    HealthStatus,
    ModelManifest,
    NetworkProfile,
    NodeProfile,
    OperationKind,
    RequestState,
    RequestStatus,
    SamplingConfig,
    SchedulerMode,
    StageDefinition,
    StageReplica,
    StrictModel,
    VerificationState,
    WorkloadClass,
)
from swarm_inference.config.product import ProductCoordinatorConfig
from swarm_inference.coordinator.deployment import DeploymentManager
from swarm_inference.coordinator.durable_state import DurableCoordinatorState
from swarm_inference.coordinator.model_catalog import ProductModelCatalog
from swarm_inference.coordinator.placement import estimate_worker_stage_rate
from swarm_inference.coordinator.registry import WorkerRegistry
from swarm_inference.coordinator.replay_log import ReplayEntry, ReplayLog
from swarm_inference.coordinator.reservations import (
    AtomicRouteAllocator,
    ReservationDecision,
)
from swarm_inference.coordinator.session_controller import ProductSessionController
from swarm_inference.coordinator.stage_planner import ProductStagePlanner
from swarm_inference.exceptions import (
    IntegrityError,
    NoValidRouteError,
    ReplayUnavailableError,
    TransportError,
)
from swarm_inference.model.synthetic import synthetic_activation
from swarm_inference.protocol.checksums import sha256_bytes
from swarm_inference.protocol.messages import (
    Ack,
    ActivationMetadata,
    ActivationRequest,
    ActivationResult,
    CacheControlRequest,
    DataPlaneEnvelope,
    FinalResultMessage,
    Heartbeat,
    RegistrationRequest,
    RegistrationResponse,
    RouteHop,
    RouteInstallRequest,
    RoutePlan,
    StageAssignmentMessage,
    StreamEventType,
    SubmitRequest,
    SubmitResponse,
    SubmitStreamEvent,
    parse_message,
    serialize_message,
)
from swarm_inference.protocol.product import (
    CancelProductRequest,
    CancelProductResponse,
    CoordinatorStatusRequest,
    CoordinatorStatusResponse,
    DeploymentPhase,
    ModelDeployRequest,
    ModelDeployResponse,
    ModelInspectRequest,
    ModelInspectResponse,
    ModelPlanRequest,
    ModelPlanResponse,
    ModelUnloadRequest,
    ModelUnloadResponse,
    ProductRequestPhase,
    ProductTokenPublication,
    SessionsRequest,
    SessionsResponse,
    TopologyStatusRequest,
    TopologyStatusResponse,
    WorkerProductStatus,
    WorkersRequest,
    WorkersResponse,
)
from swarm_inference.protocol.routes import (
    BoundedNonceCache,
    encode_route_key,
    sign_data_envelope,
    sign_route_plan,
    verify_final_result,
)
from swarm_inference.protocol.stage_worker import GetStageStatusRequest
from swarm_inference.protocol.tensor_codec import ActivationTensor, decode_tensor, encode_tensor
from swarm_inference.runtime.telemetry import ProductTelemetry
from swarm_inference.security.identity import CoordinatorIdentity, public_key_fingerprint
from swarm_inference.security.signatures import canonical_json_bytes, verify_signature
from swarm_inference.simulation.model import build_synthetic_stages
from swarm_inference.transport.base import ActivationTransport
from swarm_inference.transport.grpc_transport import GrpcTransport

ResponseT = TypeVar("ResponseT", bound=StrictModel)


@dataclass(slots=True)
class RuntimeRequestMetrics:
    request_id: str
    started_s: float
    first_token_s: float | None = None
    completed_s: float | None = None
    stage_execution_s: float = 0.0
    queue_s: float = 0.0
    transport_s: float = 0.0
    replay_s: float = 0.0
    replay_bytes: int = 0
    retries: int = 0
    route_changes: int = 0
    admission_time_ms: float = 0.0
    route_reservation_time_ms: float = 0.0
    route_id: str | None = None
    route_generation: int = 0
    data_plane_mode: str = DataPlaneMode.COORDINATOR_RELAY.value
    per_stage: list[dict[str, Any]] = field(default_factory=list)
    token_steps: list[dict[str, Any]] = field(default_factory=list)


def _product_compatibility_config() -> ExperimentConfig:
    """Provide legacy control defaults without requiring an experiment YAML."""

    return ExperimentConfig(
        name="product-stage-ring",
        execution_mode=ExecutionMode.PHYSICAL_LAN,
        seed=1,
        scheduler=SchedulerMode.STATIC,
        backend="cpu",
        data_plane=DataPlaneMode.DIRECT,
        network=NetworkProfile(
            name="product-control",
            base_latency_ms=0,
            upload_bandwidth_bytes_s=1,
            download_bandwidth_bytes_s=1,
            measured=False,
        ),
        nodes=[
            NodeProfile(
                name="product-worker",
                count=1,
                memory_bytes=1,
                compute_rate_layers_s=1,
                supported_backends=[Backend.TORCH_CPU],
                network_profile="product-control",
                measured=False,
            )
        ],
    )


class CoordinatorCore:
    """Coordinator owns metadata, routes, token commitment, and replay inputs."""

    def __init__(
        self,
        *,
        config: ExperimentConfig | None = None,
        product_config: ProductCoordinatorConfig | None = None,
        state_directory: str | Path | None = None,
        registry: WorkerRegistry | None = None,
        transport: ActivationTransport | None = None,
        model_manifest: ModelManifest | None = None,
        architecture_config: dict[str, Any] | None = None,
        runtime_dtype: str | None = None,
        tokenizer: Any | None = None,
        worker_stage_affinity: dict[str, int] | None = None,
        after_token_hook: (Callable[[dict[str, Any]], Awaitable[RoutePlan | None]] | None) = None,
    ) -> None:
        if config is None and product_config is None:
            raise ValueError("coordinator requires an experiment or product configuration")
        if (model_manifest is None) != (architecture_config is None):
            raise ValueError(
                "real-model runtime requires both model_manifest and architecture_config"
            )
        self.product_config = product_config
        self.product_mode = product_config is not None
        self.config = config or _product_compatibility_config()
        self.state_directory = Path(state_directory or ".swarm/coordinator").resolve()
        self.started_monotonic_s = time.monotonic()
        self.started_unix_ns = time.time_ns()
        self.durable_state: DurableCoordinatorState | None = None
        self.coordinator_identity: CoordinatorIdentity | None = None
        self.product_telemetry: ProductTelemetry | None = None
        self._last_heartbeat_sequences: dict[str, int] = {}
        self._registration_nonce_cache = BoundedNonceCache(
            capacity=(
                product_config.route_nonce_cache_capacity if product_config is not None else 4096
            )
        )
        self.registry = registry or WorkerRegistry(
            heartbeat_timeout_s=(
                product_config.worker_heartbeat_timeout_s if product_config is not None else 15.0
            )
        )
        self.transport = transport or GrpcTransport()
        self.model_manifest = model_manifest
        self.architecture_config = architecture_config
        self.runtime_dtype = runtime_dtype
        self.tokenizer = tokenizer
        self.worker_stage_affinity = dict(worker_stage_affinity or {})
        self.after_token_hook = after_token_hook
        if model_manifest is None:
            self.stages = build_synthetic_stages(self.config.model)
            self.runtime_model_id = self.config.model_id
            self.runtime_model_revision = self.config.model_revision
        else:
            self.stages = self._runtime_stages(model_manifest, runtime_dtype)
            self.runtime_model_id = model_manifest.model_id
            self.runtime_model_revision = model_manifest.model_revision
        self.replay = ReplayLog()
        self.events: list[dict[str, Any]] = []
        self.request_metrics: list[dict[str, Any]] = []
        self._rebalance_lock = asyncio.Lock()
        self._assigned: set[tuple[int, str]] = set()
        self.route_allocator = AtomicRouteAllocator()
        self.route_signing_key = os.urandom(32)
        self.final_result_endpoint: str | None = None
        self._pending_final_results: dict[
            tuple[str, int, int], asyncio.Future[FinalResultMessage]
        ] = {}
        self._committed_route_generations: dict[tuple[str, int], int] = {}
        self.runtime_transport_metrics: dict[str, int | float | str] = {
            "data_plane_mode": self.config.data_plane.value,
            "coordinator_control_bytes": 0,
            "coordinator_activation_bytes": 0,
            "coordinator_input_activation_bytes": 0,
            "coordinator_final_result_bytes": 0,
            "worker_to_worker_activation_bytes": 0,
            "data_messages_sent": 0,
            "data_messages_received": 0,
            "serialisation_time_ms": 0.0,
            "deserialisation_time_ms": 0.0,
            "stream_queue_time_ms": 0.0,
            "hop_transfer_time_ms": 0.0,
            "stage_execution_time_ms": 0.0,
            "admission_time_ms": 0.0,
            "route_reservation_time_ms": 0.0,
        }
        self.publication_endpoint: str | None = None
        self.product_catalog: ProductModelCatalog | None = None
        self.product_planner: ProductStagePlanner | None = None
        self.deployment_manager: DeploymentManager | None = None
        self.session_controller: ProductSessionController | None = None
        if product_config is not None:
            self.durable_state = DurableCoordinatorState(self.state_directory)
            self.coordinator_identity = CoordinatorIdentity.load_or_create(
                self.durable_state.identity_path
            )
            persisted_metadata = self.durable_state.load_metadata()
            if persisted_metadata:
                expected_metadata = {
                    "schema_version": 1,
                    "coordinator_identity": product_config.coordinator_id,
                    "coordinator_public_key": self.coordinator_identity.public_key_b64,
                    "coordinator_public_key_fingerprint": (
                        self.coordinator_identity.public_key_fingerprint
                    ),
                }
                mismatched = [
                    key
                    for key, expected in expected_metadata.items()
                    if persisted_metadata.get(key) != expected
                ]
                if mismatched:
                    raise IntegrityError(
                        "durable coordinator metadata does not match the configured "
                        "identity: " + ", ".join(sorted(mismatched))
                    )
            self.product_telemetry = ProductTelemetry(self.state_directory / "product-events.jsonl")
            self.durable_state.mark_restart_boundaries()
            self.durable_state.save_metadata(
                {
                    "schema_version": 1,
                    "coordinator_identity": product_config.coordinator_id,
                    "coordinator_public_key": self.coordinator_identity.public_key_b64,
                    "coordinator_public_key_fingerprint": (
                        self.coordinator_identity.public_key_fingerprint
                    ),
                    "last_started_unix_ns": self.started_unix_ns,
                    "state_directory": str(self.state_directory),
                }
            )
            product_transport = cast(Any, self.transport)
            self.product_catalog = ProductModelCatalog(
                registry=self.registry,
                transport=product_transport,
                maximum_active_sessions_per_worker=(
                    product_config.maximum_active_sessions_per_worker
                ),
            )
            self.product_planner = ProductStagePlanner()
            self.deployment_manager = DeploymentManager(
                registry=self.registry,
                transport=product_transport,
                state_directory=self.state_directory,
                lease_seconds=product_config.deployment_lease_seconds,
                control_timeout_s=product_config.control_timeout_s,
                coordinator_identity=self.coordinator_identity,
                coordinator_id=product_config.coordinator_id,
                telemetry=self.product_telemetry,
            )
            self.session_controller = ProductSessionController(
                deployments=self.deployment_manager,
                transport=product_transport,
                event_queue_capacity=product_config.event_queue_capacity,
                request_timeout_s=product_config.request_timeout_s,
                data_queue_capacity=product_config.token_ingress_capacity,
                state=self.durable_state,
                telemetry=self.product_telemetry,
                cleanup_timeout_s=product_config.cleanup_timeout_s,
                recovery_timeout_s=product_config.recovery_timeout_s,
                maximum_recovery_attempts=product_config.maximum_recovery_attempts,
            )

    @staticmethod
    def _runtime_stages(
        manifest: ModelManifest,
        runtime_dtype: str | None,
    ) -> list[StageDefinition]:
        source_width = {
            "F16": 2,
            "BF16": 2,
            "F32": 4,
        }.get(manifest.weight_dtype.upper())
        target_width = {
            "f16": 2,
            "float16": 2,
            "bf16": 2,
            "bfloat16": 2,
            "f32": 4,
            "float32": 4,
        }.get((runtime_dtype or manifest.weight_dtype).lower())
        if source_width is None or target_width is None:
            raise ValueError(
                f"unsupported source/runtime dtype pair: {manifest.weight_dtype}/{runtime_dtype}"
            )
        return [
            stage.model_copy(
                update={
                    "required_memory_bytes": math.ceil(
                        stage.required_memory_bytes * target_width / source_width
                    )
                }
            )
            for stage in manifest.stages
        ]

    def _worker_can_host(self, worker: Any, stage: Any) -> bool:
        desired_stage = self.worker_stage_affinity.get(str(worker.worker_id))
        if desired_stage is not None and desired_stage != int(stage.stage_id):
            return False
        if stage.required_memory_bytes > worker.effective_memory_bytes:
            return False
        if estimate_worker_stage_rate(worker, stage) <= 0:
            return False
        if self.model_manifest is None:
            return True
        if worker.backend not in self.model_manifest.compatible_worker_backends:
            return False
        normalised_dtype = (self.runtime_dtype or self.model_manifest.weight_dtype).lower()
        aliases = {
            "f16": "float16",
            "bf16": "bfloat16",
            "f32": "float32",
        }
        required_dtype = aliases.get(normalised_dtype, normalised_dtype)
        return required_dtype in worker.supported_dtypes

    async def close(self) -> None:
        if self.session_controller is not None:
            await self.session_controller.close()
        for future in self._pending_final_results.values():
            if not future.done():
                future.set_exception(TransportError("coordinator is shutting down"))
        self._pending_final_results.clear()
        self.route_allocator.release_all(reason="coordinator-shutdown")
        await self.transport.close()

    async def register(self, request: RegistrationRequest) -> RegistrationResponse:
        signed_payload = canonical_json_bytes(
            {
                "capability": request.capability.model_dump(mode="json"),
                "benchmark_nonce": request.benchmark_nonce,
            }
        )
        verify_signature(
            request.capability.public_key,
            signed_payload,
            request.signature,
        )
        fingerprint = public_key_fingerprint(request.capability.public_key)
        if self.product_config is not None:
            trusted = set(self.product_config.trusted_worker_fingerprints)
            if self.product_config.require_trusted_workers and fingerprint not in trusted:
                raise IntegrityError(
                    f"worker {request.capability.worker_id} identity is not trusted"
                )
            self._registration_nonce_cache.add(request.benchmark_nonce)
            assert self.durable_state is not None
            self.durable_state.save_worker(request.capability)
        self.registry.register(request.capability, benchmark_verified=True)
        self._last_heartbeat_sequences.pop(request.capability.worker_id, None)
        self.events.append(
            {
                "event_type": "worker_registered",
                "worker_id": request.capability.worker_id,
                "endpoint": request.capability.endpoint,
                "timestamp_monotonic_s": time.monotonic(),
                "timestamp_monotonic_ns": time.monotonic_ns(),
            }
        )
        if self.product_telemetry is not None:
            self.product_telemetry.emit(
                "worker_registered",
                worker_id=request.capability.worker_id,
                worker_public_key_fingerprint=fingerprint,
                control_endpoint=(
                    request.capability.control_endpoint or request.capability.endpoint
                ),
                data_endpoint=request.capability.data_plane_endpoint,
            )
            if self.durable_state is not None:
                self.durable_state.append_audit_event(
                    "worker_registered",
                    worker_id=request.capability.worker_id,
                    worker_public_key_fingerprint=fingerprint,
                )
        if not self.product_mode:
            await self.rebalance()
        identity = self.coordinator_identity
        return RegistrationResponse(
            accepted=True,
            heartbeat_interval_s=2.0,
            coordinator_identity=(
                self.product_config.coordinator_id if self.product_config is not None else None
            ),
            coordinator_public_key=(identity.public_key_b64 if identity is not None else None),
            coordinator_public_key_fingerprint=(
                identity.public_key_fingerprint if identity is not None else None
            ),
        )

    def remove_worker(self, worker_id: str) -> None:
        """Remove an exited process so a cached-cold replacement can own its stage."""

        self.registry.remove_worker(worker_id)
        self._assigned = {
            (stage_id, assigned_worker)
            for stage_id, assigned_worker in self._assigned
            if assigned_worker != worker_id
        }
        self.events.append(
            {
                "event_type": "worker_removed",
                "worker_id": worker_id,
                "timestamp_monotonic_ns": time.monotonic_ns(),
            }
        )

    async def heartbeat(self, request: Heartbeat) -> Ack:
        capability = self.registry.capability(request.worker_id)
        signed_payload = canonical_json_bytes(
            {
                "worker_id": request.worker_id,
                "queue_depth": request.queue_depth,
                "assignments": request.assignments,
                "monotonic_ns": request.monotonic_ns,
                "timestamp": request.timestamp.isoformat(),
            }
        )
        verify_signature(capability.public_key, signed_payload, request.signature)
        previous = self._last_heartbeat_sequences.get(request.worker_id)
        if previous is not None and request.monotonic_ns <= previous:
            raise IntegrityError("stale or replayed worker heartbeat")
        now_utc = datetime.now(UTC)
        age_s = (now_utc - request.timestamp).total_seconds()
        future_tolerance = (
            self.product_config.route_future_tolerance_s
            if self.product_config is not None
            else 30.0
        )
        if age_s < -future_tolerance:
            raise IntegrityError("worker heartbeat is future-dated outside tolerance")
        maximum_age = (
            self.product_config.worker_heartbeat_timeout_s * 2
            if self.product_config is not None
            else 30.0
        )
        if age_s > maximum_age:
            raise IntegrityError("worker heartbeat is stale")
        self.registry.heartbeat(
            request.worker_id,
            queue_depth=request.queue_depth,
            assignments=request.assignments,
        )
        self._last_heartbeat_sequences[request.worker_id] = request.monotonic_ns
        return Ack(accepted=True, detail="heartbeat recorded")

    def _require_product(
        self,
    ) -> tuple[
        ProductModelCatalog,
        ProductStagePlanner,
        DeploymentManager,
        ProductSessionController,
    ]:
        if (
            self.product_catalog is None
            or self.product_planner is None
            or self.deployment_manager is None
            or self.session_controller is None
        ):
            raise RuntimeError("coordinator was not started in product stage-ring mode")
        return (
            self.product_catalog,
            self.product_planner,
            self.deployment_manager,
            self.session_controller,
        )

    async def inspect_product_model(
        self,
        request: ModelInspectRequest,
    ) -> ModelInspectResponse:
        catalog, _, _, _ = self._require_product()
        inspected = await catalog.inspect(request.reference)
        return ModelInspectResponse(
            spec=inspected.spec,
            metadata=inspected.metadata,
            worker_eligibility=list(inspected.eligibility),
        )

    async def plan_product_model(self, request: ModelPlanRequest) -> ModelPlanResponse:
        catalog, planner, _, _ = self._require_product()
        inspected = await catalog.inspect(request.reference)
        plan = planner.build_plan(request, inspected)
        plan_directory = self.state_directory / "plans"
        plan_directory.mkdir(parents=True, exist_ok=True)
        (plan_directory / f"{plan.plan_id}.json").write_text(
            plan.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        return ModelPlanResponse(plan=plan)

    async def deploy_product_model(
        self,
        request: ModelDeployRequest,
    ) -> ModelDeployResponse:
        _, _, deployments, _ = self._require_product()
        if self.publication_endpoint is None:
            raise RuntimeError("coordinator publication endpoint is not configured")
        deployment = await deployments.deploy(
            request.plan,
            publication_destination=self.publication_endpoint,
        )
        return ModelDeployResponse(deployment=deployment)

    async def unload_product_model(
        self,
        request: ModelUnloadRequest,
    ) -> ModelUnloadResponse:
        _, _, deployments, _ = self._require_product()
        deployment = await deployments.unload(
            topology_id=request.topology_id,
            force=request.force,
        )
        return ModelUnloadResponse(
            deployment=deployment,
            detail=(
                "no deployment was active"
                if deployment is None
                else "deployment unloaded explicitly"
            ),
        )

    async def product_topology_status(
        self,
        request: TopologyStatusRequest,
    ) -> TopologyStatusResponse:
        _, _, deployments, _ = self._require_product()
        return TopologyStatusResponse(deployments=deployments.statuses(request.topology_id))

    async def product_workers(self, request: WorkersRequest) -> WorkersResponse:
        self._require_product()
        statuses: list[WorkerProductStatus] = []

        async def inspect_worker(capability: Any) -> WorkerProductStatus | None:
            healthy, age = self.registry.registration_health(capability.worker_id)
            if not request.include_unhealthy and not healthy:
                return None
            loaded_stages = []
            active_sessions = capability.active_session_count
            queue_depths = {"worker": capability.current_queue_depth}
            memory_bytes = {
                "effective": capability.effective_memory_bytes,
                "available_ram": capability.available_ram_bytes,
                "available_vram": capability.available_vram_bytes,
            }
            expert_status: dict[str, Any] = {
                "roles": [item.value for item in capability.roles],
                "expert_data_plane_endpoint": capability.expert_data_plane_endpoint,
                "owned_experts": capability.owned_experts,
                "owned_microshards": capability.owned_microshards,
                "cache_resident_bytes": capability.expert_cache_resident_bytes,
                "cache_hits": capability.expert_cache_hits,
                "cache_misses": capability.expert_cache_misses,
                "remote_whole_expert_calls": capability.remote_expert_calls,
                "remote_microshard_calls": capability.remote_microshard_calls,
                "bytes_transferred": capability.expert_bytes_transferred,
                "expert_critical_path_ns": capability.expert_critical_path_ns,
                "supported_reduction_modes": capability.supported_reduction_modes,
            }
            last_error = None
            control_endpoint = capability.control_endpoint or capability.endpoint
            if capability.stage_runtime_enabled and control_endpoint is not None:
                try:
                    worker_status = await cast(Any, self.transport).get_stage_status(
                        control_endpoint,
                        GetStageStatusRequest(
                            worker_id=capability.worker_id,
                            request_id=f"workers-status:{capability.worker_id}:{uuid4().hex}",
                        ),
                    )
                    if worker_status.loaded_stage is not None:
                        loaded_stages.append(worker_status.loaded_stage)
                    active_sessions = len(worker_status.sessions)
                    queue_depths.update(
                        {
                            "execution": worker_status.execution_queue_depth,
                            "token_publication": worker_status.token_queue_depth,
                        }
                    )
                    if worker_status.loaded_stage is not None:
                        loaded = worker_status.loaded_stage
                        memory_bytes.update(
                            {
                                "process_rss": loaded.process_rss_after_bytes,
                                "cuda_allocated": loaded.cuda_allocated_after_bytes,
                                "cuda_reserved": loaded.cuda_reserved_after_bytes,
                            }
                        )
                    if worker_status.expert_status:
                        expert_status["stage_backend"] = dict(worker_status.expert_status)
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
            if capability.expert_data_plane_endpoint is not None:
                try:
                    from swarm_inference.transport.expert import ExpertTransportClient

                    expert_client = ExpertTransportClient(
                        capability.expert_data_plane_endpoint, timeout_s=2.0
                    )
                    expert_response = await asyncio.to_thread(expert_client.control, "status")
                    live_status = expert_response.get("status")
                    if isinstance(live_status, dict):
                        expert_status.update(live_status)
                except Exception as exc:
                    expert_error = f"{type(exc).__name__}: {exc}"
                    last_error = (
                        f"{last_error}; expert status: {expert_error}"
                        if last_error is not None
                        else expert_error
                    )
            return WorkerProductStatus(
                capability=capability,
                healthy_registration=healthy,
                heartbeat_age_s=age,
                last_heartbeat_unix_ns=int(capability.last_heartbeat.timestamp() * 1e9),
                control_endpoint=control_endpoint,
                data_endpoint=capability.data_plane_endpoint,
                loaded_stages=loaded_stages,
                active_sessions=active_sessions,
                queue_depths=queue_depths,
                memory_bytes=memory_bytes,
                expert_status=expert_status,
                last_error=last_error,
                detail=(
                    "healthy"
                    if healthy and last_error is None
                    else last_error or "heartbeat expired"
                ),
            )

        inspected = await asyncio.gather(
            *(
                inspect_worker(item)
                for item in sorted(self.registry.workers(), key=lambda x: x.worker_id)
            )
        )
        statuses.extend(item for item in inspected if item is not None)
        return WorkersResponse(workers=statuses)

    async def product_sessions(self, request: SessionsRequest) -> SessionsResponse:
        _, _, _, sessions = self._require_product()
        return SessionsResponse(
            sessions=await sessions.statuses(include_terminal=request.include_terminal)
        )

    async def cancel_product_request(
        self,
        request: CancelProductRequest,
    ) -> CancelProductResponse:
        _, _, _, sessions = self._require_product()
        return await sessions.cancel(request.request_id)

    async def product_status(
        self,
        request: CoordinatorStatusRequest,
    ) -> CoordinatorStatusResponse:
        del request
        _, _, deployments, sessions = self._require_product()
        identity = self.coordinator_identity
        assert identity is not None and self.product_config is not None
        worker_response = await self.product_workers(WorkersRequest(include_unhealthy=True))
        session_values = await sessions.statuses(include_terminal=True)
        active = [
            item
            for item in session_values
            if item.status
            in {
                ProductRequestPhase.PENDING,
                ProductRequestPhase.RUNNING,
                ProductRequestPhase.RECOVERING,
            }
        ]
        ttft_values = [
            item.time_to_first_token_s
            for item in session_values
            if item.time_to_first_token_s is not None
        ]
        inter_values = [
            item.inter_token_latency_s
            for item in session_values
            if item.inter_token_latency_s is not None
        ]
        uptime = max(0.0, time.monotonic() - self.started_monotonic_s)
        queue_depths: dict[str, int] = {"client_events": sessions.queue_depth}
        memory_bytes: dict[str, int] = {}
        for worker in worker_response.workers:
            for name, value in worker.queue_depths.items():
                queue_depths[f"{worker.capability.worker_id}:{name}"] = value
            for name, value in worker.memory_bytes.items():
                memory_bytes[f"{worker.capability.worker_id}:{name}"] = value
        deployment_values = deployments.statuses()
        expert_workers = [
            item
            for item in worker_response.workers
            if item.capability.expert_data_plane_endpoint is not None
        ]

        def expert_total(name: str) -> int:
            return sum(int(item.expert_status.get(name, 0)) for item in expert_workers)

        def stage_expert_total(name: str) -> int:
            return sum(
                int(stage_status.get(name, 0))
                for item in worker_response.workers
                if isinstance(stage_status := item.expert_status.get("stage_backend"), dict)
            )

        owned_expert_count = sum(
            sum(len(experts) for experts in item.capability.owned_experts.values())
            for item in expert_workers
        )
        owned_microshard_count = sum(
            len(item.capability.owned_microshards) for item in expert_workers
        )
        reduction_modes = sorted(
            {mode for item in expert_workers for mode in item.capability.supported_reduction_modes}
        )
        last_error = next(
            (item.last_error for item in reversed(session_values) if item.last_error is not None),
            next(
                (
                    item.detail
                    for item in reversed(deployment_values)
                    if item.phase == DeploymentPhase.FAILED
                ),
                None,
            ),
        )
        return CoordinatorStatusResponse(
            coordinator_identity=self.product_config.coordinator_id,
            coordinator_public_key_fingerprint=identity.public_key_fingerprint,
            uptime_s=uptime,
            state_directory=str(self.state_directory),
            registered_worker_count=len(worker_response.workers),
            healthy_worker_count=sum(
                1 for item in worker_response.workers if item.healthy_registration
            ),
            known_deployments=len(deployment_values),
            active_topology_id=deployments.current_topology_id,
            route_generation=deployments.current_generation,
            active_session_count=len(active),
            generated_tokens=sessions.generated_token_count,
            throughput_tokens_s=(sessions.generated_token_count / uptime if uptime > 0 else 0.0),
            time_to_first_token_s=(sum(ttft_values) / len(ttft_values) if ttft_values else None),
            inter_token_latency_s=(sum(inter_values) / len(inter_values) if inter_values else None),
            recovery_count=sessions.recovery_count,
            recovering_requests=sum(
                1 for item in active if item.status == ProductRequestPhase.RECOVERING
            ),
            queue_depths=queue_depths,
            memory_bytes=memory_bytes,
            reservations={
                "deployment_worker_ids": list(deployments.reserved_worker_ids),
                "active_sessions": len(active),
            },
            expert_worker_count=len(expert_workers),
            owned_experts=owned_expert_count,
            owned_microshards=owned_microshard_count,
            expert_cache_resident_bytes=expert_total("cache_resident_bytes"),
            expert_cache_hits=expert_total("cache_hits"),
            expert_cache_misses=expert_total("cache_misses"),
            remote_expert_calls=expert_total("remote_whole_expert_calls"),
            remote_microshard_calls=expert_total("remote_microshard_calls"),
            expert_fallbacks=stage_expert_total("fallbacks"),
            expert_bytes_transferred=sum(
                int(item.expert_status.get("bytes_received", 0))
                + int(item.expert_status.get("bytes_sent", 0))
                for item in expert_workers
            ),
            expert_critical_path_ns=(
                stage_expert_total("expert_critical_path_ns") or expert_total("compute_ns")
            ),
            expert_reduction_modes=sorted(
                {
                    *reduction_modes,
                    *(
                        str(stage_status["reduction_mode"])
                        for item in worker_response.workers
                        if isinstance(
                            stage_status := item.expert_status.get("stage_backend"),
                            dict,
                        )
                        and stage_status.get("reduction_mode") not in {None, "none"}
                    ),
                }
            ),
            last_error=last_error,
        )

    async def accept_product_token(self, publication: ProductTokenPublication) -> Ack:
        _, _, _, sessions = self._require_product()
        return await sessions.publish_token(publication)

    async def rebalance(self) -> None:
        async with self._rebalance_lock:
            workers = self.registry.workers()
            if not workers:
                return
            assigned_worker_ids = {worker_id for _, worker_id in self._assigned}
            remaining = [
                worker for worker in workers if worker.worker_id not in assigned_worker_ids
            ]
            replica_counts = {
                stage.stage_id: len(self.registry.replicas(stage.stage_id)) for stage in self.stages
            }
            assignments: list[tuple[Any, Any, float]] = []

            # First establish complete coverage. Workers are selected by their
            # measured stage benchmark, not a declared relative-speed field.
            for stage in sorted(
                self.stages,
                key=lambda item: (replica_counts[item.stage_id], item.stage_id),
            ):
                if replica_counts[stage.stage_id] > 0:
                    continue
                candidates = [
                    worker for worker in remaining if self._worker_can_host(worker, stage)
                ]
                if not candidates:
                    self.events.append(
                        {
                            "event_type": "placement_pending",
                            "detail": f"no compatible worker for uncovered stage {stage.stage_id}",
                            "worker_count": len(workers),
                        }
                    )
                    return
                capability = max(
                    candidates,
                    key=lambda worker: (
                        estimate_worker_stage_rate(worker, stage),
                        worker.worker_id,
                    ),
                )
                rate = estimate_worker_stage_rate(capability, stage)
                assignments.append((capability, stage, rate))
                remaining.remove(capability)
                replica_counts[stage.stage_id] += 1

            # A tied set of bottleneck stages needs a complete replica round
            # before min pipeline capacity rises. Keep an incomplete round idle
            # rather than claiming a positive marginal throughput benefit.
            while len(remaining) >= len(self.stages):
                round_assignments: list[tuple[Any, Any, float]] = []
                round_workers = list(remaining)
                for stage in sorted(
                    self.stages,
                    key=lambda item: (replica_counts[item.stage_id], item.stage_id),
                ):
                    candidates = [
                        worker for worker in round_workers if self._worker_can_host(worker, stage)
                    ]
                    if not candidates:
                        round_assignments = []
                        break
                    capability = max(
                        candidates,
                        key=lambda worker: (
                            estimate_worker_stage_rate(worker, stage),
                            worker.worker_id,
                        ),
                    )
                    rate = estimate_worker_stage_rate(capability, stage)
                    round_assignments.append((capability, stage, rate))
                    round_workers.remove(capability)
                if not round_assignments:
                    break
                assignments.extend(round_assignments)
                for capability, stage, _ in round_assignments:
                    remaining.remove(capability)
                    replica_counts[stage.stage_id] += 1

            for capability, stage, rate in assignments:
                key = (stage.stage_id, capability.worker_id)
                if capability.endpoint is None:
                    self.events.append(
                        {
                            "event_type": "placement_rejected",
                            "worker_id": capability.worker_id,
                            "stage_id": stage.stage_id,
                            "detail": "worker has no advertised endpoint",
                        }
                    )
                    continue
                if self.model_manifest is None:
                    shard_hash = "synthetic-deterministic"
                    assignment = StageAssignmentMessage(
                        worker_id=capability.worker_id,
                        stage=stage,
                        shard_path="synthetic://deterministic",
                        shard_hash=shard_hash,
                        model_id=self.runtime_model_id,
                        model_revision=self.runtime_model_revision,
                        synthetic_model=self.config.model,
                        data_plane_mode=self.config.data_plane,
                        coordinator_data_endpoint=self.final_result_endpoint,
                        route_signing_key=(
                            encode_route_key(self.route_signing_key)
                            if self.config.data_plane == DataPlaneMode.DIRECT
                            else None
                        ),
                    )
                else:
                    shard_name = f"stage-{stage.stage_id:03d}"
                    try:
                        shard_hash = self.model_manifest.shard_hashes[shard_name]
                    except KeyError as exc:
                        raise IntegrityError(
                            f"model manifest has no hash for required shard {shard_name}"
                        ) from exc
                    assignment = StageAssignmentMessage(
                        worker_id=capability.worker_id,
                        stage=stage,
                        shard_path=shard_name,
                        shard_hash=shard_hash,
                        model_id=self.runtime_model_id,
                        model_revision=self.runtime_model_revision,
                        architecture_config=self.architecture_config,
                        model_manifest=self.model_manifest,
                        dtype=self.runtime_dtype,
                        data_plane_mode=self.config.data_plane,
                        coordinator_data_endpoint=self.final_result_endpoint,
                        route_signing_key=(
                            encode_route_key(self.route_signing_key)
                            if self.config.data_plane == DataPlaneMode.DIRECT
                            else None
                        ),
                    )
                try:
                    ack = await self.transport.assign(capability.endpoint, assignment)
                except TransportError as exc:
                    self.events.append(
                        {
                            "event_type": "assignment_failed",
                            "worker_id": capability.worker_id,
                            "stage_id": stage.stage_id,
                            "detail": str(exc),
                        }
                    )
                    continue
                if not ack.accepted:
                    continue
                replica_service_rate = (
                    1000.0 / self.config.synthetic_compute.target_stage_ms
                    if self.model_manifest is None
                    and self.config.synthetic_compute.mode == "calibrated_cpu"
                    else rate
                )
                replica = StageReplica(
                    stage_id=stage.stage_id,
                    worker_id=capability.worker_id,
                    shard_hash=shard_hash,
                    load_status="loaded",
                    warm=True,
                    measured_service_rate=replica_service_rate,
                    health=HealthStatus.HEALTHY,
                    endpoint=capability.endpoint,
                )
                self.registry.add_replica(replica)
                self._assigned.add(key)
                self.events.append(
                    {
                        "event_type": "stage_assigned",
                        "worker_id": replica.worker_id,
                        "stage_id": replica.stage_id,
                        "predicted_service_rate": replica_service_rate,
                        "marginal_basis": (
                            "required coverage"
                            if replica_counts[stage.stage_id] == 1
                            else "complete balanced replica round"
                        ),
                    }
                )
            for capability in remaining:
                self.events.append(
                    {
                        "event_type": "worker_left_idle",
                        "worker_id": capability.worker_id,
                        "detail": "incomplete replica round has non-positive immediate pipeline gain",
                    }
                )

    def stage_coverage(self) -> dict[int, int]:
        return {
            stage.stage_id: len(
                [
                    replica
                    for replica in self.registry.replicas(stage.stage_id)
                    if replica.health == HealthStatus.HEALTHY
                ]
            )
            for stage in self.stages
        }

    async def wait_for_coverage(
        self,
        *,
        minimum_replicas: int = 1,
        timeout_s: float = 30.0,
    ) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            await self.rebalance()
            if all(value >= minimum_replicas for value in self.stage_coverage().values()):
                return
            await asyncio.sleep(0.05)
        raise NoValidRouteError(
            f"stage coverage did not reach {minimum_replicas} replica(s): {self.stage_coverage()}"
        )

    def _choose_route(self) -> dict[int, StageReplica]:
        route: dict[int, StageReplica] = {}
        for stage in self.stages:
            replicas = sorted(
                [
                    replica
                    for replica in self.registry.replicas(stage.stage_id)
                    if replica.health == HealthStatus.HEALTHY and replica.endpoint is not None
                ],
                key=lambda replica: (
                    replica.queue_depth / max(replica.measured_service_rate, 1e-12),
                    -replica.reputation,
                    replica.worker_id,
                ),
            )
            if not replicas:
                raise NoValidRouteError(f"no healthy replica for stage {stage.stage_id}")
            route[stage.stage_id] = replicas[0]
        return route

    def _reserve_request_route(
        self,
        submission: SubmitRequest,
        workload_class: WorkloadClass,
    ) -> ReservationDecision:
        expired = self.route_allocator.reconcile_expired()
        if expired:
            self.events.append(
                {
                    "event_type": "route_leases_reconciled",
                    "route_ids": expired,
                }
            )
        decision = self.route_allocator.allocate(
            request_id=submission.request_id,
            stages=self.stages,
            replicas=self.registry.replicas(),
            workers=self.registry.workers(),
            token_steps=submission.max_new_tokens,
            activation_bytes=self.config.model.activation_bytes,
            workload_class=workload_class,
            lease_seconds=self.config.worker.route_lease_seconds,
        )
        self.runtime_transport_metrics["route_reservation_time_ms"] = (
            float(self.runtime_transport_metrics["route_reservation_time_ms"])
            + decision.reservation_time_ms
        )
        self.events.append(
            {
                "event_type": "route_reserved",
                "request_id": submission.request_id,
                "route_id": decision.route_id,
                "route_generation": decision.generation,
                "assignments": {
                    str(stage_id): replica.worker_id
                    for stage_id, replica in sorted(decision.assignments.items())
                },
                "candidate_costs_ms": decision.candidate_costs_ms,
                "reservation_time_ms": decision.reservation_time_ms,
                "timestamp_monotonic_ns": time.monotonic_ns(),
            }
        )
        return decision

    def _route_plan(
        self,
        submission: SubmitRequest,
        decision: ReservationDecision,
    ) -> RoutePlan:
        if self.final_result_endpoint is None:
            raise TransportError("coordinator final-result endpoint is not configured")
        stages = {stage.stage_id: stage for stage in self.stages}
        hops = [
            RouteHop(
                stage_id=stage_id,
                worker_id=replica.worker_id,
                worker_data_endpoint=replica.endpoint or "",
                expected_shard_hash=replica.shard_hash,
                expected_input_spec=stages[stage_id].input_spec,
                expected_output_spec=stages[stage_id].output_spec,
            )
            for stage_id, replica in sorted(decision.assignments.items())
        ]
        unsigned = RoutePlan(
            route_id=decision.route_id,
            route_generation=decision.generation,
            request_id=submission.request_id,
            model_id=self.runtime_model_id,
            model_revision=self.runtime_model_revision,
            assignments=hops,
            route_lease_expiry_unix_ns=(
                time.time_ns() + int(self.config.worker.route_lease_seconds * 1_000_000_000)
            ),
            workload_class=submission.workload_class,
            cancellation_generation=0,
            integrity_policy="hmac-sha256+activation-sha256+worker-ed25519",
            final_result_destination=self.final_result_endpoint,
        )
        return sign_route_plan(unsigned, self.route_signing_key)

    async def _install_direct_route(self, route: RoutePlan) -> None:
        requests = [
            (
                hop,
                RouteInstallRequest(
                    worker_id=hop.worker_id,
                    route=route,
                ),
            )
            for hop in route.assignments
        ]
        serialized_route_bytes = len(serialize_message(route))
        self.runtime_transport_metrics["coordinator_control_bytes"] = int(
            self.runtime_transport_metrics["coordinator_control_bytes"]
        ) + serialized_route_bytes * len(requests)
        try:
            acknowledgements = await asyncio.gather(
                *(
                    self.transport.install_route(hop.worker_data_endpoint, request)
                    for hop, request in requests
                )
            )
        except Exception:
            self.route_allocator.dispatch_failed(route.route_id)
            raise
        rejected = [ack.detail for ack in acknowledgements if not ack.accepted]
        if rejected:
            self.route_allocator.dispatch_failed(route.route_id)
            raise TransportError(
                f"route {route.route_id} installation rejected: {'; '.join(rejected)}"
            )

    async def _cancel_worker_request_state(
        self,
        request_id: str,
        model_revision: str,
    ) -> None:
        from swarm_inference.protocol.messages import CancelRequest

        endpoints = {
            replica.endpoint for replica in self.registry.replicas() if replica.endpoint is not None
        }
        await asyncio.gather(
            *(
                self.transport.cancel(
                    endpoint,
                    CancelRequest(
                        request_id=request_id,
                        model_revision=model_revision,
                    ),
                )
                for endpoint in endpoints
            ),
            return_exceptions=True,
        )

    async def _failed_route_stage_ids(self, route: RoutePlan) -> list[int]:
        async def probe(hop: RouteHop) -> tuple[int, bool]:
            try:
                health = await self.transport.health(hop.worker_data_endpoint)
            except Exception:
                return hop.stage_id, True
            return hop.stage_id, not health.healthy

        results = await asyncio.gather(*(probe(hop) for hop in route.assignments))
        return sorted(stage_id for stage_id, failed in results if failed)

    async def _dispatch_direct_replay(
        self,
        *,
        submission: SubmitRequest,
        route: RoutePlan,
        entry: ReplayEntry,
    ) -> None:
        first = route.assignments[0]
        decoded = decode_tensor(entry.payload, copy=False)
        metadata = ActivationMetadata(
            request_id=submission.request_id,
            tensor_id=f"{submission.request_id}:{entry.token_position}:0:replay",
            stage_id=0,
            operation=entry.operation,
            token_position=entry.token_position,
            sequence_length=decoded.sequence_length,
            cache_generation=entry.cache_generation,
            route_generation=route.route_generation,
            model_id=self.runtime_model_id,
            model_revision=self.runtime_model_revision,
        )
        replay = sign_data_envelope(
            DataPlaneEnvelope(
                message_id=(
                    f"replay:{route.route_id}:{route.route_generation}:{entry.token_position}:0"
                ),
                route_id=route.route_id,
                route_generation=route.route_generation,
                request_id=submission.request_id,
                stage_id=0,
                source_worker="coordinator",
                destination_worker=first.worker_id,
                token_position=entry.token_position,
                operation=entry.operation,
                tensor_metadata=metadata,
                tensor_payload=entry.payload,
                payload_length=len(entry.payload),
                payload_checksum=entry.checksum,
                sequence_number=0,
                timestamp_unix_ns=time.time_ns(),
                replay_only=True,
            ),
            self.route_signing_key,
        )
        ack = await self.transport.dispatch(first.worker_data_endpoint, replay)
        if not ack.accepted:
            raise TransportError(f"direct replay rejected ({ack.status}): {ack.detail}")

    async def _recover_direct_route(
        self,
        *,
        submission: SubmitRequest,
        request_state: RequestState,
        route: RoutePlan,
        metrics: RuntimeRequestMetrics,
        committed_through_token_position: int,
        failure: BaseException,
        known_failed_stage_ids: list[int] | None = None,
    ) -> RoutePlan:
        failed_stage_ids = (
            sorted(set(known_failed_stage_ids))
            if known_failed_stage_ids is not None
            else await self._failed_route_stage_ids(route)
        )
        route_stage_ids = {hop.stage_id for hop in route.assignments}
        if any(stage_id not in route_stage_ids for stage_id in failed_stage_ids):
            raise TransportError(
                f"known failed stages {failed_stage_ids} are outside route {route.route_id}"
            )
        if not failed_stage_ids:
            raise TransportError(
                f"direct hop failed but every assigned worker remains healthy: {failure}"
            ) from failure
        failed_workers: list[str] = []
        replacements: dict[int, str] = {}
        for stage_id in failed_stage_ids:
            failed_worker_id = route.assignments[stage_id].worker_id
            failed_workers.append(failed_worker_id)
            self.registry.mark_unhealthy(failed_worker_id)
            self.route_allocator.mark_failed(stage_id, failed_worker_id)
            replacement, _ = self.route_allocator.replace_failed(
                route.route_id,
                stage_id=stage_id,
                candidates=self.registry.replicas(stage_id),
            )
            replacements[stage_id] = replacement.worker_id

        lease = self.route_allocator.lease(route.route_id)
        decision = ReservationDecision(
            route_id=lease.route_id,
            generation=lease.generation,
            assignments=lease.assignments,
            candidate_costs_ms={},
            reservation_time_ms=0,
            lease_expiry_monotonic_s=lease.expires_monotonic_s,
        )
        replacement_route = self._route_plan(submission, decision)
        await self._cancel_worker_request_state(
            submission.request_id,
            self.runtime_model_revision,
        )
        await self._install_direct_route(replacement_route)

        replay_started = time.perf_counter()
        replay_bytes = 0
        if committed_through_token_position >= 0:
            entries = self.replay.entries_for(
                request_id=submission.request_id,
                model_revision=self.runtime_model_revision,
                stage_id=0,
                cache_generation=0,
                through_token_position=committed_through_token_position,
            )
            for entry in entries:
                replay_bytes += len(entry.payload)
                await self._dispatch_direct_replay(
                    submission=submission,
                    route=replacement_route,
                    entry=entry,
                )
        replay_elapsed = time.perf_counter() - replay_started
        metrics.replay_s += replay_elapsed
        metrics.replay_bytes += replay_bytes
        metrics.retries += 1
        metrics.route_changes += 1
        metrics.route_generation = replacement_route.route_generation
        request_state.retry_count += 1
        request_state.stage_route = [hop.worker_id for hop in replacement_route.assignments]
        request_state.stage_local_cache_ownership = {
            hop.stage_id: hop.worker_id for hop in replacement_route.assignments
        }
        self.events.append(
            {
                "event_type": "stage_recovered",
                "request_id": submission.request_id,
                "route_id": route.route_id,
                "old_route_generation": route.route_generation,
                "route_generation": replacement_route.route_generation,
                "failed_stage_ids": failed_stage_ids,
                "failed_worker_ids": failed_workers,
                "replacements": replacements,
                "replay_bytes": replay_bytes,
                "replay_duration_s": replay_elapsed,
                "failure": str(failure),
                "data_plane_mode": DataPlaneMode.DIRECT.value,
            }
        )
        return replacement_route

    async def accept_final_result(self, message: FinalResultMessage) -> Ack:
        try:
            verify_final_result(message, self.route_signing_key)
            if sha256_bytes(message.result.tensor_payload) != message.payload_checksum:
                raise IntegrityError(f"final result {message.message_id} checksum mismatch")
            lease = self.route_allocator.lease(message.route_id)
            if lease.released:
                raise IntegrityError(f"route {message.route_id} is already released")
            if (
                lease.request_id != message.request_id
                or lease.generation != message.route_generation
            ):
                raise IntegrityError(
                    f"stale final result route generation "
                    f"{message.route_generation}; current={lease.generation}"
                )
            final_replica = lease.assignments[max(lease.assignments)]
            if message.result.worker_id != final_replica.worker_id:
                raise IntegrityError(
                    f"final result came from {message.result.worker_id}, expected "
                    f"{final_replica.worker_id}"
                )
            if message.result.metadata.stage_id != max(lease.assignments):
                raise IntegrityError("final result did not come from the final stage")
            capability = self.registry.capability(message.result.worker_id)
            signed_payload = canonical_json_bytes(
                {
                    "worker_id": message.result.worker_id,
                    "request_id": message.result.metadata.request_id,
                    "stage_id": message.result.metadata.stage_id,
                    "token_position": message.result.metadata.token_position,
                    "checksum": message.result.checksum,
                }
            )
            verify_signature(
                capability.public_key,
                signed_payload,
                message.result.signature,
            )
            commit_key = (message.request_id, message.token_position)
            committed_generation = self._committed_route_generations.get(commit_key)
            if (
                committed_generation is not None
                and committed_generation != message.route_generation
            ):
                raise IntegrityError(
                    f"token {message.token_position} already committed by route "
                    f"generation {committed_generation}"
                )
            pending_key = (
                message.route_id,
                message.route_generation,
                message.token_position,
            )
            future = self._pending_final_results.get(pending_key)
            if future is None:
                if committed_generation == message.route_generation:
                    return Ack(accepted=True, detail="duplicate final result")
                raise IntegrityError(f"no pending token for final result {message.message_id}")
            if message.hop_telemetry:
                last = message.hop_telemetry[-1]
                transfer_ms = max(
                    0.0,
                    (time.time_ns() - message.timestamp_unix_ns) / 1_000_000,
                )
                message.hop_telemetry[-1] = last.model_copy(
                    update={
                        "transfer_ms": transfer_ms,
                        "hop_end_to_end_ms": last.hop_end_to_end_ms + transfer_ms,
                    }
                )
            self._committed_route_generations[commit_key] = message.route_generation
            if not future.done():
                future.set_result(message)
            return Ack(accepted=True, detail="final result committed")
        except Exception as exc:
            self.events.append(
                {
                    "event_type": "final_result_rejected",
                    "route_id": message.route_id,
                    "request_id": message.request_id,
                    "token_position": message.token_position,
                    "detail": str(exc),
                }
            )
            return Ack(accepted=False, detail=str(exc))

    async def _call_direct_route(
        self,
        *,
        submission: SubmitRequest,
        route: RoutePlan,
        metrics: RuntimeRequestMetrics,
        operation: OperationKind,
        token_position: int,
        sequence_length: int,
        encoded: bytes,
    ) -> ActivationResult:
        first = route.assignments[0]
        metadata = ActivationMetadata(
            request_id=submission.request_id,
            tensor_id=f"{submission.request_id}:{token_position}:0",
            stage_id=0,
            operation=operation,
            token_position=token_position,
            sequence_length=sequence_length,
            cache_generation=0,
            route_generation=route.route_generation,
            model_id=self.runtime_model_id,
            model_revision=self.runtime_model_revision,
        )
        unsigned = DataPlaneEnvelope(
            message_id=(f"{route.route_id}:{route.route_generation}:{token_position}:0"),
            route_id=route.route_id,
            route_generation=route.route_generation,
            request_id=submission.request_id,
            stage_id=0,
            source_worker="coordinator",
            destination_worker=first.worker_id,
            token_position=token_position,
            operation=operation,
            tensor_metadata=metadata,
            tensor_payload=encoded,
            payload_length=len(encoded),
            payload_checksum=sha256_bytes(encoded),
            sequence_number=0,
            timestamp_unix_ns=time.time_ns(),
        )
        envelope = sign_data_envelope(unsigned, self.route_signing_key)
        pending_key = (
            route.route_id,
            route.route_generation,
            token_position,
        )
        loop = asyncio.get_running_loop()
        future: asyncio.Future[FinalResultMessage] = loop.create_future()
        self._pending_final_results[pending_key] = future
        serialized_bytes = len(serialize_message(envelope))
        self.runtime_transport_metrics["coordinator_control_bytes"] = int(
            self.runtime_transport_metrics["coordinator_control_bytes"]
        ) + max(0, serialized_bytes - len(encoded))
        self.runtime_transport_metrics["coordinator_input_activation_bytes"] = int(
            self.runtime_transport_metrics["coordinator_input_activation_bytes"]
        ) + len(encoded)
        for hop in route.assignments:
            self.route_allocator.record_operation_start(
                hop.stage_id,
                hop.worker_id,
                len(encoded),
            )
        started = time.perf_counter_ns()
        completed_path = False
        try:
            dispatch_ack = await self.transport.dispatch(
                first.worker_data_endpoint,
                envelope,
            )
            if not dispatch_ack.accepted:
                raise TransportError(
                    f"direct route rejected ({dispatch_ack.status}): {dispatch_ack.detail}"
                )
            final = await asyncio.wait_for(
                future,
                timeout=self.config.queue.request_deadline_ms / 1000,
            )
            completed_path = True
        finally:
            self._pending_final_results.pop(pending_key, None)
            if not completed_path:
                for hop in route.assignments:
                    self.route_allocator.record_operation_aborted(
                        hop.stage_id,
                        hop.worker_id,
                        len(encoded),
                    )
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        self.runtime_transport_metrics["coordinator_final_result_bytes"] = int(
            self.runtime_transport_metrics["coordinator_final_result_bytes"]
        ) + len(final.result.tensor_payload)
        self.runtime_transport_metrics["worker_to_worker_activation_bytes"] = int(
            self.runtime_transport_metrics["worker_to_worker_activation_bytes"]
        ) + sum(item.payload_bytes for item in final.hop_telemetry[:-1])
        peer_messages = max(0, len(final.hop_telemetry) - 1)
        self.runtime_transport_metrics["data_messages_sent"] = (
            int(self.runtime_transport_metrics["data_messages_sent"]) + peer_messages
        )
        self.runtime_transport_metrics["data_messages_received"] = (
            int(self.runtime_transport_metrics["data_messages_received"]) + peer_messages
        )
        total_execution_ms = 0.0
        total_queue_ms = 0.0
        for item in final.hop_telemetry:
            total_execution_ms += item.execution_ms
            total_queue_ms += item.queue_ms
            effective_ms = item.effective_service_ms
            self.route_allocator.record_operation_complete(
                item.stage_id,
                item.worker_id,
                payload_bytes=item.payload_bytes,
                effective_service_ms=effective_ms,
                execution_ms=item.execution_ms,
            )
            metrics.per_stage.append(
                {
                    "stage_id": item.stage_id,
                    "worker_id": item.worker_id,
                    "source_worker": item.source_worker,
                    "destination_worker": item.destination_worker,
                    "token_position": token_position,
                    "execution_ms": item.execution_ms,
                    "queue_ms": item.queue_ms,
                    "serialisation_ms": item.serialisation_ms,
                    "deserialisation_ms": item.deserialisation_ms,
                    "integrity_validation_ms": item.integrity_validation_ms,
                    "cache_update_ms": item.cache_update_ms,
                    "stream_queue_ms": item.stream_queue_ms,
                    "transfer_ms": item.transfer_ms,
                    "hop_end_to_end_ms": item.hop_end_to_end_ms,
                    "effective_service_ms": effective_ms,
                    "activation_bytes_sent": item.payload_bytes,
                    "activation_bytes_received": item.payload_bytes,
                    "transport_elapsed_s": item.hop_end_to_end_ms / 1000,
                    "route_id": route.route_id,
                    "route_generation": route.route_generation,
                    "data_plane_mode": DataPlaneMode.DIRECT.value,
                }
            )
            self.runtime_transport_metrics["serialisation_time_ms"] = (
                float(self.runtime_transport_metrics["serialisation_time_ms"])
                + item.serialisation_ms
            )
            self.runtime_transport_metrics["deserialisation_time_ms"] = (
                float(self.runtime_transport_metrics["deserialisation_time_ms"])
                + item.deserialisation_ms
            )
            self.runtime_transport_metrics["stream_queue_time_ms"] = (
                float(self.runtime_transport_metrics["stream_queue_time_ms"]) + item.stream_queue_ms
            )
            self.runtime_transport_metrics["hop_transfer_time_ms"] = (
                float(self.runtime_transport_metrics["hop_transfer_time_ms"]) + item.transfer_ms
            )
            self.runtime_transport_metrics["stage_execution_time_ms"] = (
                float(self.runtime_transport_metrics["stage_execution_time_ms"]) + item.execution_ms
            )
        metrics.stage_execution_s += total_execution_ms / 1000
        metrics.queue_s += total_queue_ms / 1000
        metrics.transport_s += max(
            0.0,
            (elapsed_ms - total_execution_ms - total_queue_ms) / 1000,
        )
        return final.result

    async def _perform_direct_cache_replay(
        self,
        *,
        submission: SubmitRequest,
        route: RoutePlan,
        metrics: RuntimeRequestMetrics,
        stage_id: int,
        tokens_committed_before_failure: int,
    ) -> None:
        if not 0 <= stage_id < len(route.assignments):
            raise IntegrityError(f"cache replay stage {stage_id} is outside the installed route")
        hop = route.assignments[stage_id]
        response = await self.transport.cache_control(
            hop.worker_data_endpoint,
            CacheControlRequest(
                worker_id=hop.worker_id,
                request_id=submission.request_id,
                model_revision=self.runtime_model_revision,
                stage_id=stage_id,
                action="clear-and-replay",
            ),
        )
        if not response.accepted:
            raise IntegrityError(f"stage {stage_id} cache replay failed: {response.detail}")
        metrics.replay_s += response.replay_duration_s
        metrics.replay_bytes += response.replay_bytes
        self.events.append(
            {
                "event_type": "stage_cache_replayed",
                "request_id": submission.request_id,
                "route_id": route.route_id,
                "route_generation_before": route.route_generation,
                "route_generation_after": route.route_generation,
                "stage_id": stage_id,
                "failed_stage_id": stage_id,
                "worker_id": hop.worker_id,
                "tokens_committed_before_failure": tokens_committed_before_failure,
                "replay_input_count": response.replay_input_count,
                "replay_bytes": response.replay_bytes,
                "replay_duration_s": response.replay_duration_s,
                "cache_before": response.cache_before,
                "cache_after": response.cache_after,
            }
        )

    async def submit_stream(
        self,
        submission: SubmitRequest,
    ) -> AsyncIterator[SubmitStreamEvent]:
        if self.product_mode:
            _, _, _, sessions = self._require_product()
            stream = sessions.start(submission)
            try:
                async for stream_event in stream:
                    yield stream_event
            finally:
                if not stream.closed:
                    await sessions.disconnect(submission.request_id)
            return

        result = await self._submit_legacy(submission)
        sequence = 0

        def make_event(event_type: StreamEventType, **values: Any) -> SubmitStreamEvent:
            nonlocal sequence
            created = SubmitStreamEvent(
                event_type=event_type,
                request_id=submission.request_id,
                sequence_number=sequence,
                monotonic_timestamp_ns=time.monotonic_ns(),
                model_revision=submission.model_revision,
                **values,
            )
            sequence += 1
            return created

        yield make_event(
            StreamEventType.REQUEST_ACCEPTED,
            status_detail="legacy unary runtime accepted request",
        )
        for position, token_id in enumerate(result.output_token_ids):
            yield make_event(
                StreamEventType.TOKEN_GENERATED,
                token_position=position,
                token_id=token_id,
            )
        if result.status == "completed":
            yield make_event(
                StreamEventType.REQUEST_COMPLETED,
                final_token_ids=list(result.output_token_ids),
                timing_metrics={
                    "time_to_first_token_s": result.time_to_first_token_s or 0.0,
                    "end_to_end_s": result.end_to_end_s,
                },
            )
        else:
            yield make_event(
                StreamEventType.REQUEST_FAILED,
                status_detail=result.detail,
                final_token_ids=list(result.output_token_ids),
                timing_metrics={"end_to_end_s": result.end_to_end_s},
            )

    async def submit(self, submission: SubmitRequest) -> SubmitResponse:
        if not self.product_mode:
            return await self._submit_legacy(submission)
        started = time.perf_counter()
        output_tokens: list[int] = []
        first_token_s: float | None = None
        terminal: SubmitStreamEvent | None = None
        async for event in self.submit_stream(submission):
            if event.event_type == StreamEventType.TOKEN_GENERATED:
                assert event.token_id is not None
                output_tokens.append(event.token_id)
                if first_token_s is None:
                    first_token_s = time.perf_counter()
            if event.event_type in {
                StreamEventType.REQUEST_COMPLETED,
                StreamEventType.REQUEST_FAILED,
                StreamEventType.REQUEST_CANCELLED,
            }:
                terminal = event
        elapsed = time.perf_counter() - started
        if terminal is None:
            return SubmitResponse(
                request_id=submission.request_id,
                output_token_ids=output_tokens,
                status="failed",
                verified=False,
                time_to_first_token_s=(
                    first_token_s - started if first_token_s is not None else None
                ),
                end_to_end_s=elapsed,
                detail="submit stream ended without a terminal event",
            )
        completed = terminal.event_type == StreamEventType.REQUEST_COMPLETED
        cancelled = terminal.event_type == StreamEventType.REQUEST_CANCELLED
        final_tokens = terminal.final_token_ids or output_tokens
        return SubmitResponse(
            request_id=submission.request_id,
            output_token_ids=final_tokens,
            status="completed" if completed else "cancelled" if cancelled else "failed",
            verified=completed,
            time_to_first_token_s=terminal.timing_metrics.get(
                "time_to_first_token_s",
                first_token_s - started if first_token_s is not None else None,
            ),
            end_to_end_s=terminal.timing_metrics.get("end_to_end_s", elapsed),
            detail=terminal.status_detail,
        )

    async def _submit_legacy(self, submission: SubmitRequest) -> SubmitResponse:
        started = time.perf_counter()
        metrics = RuntimeRequestMetrics(
            request_id=submission.request_id,
            started_s=started,
            data_plane_mode=self.config.data_plane.value,
        )
        outputs: list[int] = []
        route_id: str | None = None
        request_state: RequestState | None = None
        release_reason = "request-failed"
        try:
            if submission.prompt_token_ids:
                prompt_token_ids = submission.prompt_token_ids
            elif self.model_manifest is not None:
                if self.tokenizer is None:
                    raise IntegrityError(
                        "real-model text submission requires a coordinator tokenizer; "
                        "provide prompt_token_ids or configure tokenizer_path"
                    )
                encoded_prompt = self.tokenizer(submission.prompt or "", return_tensors=None)
                prompt_token_ids = [int(value) for value in encoded_prompt["input_ids"]]
            else:
                prompt_token_ids = [byte + 1 for byte in (submission.prompt or "").encode("utf-8")]
            if self.model_manifest is not None:
                if submission.model_id not in {"synthetic", self.runtime_model_id}:
                    raise IntegrityError(
                        f"request model {submission.model_id!r} does not match "
                        f"{self.runtime_model_id!r}"
                    )
                if submission.model_revision not in {
                    "synthetic-v1",
                    self.runtime_model_revision,
                }:
                    raise IntegrityError(
                        "request model revision does not match the assigned immutable revision"
                    )
            workload_class = WorkloadClass(submission.workload_class)
            request_state = RequestState(
                request_id=submission.request_id,
                workload_class=workload_class,
                prompt_token_ids=prompt_token_ids,
                sampling=SamplingConfig(
                    temperature=0,
                    max_new_tokens=submission.max_new_tokens,
                ),
                random_seed=submission.random_seed,
                status=RequestStatus.RUNNING,
            )
            decision = self._reserve_request_route(submission, workload_class)
            route_id = decision.route_id
            route = decision.assignments
            metrics.route_id = route_id
            metrics.route_generation = decision.generation
            metrics.route_reservation_time_ms = decision.reservation_time_ms
            request_state.stage_route = [
                route[index].worker_id for index in range(len(self.stages))
            ]
            request_state.stage_local_cache_ownership = {
                stage_id: replica.worker_id for stage_id, replica in route.items()
            }
            route_plan = (
                self._route_plan(submission, decision)
                if self.config.data_plane == DataPlaneMode.DIRECT
                else None
            )
            if route_plan is not None:
                await self._install_direct_route(route_plan)
            metrics.admission_time_ms = (time.perf_counter() - started) * 1000
            self.runtime_transport_metrics["admission_time_ms"] = (
                float(self.runtime_transport_metrics["admission_time_ms"])
                + metrics.admission_time_ms
            )
            for output_position in range(submission.max_new_tokens):
                token_step_started = time.perf_counter()
                operation = OperationKind.PREFILL if output_position == 0 else OperationKind.DECODE
                token_ids = prompt_token_ids if output_position == 0 else [outputs[-1]]
                if self.model_manifest is None:
                    token_position = output_position
                    activation = synthetic_activation(
                        token_ids,
                        hidden_size=self.config.model.hidden_size,
                        dtype=self.config.model.activation_dtype,
                    )
                else:
                    token_position = (
                        0
                        if operation == OperationKind.PREFILL
                        else len(prompt_token_ids) + output_position - 1
                    )
                    activation = np.asarray([token_ids], dtype=np.int64)
                if route_plan is not None:
                    tensor = ActivationTensor(
                        tensor_id=f"{submission.request_id}:{output_position}:0",
                        request_id=submission.request_id,
                        stage_id=0,
                        token_position=token_position,
                        sequence_length=len(token_ids),
                        array=activation,
                    )
                    encoded = encode_tensor(tensor)
                    self.replay.append(
                        request_id=submission.request_id,
                        model_revision=self.runtime_model_revision,
                        stage_id=0,
                        cache_generation=0,
                        token_position=token_position,
                        operation=operation,
                        payload=encoded,
                        recorded_monotonic_ns=time.monotonic_ns(),
                    )
                    try:
                        result = await self._call_direct_route(
                            submission=submission,
                            route=route_plan,
                            metrics=metrics,
                            operation=operation,
                            token_position=token_position,
                            sequence_length=len(token_ids),
                            encoded=encoded,
                        )
                    except (TransportError, TimeoutError) as exc:
                        route_plan = await self._recover_direct_route(
                            submission=submission,
                            request_state=request_state,
                            route=route_plan,
                            metrics=metrics,
                            committed_through_token_position=token_position - 1,
                            failure=exc,
                        )
                        route = self.route_allocator.lease(route_plan.route_id).assignments
                        result = await self._call_direct_route(
                            submission=submission,
                            route=route_plan,
                            metrics=metrics,
                            operation=operation,
                            token_position=token_position,
                            sequence_length=len(token_ids),
                            encoded=encoded,
                        )
                    activation = decode_tensor(result.tensor_payload).array
                else:
                    for stage in self.stages:
                        replica = route[stage.stage_id]
                        tensor = ActivationTensor(
                            tensor_id=(
                                f"{submission.request_id}:{output_position}:{stage.stage_id}"
                            ),
                            request_id=submission.request_id,
                            stage_id=stage.stage_id,
                            token_position=token_position,
                            sequence_length=len(token_ids),
                            array=activation,
                        )
                        encoded = encode_tensor(tensor)
                        self.replay.append(
                            request_id=submission.request_id,
                            model_revision=self.runtime_model_revision,
                            stage_id=stage.stage_id,
                            cache_generation=0,
                            token_position=token_position,
                            operation=operation,
                            payload=encoded,
                            recorded_monotonic_ns=time.monotonic_ns(),
                        )
                        result, replacement = await self._execute_with_recovery(
                            request=submission,
                            request_state=request_state,
                            metrics=metrics,
                            route=route,
                            replica=replica,
                            stage_id=stage.stage_id,
                            operation=operation,
                            token_position=token_position,
                            sequence_length=len(token_ids),
                            encoded=encoded,
                        )
                        if stage.stage_id < len(self.stages) - 1:
                            self.runtime_transport_metrics["coordinator_activation_bytes"] = int(
                                self.runtime_transport_metrics["coordinator_activation_bytes"]
                            ) + 2 * len(result.tensor_payload)
                        if replacement is not None:
                            route[stage.stage_id] = replacement
                        activation = decode_tensor(result.tensor_payload).array
                token_produced_ns = time.monotonic_ns()
                if self.model_manifest is None:
                    digest = hashlib.sha256(
                        np.ascontiguousarray(activation).tobytes()
                        + submission.random_seed.to_bytes(8, "little", signed=True)
                        + output_position.to_bytes(8, "little")
                    ).digest()
                    token_id = int.from_bytes(digest[:4], "little") % 151_936
                else:
                    if (
                        activation.ndim != 3
                        or activation.shape[-1] != self.model_manifest.vocabulary_size
                    ):
                        raise IntegrityError(
                            "final real-model stage did not return [batch, sequence, vocabulary] logits"
                        )
                    coordinator_started = time.perf_counter()
                    token_logits = activation[0, -1, :].astype(np.float32, copy=False)
                    token_id = int(np.argmax(token_logits))
                    top_count = min(10, int(token_logits.shape[0]))
                    top_indices = np.argpartition(token_logits, -top_count)[-top_count:]
                    top_indices = top_indices[np.argsort(token_logits[top_indices])[::-1]]
                    coordinator_processing_ms = (time.perf_counter() - coordinator_started) * 1000
                    token_produced_ns = time.monotonic_ns()
                    metrics.token_steps.append(
                        {
                            "step": output_position,
                            "operation": operation.value,
                            "token_position": token_position,
                            "selected_token_id": token_id,
                            "selected_token_text": (
                                self.tokenizer.decode([token_id])
                                if self.tokenizer is not None
                                else None
                            ),
                            "selected_token_logit": float(token_logits[token_id]),
                            "top_logits": [
                                {
                                    "token_id": int(index),
                                    "token_text": (
                                        self.tokenizer.decode([int(index)])
                                        if self.tokenizer is not None
                                        else None
                                    ),
                                    "logit": float(token_logits[index]),
                                }
                                for index in top_indices
                            ],
                            "coordinator_processing_ms": coordinator_processing_ms,
                            "total_token_latency_ms": (time.perf_counter() - token_step_started)
                            * 1000,
                            "token_produced_monotonic_ns": token_produced_ns,
                        }
                    )
                outputs.append(token_id)
                request_state.committed_output_tokens.append(token_id)
                request_state.current_token_position += 1
                if metrics.first_token_s is None:
                    metrics.first_token_s = time.perf_counter()
                    self.events.append(
                        {
                            "event_type": "first_token_produced",
                            "request_id": submission.request_id,
                            "token_id": token_id,
                            "timestamp_monotonic_ns": token_produced_ns,
                        }
                    )
                if (
                    route_plan is not None
                    and submission.cache_replay_stage_id is not None
                    and submission.cache_replay_after_tokens == output_position + 1
                ):
                    await self._perform_direct_cache_replay(
                        submission=submission,
                        route=route_plan,
                        metrics=metrics,
                        stage_id=submission.cache_replay_stage_id,
                        tokens_committed_before_failure=output_position + 1,
                    )
                if self.after_token_hook is not None:
                    hook_route = await self.after_token_hook(
                        {
                            "core": self,
                            "submission": submission,
                            "request_state": request_state,
                            "route_plan": route_plan,
                            "metrics": metrics,
                            "output_position": output_position,
                            "token_position": token_position,
                            "token_id": token_id,
                            "committed_tokens": list(outputs),
                            "timestamp_monotonic_ns": token_produced_ns,
                        }
                    )
                    if hook_route is not None:
                        if route_plan is None:
                            raise IntegrityError(
                                "after-token hook returned a route for a non-direct request"
                            )
                        route_plan = hook_route
                        route = self.route_allocator.lease(route_plan.route_id).assignments
                if self.model_manifest is not None and self.architecture_config is not None:
                    configured_eos = self.architecture_config.get("eos_token_id")
                    eos_ids = (
                        {int(value) for value in configured_eos}
                        if isinstance(configured_eos, list)
                        else ({int(configured_eos)} if configured_eos is not None else set())
                    )
                    if token_id in eos_ids:
                        break
            request_state.status = RequestStatus.COMPLETED
            request_state.verification_state = VerificationState.VERIFIED
            metrics.completed_s = time.perf_counter()
            release_reason = "request-finished"
            self.events.append(
                {
                    "event_type": "request_completed",
                    "request_id": submission.request_id,
                    "verified_tokens": len(outputs),
                }
            )
            return SubmitResponse(
                request_id=submission.request_id,
                output_token_ids=outputs,
                status="completed",
                verified=True,
                time_to_first_token_s=(
                    metrics.first_token_s - started if metrics.first_token_s else None
                ),
                end_to_end_s=metrics.completed_s - started,
            )
        except Exception as exc:
            if request_state is not None:
                request_state.status = RequestStatus.FAILED
                request_state.verification_state = VerificationState.REJECTED
            metrics.completed_s = time.perf_counter()
            self.events.append(
                {
                    "event_type": "request_failed",
                    "request_id": submission.request_id,
                    "detail": str(exc),
                }
            )
            return SubmitResponse(
                request_id=submission.request_id,
                output_token_ids=outputs,
                status="failed",
                verified=False,
                time_to_first_token_s=(
                    metrics.first_token_s - started if metrics.first_token_s else None
                ),
                end_to_end_s=metrics.completed_s - started,
                detail=str(exc),
            )
        finally:
            self.request_metrics.append(
                {
                    "request_id": metrics.request_id,
                    "time_to_first_token_s": (
                        metrics.first_token_s - metrics.started_s
                        if metrics.first_token_s is not None
                        else None
                    ),
                    "end_to_end_s": (
                        metrics.completed_s - metrics.started_s
                        if metrics.completed_s is not None
                        else None
                    ),
                    "stage_execution_s": metrics.stage_execution_s,
                    "queue_s": metrics.queue_s,
                    "transport_s": metrics.transport_s,
                    "replay_s": metrics.replay_s,
                    "replay_bytes": metrics.replay_bytes,
                    "retry_count": metrics.retries,
                    "route_changes": metrics.route_changes,
                    "admission_time_ms": metrics.admission_time_ms,
                    "route_reservation_time_ms": metrics.route_reservation_time_ms,
                    "route_id": metrics.route_id,
                    "route_generation": metrics.route_generation,
                    "data_plane_mode": metrics.data_plane_mode,
                    "per_stage": metrics.per_stage,
                    "token_steps": metrics.token_steps,
                }
            )
            if route_id is not None:
                self.route_allocator.release(route_id, reason=release_reason)
            await self._cancel_all(submission.request_id, self.runtime_model_revision)

    async def _call_replica(
        self,
        *,
        replica: StageReplica,
        submission: SubmitRequest,
        stage_id: int,
        operation: OperationKind,
        token_position: int,
        sequence_length: int,
        encoded: bytes,
        audit: bool = False,
    ) -> tuple[ActivationResult, float]:
        if replica.endpoint is None:
            raise TransportError(f"worker {replica.worker_id} has no endpoint")
        request = ActivationRequest(
            metadata=ActivationMetadata(
                request_id=submission.request_id,
                tensor_id=f"{submission.request_id}:{token_position}:{stage_id}",
                stage_id=stage_id,
                operation=operation,
                token_position=token_position,
                sequence_length=sequence_length,
                cache_generation=0,
                model_id=submission.model_id,
                model_revision=submission.model_revision,
                audit=audit,
            ),
            tensor_payload=encoded,
        )
        started = time.perf_counter()
        result = await self.transport.execute(replica.endpoint, request)
        elapsed = time.perf_counter() - started
        if sha256_bytes(result.tensor_payload) != result.checksum:
            raise IntegrityError(f"result checksum mismatch from worker {result.worker_id}")
        capability = self.registry.capability(result.worker_id)
        signed_payload = canonical_json_bytes(
            {
                "worker_id": result.worker_id,
                "request_id": result.metadata.request_id,
                "stage_id": result.metadata.stage_id,
                "token_position": result.metadata.token_position,
                "checksum": result.checksum,
            }
        )
        verify_signature(capability.public_key, signed_payload, result.signature)
        # The tensor decoder verifies the internal activation checksum and IDs.
        decoded = decode_tensor(result.tensor_payload)
        if (
            decoded.request_id != submission.request_id
            or decoded.stage_id != stage_id
            or decoded.token_position != token_position
        ):
            raise IntegrityError("worker result metadata does not match stage request")
        return result, elapsed

    async def _execute_with_recovery(
        self,
        *,
        request: SubmitRequest,
        request_state: RequestState,
        metrics: RuntimeRequestMetrics,
        route: dict[int, StageReplica],
        replica: StageReplica,
        stage_id: int,
        operation: OperationKind,
        token_position: int,
        sequence_length: int,
        encoded: bytes,
    ) -> tuple[ActivationResult, StageReplica | None]:
        try:
            result, elapsed = await self._call_replica(
                replica=replica,
                submission=request,
                stage_id=stage_id,
                operation=operation,
                token_position=token_position,
                sequence_length=sequence_length,
                encoded=encoded,
            )
            metrics.stage_execution_s += result.execution_ms / 1000
            metrics.queue_s += result.queue_ms / 1000
            metrics.transport_s += max(
                0.0, elapsed - (result.execution_ms + result.queue_ms) / 1000
            )
            metrics.per_stage.append(
                {
                    "stage_id": stage_id,
                    "worker_id": replica.worker_id,
                    "token_position": token_position,
                    "execution_ms": result.execution_ms,
                    "queue_ms": result.queue_ms,
                    "transport_elapsed_s": elapsed,
                    "activation_bytes_sent": len(encoded),
                    "activation_bytes_received": len(result.tensor_payload),
                }
            )
            return result, None
        except Exception as original:
            alternatives = sorted(
                [
                    candidate
                    for candidate in self.registry.replicas(stage_id)
                    if candidate.worker_id != replica.worker_id
                    and candidate.health == HealthStatus.HEALTHY
                    and candidate.endpoint is not None
                ],
                key=lambda candidate: candidate.worker_id,
            )
            if not alternatives:
                raise TransportError(
                    f"stage {stage_id} worker {replica.worker_id} failed and no backup "
                    f"replica is available: {original}"
                ) from original
            replacement = alternatives[0]
            replay_started = time.perf_counter()
            replay_bytes = 0
            if token_position > 0:
                try:
                    entries = self.replay.entries_for(
                        request_id=request.request_id,
                        model_revision=request.model_revision,
                        stage_id=stage_id,
                        cache_generation=0,
                        through_token_position=token_position - 1,
                    )
                except ReplayUnavailableError:
                    entries = []
                for entry in entries:
                    replay_bytes += len(entry.payload)
                    replay_sequence = decode_tensor(entry.payload).sequence_length
                    await self._call_replica(
                        replica=replacement,
                        submission=request,
                        stage_id=stage_id,
                        operation=entry.operation,
                        token_position=entry.token_position,
                        sequence_length=replay_sequence,
                        encoded=entry.payload,
                    )
            replay_elapsed = time.perf_counter() - replay_started
            metrics.replay_s += replay_elapsed
            metrics.replay_bytes += replay_bytes
            metrics.retries += 1
            metrics.route_changes += 1
            request_state.retry_count += 1
            self.events.append(
                {
                    "event_type": "stage_recovered",
                    "request_id": request.request_id,
                    "stage_id": stage_id,
                    "failed_worker_id": replica.worker_id,
                    "replacement_worker_id": replacement.worker_id,
                    "replay_bytes": replay_bytes,
                    "replay_duration_s": replay_elapsed,
                    "failure": str(original),
                }
            )
            result, elapsed = await self._call_replica(
                replica=replacement,
                submission=request,
                stage_id=stage_id,
                operation=operation,
                token_position=token_position,
                sequence_length=sequence_length,
                encoded=encoded,
            )
            metrics.stage_execution_s += result.execution_ms / 1000
            metrics.queue_s += result.queue_ms / 1000
            metrics.transport_s += max(
                0.0, elapsed - (result.execution_ms + result.queue_ms) / 1000
            )
            metrics.per_stage.append(
                {
                    "stage_id": stage_id,
                    "worker_id": replacement.worker_id,
                    "token_position": token_position,
                    "execution_ms": result.execution_ms,
                    "queue_ms": result.queue_ms,
                    "transport_elapsed_s": elapsed,
                    "activation_bytes_sent": len(encoded),
                    "activation_bytes_received": len(result.tensor_payload),
                    "recovered": True,
                }
            )
            return result, replacement

    async def _cancel_all(self, request_id: str, model_revision: str) -> None:
        await self._cancel_worker_request_state(request_id, model_revision)
        self.replay.delete_request(request_id)


class CoordinatorRpcServer:
    def __init__(
        self,
        core: CoordinatorCore,
        *,
        maximum_message_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        self.core = core
        self.server = grpc.aio.server(
            options=[
                ("grpc.max_send_message_length", maximum_message_bytes),
                ("grpc.max_receive_message_length", maximum_message_bytes),
            ]
        )
        handlers = {
            "Register": grpc.unary_unary_rpc_method_handler(
                self._register,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "Heartbeat": grpc.unary_unary_rpc_method_handler(
                self._heartbeat,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "Submit": grpc.unary_unary_rpc_method_handler(
                self._submit,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "SubmitStream": grpc.unary_stream_rpc_method_handler(
                self._submit_stream,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "InspectModel": grpc.unary_unary_rpc_method_handler(
                self._inspect_model,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "PlanModel": grpc.unary_unary_rpc_method_handler(
                self._plan_model,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "DeployModel": grpc.unary_unary_rpc_method_handler(
                self._deploy_model,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "UnloadModel": grpc.unary_unary_rpc_method_handler(
                self._unload_model,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "TopologyStatus": grpc.unary_unary_rpc_method_handler(
                self._topology_status,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "Workers": grpc.unary_unary_rpc_method_handler(
                self._workers,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "Status": grpc.unary_unary_rpc_method_handler(
                self._status,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "Sessions": grpc.unary_unary_rpc_method_handler(
                self._sessions,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "CancelRequest": grpc.unary_unary_rpc_method_handler(
                self._cancel_request,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "PublishToken": grpc.unary_unary_rpc_method_handler(
                self._publish_token,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "FinalResult": grpc.unary_unary_rpc_method_handler(
                self._final_result,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
        }
        self.server.add_generic_rpc_handlers(
            (grpc.method_handlers_generic_handler("swarm.v1.Coordinator", handlers),)
        )
        self.bound_port: int | None = None

    async def start(
        self,
        endpoint: str,
        *,
        advertised_endpoint: str | None = None,
    ) -> int:
        self.bound_port = self.server.add_insecure_port(endpoint)
        if self.bound_port == 0:
            raise TransportError(f"could not bind coordinator endpoint {endpoint}")
        host = endpoint.rsplit(":", 1)[0].strip("[]")
        if host in {"", "0.0.0.0", "::"}:
            host = "127.0.0.1"
        self.core.final_result_endpoint = f"{host}:{self.bound_port}"
        self.core.publication_endpoint = advertised_endpoint or self.core.final_result_endpoint
        await self.server.start()
        return self.bound_port

    async def stop(self, grace_s: float = 2.0) -> None:
        await self.server.stop(grace_s)
        await self.core.close()

    async def wait_for_termination(self) -> None:
        await self.server.wait_for_termination()

    async def _register(self, data: bytes, context: grpc.aio.ServicerContext[Any, Any]) -> bytes:
        try:
            response = await self.core.register(parse_message(data, RegistrationRequest))
            return serialize_message(response)
        except Exception as exc:
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
            raise

    async def _heartbeat(self, data: bytes, context: grpc.aio.ServicerContext[Any, Any]) -> bytes:
        try:
            response = await self.core.heartbeat(parse_message(data, Heartbeat))
            return serialize_message(response)
        except Exception as exc:
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
            raise

    async def _submit(self, data: bytes, context: grpc.aio.ServicerContext[Any, Any]) -> bytes:
        response = await self.core.submit(parse_message(data, SubmitRequest))
        return serialize_message(response)

    async def _submit_stream(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> AsyncIterator[bytes]:
        request = parse_message(data, SubmitRequest)
        try:
            async for event in self.core.submit_stream(request):
                yield serialize_message(event)
        except asyncio.CancelledError:
            if self.core.session_controller is not None:
                await self.core.session_controller.disconnect(request.request_id)
            raise

    async def _inspect_model(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            response = await self.core.inspect_product_model(
                parse_message(data, ModelInspectRequest)
            )
            return serialize_message(response)
        except Exception as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            raise

    async def _plan_model(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            response = await self.core.plan_product_model(parse_message(data, ModelPlanRequest))
            return serialize_message(response)
        except Exception as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            raise

    async def _deploy_model(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            response = await self.core.deploy_product_model(parse_message(data, ModelDeployRequest))
            return serialize_message(response)
        except Exception as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            raise

    async def _unload_model(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            response = await self.core.unload_product_model(parse_message(data, ModelUnloadRequest))
            return serialize_message(response)
        except Exception as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            raise

    async def _topology_status(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            response = await self.core.product_topology_status(
                parse_message(data, TopologyStatusRequest)
            )
            return serialize_message(response)
        except Exception as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            raise

    async def _workers(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            response = await self.core.product_workers(parse_message(data, WorkersRequest))
            return serialize_message(response)
        except Exception as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            raise

    async def _status(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            response = await self.core.product_status(parse_message(data, CoordinatorStatusRequest))
            return serialize_message(response)
        except Exception as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            raise

    async def _sessions(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            response = await self.core.product_sessions(parse_message(data, SessionsRequest))
            return serialize_message(response)
        except Exception as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            raise

    async def _cancel_request(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            response = await self.core.cancel_product_request(
                parse_message(data, CancelProductRequest)
            )
            return serialize_message(response)
        except Exception as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            raise

    async def _publish_token(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            response = await self.core.accept_product_token(
                parse_message(data, ProductTokenPublication)
            )
            return serialize_message(response)
        except Exception as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            raise

    async def _final_result(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            response = await self.core.accept_final_result(parse_message(data, FinalResultMessage))
            return serialize_message(response)
        except Exception as exc:
            await context.abort(grpc.StatusCode.DATA_LOSS, str(exc))
            raise


class CoordinatorClient:
    def __init__(
        self,
        endpoint: str,
        *,
        maximum_message_bytes: int = 4 * 1024 * 1024,
        timeout_s: float = 120.0,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.channel = grpc.aio.insecure_channel(
            endpoint,
            options=[
                ("grpc.max_send_message_length", maximum_message_bytes),
                ("grpc.max_receive_message_length", maximum_message_bytes),
            ],
        )

    async def _call(
        self,
        path: str,
        request: StrictModel,
        response_type: type[ResponseT],
    ) -> ResponseT:
        call = self.channel.unary_unary(
            path,
            request_serializer=lambda value: value,
            response_deserializer=lambda value: value,
        )
        try:
            data = await call(serialize_message(request), timeout=self.timeout_s)
            return parse_message(data, response_type)
        except grpc.aio.AioRpcError as exc:
            raise TransportError(
                f"coordinator RPC {path} failed ({exc.code().name}): {exc.details()}"
            ) from exc

    async def register(self, request: RegistrationRequest) -> RegistrationResponse:
        return await self._call(
            "/swarm.v1.Coordinator/Register",
            request,
            RegistrationResponse,
        )

    async def heartbeat(self, request: Heartbeat) -> Ack:
        return await self._call("/swarm.v1.Coordinator/Heartbeat", request, Ack)

    async def submit(self, request: SubmitRequest) -> SubmitResponse:
        return await self._call(
            "/swarm.v1.Coordinator/Submit",
            request,
            SubmitResponse,
        )

    async def submit_stream(
        self,
        request: SubmitRequest,
    ) -> AsyncIterator[SubmitStreamEvent]:
        rpc = self.channel.unary_stream(
            "/swarm.v1.Coordinator/SubmitStream",
            request_serializer=lambda value: value,
            response_deserializer=lambda value: value,
        )
        call = rpc(serialize_message(request), timeout=self.timeout_s)
        try:
            async for data in call:
                yield parse_message(data, SubmitStreamEvent)
        except grpc.aio.AioRpcError as exc:
            raise TransportError(
                f"coordinator submit stream failed ({exc.code().name}): {exc.details()}"
            ) from exc
        finally:
            call.cancel()

    async def inspect_model(self, request: ModelInspectRequest) -> ModelInspectResponse:
        return await self._call(
            "/swarm.v1.Coordinator/InspectModel",
            request,
            ModelInspectResponse,
        )

    async def plan_model(self, request: ModelPlanRequest) -> ModelPlanResponse:
        return await self._call(
            "/swarm.v1.Coordinator/PlanModel",
            request,
            ModelPlanResponse,
        )

    async def deploy_model(self, request: ModelDeployRequest) -> ModelDeployResponse:
        return await self._call(
            "/swarm.v1.Coordinator/DeployModel",
            request,
            ModelDeployResponse,
        )

    async def unload_model(self, request: ModelUnloadRequest) -> ModelUnloadResponse:
        return await self._call(
            "/swarm.v1.Coordinator/UnloadModel",
            request,
            ModelUnloadResponse,
        )

    async def topology_status(self, request: TopologyStatusRequest) -> TopologyStatusResponse:
        return await self._call(
            "/swarm.v1.Coordinator/TopologyStatus",
            request,
            TopologyStatusResponse,
        )

    async def workers(self, request: WorkersRequest) -> WorkersResponse:
        return await self._call(
            "/swarm.v1.Coordinator/Workers",
            request,
            WorkersResponse,
        )

    async def status(self) -> CoordinatorStatusResponse:
        return await self._call(
            "/swarm.v1.Coordinator/Status",
            CoordinatorStatusRequest(),
            CoordinatorStatusResponse,
        )

    async def sessions(self, request: SessionsRequest) -> SessionsResponse:
        return await self._call(
            "/swarm.v1.Coordinator/Sessions",
            request,
            SessionsResponse,
        )

    async def cancel_request(self, request_id: str) -> CancelProductResponse:
        return await self._call(
            "/swarm.v1.Coordinator/CancelRequest",
            CancelProductRequest(request_id=request_id),
            CancelProductResponse,
        )

    async def publish_token(self, publication: ProductTokenPublication) -> Ack:
        return await self._call(
            "/swarm.v1.Coordinator/PublishToken",
            publication,
            Ack,
        )

    async def close(self) -> None:
        await self.channel.close()
