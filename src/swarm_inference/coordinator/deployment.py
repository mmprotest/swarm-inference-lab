"""Transactional persistent-stage deployment owned by ``CoordinatorCore``."""

from __future__ import annotations

import asyncio
import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from swarm_inference.cluster.artifacts import ArtifactManager, artifact_chunks
from swarm_inference.config.models import WorkerRole
from swarm_inference.coordinator.registry import WorkerRegistry
from swarm_inference.exceptions import IntegrityError
from swarm_inference.filesystem import replace_atomically
from swarm_inference.model.product import ModelResolutionPolicy
from swarm_inference.protocol.expert import (
    ExpertRouteParticipant,
    SignedExpertRouteLease,
    sign_expert_route_lease,
)
from swarm_inference.protocol.product import (
    DeploymentPhase,
    DeploymentStatus,
    DeploymentWorkerStatus,
    PlanWorkerAssignment,
    ProductStagePlan,
)
from swarm_inference.protocol.routes import (
    RouteLeaseParticipant,
    SignedRouteLease,
    sign_route_lease,
)
from swarm_inference.protocol.stage_worker import (
    ArtifactTransferLease,
    ArtifactTransferResponse,
    CompleteArtifactRequest,
    GetStageStatusRequest,
    InstallStageRouteRequest,
    LoadStageRequest,
    PrepareArtifactRequest,
    RemoveStageRouteRequest,
    StageActionResponse,
    StageRouteEndpoint,
    StageStatusResponse,
    UnloadStageRequest,
    VerifyArtifactRequest,
    VerifyStageRouteRequest,
    WriteArtifactChunkRequest,
    sign_artifact_transfer_lease,
)
from swarm_inference.runtime.telemetry import ProductTelemetry
from swarm_inference.security.identity import CoordinatorIdentity, public_key_fingerprint
from swarm_inference.transport.expert import ExpertTransportClient


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        replace_atomically(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class DeploymentTransport(Protocol):
    async def prepare_artifact(
        self, endpoint: str, request: PrepareArtifactRequest
    ) -> ArtifactTransferResponse: ...

    async def write_artifact_chunk(
        self, endpoint: str, request: WriteArtifactChunkRequest
    ) -> ArtifactTransferResponse: ...

    async def complete_artifact(
        self, endpoint: str, request: CompleteArtifactRequest
    ) -> ArtifactTransferResponse: ...

    async def verify_artifact(
        self, endpoint: str, request: VerifyArtifactRequest
    ) -> ArtifactTransferResponse: ...

    async def load_stage(self, endpoint: str, request: LoadStageRequest) -> StageActionResponse: ...

    async def unload_stage(
        self, endpoint: str, request: UnloadStageRequest
    ) -> StageActionResponse: ...

    async def install_stage_route(
        self, endpoint: str, request: InstallStageRouteRequest
    ) -> StageActionResponse: ...

    async def remove_stage_route(
        self, endpoint: str, request: RemoveStageRouteRequest
    ) -> StageActionResponse: ...

    async def verify_stage_route(
        self, endpoint: str, request: VerifyStageRouteRequest
    ) -> StageActionResponse: ...

    async def get_stage_status(
        self, endpoint: str, request: GetStageStatusRequest
    ) -> StageStatusResponse: ...


class DeploymentArtifactCoordinator(Protocol):
    """Artifact phases invoked inside the canonical deployment transaction."""

    async def prepare(self, plan: ProductStagePlan) -> None: ...

    async def transfer(self, plan: ProductStagePlan) -> None: ...

    async def verify(self, plan: ProductStagePlan) -> None: ...

    async def release(self, plan: ProductStagePlan) -> None: ...


class TransportArtifactCoordinator:
    """Place coordinator-built artifacts over authenticated worker control RPCs."""

    def __init__(
        self,
        *,
        transport: DeploymentTransport,
        manager: ArtifactManager,
        identity: CoordinatorIdentity,
        coordinator_id: str,
        lease_seconds: float,
        chunk_size_bytes: int = 1024 * 1024,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("artifact lease duration must be positive")
        if not 0 < chunk_size_bytes <= 2 * 1024 * 1024:
            raise ValueError("artifact RPC chunks must be in (0, 2 MiB]")
        self.transport = transport
        self.manager = manager
        self.identity = identity
        self.coordinator_id = coordinator_id
        self.lease_seconds = lease_seconds
        self.chunk_size_bytes = chunk_size_bytes
        self._leases: dict[tuple[str, str], ArtifactTransferLease] = {}

    def _lease(
        self,
        plan: ProductStagePlan,
        assignment: PlanWorkerAssignment,
    ) -> ArtifactTransferLease:
        if assignment.artifact_id is None:
            raise IntegrityError("stage assignment has no artifact identity")
        key = (plan.plan_id, assignment.worker_id)
        existing = self._leases.get(key)
        if existing is not None:
            return existing
        issued = time.time_ns()
        unsigned = ArtifactTransferLease(
            artifact_id=assignment.artifact_id,
            destination_worker_id=assignment.worker_id,
            source_node_id=self.coordinator_id,
            issued_at_unix_ns=issued,
            expires_at_unix_ns=issued + int(self.lease_seconds * 1_000_000_000),
            nonce=uuid4().hex,
            coordinator_identity=self.coordinator_id,
            coordinator_public_key=self.identity.public_key_b64,
            coordinator_fingerprint=self.identity.public_key_fingerprint,
        )
        lease = sign_artifact_transfer_lease(unsigned, self.identity)
        self._leases[key] = lease
        return lease

    @staticmethod
    def _artifact_assignments(plan: ProductStagePlan) -> list[PlanWorkerAssignment]:
        values = [item for item in plan.assignments if item.artifact_id is not None]
        if values and len(values) != len(plan.assignments):
            raise IntegrityError(
                "every stage must use an artifact when artifact deployment is used"
            )
        return values

    async def prepare(self, plan: ProductStagePlan) -> None:
        for item in self._artifact_assignments(plan):
            manifest = item.artifact_manifest
            if manifest is None:
                raise IntegrityError(f"stage {item.stage_id} has no artifact manifest")
            if (
                manifest.model_id != plan.model.model_id
                or manifest.model_revision != plan.model.model_revision
                or manifest.tokenizer_revision != plan.model.tokenizer_revision
                or manifest.dtype != plan.model.dtype
            ):
                raise IntegrityError(f"stage {item.stage_id} artifact identity differs from plan")
            self.manager.resolve(manifest.artifact_id)
            self._lease(plan, item)

    async def transfer(self, plan: ProductStagePlan) -> None:
        for item in self._artifact_assignments(plan):
            manifest = item.artifact_manifest
            assert manifest is not None
            source = self.manager.resolve(manifest.artifact_id)
            chunks_total = sum(
                math.ceil(file.size_bytes / self.chunk_size_bytes) for file in manifest.files
            )
            lease = self._lease(plan, item)
            prepared = await self.transport.prepare_artifact(
                item.control_endpoint,
                PrepareArtifactRequest(
                    worker_id=item.worker_id,
                    request_id=f"{plan.plan_id}:artifact:prepare:{item.stage_id}",
                    manifest=manifest,
                    chunks_total=chunks_total,
                    lease=lease,
                ),
            )
            if not prepared.accepted or prepared.transfer_id is None:
                raise RuntimeError(
                    f"worker {item.worker_id} rejected artifact preparation: {prepared.detail}"
                )
            if prepared.complete and prepared.verified:
                continue
            transfer_id = prepared.transfer_id
            sent = 0
            for chunk, payload in artifact_chunks(
                source,
                chunk_size_bytes=self.chunk_size_bytes,
            ):
                response = await self.transport.write_artifact_chunk(
                    item.control_endpoint,
                    WriteArtifactChunkRequest(
                        worker_id=item.worker_id,
                        request_id=(
                            f"{plan.plan_id}:artifact:chunk:{item.stage_id}:{chunk.chunk_index}"
                        ),
                        transfer_id=transfer_id,
                        chunk=chunk,
                        payload=payload,
                        lease=lease,
                    ),
                )
                if not response.accepted:
                    raise RuntimeError(
                        f"worker {item.worker_id} rejected artifact chunk "
                        f"{chunk.chunk_index}: {response.detail}"
                    )
                sent += 1
            if sent != chunks_total:
                raise IntegrityError("artifact transport emitted an unexpected chunk count")
            completed = await self.transport.complete_artifact(
                item.control_endpoint,
                CompleteArtifactRequest(
                    worker_id=item.worker_id,
                    request_id=f"{plan.plan_id}:artifact:complete:{item.stage_id}",
                    transfer_id=transfer_id,
                    manifest=manifest,
                    lease=lease,
                ),
            )
            if not completed.accepted or not completed.complete or not completed.verified:
                raise RuntimeError(
                    f"worker {item.worker_id} did not verify artifact: {completed.detail}"
                )

    async def verify(self, plan: ProductStagePlan) -> None:
        for item in self._artifact_assignments(plan):
            assert item.artifact_id is not None
            response = await self.transport.verify_artifact(
                item.control_endpoint,
                VerifyArtifactRequest(
                    worker_id=item.worker_id,
                    request_id=f"{plan.plan_id}:artifact:verify:{item.stage_id}",
                    artifact_id=item.artifact_id,
                    lease=self._lease(plan, item),
                ),
            )
            if not response.accepted or not response.verified:
                raise RuntimeError(
                    f"worker {item.worker_id} artifact verification failed: {response.detail}"
                )

    async def release(self, plan: ProductStagePlan) -> None:
        for item in plan.assignments:
            self._leases.pop((plan.plan_id, item.worker_id), None)


class DeploymentManager:
    """Reserve, load, connect, verify, and explicitly unload one product topology."""

    def __init__(
        self,
        *,
        registry: WorkerRegistry,
        transport: DeploymentTransport,
        state_directory: Path,
        lease_seconds: float,
        control_timeout_s: float,
        coordinator_identity: CoordinatorIdentity | None = None,
        coordinator_id: str = "coordinator",
        telemetry: ProductTelemetry | None = None,
        worker_trust_checker: Callable[[str], bool] | None = None,
        artifact_coordinator: DeploymentArtifactCoordinator | None = None,
    ) -> None:
        self.registry = registry
        self.transport = transport
        self.state_directory = state_directory.resolve()
        self.lease_seconds = lease_seconds
        self.control_timeout_s = control_timeout_s
        self.coordinator_identity = coordinator_identity
        self.coordinator_id = coordinator_id
        self.telemetry = telemetry
        self.worker_trust_checker = worker_trust_checker
        self.artifact_coordinator = artifact_coordinator
        self._lock = asyncio.Lock()
        self._reserved_workers: set[str] = set()
        self._deployments: dict[str, DeploymentStatus] = {}
        self._plans: dict[str, ProductStagePlan] = {}
        self._current_topology_id: str | None = None
        self._publication_destination: str | None = None
        self._load_persisted_state()

    @property
    def reserved_worker_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._reserved_workers))

    def ready_plan(self, *, model_id: str, model_revision: str) -> ProductStagePlan:
        topology_id = self._current_topology_id
        if topology_id is None:
            raise RuntimeError("no product topology is deployed")
        status = self._deployments[topology_id]
        plan = self._plans[topology_id]
        if not status.ready or status.phase != DeploymentPhase.READY:
            raise RuntimeError(f"topology {topology_id} is not ready")
        if plan.model.model_id != model_id or plan.model.model_revision != model_revision:
            raise RuntimeError(
                f"ready topology owns {plan.model.model_id}@{plan.model.model_revision}, "
                f"not {model_id}@{model_revision}"
            )
        self._require_trusted_plan(plan, action="new session admission")
        return plan

    def _require_trusted_plan(self, plan: ProductStagePlan, *, action: str) -> None:
        checker = self.worker_trust_checker
        if checker is None:
            return
        worker_ids = {
            *(item.worker_id for item in plan.assignments),
            *self._expert_worker_ids(plan),
        }
        untrusted = sorted(worker_id for worker_id in worker_ids if not checker(worker_id))
        if untrusted:
            raise IntegrityError(
                f"{action} rejected because worker trust was removed or is absent: "
                + ", ".join(untrusted)
            )

    @property
    def current_topology_id(self) -> str | None:
        return self._current_topology_id

    @property
    def current_generation(self) -> int | None:
        if self._current_topology_id is None:
            return None
        return self._deployments[self._current_topology_id].generation

    def statuses(self, topology_id: str | None = None) -> list[DeploymentStatus]:
        values = list(self._deployments.values())
        if topology_id is not None:
            values = [item for item in values if item.topology_id == topology_id]
        return [item.model_copy(deep=True) for item in sorted(values, key=lambda x: x.topology_id)]

    def _save(self, status: DeploymentStatus) -> None:
        directory = self.state_directory / "deployments"
        _atomic_write_text(
            directory / f"{status.deployment_id}.json",
            status.model_dump_json(indent=2) + "\n",
        )

    def _save_plan(self, plan: ProductStagePlan) -> None:
        directory = self.state_directory / "plans"
        _atomic_write_text(
            directory / f"{plan.plan_id}.json",
            plan.model_dump_json(indent=2) + "\n",
        )

    def _load_persisted_state(self) -> None:
        deployment_directory = self.state_directory / "deployments"
        plan_directory = self.state_directory / "plans"
        if not deployment_directory.is_dir() or not plan_directory.is_dir():
            return
        plans: dict[str, ProductStagePlan] = {}
        for path in sorted(plan_directory.glob("*.json")):
            plan = ProductStagePlan.model_validate_json(path.read_text(encoding="utf-8"))
            plans[plan.plan_id] = plan
        for path in sorted(deployment_directory.glob("*.json")):
            status = DeploymentStatus.model_validate_json(path.read_text(encoding="utf-8"))
            loaded_plan = plans.get(status.plan_id)
            if loaded_plan is None:
                continue
            # Connections and worker registrations are process-local. Keep the
            # evidence inspectable, but never advertise a pre-restart route as live.
            if status.ready:
                status = status.model_copy(
                    update={
                        "ready": False,
                        "phase": DeploymentPhase.FAILED,
                        "detail": (
                            "coordinator restarted; workers must re-register before route recovery"
                        ),
                        "updated_monotonic_ns": time.monotonic_ns(),
                    }
                )
            self._deployments[status.topology_id] = status
            self._plans[status.topology_id] = loaded_plan
            self._save(status)

    def _signed_route_lease(
        self,
        plan: ProductStagePlan,
        *,
        lease_expiry_unix_ns: int,
    ) -> SignedRouteLease | None:
        identity = self.coordinator_identity
        if identity is None:
            return None
        participants: list[RouteLeaseParticipant] = []
        for item in plan.assignments:
            capability = self.registry.capability(item.worker_id)
            participants.append(
                RouteLeaseParticipant(
                    worker_id=item.worker_id,
                    worker_public_key=capability.public_key,
                    worker_public_key_fingerprint=public_key_fingerprint(capability.public_key),
                    control_endpoint=item.control_endpoint,
                    data_endpoint=item.data_endpoint,
                    stage_id=item.stage_id,
                    assignment=item.assignment,
                    device=item.device,
                    dtype=plan.model.dtype,
                )
            )
        issued = time.time_ns()
        unsigned = SignedRouteLease(
            topology_id=plan.topology_id,
            route_generation=plan.generation,
            model_id=plan.model.model_id,
            model_revision=plan.model.model_revision,
            tokenizer_revision=plan.model.tokenizer_revision,
            adapter_id=plan.model.adapter_id,
            dtype=plan.model.dtype,
            participants=participants,
            lease_issued_unix_ns=issued,
            lease_expiry_unix_ns=lease_expiry_unix_ns,
            nonce=uuid4().hex,
            coordinator_identity=self.coordinator_id,
            coordinator_public_key=identity.public_key_b64,
            coordinator_public_key_fingerprint=identity.public_key_fingerprint,
        )
        return sign_route_lease(unsigned, identity)

    @staticmethod
    def _expert_worker_ids(plan: ProductStagePlan) -> set[str]:
        return {
            worker_id
            for stage in plan.expert_plans
            for placement in stage.placements
            if placement.strategy != "local"
            for worker_id in placement.worker_ids
        }

    def _signed_expert_route_lease(
        self,
        plan: ProductStagePlan,
        *,
        lease_expiry_unix_ns: int,
    ) -> SignedExpertRouteLease | None:
        expert_worker_ids = self._expert_worker_ids(plan)
        if not expert_worker_ids:
            return None
        identity = self.coordinator_identity
        if identity is None:
            raise RuntimeError("remote expert deployment requires a coordinator identity")
        if not plan.expert_model_fingerprint or not plan.expert_quantization_fingerprint:
            raise RuntimeError("remote expert deployment requires exact model identities")
        stage_assignments = {item.worker_id: item for item in plan.assignments}
        participant_ids = set(stage_assignments) | expert_worker_ids
        participants: list[ExpertRouteParticipant] = []
        for worker_id in sorted(participant_ids):
            capability = self.registry.capability(worker_id)
            roles: list[str] = []
            if worker_id in stage_assignments:
                roles.append(WorkerRole.CONTIGUOUS_STAGE.value)
            roles.extend(
                role.value
                for role in capability.roles
                if role
                in {
                    WorkerRole.WHOLE_EXPERT,
                    WorkerRole.EXPERT_MICROSHARD,
                    WorkerRole.REDUCER,
                }
            )
            assignment = stage_assignments.get(worker_id)
            endpoint = capability.expert_data_plane_endpoint or (
                assignment.data_endpoint if assignment is not None else ""
            )
            if not endpoint:
                raise RuntimeError(f"expert route participant {worker_id} has no endpoint")
            participants.append(
                ExpertRouteParticipant(
                    worker_id=worker_id,
                    worker_public_key=capability.public_key,
                    worker_public_key_fingerprint=public_key_fingerprint(capability.public_key),
                    endpoint=endpoint,
                    roles=roles,
                    owned_experts={
                        int(layer_id): [int(expert_id) for expert_id in expert_ids]
                        for layer_id, expert_ids in capability.owned_experts.items()
                    },
                    owned_microshards=list(capability.owned_microshards),
                    model_fingerprint=plan.expert_model_fingerprint,
                    quantization_fingerprint=plan.expert_quantization_fingerprint,
                )
            )
        issued = time.time_ns()
        unsigned = SignedExpertRouteLease(
            topology_id=plan.topology_id,
            route_generation=plan.generation,
            model_id=plan.model.model_id,
            model_revision=plan.model.model_revision,
            model_fingerprint=plan.expert_model_fingerprint,
            quantization_fingerprint=plan.expert_quantization_fingerprint,
            participants=participants,
            lease_issued_unix_ns=issued,
            lease_expiry_unix_ns=lease_expiry_unix_ns,
            nonce=uuid4().hex,
            coordinator_identity=self.coordinator_id,
            coordinator_public_key=identity.public_key_b64,
            coordinator_public_key_fingerprint=identity.public_key_fingerprint,
        )
        return sign_expert_route_lease(unsigned, identity)

    async def _install_expert_routes(
        self,
        plan: ProductStagePlan,
        lease: SignedExpertRouteLease | None,
    ) -> None:
        if lease is None:
            return

        async def install(worker_id: str) -> None:
            capability = self.registry.capability(worker_id)
            endpoint = capability.expert_data_plane_endpoint
            if endpoint is None:
                raise RuntimeError(f"expert worker {worker_id} has no data endpoint")
            client = ExpertTransportClient(endpoint, timeout_s=self.control_timeout_s)
            await asyncio.to_thread(
                client.control,
                "install_route",
                route_lease=lease.model_dump(mode="json"),
            )

        results = await asyncio.gather(
            *(install(worker_id) for worker_id in sorted(self._expert_worker_ids(plan))),
            return_exceptions=True,
        )
        errors = [str(item) for item in results if isinstance(item, BaseException)]
        if errors:
            raise RuntimeError("expert route installation failed: " + "; ".join(errors))

    async def _remove_expert_routes(self, plan: ProductStagePlan) -> list[str]:
        async def remove(worker_id: str) -> None:
            capability = self.registry.capability(worker_id)
            endpoint = capability.expert_data_plane_endpoint
            if endpoint is None:
                return
            client = ExpertTransportClient(endpoint, timeout_s=self.control_timeout_s)
            await asyncio.to_thread(
                client.control,
                "remove_route",
                topology_id=plan.topology_id,
                route_generation=plan.generation,
            )

        results = await asyncio.gather(
            *(remove(worker_id) for worker_id in sorted(self._expert_worker_ids(plan))),
            return_exceptions=True,
        )
        return [
            f"expert route removal failed: {type(item).__name__}: {item}"
            for item in results
            if isinstance(item, BaseException)
        ]

    def _transition(
        self,
        status: DeploymentStatus,
        phase: DeploymentPhase,
        *,
        detail: str = "",
        ready: bool = False,
    ) -> DeploymentStatus:
        updated = status.model_copy(
            update={
                "phase": phase,
                "detail": detail,
                "ready": ready,
                "updated_monotonic_ns": time.monotonic_ns(),
            },
            deep=True,
        )
        self._deployments[updated.topology_id] = updated
        self._save(updated)
        if self.telemetry is not None:
            self.telemetry.emit(
                "deployment_progress",
                deployment_id=updated.deployment_id,
                topology_id=updated.topology_id,
                phase=phase.value,
                detail=detail,
            )
        return updated

    async def deploy(
        self,
        plan: ProductStagePlan,
        *,
        publication_destination: str,
    ) -> DeploymentStatus:
        async with self._lock:
            self._require_trusted_plan(plan, action="route deployment")
            self._publication_destination = publication_destination
            existing = self._deployments.get(plan.topology_id)
            existing_plan = self._plans.get(plan.topology_id)
            if (
                existing is not None
                and existing.ready
                and existing.phase == DeploymentPhase.READY
                and existing.generation == plan.generation
                and existing.plan_id == plan.plan_id
                and existing_plan is not None
                and existing_plan.model == plan.model
                and existing_plan.assignments == plan.assignments
                and existing_plan.expert_plans == plan.expert_plans
            ):
                return existing.model_copy(update={"idempotent": True}, deep=True)
            if self._current_topology_id is not None:
                current = self._deployments[self._current_topology_id]
                await self._unload_locked(current, force=True, replacement=True)

            now = time.monotonic_ns()
            workers = [
                DeploymentWorkerStatus(
                    worker_id=item.worker_id,
                    stage_id=item.stage_id,
                    control_endpoint=item.control_endpoint,
                    data_endpoint=item.data_endpoint,
                )
                for item in plan.assignments
            ]
            status = DeploymentStatus(
                deployment_id=f"deployment-{uuid4().hex}",
                plan_id=plan.plan_id,
                topology_id=plan.topology_id,
                generation=plan.generation,
                model=plan.model,
                phase=DeploymentPhase.RESERVING,
                ready=False,
                workers=workers,
                created_monotonic_ns=now,
                updated_monotonic_ns=now,
            )
            self._deployments[plan.topology_id] = status
            self._plans[plan.topology_id] = plan
            self._save_plan(plan)
            self._save(status)
            if self.telemetry is not None:
                self.telemetry.emit(
                    "deployment_started",
                    deployment_id=status.deployment_id,
                    topology_id=plan.topology_id,
                    route_generation=plan.generation,
                    model_revision=plan.model.model_revision,
                )
            loaded_stage_ids: set[int] = set()
            installed_stage_ids: set[int] = set()
            try:
                all_worker_ids = {
                    *(item.worker_id for item in plan.assignments),
                    *self._expert_worker_ids(plan),
                }
                for worker_id in sorted(all_worker_ids):
                    healthy, _ = self.registry.registration_health(worker_id)
                    if not healthy:
                        raise RuntimeError(f"worker {worker_id} registration became unhealthy")
                    if worker_id in self._reserved_workers:
                        raise RuntimeError(f"worker {worker_id} is already reserved")
                    self._reserved_workers.add(worker_id)
                for item in plan.assignments:
                    status.workers[item.stage_id].reserved = True
                status = self._transition(
                    status,
                    DeploymentPhase.PREPARING_ARTIFACTS,
                    detail="preparing exact stage artifact identities",
                )
                if self.artifact_coordinator is not None:
                    await self.artifact_coordinator.prepare(plan)
                elif any(item.artifact_id is not None for item in plan.assignments):
                    raise RuntimeError(
                        "stage artifacts were requested but no artifact manager exists"
                    )
                status = self._transition(
                    status,
                    DeploymentPhase.TRANSFERRING_ARTIFACTS,
                    detail="placing stage artifacts on their assigned workers",
                )
                if self.artifact_coordinator is not None:
                    await self.artifact_coordinator.transfer(plan)
                status = self._transition(
                    status,
                    DeploymentPhase.VERIFYING_ARTIFACTS,
                    detail="verifying complete artifact and ownership hashes",
                )
                if self.artifact_coordinator is not None:
                    await self.artifact_coordinator.verify(plan)
                status = self._transition(status, DeploymentPhase.LOADING)
                lease_expiry = time.time_ns() + int(self.lease_seconds * 1_000_000_000)
                load_requests = [
                    LoadStageRequest(
                        worker_id=item.worker_id,
                        request_id=f"{status.deployment_id}:load:{item.stage_id}",
                        model_id=plan.model.model_id,
                        model_revision=plan.model.model_revision,
                        tokenizer_revision=plan.model.tokenizer_revision,
                        topology_id=plan.topology_id,
                        route_generation=plan.generation,
                        stage_count=plan.stage_count,
                        assignment=item.assignment,
                        device=item.device,
                        dtype=plan.model.dtype,
                        artifact_id=item.artifact_id,
                        model_path=None,
                        allow_download=(
                            plan.model.resolution_policy == ModelResolutionPolicy.ALLOW_DOWNLOAD
                        ),
                        lease_expiry_unix_ns=lease_expiry,
                        deadline_unix_ns=(
                            time.time_ns() + int(self.control_timeout_s * 1_000_000_000)
                        ),
                        expert_plan=(
                            plan.expert_plans[item.stage_id].model_dump(mode="json")
                            if plan.expert_plans
                            else None
                        ),
                        expert_model_fingerprint=(
                            plan.expert_model_fingerprint if plan.expert_plans else None
                        ),
                        expert_quantization_fingerprint=(
                            plan.expert_quantization_fingerprint if plan.expert_plans else None
                        ),
                    )
                    for item in plan.assignments
                ]
                load_results = await asyncio.gather(
                    *(
                        self.transport.load_stage(item.control_endpoint, request)
                        for item, request in zip(plan.assignments, load_requests, strict=True)
                    ),
                    return_exceptions=True,
                )
                load_errors: list[str] = []
                for item, result in zip(plan.assignments, load_results, strict=True):
                    if isinstance(result, BaseException):
                        load_errors.append(
                            f"stage {item.stage_id} load failed on {item.worker_id}: {result}"
                        )
                    elif not result.accepted:
                        load_errors.append(f"stage {item.stage_id} load rejected: {result.detail}")
                    else:
                        loaded_stage_ids.add(item.stage_id)
                        status.workers[item.stage_id].loaded = True
                if load_errors:
                    raise RuntimeError("; ".join(load_errors))

                status = self._transition(status, DeploymentPhase.VERIFYING_LOADS)
                stage_statuses = await asyncio.gather(
                    *(
                        self.transport.get_stage_status(
                            item.control_endpoint,
                            GetStageStatusRequest(
                                worker_id=item.worker_id,
                                request_id=f"{status.deployment_id}:verify-load:{item.stage_id}",
                                topology_id=plan.topology_id,
                            ),
                        )
                        for item in plan.assignments
                    )
                )
                for item, worker_status in zip(plan.assignments, stage_statuses, strict=True):
                    loaded = worker_status.loaded_stage
                    if (
                        loaded is None
                        or loaded.model_id != plan.model.model_id
                        or loaded.model_revision != plan.model.model_revision
                        or loaded.tokenizer_revision != plan.model.tokenizer_revision
                        or loaded.topology_id != plan.topology_id
                        or loaded.assignment != item.assignment
                        or loaded.artifact_id != item.artifact_id
                        or int(loaded.ownership.get("stage_id", -1)) != item.stage_id
                        or int(loaded.ownership.get("layer_start", -1))
                        != item.assignment.layer_start
                        or int(loaded.ownership.get("layer_end", -1)) != item.assignment.layer_end
                    ):
                        raise RuntimeError(
                            f"worker {item.worker_id} did not prove exact stage ownership"
                        )
                    worker_record = status.workers[item.stage_id]
                    worker_record.ownership_verified = True
                    worker_record.process_id = worker_status.process_id
                    worker_record.load_count = loaded.load_count

                status = self._transition(status, DeploymentPhase.INSTALLING_ROUTES)
                signed_lease = self._signed_route_lease(
                    plan,
                    lease_expiry_unix_ns=lease_expiry,
                )
                signed_expert_lease = self._signed_expert_route_lease(
                    plan,
                    lease_expiry_unix_ns=lease_expiry,
                )
                await self._install_expert_routes(plan, signed_expert_lease)
                route_requests: list[InstallStageRouteRequest] = []
                for index, item in enumerate(plan.assignments):
                    previous = plan.assignments[index - 1] if index > 0 else None
                    following = (
                        plan.assignments[index + 1] if index + 1 < plan.stage_count else None
                    )
                    route_requests.append(
                        InstallStageRouteRequest(
                            worker_id=item.worker_id,
                            request_id=f"{status.deployment_id}:route:{item.stage_id}",
                            model_id=plan.model.model_id,
                            model_revision=plan.model.model_revision,
                            tokenizer_revision=plan.model.tokenizer_revision,
                            topology_id=plan.topology_id,
                            route_generation=plan.generation,
                            assignment=item.assignment,
                            device=item.device,
                            dtype=plan.model.dtype,
                            previous_stage=(
                                StageRouteEndpoint(
                                    worker_id=previous.worker_id,
                                    stage_id=previous.stage_id,
                                    data_endpoint=previous.data_endpoint,
                                    assignment=previous.assignment,
                                )
                                if previous is not None
                                else None
                            ),
                            next_stage=(
                                StageRouteEndpoint(
                                    worker_id=following.worker_id,
                                    stage_id=following.stage_id,
                                    data_endpoint=following.data_endpoint,
                                    assignment=following.assignment,
                                )
                                if following is not None
                                else None
                            ),
                            stage_count=plan.stage_count,
                            stage_zero_publication_destination=(
                                publication_destination if item.stage_id == 0 else None
                            ),
                            lease_expiry_unix_ns=lease_expiry,
                            deadline_unix_ns=(
                                time.time_ns() + int(self.control_timeout_s * 1_000_000_000)
                            ),
                            route_lease=signed_lease,
                            expert_route_lease=signed_expert_lease,
                        )
                    )
                route_results = await asyncio.gather(
                    *(
                        self.transport.install_stage_route(item.control_endpoint, request)
                        for item, request in zip(plan.assignments, route_requests, strict=True)
                    ),
                    return_exceptions=True,
                )
                route_errors: list[str] = []
                for item, result in zip(plan.assignments, route_results, strict=True):
                    if isinstance(result, BaseException):
                        route_errors.append(
                            f"stage {item.stage_id} route installation failed: {result}"
                        )
                    elif not result.accepted:
                        route_errors.append(
                            f"stage {item.stage_id} route installation rejected: {result.detail}"
                        )
                    else:
                        installed_stage_ids.add(item.stage_id)
                        status.workers[item.stage_id].route_installed = True
                if route_errors:
                    raise RuntimeError("; ".join(route_errors))

                status = self._transition(status, DeploymentPhase.VERIFYING_PEERS)
                for item in reversed(plan.assignments):
                    result = await self.transport.verify_stage_route(
                        item.control_endpoint,
                        VerifyStageRouteRequest(
                            worker_id=item.worker_id,
                            request_id=f"{status.deployment_id}:peer:{item.stage_id}",
                            model_id=plan.model.model_id,
                            model_revision=plan.model.model_revision,
                            tokenizer_revision=plan.model.tokenizer_revision,
                            topology_id=plan.topology_id,
                            route_generation=plan.generation,
                            stage_id=item.stage_id,
                            device=item.device,
                            dtype=plan.model.dtype,
                            deadline_unix_ns=(
                                time.time_ns() + int(self.control_timeout_s * 1_000_000_000)
                            ),
                        ),
                    )
                    if not result.accepted:
                        raise RuntimeError(
                            f"stage {item.stage_id} peer verification rejected: {result.detail}"
                        )
                    status.workers[item.stage_id].peer_verified = True
                status = self._transition(
                    status,
                    DeploymentPhase.READY,
                    detail="all stages loaded, ownership-verified, routed, and connected",
                    ready=True,
                )
                self._current_topology_id = plan.topology_id
                if self.telemetry is not None:
                    self.telemetry.emit(
                        "deployment_ready",
                        deployment_id=status.deployment_id,
                        topology_id=plan.topology_id,
                        route_generation=plan.generation,
                    )
                    self.telemetry.emit(
                        "route_generation_installed",
                        topology_id=plan.topology_id,
                        route_generation=plan.generation,
                        worker_ids=[item.worker_id for item in plan.assignments],
                    )
                return status.model_copy(deep=True)
            except BaseException as exc:
                status = self._transition(
                    status,
                    DeploymentPhase.ROLLING_BACK,
                    detail=f"{type(exc).__name__}: {exc}",
                )
                rollback_errors = await self._rollback(
                    plan,
                    status,
                    loaded_stage_ids=loaded_stage_ids,
                    installed_stage_ids=installed_stage_ids,
                )
                detail = f"deployment rolled back: {type(exc).__name__}: {exc}"
                if rollback_errors:
                    detail += "; rollback incomplete: " + "; ".join(rollback_errors)
                status = self._transition(
                    status,
                    DeploymentPhase.FAILED,
                    detail=detail,
                )
                raise RuntimeError(status.detail) from exc

    def _eligible_replacements(
        self,
        *,
        plan: ProductStagePlan,
        failed: PlanWorkerAssignment,
        excluded_worker_ids: set[str],
    ) -> list[PlanWorkerAssignment]:
        failed_capability = self.registry.capability(failed.worker_id)
        candidates: list[PlanWorkerAssignment] = []
        for capability in self.registry.healthy_workers():
            if capability.worker_id in excluded_worker_ids:
                continue
            if self.worker_trust_checker is not None and not self.worker_trust_checker(
                capability.worker_id
            ):
                continue
            control_endpoint = capability.control_endpoint or capability.endpoint
            data_endpoint = capability.data_plane_endpoint
            same_device = capability.device_identifier == failed.device
            same_protocol = (
                capability.stage_ring_protocol_version
                == failed_capability.stage_ring_protocol_version
            )
            adapter_ok = plan.model.adapter_id in capability.supported_model_adapters
            dtype_ok = plan.model.dtype in {
                *capability.supported_dtypes,
                *capability.supported_activation_dtypes,
            }
            if (
                not capability.stage_runtime_enabled
                or control_endpoint is None
                or data_endpoint is None
                or not same_device
                or not same_protocol
                or not adapter_ok
                or not dtype_ok
                or capability.backend != failed_capability.backend
                or capability.effective_memory_bytes < failed.required_memory_bytes
            ):
                continue
            candidates.append(
                failed.model_copy(
                    update={
                        "worker_id": capability.worker_id,
                        "control_endpoint": control_endpoint,
                        "data_endpoint": data_endpoint,
                        "device": capability.device_identifier,
                        "effective_memory_bytes": capability.effective_memory_bytes,
                    },
                    deep=True,
                )
            )
        return sorted(
            candidates,
            key=lambda item: (
                self.registry.capability(item.worker_id).active_session_count,
                self.registry.capability(item.worker_id).current_queue_depth,
                item.worker_id,
            ),
        )

    async def recover(
        self,
        *,
        failed_worker_ids: set[str],
        publication_destination: str | None = None,
    ) -> ProductStagePlan:
        """Replace failed stages, install one newer route, and keep weights resident."""

        async with self._lock:
            topology_id = self._current_topology_id
            if topology_id is None:
                raise RuntimeError("no active topology is available for recovery")
            status = self._deployments[topology_id]
            plan = self._plans[topology_id]
            destination = publication_destination or self._publication_destination
            if destination is None:
                raise RuntimeError("coordinator publication endpoint is unavailable")
            status = self._transition(
                status,
                DeploymentPhase.RECOVERING,
                detail="rebuilding route generation after a stage-ring failure",
            )
            selected_assignments = list(plan.assignments)
            replacement_stage_ids: set[int] = set()
            excluded = {
                item.worker_id
                for item in plan.assignments
                if item.worker_id not in failed_worker_ids
            } | set(failed_worker_ids)
            for item in plan.assignments:
                if item.worker_id not in failed_worker_ids:
                    continue
                candidates = self._eligible_replacements(
                    plan=plan,
                    failed=item,
                    excluded_worker_ids=excluded,
                )
                if not candidates:
                    raise RuntimeError(
                        f"no exact eligible replacement for stage {item.stage_id} "
                        f"worker {item.worker_id}"
                    )
                replacement = candidates[0]
                selected_assignments[item.stage_id] = replacement
                replacement_stage_ids.add(item.stage_id)
                excluded.add(replacement.worker_id)
                self._reserved_workers.discard(item.worker_id)
                self._reserved_workers.add(replacement.worker_id)
                if self.telemetry is not None:
                    self.telemetry.emit(
                        "replacement_selected",
                        topology_id=plan.topology_id,
                        failed_worker_id=item.worker_id,
                        replacement_worker_id=replacement.worker_id,
                        stage_id=item.stage_id,
                    )

            generation = plan.generation + 1
            new_plan = plan.model_copy(
                update={
                    "generation": generation,
                    "assignments": selected_assignments,
                    "report": plan.report.model_copy(
                        update={"worker_assignments": selected_assignments},
                        deep=True,
                    ),
                },
                deep=True,
            )
            self._require_trusted_plan(new_plan, action="replacement route deployment")
            lease_expiry = time.time_ns() + int(self.lease_seconds * 1_000_000_000)
            deadline = time.time_ns() + int(self.control_timeout_s * 1_000_000_000)
            replacement_items = [
                item for item in new_plan.assignments if item.stage_id in replacement_stage_ids
            ]
            try:
                load_results = await asyncio.wait_for(
                    asyncio.gather(
                        *(
                            self.transport.load_stage(
                                item.control_endpoint,
                                LoadStageRequest(
                                    worker_id=item.worker_id,
                                    request_id=(
                                        f"{status.deployment_id}:recover-load:{generation}:"
                                        f"{item.stage_id}"
                                    ),
                                    model_id=new_plan.model.model_id,
                                    model_revision=new_plan.model.model_revision,
                                    tokenizer_revision=new_plan.model.tokenizer_revision,
                                    topology_id=new_plan.topology_id,
                                    route_generation=generation,
                                    stage_count=new_plan.stage_count,
                                    assignment=item.assignment,
                                    device=item.device,
                                    dtype=new_plan.model.dtype,
                                    allow_download=(
                                        new_plan.model.resolution_policy
                                        == ModelResolutionPolicy.ALLOW_DOWNLOAD
                                    ),
                                    lease_expiry_unix_ns=lease_expiry,
                                    deadline_unix_ns=deadline,
                                    expert_plan=(
                                        new_plan.expert_plans[item.stage_id].model_dump(mode="json")
                                        if new_plan.expert_plans
                                        else None
                                    ),
                                    expert_model_fingerprint=(
                                        new_plan.expert_model_fingerprint
                                        if new_plan.expert_plans
                                        else None
                                    ),
                                    expert_quantization_fingerprint=(
                                        new_plan.expert_quantization_fingerprint
                                        if new_plan.expert_plans
                                        else None
                                    ),
                                ),
                            )
                            for item in replacement_items
                        ),
                        return_exceptions=True,
                    ),
                    timeout=self.control_timeout_s,
                )
                errors = [
                    str(result)
                    for result in load_results
                    if isinstance(result, BaseException) or not result.accepted
                ]
                if errors:
                    raise RuntimeError("replacement stage load failed: " + "; ".join(errors))

                for item in replacement_items:
                    worker_status = await asyncio.wait_for(
                        self.transport.get_stage_status(
                            item.control_endpoint,
                            GetStageStatusRequest(
                                worker_id=item.worker_id,
                                request_id=(
                                    f"{status.deployment_id}:recover-verify:{generation}:"
                                    f"{item.stage_id}"
                                ),
                                topology_id=new_plan.topology_id,
                                deadline_unix_ns=deadline,
                            ),
                        ),
                        timeout=self.control_timeout_s,
                    )
                    loaded = worker_status.loaded_stage
                    if (
                        loaded is None
                        or loaded.model_id != new_plan.model.model_id
                        or loaded.model_revision != new_plan.model.model_revision
                        or loaded.tokenizer_revision != new_plan.model.tokenizer_revision
                        or loaded.assignment != item.assignment
                        or loaded.device != item.device
                        or loaded.dtype != new_plan.model.dtype
                    ):
                        raise RuntimeError(
                            f"replacement worker {item.worker_id} failed exact load verification"
                        )

                signed_lease = self._signed_route_lease(
                    new_plan,
                    lease_expiry_unix_ns=lease_expiry,
                )
                signed_expert_lease = self._signed_expert_route_lease(
                    new_plan,
                    lease_expiry_unix_ns=lease_expiry,
                )
                await self._install_expert_routes(new_plan, signed_expert_lease)
                route_requests: list[InstallStageRouteRequest] = []
                for index, item in enumerate(new_plan.assignments):
                    previous = new_plan.assignments[index - 1] if index > 0 else None
                    following = (
                        new_plan.assignments[index + 1]
                        if index + 1 < new_plan.stage_count
                        else None
                    )
                    route_requests.append(
                        InstallStageRouteRequest(
                            worker_id=item.worker_id,
                            request_id=(
                                f"{status.deployment_id}:recover-route:{generation}:{item.stage_id}"
                            ),
                            model_id=new_plan.model.model_id,
                            model_revision=new_plan.model.model_revision,
                            tokenizer_revision=new_plan.model.tokenizer_revision,
                            topology_id=new_plan.topology_id,
                            route_generation=generation,
                            assignment=item.assignment,
                            device=item.device,
                            dtype=new_plan.model.dtype,
                            previous_stage=(
                                StageRouteEndpoint(
                                    worker_id=previous.worker_id,
                                    stage_id=previous.stage_id,
                                    data_endpoint=previous.data_endpoint,
                                    assignment=previous.assignment,
                                )
                                if previous is not None
                                else None
                            ),
                            next_stage=(
                                StageRouteEndpoint(
                                    worker_id=following.worker_id,
                                    stage_id=following.stage_id,
                                    data_endpoint=following.data_endpoint,
                                    assignment=following.assignment,
                                )
                                if following is not None
                                else None
                            ),
                            stage_count=new_plan.stage_count,
                            stage_zero_publication_destination=(
                                destination if item.stage_id == 0 else None
                            ),
                            lease_expiry_unix_ns=lease_expiry,
                            deadline_unix_ns=deadline,
                            replace=True,
                            route_lease=signed_lease,
                            expert_route_lease=signed_expert_lease,
                        )
                    )
                route_results = await asyncio.wait_for(
                    asyncio.gather(
                        *(
                            self.transport.install_stage_route(item.control_endpoint, request)
                            for item, request in zip(
                                new_plan.assignments,
                                route_requests,
                                strict=True,
                            )
                        ),
                        return_exceptions=True,
                    ),
                    timeout=self.control_timeout_s,
                )
                route_errors = [
                    str(result)
                    for result in route_results
                    if isinstance(result, BaseException) or not result.accepted
                ]
                if route_errors:
                    raise RuntimeError(
                        "replacement route installation failed: " + "; ".join(route_errors)
                    )
                for item in reversed(new_plan.assignments):
                    result = await asyncio.wait_for(
                        self.transport.verify_stage_route(
                            item.control_endpoint,
                            VerifyStageRouteRequest(
                                worker_id=item.worker_id,
                                request_id=(
                                    f"{status.deployment_id}:recover-peer:{generation}:"
                                    f"{item.stage_id}"
                                ),
                                model_id=new_plan.model.model_id,
                                model_revision=new_plan.model.model_revision,
                                tokenizer_revision=new_plan.model.tokenizer_revision,
                                topology_id=new_plan.topology_id,
                                route_generation=generation,
                                stage_id=item.stage_id,
                                device=item.device,
                                dtype=new_plan.model.dtype,
                                deadline_unix_ns=deadline,
                            ),
                        ),
                        timeout=self.control_timeout_s,
                    )
                    if not result.accepted:
                        raise RuntimeError(
                            f"replacement route peer verification failed at stage "
                            f"{item.stage_id}: {result.detail}"
                        )
            except asyncio.CancelledError:
                # Request cancellation owns session cleanup.  Do not roll back the
                # deployment here: loaded stages are shared product resources and
                # must remain resident when one request is cancelled.
                raise
            except BaseException as exc:
                all_stage_ids = {item.stage_id for item in new_plan.assignments}
                rollback_errors = await self._rollback(
                    new_plan,
                    status,
                    loaded_stage_ids=all_stage_ids,
                    installed_stage_ids=all_stage_ids,
                )
                for item in [*plan.assignments, *new_plan.assignments]:
                    self._reserved_workers.discard(item.worker_id)
                self._current_topology_id = None
                detail = f"route recovery failed: {type(exc).__name__}: {exc}"
                if rollback_errors:
                    detail += "; bounded cleanup incomplete: " + "; ".join(rollback_errors)
                self._transition(
                    status,
                    DeploymentPhase.FAILED,
                    detail=detail,
                )
                raise

            workers = [
                DeploymentWorkerStatus(
                    worker_id=item.worker_id,
                    stage_id=item.stage_id,
                    control_endpoint=item.control_endpoint,
                    data_endpoint=item.data_endpoint,
                    reserved=True,
                    loaded=True,
                    ownership_verified=True,
                    route_installed=True,
                    peer_verified=True,
                    detail=(
                        "replacement loaded and verified"
                        if item.stage_id in replacement_stage_ids
                        else "resident stage reused for newer route generation"
                    ),
                )
                for item in new_plan.assignments
            ]
            status = status.model_copy(
                update={
                    "generation": generation,
                    "workers": workers,
                    "updated_monotonic_ns": time.monotonic_ns(),
                },
                deep=True,
            )
            self._plans[topology_id] = new_plan
            self._save_plan(new_plan)
            status = self._transition(
                status,
                DeploymentPhase.READY,
                detail="replacement topology installed and peer-verified",
                ready=True,
            )
            if self.telemetry is not None:
                self.telemetry.emit(
                    "route_generation_installed",
                    topology_id=new_plan.topology_id,
                    route_generation=new_plan.generation,
                    worker_ids=[item.worker_id for item in new_plan.assignments],
                )
            return new_plan.model_copy(deep=True)

    async def _rollback(
        self,
        plan: ProductStagePlan,
        status: DeploymentStatus,
        *,
        loaded_stage_ids: set[int],
        installed_stage_ids: set[int],
        force_unload: bool = True,
    ) -> list[str]:
        errors: list[str] = []
        route_targets = [item for item in plan.assignments if item.stage_id in installed_stage_ids]
        route_results = await asyncio.gather(
            *(
                self.transport.remove_stage_route(
                    item.control_endpoint,
                    RemoveStageRouteRequest(
                        worker_id=item.worker_id,
                        request_id=f"{status.deployment_id}:rollback-route:{item.stage_id}",
                        model_id=plan.model.model_id,
                        model_revision=plan.model.model_revision,
                        tokenizer_revision=plan.model.tokenizer_revision,
                        topology_id=plan.topology_id,
                        route_generation=plan.generation,
                        stage_id=item.stage_id,
                        device=item.device,
                        dtype=plan.model.dtype,
                    ),
                )
                for item in route_targets
            ),
            return_exceptions=True,
        )
        removed_stage_ids: set[int] = set()
        for item, result in zip(route_targets, route_results, strict=True):
            if isinstance(result, BaseException):
                errors.append(
                    f"stage {item.stage_id} route removal failed: {type(result).__name__}: {result}"
                )
            elif not result.accepted:
                errors.append(f"stage {item.stage_id} route removal rejected: {result.detail}")
            else:
                removed_stage_ids.add(item.stage_id)

        load_targets = [item for item in plan.assignments if item.stage_id in loaded_stage_ids]
        unload_results = await asyncio.gather(
            *(
                self.transport.unload_stage(
                    item.control_endpoint,
                    UnloadStageRequest(
                        worker_id=item.worker_id,
                        request_id=f"{status.deployment_id}:rollback-load:{item.stage_id}",
                        model_id=plan.model.model_id,
                        model_revision=plan.model.model_revision,
                        tokenizer_revision=plan.model.tokenizer_revision,
                        topology_id=plan.topology_id,
                        route_generation=plan.generation,
                        stage_count=plan.stage_count,
                        assignment=item.assignment,
                        device=item.device,
                        dtype=plan.model.dtype,
                        force=force_unload,
                    ),
                )
                for item in load_targets
            ),
            return_exceptions=True,
        )
        unloaded_stage_ids: set[int] = set()
        for item, result in zip(load_targets, unload_results, strict=True):
            if isinstance(result, BaseException):
                errors.append(
                    f"stage {item.stage_id} unload failed: {type(result).__name__}: {result}"
                )
            elif not result.accepted:
                errors.append(f"stage {item.stage_id} unload rejected: {result.detail}")
            else:
                unloaded_stage_ids.add(item.stage_id)

        for item in plan.assignments:
            worker = status.workers[item.stage_id]
            if item.stage_id not in loaded_stage_ids or item.stage_id in unloaded_stage_ids:
                self._reserved_workers.discard(item.worker_id)
                worker.reserved = False
                worker.loaded = False
                worker.ownership_verified = False
            if (
                item.stage_id not in installed_stage_ids
                or item.stage_id in removed_stage_ids
                or item.stage_id in unloaded_stage_ids
            ):
                worker.route_installed = False
                worker.peer_verified = False
        errors.extend(await self._remove_expert_routes(plan))
        if self.artifact_coordinator is not None:
            try:
                await self.artifact_coordinator.release(plan)
            except Exception as exc:
                errors.append(f"artifact lease release failed: {type(exc).__name__}: {exc}")
        for worker_id in self._expert_worker_ids(plan):
            self._reserved_workers.discard(worker_id)
        return errors

    async def unload(
        self,
        *,
        topology_id: str | None,
        force: bool,
    ) -> DeploymentStatus | None:
        async with self._lock:
            selected = topology_id or self._current_topology_id
            if selected is None:
                return None
            try:
                status = self._deployments[selected]
            except KeyError as exc:
                raise RuntimeError(f"unknown topology {selected}") from exc
            return await self._unload_locked(status, force=force, replacement=False)

    async def _unload_locked(
        self,
        status: DeploymentStatus,
        *,
        force: bool,
        replacement: bool,
    ) -> DeploymentStatus:
        if status.phase == DeploymentPhase.UNLOADED:
            return status.model_copy(update={"idempotent": True}, deep=True)
        plan = self._plans[status.topology_id]
        if not force:
            worker_statuses = await asyncio.gather(
                *(
                    self.transport.get_stage_status(
                        item.control_endpoint,
                        GetStageStatusRequest(
                            worker_id=item.worker_id,
                            request_id=(f"{status.deployment_id}:unload-preflight:{item.stage_id}"),
                            topology_id=plan.topology_id,
                        ),
                    )
                    for item in plan.assignments
                )
            )
            active = [
                f"stage {item.stage_id} ({len(worker_status.sessions)} sessions)"
                for item, worker_status in zip(plan.assignments, worker_statuses, strict=True)
                if worker_status.sessions
            ]
            if active:
                raise RuntimeError(
                    "deployment has active sessions; use force to cancel them: " + ", ".join(active)
                )
        status = self._transition(
            status,
            DeploymentPhase.UNLOADING,
            detail="deployment replacement" if replacement else "explicit unload",
        )
        rollback_errors = await self._rollback(
            plan,
            status,
            loaded_stage_ids={item.stage_id for item in plan.assignments},
            installed_stage_ids={item.stage_id for item in plan.assignments},
            force_unload=force,
        )
        if rollback_errors:
            detail = "explicit unload incomplete: " + "; ".join(rollback_errors)
            status = self._transition(status, DeploymentPhase.FAILED, detail=detail)
            raise RuntimeError(status.detail)
        status = self._transition(
            status,
            DeploymentPhase.UNLOADED,
            detail="resident stages explicitly unloaded",
        )
        if self._current_topology_id == status.topology_id:
            self._current_topology_id = None
        if self.telemetry is not None:
            for item in plan.assignments:
                self.telemetry.emit(
                    "stage_unloaded",
                    topology_id=plan.topology_id,
                    route_generation=plan.generation,
                    worker_id=item.worker_id,
                    stage_id=item.stage_id,
                )
        return status.model_copy(deep=True)


__all__ = [
    "DeploymentArtifactCoordinator",
    "DeploymentManager",
    "DeploymentTransport",
    "TransportArtifactCoordinator",
]
