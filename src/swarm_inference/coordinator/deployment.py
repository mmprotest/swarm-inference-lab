"""Transactional persistent-stage deployment owned by ``CoordinatorCore``."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from swarm_inference.coordinator.registry import WorkerRegistry
from swarm_inference.model.product import ModelResolutionPolicy
from swarm_inference.protocol.product import (
    DeploymentPhase,
    DeploymentStatus,
    DeploymentWorkerStatus,
    ProductStagePlan,
)
from swarm_inference.protocol.stage_worker import (
    GetStageStatusRequest,
    InstallStageRouteRequest,
    LoadStageRequest,
    RemoveStageRouteRequest,
    StageActionResponse,
    StageRouteEndpoint,
    StageStatusResponse,
    UnloadStageRequest,
    VerifyStageRouteRequest,
)


class DeploymentTransport(Protocol):
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
    ) -> None:
        self.registry = registry
        self.transport = transport
        self.state_directory = state_directory.resolve()
        self.lease_seconds = lease_seconds
        self.control_timeout_s = control_timeout_s
        self._lock = asyncio.Lock()
        self._reserved_workers: set[str] = set()
        self._deployments: dict[str, DeploymentStatus] = {}
        self._plans: dict[str, ProductStagePlan] = {}
        self._current_topology_id: str | None = None

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
        return plan

    def statuses(self, topology_id: str | None = None) -> list[DeploymentStatus]:
        values = list(self._deployments.values())
        if topology_id is not None:
            values = [item for item in values if item.topology_id == topology_id]
        return [item.model_copy(deep=True) for item in sorted(values, key=lambda x: x.topology_id)]

    def _save(self, status: DeploymentStatus) -> None:
        directory = self.state_directory / "deployments"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{status.deployment_id}.json").write_text(
            status.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

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
        return updated

    async def deploy(
        self,
        plan: ProductStagePlan,
        *,
        publication_destination: str,
    ) -> DeploymentStatus:
        async with self._lock:
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
            self._save(status)
            loaded_stage_ids: set[int] = set()
            installed_stage_ids: set[int] = set()
            try:
                for item in plan.assignments:
                    healthy, _ = self.registry.registration_health(item.worker_id)
                    if not healthy:
                        raise RuntimeError(f"worker {item.worker_id} registration became unhealthy")
                    if item.worker_id in self._reserved_workers:
                        raise RuntimeError(f"worker {item.worker_id} is already reserved")
                    self._reserved_workers.add(item.worker_id)
                    status.workers[item.stage_id].reserved = True
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
                        model_path=None,
                        allow_download=(
                            plan.model.resolution_policy == ModelResolutionPolicy.ALLOW_DOWNLOAD
                        ),
                        lease_expiry_unix_ns=lease_expiry,
                        deadline_unix_ns=(
                            time.time_ns() + int(self.control_timeout_s * 1_000_000_000)
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
        return status.model_copy(deep=True)


__all__ = ["DeploymentManager", "DeploymentTransport"]
