from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from swarm_inference.config.models import Backend, WorkerCapability
from swarm_inference.coordinator.deployment import DeploymentManager
from swarm_inference.coordinator.registry import WorkerRegistry
from swarm_inference.model.partition import StageAssignment
from swarm_inference.model.product import ModelResolutionPolicy, ProductModelSpec
from swarm_inference.protocol.product import (
    DeploymentPhase,
    PlanCandidateReport,
    PlanWorkerAssignment,
    ProductStagePlan,
    StagePlanReport,
)
from swarm_inference.protocol.stage_worker import (
    GetStageStatusRequest,
    InstalledStageRouteStatus,
    InstallStageRouteRequest,
    LoadedStageStatus,
    LoadStageRequest,
    RemoveStageRouteRequest,
    StageActionResponse,
    StageSessionStatus,
    StageStatusResponse,
    UnloadStageRequest,
    VerifyStageRouteRequest,
)


def _assignment(stage_id: int) -> StageAssignment:
    return StageAssignment(
        stage_id=stage_id,
        layer_start=stage_id,
        layer_end=stage_id + 1,
        layer_ids=(stage_id,),
        weight_bytes=100,
        estimated_compute_ns=10,
        measured_compute_ns=10,
        kv_cache_bytes_per_token=1,
        peak_temporary_bytes=10,
        activation_bytes=8,
        device="cpu",
        owns_embeddings=stage_id == 0,
        owns_final_norm=stage_id == 1,
        owns_output_projection=stage_id == 1,
    )


def _capability(stage_id: int) -> WorkerCapability:
    return WorkerCapability(
        worker_id=f"worker-{stage_id}",
        public_key=f"key-{stage_id}",
        hostname="localhost",
        operating_system="test",
        architecture="test",
        backend=Backend.TORCH_CPU,
        cpu_model="test",
        logical_cpu_count=1,
        total_ram_bytes=10_000,
        available_ram_bytes=10_000,
        supported_dtypes=["float32"],
        upload_bandwidth_bytes_s=1_000,
        download_bandwidth_bytes_s=1_000,
        coordinator_latency_ms=0,
        memory_limit_bytes=10_000,
        endpoint=f"127.0.0.1:{50_000 + stage_id}",
        control_endpoint=f"127.0.0.1:{50_000 + stage_id}",
        data_plane_endpoint=f"127.0.0.1:{51_000 + stage_id}",
        device_identifier="cpu",
        stage_runtime_enabled=True,
    )


def _plan() -> ProductStagePlan:
    assignments = [
        PlanWorkerAssignment(
            stage_id=stage_id,
            worker_id=f"worker-{stage_id}",
            control_endpoint=f"127.0.0.1:{50_000 + stage_id}",
            data_endpoint=f"127.0.0.1:{51_000 + stage_id}",
            device="cpu",
            effective_memory_bytes=10_000,
            required_memory_bytes=200,
            assignment=_assignment(stage_id),
        )
        for stage_id in range(2)
    ]
    selected = PlanCandidateReport(
        name="two-stage-equal-ring",
        topology="two-stage-equal-ring",
        stage_count=2,
        partition_method="equal",
        feasible=True,
        selected=True,
        worker_ids=["worker-0", "worker-1"],
        expected_critical_path_ms=2,
        expected_utility_tokens_s=500,
    )
    return ProductStagePlan(
        plan_id="plan-test",
        topology_id="topology-test",
        generation=1,
        created_monotonic_ns=time.monotonic_ns(),
        model=ProductModelSpec(
            model_id="test/olmoe",
            model_revision="model-commit",
            tokenizer_revision="tokenizer-commit",
            adapter_id="olmoe",
            dtype="float32",
            layer_count=2,
            hidden_size=4,
            metadata_hash="metadata",
            resolution_policy=ModelResolutionPolicy.LOCAL_ONLY,
        ),
        stage_count=2,
        partition_method="equal",
        max_sequence_tokens=128,
        assignments=assignments,
        report=StagePlanReport(
            selected_topology="two-stage-equal-ring",
            worker_assignments=assignments,
            reason_for_selection="test measured plan",
            candidates=[selected],
            worker_eligibility=[],
        ),
    )


class _DeploymentTransport:
    def __init__(
        self,
        *,
        fail_load_stage: int | None = None,
        fail_peer_stage: int | None = None,
    ) -> None:
        self.fail_load_stage = fail_load_stage
        self.fail_peer_stage = fail_peer_stage
        self.loads: dict[int, LoadStageRequest] = {}
        self.routes: dict[int, InstallStageRouteRequest] = {}
        self.load_calls: list[int] = []
        self.unload_calls: list[int] = []
        self.remove_route_calls: list[int] = []
        self.sessions: dict[int, list[StageSessionStatus]] = {}

    @staticmethod
    def _response(request: Any, *, accepted: bool, detail: str) -> StageActionResponse:
        return StageActionResponse(
            worker_id=request.worker_id,
            request_id=request.request_id,
            accepted=accepted,
            detail=detail,
        )

    async def load_stage(self, _endpoint: str, request: LoadStageRequest) -> StageActionResponse:
        stage_id = request.assignment.stage_id
        self.load_calls.append(stage_id)
        if stage_id == self.fail_load_stage:
            return self._response(request, accepted=False, detail="injected load failure")
        self.loads[stage_id] = request
        return self._response(request, accepted=True, detail="loaded")

    async def unload_stage(
        self, _endpoint: str, request: UnloadStageRequest
    ) -> StageActionResponse:
        stage_id = request.assignment.stage_id
        self.unload_calls.append(stage_id)
        self.loads.pop(stage_id, None)
        return self._response(request, accepted=True, detail="unloaded")

    async def install_stage_route(
        self, _endpoint: str, request: InstallStageRouteRequest
    ) -> StageActionResponse:
        stage_id = request.assignment.stage_id
        self.routes[stage_id] = request
        return self._response(request, accepted=True, detail="installed")

    async def remove_stage_route(
        self, _endpoint: str, request: RemoveStageRouteRequest
    ) -> StageActionResponse:
        self.remove_route_calls.append(request.stage_id)
        self.routes.pop(request.stage_id, None)
        return self._response(request, accepted=True, detail="removed")

    async def verify_stage_route(
        self, _endpoint: str, request: VerifyStageRouteRequest
    ) -> StageActionResponse:
        if request.stage_id == self.fail_peer_stage:
            return self._response(request, accepted=False, detail="injected peer failure")
        return self._response(request, accepted=True, detail="connected")

    async def get_stage_status(
        self, _endpoint: str, request: GetStageStatusRequest
    ) -> StageStatusResponse:
        stage_id = int(request.worker_id.rsplit("-", 1)[-1])
        loaded_request = self.loads.get(stage_id)
        route = self.routes.get(stage_id)
        loaded = None
        if loaded_request is not None:
            assignment = loaded_request.assignment
            loaded = LoadedStageStatus(
                model_id=loaded_request.model_id,
                model_revision=loaded_request.model_revision,
                tokenizer_revision=loaded_request.tokenizer_revision,
                topology_id=loaded_request.topology_id,
                assignment=assignment,
                device=loaded_request.device,
                dtype=loaded_request.dtype,
                model_path="fake://model",
                ownership={
                    "stage_id": assignment.stage_id,
                    "layer_start": assignment.layer_start,
                    "layer_end": assignment.layer_end,
                },
                loaded_monotonic_ns=1,
                load_count=1,
                process_rss_before_bytes=0,
                process_rss_after_bytes=100,
                cuda_allocated_before_bytes=0,
                cuda_allocated_after_bytes=0,
                cuda_reserved_before_bytes=0,
                cuda_reserved_after_bytes=0,
            )
        installed = None
        if route is not None:
            installed = InstalledStageRouteStatus(
                topology_id=route.topology_id,
                route_generation=route.route_generation,
                previous_stage=route.previous_stage,
                next_stage=route.next_stage,
                stage_count=route.stage_count,
                stage_zero_publication_destination=(route.stage_zero_publication_destination),
                lease_expiry_unix_ns=route.lease_expiry_unix_ns,
            )
        return StageStatusResponse(
            worker_id=request.worker_id,
            request_id=request.request_id,
            process_id=1_000 + stage_id,
            draining=False,
            loaded_stage=loaded,
            installed_route=installed,
            sessions=self.sessions.get(stage_id, []),
            execution_queue_depth=0,
            execution_queue_capacity=8,
            token_queue_depth=0,
            token_queue_capacity=8,
            dropped_token_publications=0,
        )


def _manager(tmp_path: Path, transport: _DeploymentTransport) -> DeploymentManager:
    registry = WorkerRegistry()
    for stage_id in range(2):
        registry.register(_capability(stage_id), benchmark_verified=True)
    return DeploymentManager(
        registry=registry,
        transport=transport,
        state_directory=tmp_path,
        lease_seconds=60,
        control_timeout_s=5,
    )


@pytest.mark.asyncio
async def test_deployment_is_idempotent_installs_every_route_and_reports_status(
    tmp_path: Path,
) -> None:
    transport = _DeploymentTransport()
    manager = _manager(tmp_path, transport)
    plan = _plan()

    first = await manager.deploy(plan, publication_destination="127.0.0.1:50051")
    same_semantic_plan = plan.model_copy(update={"created_monotonic_ns": time.monotonic_ns()})
    second = await manager.deploy(
        same_semantic_plan,
        publication_destination="127.0.0.1:50051",
    )

    assert first.phase == DeploymentPhase.READY
    assert first.ready
    assert second.idempotent
    assert transport.load_calls == [0, 1]
    assert set(transport.routes) == {0, 1}
    assert transport.routes[0].next_stage is not None
    assert transport.routes[1].previous_stage is not None
    assert transport.routes[0].stage_zero_publication_destination == "127.0.0.1:50051"
    assert transport.routes[1].stage_zero_publication_destination is None
    assert all(item.route_installed and item.peer_verified for item in first.workers)
    assert manager.statuses()[0].phase == DeploymentPhase.READY
    assert (tmp_path / "deployments" / f"{first.deployment_id}.json").is_file()


@pytest.mark.asyncio
async def test_deployment_reload_keeps_evidence_but_never_assumes_old_routes_are_live(
    tmp_path: Path,
) -> None:
    first = _manager(tmp_path, _DeploymentTransport())
    plan = _plan()
    deployed = await first.deploy(
        plan,
        publication_destination="127.0.0.1:50051",
    )
    assert deployed.ready

    reloaded = _manager(tmp_path, _DeploymentTransport())
    status = reloaded.statuses()[0]
    assert status.topology_id == deployed.topology_id
    assert status.generation == deployed.generation
    assert not status.ready
    assert status.phase == DeploymentPhase.FAILED
    assert "workers must re-register" in status.detail
    assert reloaded.current_topology_id is None
    with pytest.raises(RuntimeError, match="no product topology is deployed"):
        reloaded.ready_plan(
            model_id=plan.model.model_id,
            model_revision=plan.model.model_revision,
        )


@pytest.mark.asyncio
async def test_partial_load_failure_rolls_back_every_success_and_reservation(
    tmp_path: Path,
) -> None:
    transport = _DeploymentTransport(fail_load_stage=0)
    manager = _manager(tmp_path, transport)

    with pytest.raises(RuntimeError, match="load rejected"):
        await manager.deploy(_plan(), publication_destination="127.0.0.1:50051")

    assert transport.unload_calls == [1]
    assert not transport.loads
    assert not transport.routes
    assert manager.reserved_worker_ids == ()
    status = manager.statuses()[0]
    assert status.phase == DeploymentPhase.FAILED
    assert not status.ready
    assert not any(item.reserved or item.loaded for item in status.workers)


@pytest.mark.asyncio
async def test_peer_connection_failure_removes_all_routes_and_loaded_stages(
    tmp_path: Path,
) -> None:
    transport = _DeploymentTransport(fail_peer_stage=0)
    manager = _manager(tmp_path, transport)

    with pytest.raises(RuntimeError, match="peer verification rejected"):
        await manager.deploy(_plan(), publication_destination="127.0.0.1:50051")

    assert sorted(transport.remove_route_calls) == [0, 1]
    assert sorted(transport.unload_calls) == [0, 1]
    assert not transport.loads
    assert not transport.routes
    assert manager.reserved_worker_ids == ()
    assert manager.statuses()[0].phase == DeploymentPhase.FAILED
