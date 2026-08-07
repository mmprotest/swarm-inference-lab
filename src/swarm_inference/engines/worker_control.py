"""Coordinator-authorized direct control of persistent worker engine runtimes."""

from __future__ import annotations

import math
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from swarm_inference.cluster.artifacts import ArtifactManager, artifact_chunks
from swarm_inference.cluster.models import ArtifactManifest
from swarm_inference.cluster.pairing import create_cluster_authentication
from swarm_inference.engines.interfaces import (
    Deployment,
    ExecutionPlan,
    InferenceEvent,
    InferenceRequest,
)
from swarm_inference.protocol.cluster import (
    EngineArtifactLeaseRequest,
    EngineArtifactLeaseResponse,
    EngineLeaseRequest,
    EngineLeaseResponse,
)
from swarm_inference.protocol.engine_worker import (
    EngineActionResponse,
    EngineInferenceChunk,
    PrepareEngineRequest,
    SubmitEngineRequest,
    UnloadEngineRequest,
    execution_plan_hash,
)
from swarm_inference.protocol.stage_worker import (
    ArtifactTransferResponse,
    CompleteArtifactRequest,
    PrepareArtifactRequest,
    VerifyArtifactRequest,
    WriteArtifactChunkRequest,
)
from swarm_inference.security.identity import WorkerIdentity


class EngineAuthorizationClient(Protocol):
    async def engine_lease(self, request: EngineLeaseRequest) -> EngineLeaseResponse: ...

    async def engine_artifact_lease(
        self,
        request: EngineArtifactLeaseRequest,
    ) -> EngineArtifactLeaseResponse: ...


class WorkerEngineTransport(Protocol):
    async def prepare_engine(
        self, endpoint: str, request: PrepareEngineRequest
    ) -> EngineActionResponse: ...

    def submit_engine_stream(
        self, endpoint: str, request: SubmitEngineRequest
    ) -> AsyncIterator[EngineInferenceChunk]: ...

    async def unload_engine(
        self, endpoint: str, request: UnloadEngineRequest
    ) -> EngineActionResponse: ...

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


@dataclass(slots=True)
class _ResidentDeployment:
    owner_worker_id: str
    worker_plans: dict[str, ExecutionPlan]


class CoordinatorAuthorizedEngineLifecycle:
    """Keep backend processes on workers while the coordinator authorizes control."""

    def __init__(
        self,
        *,
        coordinator: EngineAuthorizationClient,
        transport: WorkerEngineTransport,
        identity: WorkerIdentity,
        node_id: str,
        worker_endpoints: dict[str, str],
        artifact_manager: ArtifactManager,
        artifact_manifest: ArtifactManifest,
        lease_seconds: int = 300,
        artifact_chunk_size_bytes: int = 1024 * 1024,
    ) -> None:
        if not worker_endpoints:
            raise ValueError("worker engine lifecycle requires control endpoints")
        if not 0 < lease_seconds <= 3600:
            raise ValueError("engine lifecycle lease duration must be in (0, 3600]")
        if not 0 < artifact_chunk_size_bytes <= 2 * 1024 * 1024:
            raise ValueError("engine artifact chunks must be in (0, 2 MiB]")
        self.coordinator = coordinator
        self.transport = transport
        self.identity = identity
        self.node_id = node_id
        self.worker_endpoints = dict(worker_endpoints)
        self.artifact_manager = artifact_manager
        self.artifact_manifest = artifact_manifest
        self.lease_seconds = lease_seconds
        self.artifact_chunk_size_bytes = artifact_chunk_size_bytes
        self._deployments: dict[str, _ResidentDeployment] = {}

    def _endpoint(self, worker_id: str) -> str:
        try:
            return self.worker_endpoints[worker_id]
        except KeyError as exc:
            raise RuntimeError(f"selected engine worker {worker_id!r} has no control endpoint") from exc

    def _authentication(self, *, action: str, body: dict[str, Any]):
        return create_cluster_authentication(
            identity=self.identity,
            node_id=self.node_id,
            action=action,
            body=body,
        )

    async def _engine_lease(
        self,
        *,
        action: str,
        worker_id: str,
        deployment_id: str,
        plan: ExecutionPlan,
    ):
        body = {
            "action": action,
            "worker_id": worker_id,
            "deployment_id": deployment_id,
            "plan": plan.model_dump(mode="json"),
            "ttl_seconds": self.lease_seconds,
        }
        request = EngineLeaseRequest(
            authentication=self._authentication(action="engine-lease", body=body),
            action=action,
            worker_id=worker_id,
            deployment_id=deployment_id,
            plan=plan,
            ttl_seconds=self.lease_seconds,
        )
        return (await self.coordinator.engine_lease(request)).lease

    async def _artifact_lease(
        self,
        *,
        worker_id: str,
        deployment_id: str,
        plan: ExecutionPlan,
    ):
        manifest = self.artifact_manifest
        body = {
            "worker_id": worker_id,
            "deployment_id": deployment_id,
            "plan": plan.model_dump(mode="json"),
            "manifest": manifest.model_dump(mode="json"),
            "ttl_seconds": self.lease_seconds,
        }
        request = EngineArtifactLeaseRequest(
            authentication=self._authentication(
                action="engine-artifact-lease",
                body=body,
            ),
            worker_id=worker_id,
            deployment_id=deployment_id,
            plan=plan,
            manifest=manifest,
            ttl_seconds=self.lease_seconds,
        )
        return (await self.coordinator.engine_artifact_lease(request)).lease

    async def _transfer_artifact(
        self,
        *,
        worker_id: str,
        deployment_id: str,
        plan: ExecutionPlan,
    ) -> None:
        manifest = self.artifact_manifest
        source = self.artifact_manager.resolve(manifest.artifact_id)
        chunks_total = sum(
            math.ceil(item.size_bytes / self.artifact_chunk_size_bytes)
            for item in manifest.files
        )
        lease = await self._artifact_lease(
            worker_id=worker_id,
            deployment_id=deployment_id,
            plan=plan,
        )
        endpoint = self._endpoint(worker_id)
        prepared = await self.transport.prepare_artifact(
            endpoint,
            PrepareArtifactRequest(
                worker_id=worker_id,
                request_id=f"artifact-prepare-{uuid4().hex}",
                manifest=manifest,
                chunks_total=chunks_total,
                lease=lease,
            ),
        )
        if not prepared.accepted or prepared.transfer_id is None:
            raise RuntimeError(f"worker rejected engine artifact: {prepared.detail}")
        if not (prepared.complete and prepared.verified):
            transfer_id = prepared.transfer_id
            sent = 0
            for chunk, payload in artifact_chunks(
                source,
                chunk_size_bytes=self.artifact_chunk_size_bytes,
            ):
                response = await self.transport.write_artifact_chunk(
                    endpoint,
                    WriteArtifactChunkRequest(
                        worker_id=worker_id,
                        request_id=f"artifact-chunk-{uuid4().hex}",
                        transfer_id=transfer_id,
                        chunk=chunk,
                        payload=payload,
                        lease=lease,
                    ),
                )
                if not response.accepted:
                    raise RuntimeError(f"worker rejected engine artifact chunk: {response.detail}")
                sent += 1
            if sent != chunks_total:
                raise RuntimeError("engine artifact transport emitted an unexpected chunk count")
            completed = await self.transport.complete_artifact(
                endpoint,
                CompleteArtifactRequest(
                    worker_id=worker_id,
                    request_id=f"artifact-complete-{uuid4().hex}",
                    transfer_id=transfer_id,
                    manifest=manifest,
                    lease=lease,
                ),
            )
            if not completed.accepted or not completed.complete or not completed.verified:
                raise RuntimeError(f"worker did not verify engine artifact: {completed.detail}")
        verified = await self.transport.verify_artifact(
            endpoint,
            VerifyArtifactRequest(
                worker_id=worker_id,
                request_id=f"artifact-verify-{uuid4().hex}",
                artifact_id=manifest.artifact_id,
                lease=lease,
            ),
        )
        if not verified.accepted or not verified.verified:
            raise RuntimeError(f"worker engine artifact verification failed: {verified.detail}")

    async def _prepare_worker(
        self,
        *,
        worker_id: str,
        deployment_id: str,
        plan: ExecutionPlan,
        artifact_ids: tuple[str, ...],
    ) -> EngineActionResponse:
        role = plan.worker_roles[worker_id]
        lease = await self._engine_lease(
            action="prepare",
            worker_id=worker_id,
            deployment_id=deployment_id,
            plan=plan,
        )
        response = await self.transport.prepare_engine(
            self._endpoint(worker_id),
            PrepareEngineRequest(
                worker_id=worker_id,
                request_id=f"engine-prepare-{uuid4().hex}",
                deployment_id=deployment_id,
                assigned_role=role,
                plan=plan,
                artifact_ids=artifact_ids,
                lease=lease,
            ),
        )
        if not response.accepted or response.deployment is None:
            raise RuntimeError(f"worker {worker_id} rejected engine prepare: {response.detail}")
        return response

    async def prepare(self, plan: ExecutionPlan) -> Deployment:
        active = {
            worker_id: role
            for worker_id, role in plan.worker_roles.items()
            if role not in {"idle", "background_replica", "storage_cache", "verification"}
        }
        compute_workers = sorted(
            worker_id for worker_id, role in active.items() if role == "tensor_rpc_compute"
        )
        owners = sorted(worker_id for worker_id in active if worker_id not in compute_workers)
        if len(owners) != 1:
            raise RuntimeError("general engine deployment requires exactly one submission owner")
        if compute_workers and plan.engine_id != "llamacpp-rpc":
            raise RuntimeError("only llama.cpp advertises distributed tensor RPC roles")
        owner = owners[0]
        deployment_id = "engine-" + execution_plan_hash(plan).removeprefix("sha256:")[:32]
        worker_plans: dict[str, ExecutionPlan] = {}
        responses: dict[str, EngineActionResponse] = {}
        try:
            for worker_id in compute_workers:
                worker_plan = plan
                responses[worker_id] = await self._prepare_worker(
                    worker_id=worker_id,
                    deployment_id=deployment_id,
                    plan=worker_plan,
                    artifact_ids=(),
                )
                worker_plans[worker_id] = worker_plan
            rpc_endpoints = {
                worker_id: responses[worker_id].endpoints[worker_id]
                for worker_id in compute_workers
            }
            owner_plan = plan
            if rpc_endpoints:
                fractions = dict(plan.engine_parameters.get("tensor_split", {}))
                owner_plan = plan.model_copy(
                    update={
                        "engine_parameters": {
                            **plan.engine_parameters,
                            "rpc_endpoints": rpc_endpoints,
                            "tensor_split_order": [*compute_workers, owner],
                            "tensor_split_values": [
                                float(fractions[worker_id])
                                for worker_id in [*compute_workers, owner]
                            ],
                        }
                    }
                )
            await self._transfer_artifact(
                worker_id=owner,
                deployment_id=deployment_id,
                plan=owner_plan,
            )
            responses[owner] = await self._prepare_worker(
                worker_id=owner,
                deployment_id=deployment_id,
                plan=owner_plan,
                artifact_ids=(self.artifact_manifest.artifact_id,),
            )
            worker_plans[owner] = owner_plan
        except BaseException:
            for worker_id, worker_plan in reversed(tuple(worker_plans.items())):
                with suppress(Exception):
                    await self._unload_worker(
                        worker_id=worker_id,
                        deployment_id=deployment_id,
                        plan=worker_plan,
                        force=True,
                    )
            raise
        combined_endpoints = {
            key: value
            for response in responses.values()
            for key, value in response.endpoints.items()
        }
        combined_process_ids = {
            key: value
            for response in responses.values()
            for key, value in response.process_ids.items()
        }
        self._deployments[deployment_id] = _ResidentDeployment(
            owner_worker_id=owner,
            worker_plans=worker_plans,
        )
        return Deployment(
            deployment_id=deployment_id,
            engine_id=plan.engine_id,
            execution_identity=plan.execution_identity,
            plan=plan,
            ready=True,
            endpoints=combined_endpoints,
            process_ids=combined_process_ids,
            metadata={
                "worker_owned": True,
                "coordinator_authorized": True,
                "artifact_id": self.artifact_manifest.artifact_id,
                "submission_owner": owner,
            },
        )

    async def submit(
        self,
        deployment: Deployment,
        request: InferenceRequest,
    ) -> AsyncIterator[InferenceEvent]:
        try:
            resident = self._deployments[deployment.deployment_id]
        except KeyError as exc:
            raise RuntimeError("worker engine deployment is not controlled by this lifecycle") from exc
        worker_id = resident.owner_worker_id
        plan = resident.worker_plans[worker_id]
        lease = await self._engine_lease(
            action="submit",
            worker_id=worker_id,
            deployment_id=deployment.deployment_id,
            plan=plan,
        )
        submit_request = SubmitEngineRequest(
            worker_id=worker_id,
            request_id=f"engine-submit-{uuid4().hex}",
            deployment_id=deployment.deployment_id,
            plan=plan,
            inference=request,
            lease=lease,
        )
        received = False
        async for chunk in self.transport.submit_engine_stream(
            self._endpoint(worker_id),
            submit_request,
        ):
            expected = (
                worker_id,
                submit_request.request_id,
                deployment.deployment_id,
                request.request_id,
            )
            actual = (
                chunk.worker_id,
                chunk.request_id,
                chunk.deployment_id,
                chunk.event.request_id,
            )
            if actual != expected:
                raise RuntimeError("worker engine stream identity changed in flight")
            received = True
            yield chunk.event
        if not received:
            raise RuntimeError("worker engine submission returned no events")

    async def _unload_worker(
        self,
        *,
        worker_id: str,
        deployment_id: str,
        plan: ExecutionPlan,
        force: bool,
    ) -> None:
        lease = await self._engine_lease(
            action="unload",
            worker_id=worker_id,
            deployment_id=deployment_id,
            plan=plan,
        )
        response = await self.transport.unload_engine(
            self._endpoint(worker_id),
            UnloadEngineRequest(
                worker_id=worker_id,
                request_id=f"engine-unload-{uuid4().hex}",
                deployment_id=deployment_id,
                plan=plan,
                force=force,
                lease=lease,
            ),
        )
        if not response.accepted:
            raise RuntimeError(f"worker {worker_id} rejected engine unload: {response.detail}")

    async def unload(self, deployment: Deployment) -> None:
        resident = self._deployments.get(deployment.deployment_id)
        if resident is None:
            return
        errors: list[BaseException] = []
        ordered = [
            resident.owner_worker_id,
            *sorted(
                worker_id
                for worker_id in resident.worker_plans
                if worker_id != resident.owner_worker_id
            ),
        ]
        for worker_id in ordered:
            try:
                await self._unload_worker(
                    worker_id=worker_id,
                    deployment_id=deployment.deployment_id,
                    plan=resident.worker_plans[worker_id],
                    force=False,
                )
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(f"{len(errors)} worker engine unload action(s) failed")
        self._deployments.pop(deployment.deployment_id, None)


__all__ = [
    "CoordinatorAuthorizedEngineLifecycle",
    "EngineAuthorizationClient",
    "WorkerEngineTransport",
]
