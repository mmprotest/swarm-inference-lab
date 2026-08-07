from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from swarm_inference.config.models import Backend, WorkerCapability
from swarm_inference.config.product import ProductCoordinatorConfig
from swarm_inference.coordinator.service import (
    CoordinatorClient,
    CoordinatorCore,
    CoordinatorRpcServer,
)
from swarm_inference.engines.interfaces import ExecutionPlan, PhasePlan
from swarm_inference.exceptions import TransportError
from swarm_inference.protocol.cluster import (
    ClusterRequestAuthentication,
    EngineLeaseRequest,
)
from swarm_inference.protocol.engine_worker import verify_engine_deployment_lease
from swarm_inference.security.identity import WorkerIdentity


def _plan(*, engine_id: str = "fake-engine") -> ExecutionPlan:
    roles = {"worker-a": "critical_path_stage"}
    return ExecutionPlan(
        plan_id="plan-a",
        engine_id=engine_id,
        model_fingerprint="sha256:" + "1" * 64,
        execution_identity="sha256:" + "2" * 64,
        objective="speed",
        topology="local-complete-model",
        worker_roles=roles,
        prefill_plan=PhasePlan(phase="prefill", worker_roles=roles),
        decode_plan=PhasePlan(phase="decode", worker_roles=roles),
        predicted_ttft_ms=1,
        predicted_decode_tokens_s=10,
        predicted_aggregate_tokens_s=10,
        score=10,
    )


def _capability(identity: WorkerIdentity) -> WorkerCapability:
    return WorkerCapability(
        worker_id="worker-a",
        public_key=identity.public_key_b64,
        hostname="localhost",
        operating_system="test",
        architecture="test",
        backend=Backend.TORCH_CPU,
        cpu_model="test",
        logical_cpu_count=1,
        total_ram_bytes=1_000_000,
        available_ram_bytes=1_000_000,
        upload_bandwidth_bytes_s=1_000_000,
        download_bandwidth_bytes_s=1_000_000,
        coordinator_latency_ms=0,
        execution_engines=[
            {
                "engine_id": "fake-engine",
                "enabled": True,
                "formats": ["GGUF"],
                "roles": ["critical_path_stage"],
            }
        ],
    )


class _ClusterControl:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def verify_authentication(
        self,
        _authentication: ClusterRequestAuthentication,
        *,
        action: str,
        body: dict[str, Any],
    ) -> SimpleNamespace:
        self.calls.append((action, body))
        return SimpleNamespace(node_id="node-client")


def _authentication() -> ClusterRequestAuthentication:
    return ClusterRequestAuthentication(
        node_id="node-client",
        timestamp_unix_ns=time.time_ns(),
        nonce="test-nonce",
        signature="test-signature",
    )


@pytest.mark.asyncio
async def test_coordinator_issues_plan_bound_engine_lease(tmp_path: Path) -> None:
    core = CoordinatorCore(
        product_config=ProductCoordinatorConfig(
            require_trusted_workers=False,
            engine_action_lease_seconds=2,
        ),
        state_directory=tmp_path / "coordinator",
    )
    worker_identity = WorkerIdentity.generate()
    core.registry.register(_capability(worker_identity), benchmark_verified=True)
    control = _ClusterControl()
    core.cluster_control = control  # type: ignore[assignment]
    server = CoordinatorRpcServer(core)
    port = await server.start("127.0.0.1:0")
    client = CoordinatorClient(f"127.0.0.1:{port}")
    plan = _plan()
    request = EngineLeaseRequest(
        authentication=_authentication(),
        action="prepare",
        worker_id="worker-a",
        deployment_id="deployment-a",
        plan=plan,
        ttl_seconds=300,
    )
    try:
        response = await client.engine_lease(request)
        assert core.coordinator_identity is not None
        lease = response.lease
        assert lease.expires_at_unix_ns - lease.issued_at_unix_ns == 2_000_000_000
        verify_engine_deployment_lease(
            lease,
            action="prepare",
            worker_id="worker-a",
            deployment_id="deployment-a",
            engine_id="fake-engine",
            execution_identity=plan.execution_identity,
            plan=plan,
            trusted_coordinator_public_key=core.coordinator_identity.public_key_b64,
            trusted_coordinator_fingerprint=(
                core.coordinator_identity.public_key_fingerprint
            ),
        )
        assert control.calls == [
            (
                "engine-lease",
                request.model_dump(
                    mode="json",
                    exclude={"authentication", "schema_version"},
                ),
            )
        ]

        unsupported = request.model_copy(update={"plan": _plan(engine_id="missing")})
        with pytest.raises(TransportError, match="does not advertise enabled engine"):
            await client.engine_lease(unsupported)
    finally:
        await client.close()
        await server.stop(grace_s=0)
