from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from swarm_inference.cluster.agent import NodeAgent, NodeAgentOptions
from swarm_inference.cluster.models import (
    ClusterMetadata,
    NodeMembership,
    VersionCompatibility,
    node_id_from_fingerprint,
)
from swarm_inference.cluster.runtime_manager import RuntimeManager
from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.config.models import Backend
from swarm_inference.platforms.base import (
    BackendProbeResult,
    InterfaceAddress,
    PlatformIdentity,
)
from swarm_inference.protocol.cluster import NodeUpdateResponse, ReachabilityCheckResponse
from swarm_inference.security.identity import CoordinatorIdentity, WorkerIdentity
from swarm_inference.testing.process_harness import ProductCluster


class _InjectedPlatform:
    service_mode = "foreground"

    def identity(self) -> PlatformIdentity:
        return PlatformIdentity(
            system="windows",
            release="11",
            architecture="AMD64",
            support_status="validated",
            support_reason="process-injected platform",
        )

    def accelerator_probes(self) -> list[BackendProbeResult]:
        return [
            BackendProbeResult(
                backend=Backend.TORCH_CPU,
                device="cpu",
                detected=True,
                operational=True,
                reason="injected tensor probe passed",
                total_memory_bytes=2 * 1024**3,
                available_memory_bytes=1024**3,
                supported_dtypes=["float32"],
            )
        ]

    def interface_addresses(self) -> list[InterfaceAddress]:
        return [
            InterfaceAddress(
                interface="Loopback acceptance adapter",
                address="127.0.0.1",
                prefix_length=8,
                is_private=True,
                is_loopback=True,
                is_up=True,
                mtu=1500,
            )
        ]

    def routed_source_address(self, destination_endpoint: str) -> str:
        del destination_endpoint
        return "127.0.0.1"


class _Lifecycle:
    def __init__(self) -> None:
        self.termination: asyncio.Future[None] | None = None

    async def start(self) -> object:
        self.termination = asyncio.get_running_loop().create_future()
        return object()

    async def wait(self) -> None:
        assert self.termination is not None
        await self.termination

    async def stop(self) -> None:
        if self.termination is not None and not self.termination.done():
            self.termination.set_result(None)


class _CoordinatorClient:
    async def node_update(self, request: Any) -> NodeUpdateResponse:
        return NodeUpdateResponse(
            accepted=True,
            node_id=request.metadata.node_id,
            endpoint_changed=False,
        )

    async def reachability_check(self, request: Any) -> ReachabilityCheckResponse:
        return ReachabilityCheckResponse(
            node_id=request.node_id,
            control_endpoint="127.0.0.1:52001",
            data_endpoint="127.0.0.1:52002",
            control_reachable=True,
            data_reachable=True,
            coordinator_source_address="127.0.0.1",
            detail="injected bidirectional reachability passed",
        )

    async def close(self) -> None:
        return None


def _agent_process(state_root: str, result_queue: Any) -> None:
    async def run() -> None:
        state = ClusterStateStore(Path(state_root))
        platform = _InjectedPlatform()
        manager = RuntimeManager(
            state=state,
            platform=platform,  # type: ignore[arg-type]
            allow_loopback=True,
        )
        agent = NodeAgent(
            state=state,
            platform=platform,  # type: ignore[arg-type]
            runtime_manager=manager,
            options=NodeAgentOptions(health_refresh_seconds=3600),
            worker_factory=lambda configuration: _Lifecycle(),
            client_factory=lambda endpoint: _CoordinatorClient(),  # type: ignore[arg-type,return-value]
        )
        try:
            status = await agent.start()
            identity = state.load_or_create_node_identity()
            membership = state.membership(status.node_id)
            result_queue.put(
                {
                    "state": status.state,
                    "node_id": status.node_id,
                    "fingerprint": identity.public_key_fingerprint,
                    "membership": membership.model_dump_json() if membership else None,
                    "error": status.last_error,
                }
            )
        except BaseException as exc:
            result_queue.put({"error": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            await agent.stop()

    asyncio.run(run())


async def _one_process_start(state_root: Path) -> dict[str, Any]:
    cluster = ProductCluster()
    queue = cluster.queue(maxsize=2)
    cluster.process(
        "node-agent",
        target=_agent_process,
        args=(str(state_root), queue),
    )
    try:
        cluster.start()
        values = await cluster.wait_ready(queue, count=1, timeout=30)
        return values[0]
    finally:
        await cluster.close()


@pytest.mark.asyncio
async def test_process_isolated_agent_restart_reuses_identity_and_membership(
    tmp_path: Path,
) -> None:
    state = ClusterStateStore(tmp_path / "node-state")
    identity = WorkerIdentity.load_or_create(state.paths.node_identity)
    coordinator = CoordinatorIdentity.generate()
    node_id = node_id_from_fingerprint(identity.public_key_fingerprint)
    cluster = ClusterMetadata(
        cluster_id="cluster-agent-process",
        name="agent-process",
        coordinator_id="node-coordinator",
        coordinator_endpoint="127.0.0.1:59999",
        coordinator_public_key=coordinator.public_key_b64,
        coordinator_fingerprint=coordinator.public_key_fingerprint,
        created_at_unix_ns=time.time_ns(),
        runtime_compatibility=VersionCompatibility(
            minimum_runtime_version="0.1.0",
            maximum_runtime_version_exclusive="1.0.0",
        ),
    )
    membership = NodeMembership(
        cluster_id=cluster.cluster_id,
        node_id=node_id,
        node_public_key=identity.public_key_b64,
        node_fingerprint=identity.public_key_fingerprint,
        coordinator_public_key=cluster.coordinator_public_key,
        coordinator_fingerprint=cluster.coordinator_fingerprint,
        joined_at_unix_ns=time.time_ns(),
    )
    state.save_cluster(cluster)
    state.save_membership(membership)

    first = await _one_process_start(state.paths.root)
    second = await _one_process_start(state.paths.root)

    assert first["state"] == second["state"] == "ready"
    assert first["fingerprint"] == second["fingerprint"]
    assert first["node_id"] == second["node_id"] == node_id
    assert first["membership"] == second["membership"]
