from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from swarm_inference.cluster.agent import NodeAgent, NodeAgentOptions
from swarm_inference.cluster.models import (
    ClusterMetadata,
    NodeMembership,
    NodeMetadata,
    VersionCompatibility,
    node_id_from_fingerprint,
)
from swarm_inference.cluster.runtime_manager import RuntimeManager
from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.config.models import Backend
from swarm_inference.exceptions import ConfigurationError
from swarm_inference.platforms.base import (
    BackendProbeResult,
    FirewallStatus,
    InterfaceAddress,
    PlatformIdentity,
)
from swarm_inference.protocol.cluster import (
    NodeUpdateResponse,
    ReachabilityCheckResponse,
)
from swarm_inference.security.identity import CoordinatorIdentity, WorkerIdentity


class _Platform:
    service_mode = "foreground"

    def identity(self) -> PlatformIdentity:
        return PlatformIdentity(
            system="windows",
            release="11",
            architecture="AMD64",
            support_status="validated",
            support_reason="injected platform",
        )

    def accelerator_probes(self):
        return [
            BackendProbeResult(
                backend=Backend.TORCH_CPU,
                device="cpu",
                detected=True,
                operational=True,
                reason="correct tensor probe passed",
                total_memory_bytes=16 * 1024**3,
                available_memory_bytes=8 * 1024**3,
                supported_dtypes=["float32"],
            )
        ]

    def interface_addresses(self):
        return [
            InterfaceAddress(
                interface="Ethernet",
                address="192.168.1.20",
                prefix_length=24,
                is_private=True,
                is_loopback=False,
                is_up=True,
                mtu=1500,
            )
        ]

    def routed_source_address(self, destination_endpoint: str) -> str:
        del destination_endpoint
        return "192.168.1.20"


class _Runtime:
    def __init__(self, *, fail_start: BaseException | None = None) -> None:
        self.fail_start = fail_start
        self.started = 0
        self.stopped = 0
        self.termination: asyncio.Future[None] | None = None

    async def start(self):
        self.started += 1
        if self.fail_start is not None:
            raise self.fail_start
        self.termination = asyncio.get_running_loop().create_future()
        return object()

    async def wait(self) -> None:
        assert self.termination is not None
        await self.termination

    async def stop(self) -> None:
        self.stopped += 1
        if self.termination is not None and not self.termination.done():
            self.termination.set_result(None)

    def fail(self, exc: BaseException) -> None:
        assert self.termination is not None
        self.termination.set_exception(exc)


class _NetworkProbe:
    async def start(self, endpoint: str) -> int:
        del endpoint
        return 1

    async def stop(self) -> None:
        return None


class _Client:
    def __init__(self, *, reachable: bool = True) -> None:
        self.reachable = reachable
        self.closed = False

    async def node_update(self, request):
        return NodeUpdateResponse(
            accepted=True,
            node_id=request.metadata.node_id,
            endpoint_changed=False,
        )

    async def reachability_check(self, request):
        return ReachabilityCheckResponse(
            node_id=request.node_id,
            control_endpoint="192.168.1.20:51001",
            data_endpoint="192.168.1.20:51002",
            control_reachable=self.reachable,
            data_reachable=self.reachable,
            coordinator_source_address="192.168.1.10",
            detail="reachable" if self.reachable else "blocked by firewall",
        )

    async def close(self) -> None:
        self.closed = True


class _BlockedFirewallManager:
    async def configure_firewall(self, specification):
        return FirewallStatus(
            owner_label=specification.owner_label,
            configured=False,
            private_only=True,
            blocked=True,
            detail="firewall permission denied",
            remediation_command="exact-private-firewall-command",
        )


def _state(tmp_path: Path) -> ClusterStateStore:
    state = ClusterStateStore(tmp_path)
    identity = WorkerIdentity.load_or_create(state.paths.node_identity)
    coordinator = CoordinatorIdentity.generate()
    node_id = node_id_from_fingerprint(identity.public_key_fingerprint)
    cluster = ClusterMetadata(
        cluster_id="cluster-agent",
        name="agent-test",
        coordinator_id="node-00000000",
        coordinator_endpoint="192.168.1.10:50051",
        coordinator_public_key=coordinator.public_key_b64,
        coordinator_fingerprint=coordinator.public_key_fingerprint,
        created_at_unix_ns=time.time_ns(),
        runtime_compatibility=VersionCompatibility(
            minimum_runtime_version="0.1.0",
            maximum_runtime_version_exclusive="0.2.0",
        ),
    )
    state.save_cluster(cluster)
    membership = NodeMembership(
        cluster_id=cluster.cluster_id,
        node_id=node_id,
        node_public_key=identity.public_key_b64,
        node_fingerprint=identity.public_key_fingerprint,
        coordinator_public_key=coordinator.public_key_b64,
        coordinator_fingerprint=coordinator.public_key_fingerprint,
        joined_at_unix_ns=time.time_ns(),
    )
    state.save_membership(membership)
    state.save_node(
        NodeMetadata(
            node_id=node_id,
            public_key=identity.public_key_b64,
            fingerprint=identity.public_key_fingerprint,
            hostname="node",
            operating_system="test",
            architecture="test",
            agent_version="0.1.0",
            runtime_version="0.1.0",
            build_id="test",
            package_lock_hash="5" * 64,
            worker_ids=[],
            joined_at_unix_ns=membership.joined_at_unix_ns,
            last_seen_at_unix_ns=membership.joined_at_unix_ns,
        )
    )
    return state


def _manager(state: ClusterStateStore) -> RuntimeManager:
    return RuntimeManager(
        state=state,
        platform=_Platform(),  # type: ignore[arg-type]
        port_available=lambda host, port: True,
    )


@pytest.mark.asyncio
async def test_node_agent_starts_ready_and_reuses_membership_after_restart(tmp_path: Path) -> None:
    state = _state(tmp_path / "state")
    runtimes: list[_Runtime] = []

    def worker_factory(configuration):
        del configuration
        runtime = _Runtime()
        runtimes.append(runtime)
        return runtime

    options = NodeAgentOptions(health_refresh_seconds=3600)
    agent = NodeAgent(
        state=state,
        platform=_Platform(),  # type: ignore[arg-type]
        runtime_manager=_manager(state),
        options=options,
        worker_factory=worker_factory,
        network_probe_factory=lambda configuration: _NetworkProbe(),
        client_factory=lambda endpoint: _Client(),  # type: ignore[arg-type,return-value]
    )
    status = await agent.start()
    assert status.state == "ready"
    membership_before = state.load_memberships()
    await agent.stop()

    restarted = NodeAgent(
        state=state,
        platform=_Platform(),  # type: ignore[arg-type]
        runtime_manager=_manager(state),
        options=options,
        worker_factory=worker_factory,
        network_probe_factory=lambda configuration: _NetworkProbe(),
        client_factory=lambda endpoint: _Client(),  # type: ignore[arg-type,return-value]
    )
    assert (await restarted.start()).state == "ready"
    assert state.load_memberships() == membership_before
    assert len(runtimes) == 2
    await restarted.stop()


@pytest.mark.asyncio
async def test_node_agent_restarts_failed_worker_with_bounded_backoff(tmp_path: Path) -> None:
    state = _state(tmp_path / "state")
    runtimes: list[_Runtime] = []

    def worker_factory(configuration):
        del configuration
        runtime = _Runtime()
        runtimes.append(runtime)
        return runtime

    agent = NodeAgent(
        state=state,
        platform=_Platform(),  # type: ignore[arg-type]
        runtime_manager=_manager(state),
        options=NodeAgentOptions(
            maximum_restart_attempts=2,
            restart_initial_backoff_seconds=0.001,
            restart_maximum_backoff_seconds=0.001,
            health_refresh_seconds=3600,
        ),
        worker_factory=worker_factory,
        network_probe_factory=lambda configuration: _NetworkProbe(),
        client_factory=lambda endpoint: _Client(),  # type: ignore[arg-type,return-value]
        sleep=lambda delay: asyncio.sleep(0),
    )
    assert (await agent.start()).state == "ready"
    runtimes[0].fail(OSError("injected worker failure"))
    for _ in range(50):
        if len(runtimes) == 2 and agent.status.state == "ready":
            break
        await asyncio.sleep(0)
    assert len(runtimes) == 2
    assert agent.status.state == "ready"
    await agent.stop()


@pytest.mark.asyncio
async def test_permanent_configuration_failure_does_not_spin(tmp_path: Path) -> None:
    state = _state(tmp_path / "state")
    calls = 0

    def worker_factory(configuration):
        nonlocal calls
        del configuration
        calls += 1
        return _Runtime(fail_start=ConfigurationError("permanent backend mismatch"))

    agent = NodeAgent(
        state=state,
        platform=_Platform(),  # type: ignore[arg-type]
        runtime_manager=_manager(state),
        options=NodeAgentOptions(health_refresh_seconds=3600),
        worker_factory=worker_factory,
        network_probe_factory=lambda configuration: _NetworkProbe(),
        client_factory=lambda endpoint: _Client(),  # type: ignore[arg-type,return-value]
    )
    status = await agent.start()
    assert status.state == "failed"
    assert "permanent backend mismatch" in status.last_error
    assert calls == 1


@pytest.mark.asyncio
async def test_firewall_permission_denial_keeps_node_blocked_with_remediation(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path / "state")
    agent = NodeAgent(
        state=state,
        platform=_Platform(),  # type: ignore[arg-type]
        runtime_manager=_manager(state),
        service_manager=_BlockedFirewallManager(),  # type: ignore[arg-type]
        options=NodeAgentOptions(health_refresh_seconds=3600),
        worker_factory=lambda configuration: _Runtime(),
        network_probe_factory=lambda configuration: _NetworkProbe(),
        client_factory=lambda endpoint: _Client(reachable=False),  # type: ignore[arg-type,return-value]
    )
    status = await agent.start()
    assert status.state == "blocked"
    assert status.error_category == "permission"
    assert "exact-private-firewall-command" in status.last_error
    await agent.stop()
