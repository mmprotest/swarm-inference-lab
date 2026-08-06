"""Persistent node agent orchestrating canonical coordinator and worker lifecycles."""

from __future__ import annotations

import asyncio
import hashlib
import os
import socket
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import Field, PositiveInt

from swarm_inference import __version__
from swarm_inference.cluster.models import (
    BackendValidationRecord,
    ClusterAuditEvent,
    ClusterMetadata,
    NodeAgentState,
    NodeConfiguration,
    NodeMembership,
    NodeMetadata,
    NodeRuntimeMetadata,
    aggregate_validation_status,
    node_id_from_fingerprint,
)
from swarm_inference.cluster.network import (
    DirectedNetworkMeasurer,
    DirectNetworkProbeServer,
    NetworkMeasurementRepository,
)
from swarm_inference.cluster.pairing import create_cluster_authentication
from swarm_inference.cluster.runtime_manager import RuntimeManager
from swarm_inference.cluster.service_manager import ServiceManager
from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.config.models import StrictModel
from swarm_inference.coordinator.runtime import CoordinatorRuntime
from swarm_inference.coordinator.service import CoordinatorClient
from swarm_inference.exceptions import (
    BackendIncompatibleError,
    ConfigurationError,
    IntegrityError,
    TransportError,
)
from swarm_inference.host import split_endpoint
from swarm_inference.platforms.base import FirewallRuleSpec, PlatformAdapter
from swarm_inference.protocol.cluster import (
    ClusterStatusRequest,
    ClusterStatusResponse,
    NetworkProbeControlRequest,
    NetworkProbeControlResponse,
    NodeUpdateRequest,
    NodeUpdateResponse,
    ReachabilityCheckRequest,
    ReachabilityCheckResponse,
)
from swarm_inference.security.identity import WorkerIdentity

NodeAgentRole = Literal["coordinator", "worker"]
ErrorCategory = Literal[
    "permission",
    "connectivity",
    "compatibility",
    "capacity",
    "artifact-integrity",
    "execution",
]


def _default_roles() -> set[NodeAgentRole]:
    return {"worker"}


class NodeAgentOptions(StrictModel):
    schema_version: Literal[1] = 1
    roles: set[NodeAgentRole] = Field(default_factory=_default_roles)
    maximum_restart_attempts: PositiveInt = 5
    restart_initial_backoff_seconds: float = Field(default=1.0, gt=0, le=60)
    restart_maximum_backoff_seconds: float = Field(default=30.0, gt=0, le=300)
    health_refresh_seconds: float = Field(default=30.0, gt=0, le=3600)
    reachability_timeout_ms: PositiveInt = 3000
    network_measurement_ttl_seconds: PositiveInt = 900
    network_probe_max_bytes: PositiveInt = 16 * 1024 * 1024
    network_probe_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    network_probe_payload_sizes: list[PositiveInt] = Field(
        default_factory=lambda: [4096, 256 * 1024, 1024 * 1024]
    )
    network_probe_sample_count: PositiveInt = 3
    private_subnets: list[str] = Field(
        default_factory=lambda: [
            "10.0.0.0/8",
            "172.16.0.0/12",
            "192.168.0.0/16",
        ]
    )


class _Lifecycle(Protocol):
    async def start(self) -> Any: ...

    async def wait(self) -> None: ...

    async def stop(self) -> None: ...


class _NetworkProbeLifecycle(Protocol):
    async def start(self, endpoint: str) -> int: ...

    async def stop(self) -> None: ...


class _ClusterClient(Protocol):
    async def node_update(self, request: NodeUpdateRequest) -> NodeUpdateResponse: ...

    async def reachability_check(
        self,
        request: ReachabilityCheckRequest,
    ) -> ReachabilityCheckResponse: ...

    async def cluster_status(self, request: ClusterStatusRequest) -> ClusterStatusResponse: ...

    async def network_probe_control(
        self,
        request: NetworkProbeControlRequest,
    ) -> NetworkProbeControlResponse: ...

    async def close(self) -> None: ...


WorkerFactory = Callable[[NodeConfiguration], _Lifecycle]
CoordinatorFactory = Callable[[ClusterMetadata], _Lifecycle]
NetworkProbeFactory = Callable[[NodeConfiguration], _NetworkProbeLifecycle]
ClusterClientFactory = Callable[[str], _ClusterClient]
Sleep = Callable[[float], Awaitable[None]]


def _default_client_factory(endpoint: str) -> _ClusterClient:
    return CoordinatorClient(endpoint)


def _build_identity() -> tuple[str, str]:
    build_id = os.environ.get("SWARM_BUILD_ID", f"swarm-inference-lab-{__version__}")
    lock = os.environ.get("SWARM_PACKAGE_LOCK_HASH")
    if lock is None:
        lock = hashlib.sha256(f"{__version__}:{build_id}".encode()).hexdigest()
    return build_id, lock


def _error_category(
    exc: BaseException,
) -> ErrorCategory:
    if isinstance(exc, PermissionError):
        return "permission"
    if isinstance(exc, (ConfigurationError, BackendIncompatibleError)):
        return "compatibility"
    if isinstance(exc, IntegrityError):
        return "permission"
    if isinstance(exc, (TransportError, OSError, TimeoutError)):
        return "connectivity"
    return "execution"


class NodeAgent:
    """Long-lived owner of identity, configuration, runtimes, and reconnect policy."""

    def __init__(
        self,
        *,
        state: ClusterStateStore,
        platform: PlatformAdapter,
        runtime_manager: RuntimeManager,
        service_manager: ServiceManager | None = None,
        options: NodeAgentOptions | None = None,
        worker_factory: WorkerFactory | None = None,
        coordinator_factory: CoordinatorFactory | None = None,
        network_probe_factory: NetworkProbeFactory | None = None,
        client_factory: ClusterClientFactory = _default_client_factory,
        clock_ns: Callable[[], int] = time.time_ns,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.state = state
        self.platform = platform
        self.runtime_manager = runtime_manager
        self.service_manager = service_manager or ServiceManager(
            platform=platform,
            state=state,
        )
        self.options = options or NodeAgentOptions()
        self.worker_factory = worker_factory or runtime_manager.build_worker_runtime
        self.coordinator_factory = coordinator_factory or self._default_coordinator_factory
        self.network_probe_factory = network_probe_factory or self._default_network_probe_factory
        self.client_factory = client_factory
        self.clock_ns = clock_ns
        self.sleep = sleep
        self._lifecycle_lock = asyncio.Lock()
        self._runtime_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._worker: _Lifecycle | None = None
        self._coordinator: _Lifecycle | None = None
        self._network_probe: _NetworkProbeLifecycle | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._health_task: asyncio.Task[None] | None = None
        self._configuration: NodeConfiguration | None = None
        self._identity: WorkerIdentity | None = None
        self._cluster: ClusterMetadata | None = None
        self._membership: NodeMembership | None = None
        self._stopping = False
        self._status: NodeRuntimeMetadata | None = None

    def _default_coordinator_factory(self, cluster: ClusterMetadata) -> CoordinatorRuntime:
        host, port = split_endpoint(cluster.coordinator_endpoint)
        del host
        return self.runtime_manager.build_coordinator_runtime(
            cluster=cluster,
            listen_endpoint=f"0.0.0.0:{port}",
            advertised_endpoint=cluster.coordinator_endpoint,
        )

    def _default_network_probe_factory(
        self,
        configuration: NodeConfiguration,
    ) -> DirectNetworkProbeServer:
        assert self._identity is not None
        assert self._cluster is not None
        return DirectNetworkProbeServer(
            state=self.state,
            cluster=self._cluster,
            identity=self._identity,
            node_id=configuration.node_id,
            worker_id=self.runtime_manager.worker_id(configuration),
            maximum_bytes=self.options.network_probe_max_bytes,
            timeout_seconds=self.options.network_probe_timeout_seconds,
            clock_ns=self.clock_ns,
        )

    @property
    def status(self) -> NodeRuntimeMetadata:
        if self._status is None:
            identity = self.state.load_or_create_node_identity()
            return NodeRuntimeMetadata(
                node_id=node_id_from_fingerprint(identity.public_key_fingerprint),
                state="stopped",
                service_state="stopped",
                last_refresh_unix_ns=self.clock_ns(),
            )
        return self._status.model_copy(deep=True)

    def _set_status(
        self,
        state_value: NodeAgentState,
        *,
        reason: str | None = None,
        last_error: str | None = None,
        error_category: ErrorCategory | None = None,
        coordinator_reachable: bool | None = None,
        data_reachable: bool | None = None,
    ) -> None:
        assert self._identity is not None
        configuration = self._configuration
        existing = self._status
        selected_backend = (
            configuration.backend_selection.selected_backend if configuration else None
        )
        selected_device = configuration.backend_selection.selected_device if configuration else None
        selected_dtype = configuration.backend_selection.selected_dtype if configuration else None
        value = NodeRuntimeMetadata(
            node_id=node_id_from_fingerprint(self._identity.public_key_fingerprint),
            cluster_id=self._cluster.cluster_id if self._cluster is not None else None,
            state=state_value,
            reason=reason,
            service_state="running" if state_value not in {"stopped", "failed"} else state_value,
            process_id=os.getpid(),
            selected_backend=selected_backend,
            selected_device=selected_device,
            selected_dtype=selected_dtype,
            memory_limit_bytes=(configuration.memory_budget.limit_bytes if configuration else None),
            storage_limit_bytes=(configuration.storage_limit_bytes if configuration else None),
            control_endpoint=(
                configuration.endpoints.control_advertised_endpoint if configuration else None
            ),
            data_endpoint=(
                configuration.endpoints.data_advertised_endpoint if configuration else None
            ),
            probe_endpoint=(
                configuration.endpoints.probe_advertised_endpoint if configuration else None
            ),
            coordinator_reachable=(
                coordinator_reachable
                if coordinator_reachable is not None
                else bool(existing and existing.coordinator_reachable)
            ),
            data_reachable=(
                data_reachable
                if data_reachable is not None
                else bool(existing and existing.data_reachable)
            ),
            probe_reachable=(existing.probe_reachable if existing else None),
            artifact_cache_bytes=sum(
                item.size_bytes for item in self.state.load_artifact_cache().entries
            ),
            loaded_stage_ids=existing.loaded_stage_ids if existing else [],
            current_role=existing.current_role if existing else "idle",
            last_refresh_unix_ns=self.clock_ns(),
            last_error=last_error,
            error_category=error_category,
        )
        self._status = value
        self.state.save_runtime(value)

    def _audit(self, event_type: str, *, detail: str | None = None) -> None:
        assert self._cluster is not None
        assert self._identity is not None
        self.state.append_audit(
            ClusterAuditEvent(
                event_id=uuid4().hex,
                event_type=event_type,
                timestamp_unix_ns=self.clock_ns(),
                cluster_id=self._cluster.cluster_id,
                node_id=node_id_from_fingerprint(self._identity.public_key_fingerprint),
                detail=detail,
            )
        )

    def _load_security_state(self) -> tuple[WorkerIdentity, ClusterMetadata, NodeMembership]:
        identity = self.state.load_or_create_node_identity()
        node_id = node_id_from_fingerprint(identity.public_key_fingerprint)
        cluster = self.state.load_cluster()
        membership = self.state.membership(node_id)
        if cluster is None:
            raise ConfigurationError("node has no pinned cluster metadata; run 'swarm node join'")
        if membership is None:
            raise ConfigurationError("node has no cluster membership; run 'swarm node join'")
        membership.verify_identity_bindings()
        if membership.status != "active":
            raise IntegrityError(f"node membership is {membership.status}")
        if membership.node_public_key != identity.public_key_b64:
            raise IntegrityError("durable node identity does not match cluster membership")
        if membership.coordinator_fingerprint != cluster.coordinator_fingerprint:
            raise IntegrityError("membership coordinator pin does not match cluster metadata")
        return identity, cluster, membership

    def _node_metadata(self, configuration: NodeConfiguration) -> NodeMetadata:
        assert self._identity is not None
        assert self._membership is not None
        platform_identity = self.platform.identity()
        existing = self.state.node(configuration.node_id)
        build_id, lock_hash = _build_identity()
        operating_system = f"{platform_identity.system} {platform_identity.release}"
        records = list(existing.backend_validations if existing is not None else [])
        selected_backend = configuration.backend_selection.selected_backend
        current_record = next(
            (
                item
                for item in records
                if item.backend == selected_backend
                and item.platform_system == platform_identity.system
                and item.platform_release == platform_identity.release
                and item.platform_architecture.lower() == platform_identity.architecture.lower()
            ),
            None,
        )
        if current_record is None:
            records.append(
                BackendValidationRecord.not_run(
                    backend=selected_backend,
                    platform=platform_identity,
                )
            )
        software_status, physical_status = aggregate_validation_status(
            records,
            backend=selected_backend,
            architecture=platform_identity.architecture,
            operating_system=operating_system,
        )
        return (
            NodeMetadata(
                node_id=configuration.node_id,
                public_key=self._identity.public_key_b64,
                fingerprint=self._identity.public_key_fingerprint,
                hostname=socket.gethostname(),
                operating_system=operating_system,
                architecture=platform_identity.architecture,
                agent_version=__version__,
                runtime_version=__version__,
                build_id=build_id,
                package_lock_hash=lock_hash,
                selected_backend=configuration.backend_selection.selected_backend,
                selected_device=configuration.backend_selection.selected_device,
                worker_ids=[self.runtime_manager.worker_id(configuration)],
                control_endpoint=configuration.endpoints.control_advertised_endpoint,
                data_endpoint=configuration.endpoints.data_advertised_endpoint,
                probe_endpoint=configuration.endpoints.probe_advertised_endpoint,
                joined_at_unix_ns=self._membership.joined_at_unix_ns,
                last_seen_at_unix_ns=self.clock_ns(),
                service_mode=configuration.service_mode,
                implementation_status=platform_identity.implementation_status,
                implementation_reason=platform_identity.implementation_reason,
                software_validation_status=software_status,
                physical_validation_status=physical_status,
                backend_validations=records,
                revoked=False,
                revoked_at_unix_ns=None,
                revocation_reason=None,
            )
            if existing is None
            else existing.model_copy(
                update={
                    "hostname": socket.gethostname(),
                    "operating_system": operating_system,
                    "architecture": platform_identity.architecture,
                    "agent_version": __version__,
                    "runtime_version": __version__,
                    "build_id": build_id,
                    "package_lock_hash": lock_hash,
                    "selected_backend": configuration.backend_selection.selected_backend,
                    "selected_device": configuration.backend_selection.selected_device,
                    "worker_ids": [self.runtime_manager.worker_id(configuration)],
                    "control_endpoint": configuration.endpoints.control_advertised_endpoint,
                    "data_endpoint": configuration.endpoints.data_advertised_endpoint,
                    "probe_endpoint": configuration.endpoints.probe_advertised_endpoint,
                    "last_seen_at_unix_ns": self.clock_ns(),
                    "service_mode": configuration.service_mode,
                    "implementation_status": platform_identity.implementation_status,
                    "implementation_reason": platform_identity.implementation_reason,
                    "software_validation_status": software_status,
                    "physical_validation_status": physical_status,
                    "backend_validations": records,
                }
            )
        )

    async def _publish_node_and_verify(
        self,
        metadata: NodeMetadata,
    ) -> ReachabilityCheckResponse:
        assert self._identity is not None
        assert self._cluster is not None
        runtime_status = self.status if self._status is not None else None
        body: dict[str, Any] = {"metadata": metadata.model_dump(mode="json")}
        if runtime_status is not None:
            body["runtime"] = runtime_status.model_dump(mode="json")
        update_auth = create_cluster_authentication(
            identity=self._identity,
            node_id=metadata.node_id,
            action="node-update",
            body=body,
            timestamp_unix_ns=self.clock_ns(),
        )
        client = self.client_factory(self._cluster.coordinator_endpoint)
        try:
            update = await client.node_update(
                NodeUpdateRequest(
                    authentication=update_auth,
                    metadata=metadata,
                    runtime=runtime_status,
                )
            )
            if not update.accepted:
                raise IntegrityError("coordinator rejected node metadata update")
            reach_body = {
                "node_id": metadata.node_id,
                "timeout_ms": self.options.reachability_timeout_ms,
            }
            reach_auth = create_cluster_authentication(
                identity=self._identity,
                node_id=metadata.node_id,
                action="reachability-check",
                body=reach_body,
                timestamp_unix_ns=self.clock_ns(),
            )
            return await client.reachability_check(
                ReachabilityCheckRequest(
                    authentication=reach_auth,
                    node_id=metadata.node_id,
                    timeout_ms=self.options.reachability_timeout_ms,
                )
            )
        finally:
            await client.close()

    async def _publish_runtime_status(self, metadata: NodeMetadata) -> None:
        """Publish the post-transition state without another reachability cycle."""

        assert self._identity is not None
        assert self._cluster is not None
        runtime_status = self.status
        body = {
            "metadata": metadata.model_dump(mode="json"),
            "runtime": runtime_status.model_dump(mode="json"),
        }
        authentication = create_cluster_authentication(
            identity=self._identity,
            node_id=metadata.node_id,
            action="node-update",
            body=body,
            timestamp_unix_ns=self.clock_ns(),
        )
        client = self.client_factory(self._cluster.coordinator_endpoint)
        try:
            update = await client.node_update(
                NodeUpdateRequest(
                    authentication=authentication,
                    metadata=metadata,
                    runtime=runtime_status,
                )
            )
            if not update.accepted:
                raise IntegrityError("coordinator rejected node runtime status update")
        finally:
            await client.close()

    async def _apply_firewall_and_retry(
        self,
        metadata: NodeMetadata,
        response: ReachabilityCheckResponse,
    ) -> ReachabilityCheckResponse:
        assert self._cluster is not None
        control_port = split_endpoint(response.control_endpoint)[1]
        data_port = split_endpoint(response.data_endpoint)[1]
        probe_ports = (
            [split_endpoint(response.probe_endpoint)[1]]
            if response.probe_endpoint is not None
            else []
        )
        specification = FirewallRuleSpec(
            cluster_id=self._cluster.cluster_id,
            node_id=metadata.node_id,
            control_ports=[control_port],
            data_ports=[data_port, *probe_ports],
            private_subnets=self.options.private_subnets,
        )
        firewall = await self.service_manager.configure_firewall(specification)
        if firewall.blocked:
            action = firewall.remediation_command or "configure the private firewall rule"
            raise PermissionError(f"{firewall.detail}; corrective action: {action}")
        return await self._publish_node_and_verify(metadata)

    async def _refresh_directed_links(self) -> int:
        """Measure every stale outgoing product-node link through direct sockets."""

        assert self._identity is not None
        assert self._cluster is not None
        assert self._configuration is not None
        node_id = self._configuration.node_id
        source_worker_id = self.runtime_manager.worker_id(self._configuration)
        client = self.client_factory(self._cluster.coordinator_endpoint)
        # Compatibility clients used by low-level/manual deployments predate
        # cluster measurement. They remain usable but cannot create evidence.
        if not hasattr(client, "cluster_status") or not hasattr(client, "network_probe_control"):
            await client.close()
            return 0
        try:
            status_body = {"include_artifacts": False, "include_network": True}
            status_auth = create_cluster_authentication(
                identity=self._identity,
                node_id=node_id,
                action="cluster-status",
                body=status_body,
                timestamp_unix_ns=self.clock_ns(),
            )
            status = await client.cluster_status(
                ClusterStatusRequest(
                    authentication=status_auth,
                    include_artifacts=False,
                    include_network=True,
                )
            )
            repository = NetworkMeasurementRepository(
                state=self.state,
                ttl_seconds=self.options.network_measurement_ttl_seconds,
                clock_ns=self.clock_ns,
            )
            interface = next(
                (
                    item
                    for item in self.platform.interface_addresses()
                    if item.address == self._configuration.endpoints.source_address
                ),
                None,
            )
            measurer = DirectedNetworkMeasurer(
                state=self.state,
                cluster=self._cluster,
                identity=self._identity,
                node_id=node_id,
                worker_id=source_worker_id,
                source_interface=(interface.interface if interface is not None else None),
                source_mtu=(interface.mtu if interface is not None else None),
                timeout_seconds=self.options.network_probe_timeout_seconds,
                clock_ns=self.clock_ns,
            )
            destinations = sorted(
                (
                    node.metadata
                    for node in status.nodes
                    if node.metadata.node_id != node_id
                    and not node.metadata.revoked
                    and node.metadata.probe_endpoint is not None
                    and node.metadata.worker_ids
                ),
                key=lambda node: node.node_id,
            )[:64]
            completed = 0
            for destination in destinations:
                destination_worker_id = destination.worker_ids[0]
                if repository.get(source_worker_id, destination_worker_id) is not None:
                    continue
                issue = NetworkProbeControlRequest(
                    authentication=create_cluster_authentication(
                        identity=self._identity,
                        node_id=node_id,
                        action="network-probe-control",
                        body={
                            "operation": "issue",
                            "source_worker_id": source_worker_id,
                            "destination_worker_id": destination_worker_id,
                            "payload_sizes": self.options.network_probe_payload_sizes,
                            "sample_count": self.options.network_probe_sample_count,
                            "maximum_bytes": self.options.network_probe_max_bytes,
                            "timeout_ms": int(self.options.network_probe_timeout_seconds * 1000),
                            "measurement": None,
                        },
                        timestamp_unix_ns=self.clock_ns(),
                    ),
                    source_worker_id=source_worker_id,
                    destination_worker_id=destination_worker_id,
                    payload_sizes=self.options.network_probe_payload_sizes,
                    sample_count=self.options.network_probe_sample_count,
                    maximum_bytes=self.options.network_probe_max_bytes,
                    timeout_ms=int(self.options.network_probe_timeout_seconds * 1000),
                )
                issued = await client.network_probe_control(issue)
                if not issued.accepted or issued.ticket is None:
                    raise ConfigurationError(
                        issued.detail
                        or f"coordinator did not issue a probe ticket for {destination.node_id}"
                    )
                measurement = await measurer.measure(issued.ticket)
                record_body = {
                    "operation": "record",
                    "source_worker_id": source_worker_id,
                    "destination_worker_id": destination_worker_id,
                    "payload_sizes": self.options.network_probe_payload_sizes,
                    "sample_count": self.options.network_probe_sample_count,
                    "maximum_bytes": self.options.network_probe_max_bytes,
                    "timeout_ms": int(self.options.network_probe_timeout_seconds * 1000),
                    "measurement": measurement.model_dump(mode="json"),
                }
                recorded = await client.network_probe_control(
                    NetworkProbeControlRequest(
                        authentication=create_cluster_authentication(
                            identity=self._identity,
                            node_id=node_id,
                            action="network-probe-control",
                            body=record_body,
                            timestamp_unix_ns=self.clock_ns(),
                        ),
                        operation="record",
                        source_worker_id=source_worker_id,
                        destination_worker_id=destination_worker_id,
                        payload_sizes=self.options.network_probe_payload_sizes,
                        sample_count=self.options.network_probe_sample_count,
                        maximum_bytes=self.options.network_probe_max_bytes,
                        timeout_ms=int(self.options.network_probe_timeout_seconds * 1000),
                        measurement=measurement,
                    )
                )
                if not recorded.accepted:
                    raise IntegrityError(
                        recorded.detail
                        or f"coordinator rejected measurement for {destination.node_id}"
                    )
                completed += 1
            return completed
        finally:
            await client.close()

    async def _start_runtimes(self, configuration: NodeConfiguration) -> None:
        assert self._cluster is not None
        if "coordinator" in self.options.roles:
            self._coordinator = self.coordinator_factory(self._cluster)
            await self._coordinator.start()
        if "worker" in self.options.roles:
            self._worker = self.worker_factory(configuration)
            await self._worker.start()
            await self._start_network_probe(configuration)

    async def _start_network_probe(self, configuration: NodeConfiguration) -> None:
        if configuration.endpoints.probe_listen_endpoint is None:
            raise ConfigurationError("node configuration has no network probe endpoint")
        self._network_probe = self.network_probe_factory(configuration)
        await self._network_probe.start(configuration.endpoints.probe_listen_endpoint)

    async def _stop_runtimes(self) -> None:
        errors: list[Exception] = []
        if self._network_probe is not None:
            try:
                await self._network_probe.stop()
            except Exception as exc:
                errors.append(exc)
            self._network_probe = None
        if self._worker is not None:
            try:
                await self._worker.stop()
            except Exception as exc:
                errors.append(exc)
            self._worker = None
        if self._coordinator is not None:
            try:
                await self._coordinator.stop()
            except Exception as exc:
                errors.append(exc)
            self._coordinator = None
        if errors:
            raise ExceptionGroup("node runtime shutdown failed", errors)

    async def start(self) -> NodeRuntimeMetadata:
        async with self._lifecycle_lock:
            if self._status is not None and self._status.state in {"ready", "degraded", "blocked"}:
                return self.status
            self._stopping = False
            self._stop_event = asyncio.Event()
            try:
                self._identity, self._cluster, self._membership = self._load_security_state()
                node_id = node_id_from_fingerprint(self._identity.public_key_fingerprint)
                previous = self.state.load_node_configuration()
                self._configuration = self.runtime_manager.prepare_configuration(
                    node_id=node_id,
                    cluster=self._cluster,
                    backend_override=previous.backend_override if previous else None,
                    memory_limit_override_bytes=(
                        previous.memory_limit_override_bytes if previous else None
                    ),
                    memory_percent_override=(
                        previous.memory_percent_override if previous else None
                    ),
                    storage_limit_bytes=previous.storage_limit_bytes if previous else None,
                    control_endpoint_override=(
                        previous.control_endpoint_override if previous else None
                    ),
                    data_endpoint_override=(previous.data_endpoint_override if previous else None),
                    interface_override=previous.interface_override if previous else None,
                    service_mode=previous.service_mode if previous else None,
                )
                self._set_status("degraded", reason="starting canonical runtimes")
                await self._start_runtimes(self._configuration)
                metadata = self._node_metadata(self._configuration)
                self.state.save_node(metadata)
                reachability = await self._publish_node_and_verify(metadata)
                endpoints_reachable = (
                    reachability.control_reachable
                    and reachability.data_reachable
                    and reachability.probe_reachable is not False
                )
                if not endpoints_reachable:
                    reachability = await self._apply_firewall_and_retry(metadata, reachability)
                endpoints_reachable = (
                    reachability.control_reachable
                    and reachability.data_reachable
                    and reachability.probe_reachable is not False
                )
                if not endpoints_reachable:
                    self._set_status(
                        "blocked",
                        reason=reachability.detail,
                        last_error=reachability.detail,
                        error_category="connectivity",
                        coordinator_reachable=True,
                        data_reachable=False,
                    )
                    await self._publish_runtime_status(metadata)
                    return self.status
                await self._refresh_directed_links()
                self._set_status(
                    "ready",
                    reason="worker registered and bidirectional reachability verified",
                    coordinator_reachable=True,
                    data_reachable=True,
                )
                if self._status is not None:
                    self._status = self._status.model_copy(
                        update={"probe_reachable": reachability.probe_reachable}
                    )
                    self.state.save_runtime(self._status)
                await self._publish_runtime_status(metadata)
                self._audit("node_reconnected")
                if self._worker is not None:
                    self._monitor_task = asyncio.create_task(
                        self._monitor_worker(),
                        name=f"node-agent-monitor:{node_id}",
                    )
                self._health_task = asyncio.create_task(
                    self._health_loop(),
                    name=f"node-agent-health:{node_id}",
                )
                return self.status
            except BaseException as exc:
                with suppress(BaseException):
                    await self._stop_runtimes()
                if self._identity is None:
                    self._identity = self.state.load_or_create_node_identity()
                category = _error_category(exc)
                state_value: NodeAgentState = (
                    "blocked"
                    if isinstance(exc, (PermissionError, TransportError, OSError))
                    else "failed"
                )
                self._set_status(
                    state_value,
                    reason=str(exc),
                    last_error=str(exc),
                    error_category=category,
                )
                return self.status

    async def _restart_worker(self, reason: str) -> None:
        async with self._runtime_lock:
            if self._stopping or self._cluster is None or self._identity is None:
                return
            if self._worker is not None:
                await self._worker.stop()
            if self._network_probe is not None:
                await self._network_probe.stop()
                self._network_probe = None
            previous = self._configuration
            self._configuration = self.runtime_manager.prepare_configuration(
                node_id=node_id_from_fingerprint(self._identity.public_key_fingerprint),
                cluster=self._cluster,
                backend_override=previous.backend_override if previous else None,
                memory_limit_override_bytes=(
                    previous.memory_limit_override_bytes if previous else None
                ),
                memory_percent_override=(previous.memory_percent_override if previous else None),
                storage_limit_bytes=previous.storage_limit_bytes if previous else None,
                control_endpoint_override=(
                    previous.control_endpoint_override if previous else None
                ),
                data_endpoint_override=previous.data_endpoint_override if previous else None,
                interface_override=previous.interface_override if previous else None,
                service_mode=previous.service_mode if previous else None,
            )
            self._worker = self.worker_factory(self._configuration)
            await self._worker.start()
            await self._start_network_probe(self._configuration)
            metadata = self._node_metadata(self._configuration)
            self.state.save_node(metadata)
            reachability = await self._publish_node_and_verify(metadata)
            if not (
                reachability.control_reachable
                and reachability.data_reachable
                and reachability.probe_reachable is not False
            ):
                raise OSError(reachability.detail)
            self._set_status(
                "ready",
                reason=reason,
                coordinator_reachable=True,
                data_reachable=True,
            )

    async def _monitor_worker(self) -> None:
        attempts = 0
        while not self._stopping and self._worker is not None:
            worker = self._worker
            failure: BaseException | None = None
            try:
                await worker.wait()
                failure = RuntimeError("worker runtime stopped unexpectedly")
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                failure = exc
            if self._stopping:
                return
            assert failure is not None
            if isinstance(
                failure,
                (ConfigurationError, BackendIncompatibleError, IntegrityError, ValueError),
            ):
                self._set_status(
                    "failed",
                    reason="permanent worker configuration failure",
                    last_error=str(failure),
                    error_category=_error_category(failure),
                )
                return
            attempts += 1
            if attempts > self.options.maximum_restart_attempts:
                self._set_status(
                    "failed",
                    reason="bounded worker restart policy exhausted",
                    last_error=str(failure),
                    error_category=_error_category(failure),
                )
                return
            delay = min(
                self.options.restart_initial_backoff_seconds * (2 ** (attempts - 1)),
                self.options.restart_maximum_backoff_seconds,
            )
            self._set_status(
                "degraded",
                reason=f"worker restart {attempts}/{self.options.maximum_restart_attempts} in {delay:.1f}s",
                last_error=str(failure),
                error_category=_error_category(failure),
            )
            await self.sleep(delay)
            try:
                await self._restart_worker("worker restarted after bounded backoff")
            except BaseException as exc:
                failure = exc
                continue

    async def _health_loop(self) -> None:
        while not self._stopping:
            await self.sleep(self.options.health_refresh_seconds)
            if self._stopping or self._configuration is None:
                return
            try:
                current = self.runtime_manager.current_network_fingerprint(
                    coordinator_endpoint=self._configuration.coordinator_endpoint,
                    interface_override=self._configuration.interface_override,
                )
                if current != self._configuration.endpoints.network_fingerprint:
                    self._set_status("degraded", reason="network change detected")
                    await self._restart_worker("network endpoints reselected after change")
                elif self._status is not None:
                    self._set_status(
                        self._status.state,
                        reason=self._status.reason,
                        coordinator_reachable=self._status.coordinator_reachable,
                        data_reachable=self._status.data_reachable,
                    )
                await self._refresh_directed_links()
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                self._set_status(
                    "degraded",
                    reason="capability or network refresh failed",
                    last_error=str(exc),
                    error_category=_error_category(exc),
                )

    async def refresh_before_deployment(self) -> NodeRuntimeMetadata:
        if self._configuration is None:
            raise RuntimeError("node agent is not started")
        await self._restart_worker("memory and endpoint choices refreshed before deployment")
        return self.status

    async def wait(self) -> None:
        await self._stop_event.wait()

    async def stop(self) -> None:
        async with self._lifecycle_lock:
            if self._stopping:
                return
            self._stopping = True
            for task in (self._monitor_task, self._health_task):
                if task is not None:
                    task.cancel()
            await asyncio.gather(
                *(task for task in (self._monitor_task, self._health_task) if task is not None),
                return_exceptions=True,
            )
            self._monitor_task = None
            self._health_task = None
            try:
                await self._stop_runtimes()
            finally:
                if self._identity is None:
                    self._identity = self.state.load_or_create_node_identity()
                self._set_status("stopped", reason="node agent stopped")
                self._stop_event.set()


__all__ = ["NodeAgent", "NodeAgentOptions"]
