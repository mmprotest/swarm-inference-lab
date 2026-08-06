"""Automatic product configuration and canonical runtime construction."""

from __future__ import annotations

import hashlib
import ipaddress
import shutil
import socket
import time
from collections.abc import Callable, Sequence
from uuid import uuid4

from swarm_inference.cluster.artifacts import ArtifactManager, ArtifactOperationCoordinator
from swarm_inference.cluster.models import (
    BackendCandidateRecord,
    BackendSelectionReport,
    ClusterAuditEvent,
    ClusterMetadata,
    EndpointSelection,
    MemoryBudget,
    NodeConfiguration,
    NodeServiceMode,
    aggregate_validation_status,
)
from swarm_inference.cluster.network import NetworkProbeCoordinator
from swarm_inference.cluster.pairing import PairingManager
from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.config.models import Backend, WorkerRole
from swarm_inference.config.product import ProductCoordinatorConfig
from swarm_inference.coordinator.deployment import TransportArtifactCoordinator
from swarm_inference.coordinator.runtime import CoordinatorRuntime
from swarm_inference.coordinator.service import CoordinatorCore
from swarm_inference.exceptions import BackendIncompatibleError, ConfigurationError
from swarm_inference.host import format_endpoint, is_loopback_host, is_wildcard_host, split_endpoint
from swarm_inference.platforms.base import BackendProbeResult, PlatformAdapter
from swarm_inference.security.trust_store import WorkerTrustStore
from swarm_inference.worker.runtime import WorkerRuntime, WorkerRuntimeConfig

_DYNAMIC_PORT_START = 49152
_DYNAMIC_PORT_END = 65535
_MAXIMUM_PORT_PROBES = 1024
_GIB = 1024**3


def _port_is_available(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    bind_host = "::" if family == socket.AF_INET6 else "0.0.0.0"
    try:
        with socket.socket(family, socket.SOCK_STREAM) as listener:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            listener.bind((bind_host, port))
    except OSError:
        return False
    return True


def _network_fingerprint(addresses: Sequence[object], source_address: str) -> str:
    rows = sorted(str(value) for value in addresses)
    payload = "\n".join([source_address, *rows]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class RuntimeManager:
    """Own every automatic choice formerly exposed as a required worker flag."""

    def __init__(
        self,
        *,
        state: ClusterStateStore,
        platform: PlatformAdapter,
        clock_ns: Callable[[], int] = time.time_ns,
        port_available: Callable[[str, int], bool] = _port_is_available,
        maximum_port_probes: int = _MAXIMUM_PORT_PROBES,
        allow_loopback: bool = False,
    ) -> None:
        if maximum_port_probes <= 0 or maximum_port_probes > (
            _DYNAMIC_PORT_END - _DYNAMIC_PORT_START
        ):
            raise ValueError("port probe bound is invalid")
        self.state = state
        self.platform = platform
        self.clock_ns = clock_ns
        self.port_available = port_available
        self.maximum_port_probes = maximum_port_probes
        self.allow_loopback = allow_loopback
        self.artifact_manager: ArtifactManager | None = None

    def _audit(
        self,
        event_type: str,
        *,
        cluster_id: str,
        node_id: str,
        detail: str,
        category: str | None = None,
    ) -> None:
        self.state.append_audit(
            ClusterAuditEvent(
                event_id=uuid4().hex,
                event_type=event_type,
                timestamp_unix_ns=self.clock_ns(),
                cluster_id=cluster_id,
                node_id=node_id,
                category=category,
                detail=detail,
            )
        )

    def select_backend(
        self,
        *,
        override: Backend | None = None,
    ) -> tuple[BackendSelectionReport, BackendProbeResult]:
        probes = self.platform.accelerator_probes()
        by_backend = {item.backend: item for item in probes}
        order = [Backend.TORCH_CUDA, Backend.TORCH_MPS, Backend.TORCH_CPU]
        if override is not None:
            order = [override]
        selected = next(
            (
                by_backend[item]
                for item in order
                if item in by_backend and by_backend[item].operational
            ),
            None,
        )
        if selected is None:
            if override is not None:
                probe = by_backend.get(override)
                reason = (
                    probe.reason if probe is not None else "backend was not probed on this platform"
                )
                raise BackendIncompatibleError(
                    f"configured backend {override.value} is not operational: {reason}"
                )
            details = "; ".join(f"{item.backend.value}: {item.reason}" for item in probes)
            raise BackendIncompatibleError(f"no operational torch backend was found: {details}")
        dtype_order = (
            ["bfloat16", "float16", "float32"]
            if selected.backend != Backend.TORCH_CPU
            else ["bfloat16", "float32", "float16"]
        )
        selected_dtype = next(
            (item for item in dtype_order if item in selected.supported_dtypes),
            None,
        )
        if selected_dtype is None:
            raise BackendIncompatibleError(
                f"backend {selected.backend.value} passed no supported dtype probe"
            )
        candidates = [
            BackendCandidateRecord.model_validate(probe.model_dump(exclude={"probe_version"}))
            for probe in sorted(
                probes, key=lambda item: order.index(item.backend) if item.backend in order else 99
            )
        ]
        reason = (
            f"explicit override {selected.backend.value} passed its operational tensor probe"
            if override is not None
            else f"selected highest-priority operational backend {selected.backend.value}"
        )
        return (
            BackendSelectionReport(
                candidates=candidates,
                selected_backend=selected.backend,
                selected_device=selected.device,
                selected_dtype=selected_dtype,
                reason=reason,
                measured_at_unix_ns=self.clock_ns(),
            ),
            selected,
        )

    def calculate_memory_budget(
        self,
        probe: BackendProbeResult,
        *,
        explicit_bytes: int | None = None,
        explicit_percent: float | None = None,
        cpu_total_fraction: float = 0.75,
    ) -> MemoryBudget:
        if explicit_bytes is not None and explicit_percent is not None:
            raise ConfigurationError("memory byte and percentage overrides are mutually exclusive")
        if not 0 < cpu_total_fraction <= 1:
            raise ValueError("CPU total-memory fraction must be in (0, 1]")
        available = int(probe.available_memory_bytes)
        total = int(probe.total_memory_bytes)
        if available <= 0 or total <= 0:
            raise ConfigurationError("selected backend did not report usable memory")
        if explicit_bytes is not None:
            if explicit_bytes <= 0 or explicit_bytes > available:
                raise ConfigurationError(f"explicit memory limit must be in (0, {available}] bytes")
            limit = explicit_bytes
            reserve = available - limit
            source = "explicit-bytes"
        elif explicit_percent is not None:
            if not 0 < explicit_percent <= 100:
                raise ConfigurationError("memory percent must be in (0, 100]")
            limit = max(1, int(available * explicit_percent / 100))
            reserve = available - limit
            source = "explicit-percent"
        elif probe.backend == Backend.TORCH_CPU:
            limit = max(1, min(int(available * 0.75), int(total * cpu_total_fraction)))
            reserve = available - limit
            source = "automatic"
        elif probe.backend == Backend.TORCH_CUDA:
            required_reserve = max(512 * 1024**2, int(total * 0.05))
            if available <= required_reserve:
                raise ConfigurationError("free CUDA memory is below the required runtime reserve")
            limit = min(int(available * 0.85), available - required_reserve)
            reserve = available - limit
            source = "automatic"
        elif probe.backend == Backend.TORCH_MPS:
            required_reserve = max(2 * _GIB, int(total * 0.20))
            if available <= required_reserve:
                raise ConfigurationError("available unified memory is below the macOS reserve")
            limit = min(int(available * 0.70), available - required_reserve)
            reserve = available - limit
            source = "automatic"
        else:
            raise BackendIncompatibleError(
                f"automatic memory budgeting does not support {probe.backend.value}"
            )
        return MemoryBudget(
            backend=probe.backend,
            available_bytes=available,
            total_bytes=total,
            reserve_bytes=reserve,
            limit_bytes=limit,
            source=source,
            fraction_of_available=limit / available,
        )

    def _validate_advertised_host(self, host: str) -> None:
        if is_wildcard_host(host):
            raise ConfigurationError("advertised endpoints cannot use a wildcard address")
        if is_loopback_host(host) and not self.allow_loopback:
            raise ConfigurationError("physical nodes cannot advertise a loopback address")
        try:
            address = ipaddress.ip_address(host.split("%", 1)[0])
        except ValueError:
            return
        if address.is_unspecified:
            raise ConfigurationError("advertised endpoints cannot use an unspecified address")

    def _find_port(self, *, node_id: str, purpose: str, excluded: set[int]) -> int:
        span = _DYNAMIC_PORT_END - _DYNAMIC_PORT_START + 1
        seed = int(hashlib.sha256(f"{node_id}:{purpose}".encode()).hexdigest()[:8], 16)
        for offset in range(self.maximum_port_probes):
            port = _DYNAMIC_PORT_START + ((seed + offset) % span)
            if port in excluded:
                continue
            if self.port_available("0.0.0.0", port):
                return port
        raise ConfigurationError(
            f"no available {purpose} port found after {self.maximum_port_probes} probes"
        )

    def _explicit_endpoint(
        self,
        value: str,
        *,
        node_id: str,
        purpose: str,
        excluded: set[int],
    ) -> tuple[str, int]:
        host, port = split_endpoint(value)
        self._validate_advertised_host(host)
        selected_port = port or self._find_port(
            node_id=node_id,
            purpose=purpose,
            excluded=excluded,
        )
        if port and (port in excluded or not self.port_available("0.0.0.0", port)):
            raise ConfigurationError(f"explicit {purpose} port {port} is already in use")
        return host, selected_port

    def select_endpoints(
        self,
        *,
        node_id: str,
        coordinator_endpoint: str,
        control_override: str | None = None,
        data_override: str | None = None,
        interface_override: str | None = None,
    ) -> EndpointSelection:
        addresses = self.platform.interface_addresses()
        source_address: str
        interface_name: str | None = None
        if interface_override is not None:
            candidates = [
                item
                for item in addresses
                if item.interface == interface_override
                and item.is_up
                and not item.is_loopback
                and item.is_private
            ]
            if not candidates:
                raise ConfigurationError(
                    f"interface override {interface_override!r} has no up private address"
                )
            selected_address = sorted(
                candidates,
                key=lambda item: (":" in item.address, item.address),
            )[0]
            source_address = selected_address.address
            interface_name = selected_address.interface
        else:
            source_address = self.platform.routed_source_address(coordinator_endpoint)
            interface_name = next(
                (
                    item.interface
                    for item in addresses
                    if item.address == source_address and item.is_up
                ),
                None,
            )
        self._validate_advertised_host(source_address)
        excluded: set[int] = set()
        if control_override is not None:
            control_host, control_port = self._explicit_endpoint(
                control_override,
                node_id=node_id,
                purpose="control",
                excluded=excluded,
            )
        else:
            control_host = source_address
            control_port = self._find_port(
                node_id=node_id,
                purpose="control",
                excluded=excluded,
            )
        excluded.add(control_port)
        if data_override is not None:
            data_host, data_port = self._explicit_endpoint(
                data_override,
                node_id=node_id,
                purpose="data",
                excluded=excluded,
            )
        else:
            data_host = source_address
            data_port = self._find_port(
                node_id=node_id,
                purpose="data",
                excluded=excluded,
            )
        excluded.add(data_port)
        probe_port = self._find_port(
            node_id=node_id,
            purpose="network-probe",
            excluded=excluded,
        )
        fingerprint = _network_fingerprint(addresses, source_address)
        return EndpointSelection(
            control_listen_endpoint=format_endpoint("0.0.0.0", control_port),
            control_advertised_endpoint=format_endpoint(control_host, control_port),
            data_listen_endpoint=format_endpoint("0.0.0.0", data_port),
            data_advertised_endpoint=format_endpoint(data_host, data_port),
            probe_listen_endpoint=format_endpoint("0.0.0.0", probe_port),
            probe_advertised_endpoint=format_endpoint(source_address, probe_port),
            source_address=source_address,
            interface_name=interface_name,
            selected_at_unix_ns=self.clock_ns(),
            selection_reason=(
                "explicit interface/endpoint override"
                if any((control_override, data_override, interface_override))
                else "routed-source discovery toward coordinator with available dynamic ports"
            ),
            network_fingerprint=fingerprint,
        )

    def select_coordinator_endpoint(
        self,
        *,
        node_id: str,
        override: str | None = None,
    ) -> str:
        """Choose a stable private coordinator address and an available port."""

        if override is not None:
            host, port = self._explicit_endpoint(
                override,
                node_id=node_id,
                purpose="coordinator",
                excluded=set(),
            )
            return format_endpoint(host, port)
        try:
            host = self.platform.routed_source_address("192.0.2.1:9")
            self._validate_advertised_host(host)
        except (OSError, ConfigurationError):
            candidates = sorted(
                (
                    item
                    for item in self.platform.interface_addresses()
                    if item.is_up and item.is_private and not item.is_loopback
                ),
                key=lambda item: (":" in item.address, item.interface, item.address),
            )
            if not candidates:
                raise ConfigurationError(
                    "no up private interface can advertise the coordinator"
                ) from None
            host = candidates[0].address
            self._validate_advertised_host(host)
        port = self._find_port(node_id=node_id, purpose="coordinator", excluded=set())
        return format_endpoint(host, port)

    def current_network_fingerprint(
        self,
        *,
        coordinator_endpoint: str,
        interface_override: str | None = None,
    ) -> str:
        addresses = self.platform.interface_addresses()
        if interface_override is not None:
            selected = next(
                (
                    item.address
                    for item in addresses
                    if item.interface == interface_override
                    and item.is_up
                    and item.is_private
                    and not item.is_loopback
                ),
                None,
            )
            if selected is None:
                raise ConfigurationError(
                    f"interface override {interface_override!r} is no longer available"
                )
            source = selected
        else:
            source = self.platform.routed_source_address(coordinator_endpoint)
        return _network_fingerprint(addresses, source)

    def _storage_limit(self, explicit_bytes: int | None) -> int:
        self.state.paths.artifacts.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.state.paths.artifacts).free
        if explicit_bytes is not None:
            if explicit_bytes <= 0 or explicit_bytes > free:
                raise ConfigurationError(f"storage limit must be in (0, {free}] bytes")
            return explicit_bytes
        return max(1, min(100 * _GIB, int(free * 0.50)))

    def prepare_configuration(
        self,
        *,
        node_id: str,
        cluster: ClusterMetadata,
        backend_override: Backend | None = None,
        memory_limit_override_bytes: int | None = None,
        memory_percent_override: float | None = None,
        storage_limit_bytes: int | None = None,
        control_endpoint_override: str | None = None,
        data_endpoint_override: str | None = None,
        interface_override: str | None = None,
        service_mode: NodeServiceMode | None = None,
    ) -> NodeConfiguration:
        previous = self.state.load_node_configuration()
        backend_override = backend_override or (previous.backend_override if previous else None)
        memory_limit_override_bytes = memory_limit_override_bytes or (
            previous.memory_limit_override_bytes if previous else None
        )
        memory_percent_override = memory_percent_override or (
            previous.memory_percent_override if previous else None
        )
        storage_limit_bytes = storage_limit_bytes or (
            previous.storage_limit_bytes if previous else None
        )
        control_endpoint_override = control_endpoint_override or (
            previous.control_endpoint_override if previous else None
        )
        data_endpoint_override = data_endpoint_override or (
            previous.data_endpoint_override if previous else None
        )
        interface_override = interface_override or (
            previous.interface_override if previous else None
        )
        backend_selection, probe = self.select_backend(override=backend_override)
        memory = self.calculate_memory_budget(
            probe,
            explicit_bytes=memory_limit_override_bytes,
            explicit_percent=memory_percent_override,
        )
        endpoints = self.select_endpoints(
            node_id=node_id,
            coordinator_endpoint=cluster.coordinator_endpoint,
            control_override=control_endpoint_override,
            data_override=data_endpoint_override,
            interface_override=interface_override,
        )
        selected_mode = service_mode or self.platform.service_mode
        configuration = NodeConfiguration(
            node_id=node_id,
            cluster_id=cluster.cluster_id,
            coordinator_endpoint=cluster.coordinator_endpoint,
            coordinator_fingerprint=cluster.coordinator_fingerprint,
            backend_override=backend_override,
            memory_limit_override_bytes=memory_limit_override_bytes,
            memory_percent_override=memory_percent_override,
            storage_limit_bytes=self._storage_limit(storage_limit_bytes),
            control_endpoint_override=control_endpoint_override,
            data_endpoint_override=data_endpoint_override,
            interface_override=interface_override,
            backend_selection=backend_selection,
            memory_budget=memory,
            endpoints=endpoints,
            service_mode=selected_mode,
            updated_at_unix_ns=self.clock_ns(),
        )
        self.state.save_node_configuration(configuration)
        self._audit(
            "backend_selected",
            cluster_id=cluster.cluster_id,
            node_id=node_id,
            detail=backend_selection.reason,
        )
        self._audit(
            "endpoint_selected",
            cluster_id=cluster.cluster_id,
            node_id=node_id,
            detail=endpoints.selection_reason,
        )
        return configuration

    @staticmethod
    def worker_id(configuration: NodeConfiguration) -> str:
        device = configuration.backend_selection.selected_device.lower()
        if device.startswith("cuda"):
            index = device.partition(":")[2] or "0"
            suffix = f"cuda-{index}"
        elif device.startswith("mps"):
            suffix = "mps-0"
        else:
            suffix = "cpu-0"
        return f"{configuration.node_id}/{suffix}"

    def build_worker_runtime(self, configuration: NodeConfiguration) -> WorkerRuntime:
        if self.artifact_manager is None:
            self.artifact_manager = ArtifactManager(
                state=self.state,
                node_id=configuration.node_id,
                storage_limit_bytes=configuration.storage_limit_bytes,
                clock_ns=self.clock_ns,
            )
        platform_identity = self.platform.identity()
        metadata = self.state.node(configuration.node_id)
        records = list(metadata.backend_validations if metadata is not None else [])
        software_status, physical_status = aggregate_validation_status(
            records,
            backend=configuration.backend_selection.selected_backend,
            architecture=platform_identity.architecture,
            operating_system=f"{platform_identity.system} {platform_identity.release}",
        )
        scoped_records = [
            item
            for item in records
            if item.backend == configuration.backend_selection.selected_backend
            and item.platform_system == platform_identity.system
            and item.platform_release == platform_identity.release
            and item.platform_architecture.lower() == platform_identity.architecture.lower()
        ]
        evidence_ids = sorted(
            {item.evidence_id for item in scoped_records if item.evidence_id is not None}
        )
        timestamps = [
            item.validated_at_unix_ns
            for item in scoped_records
            if item.validated_at_unix_ns is not None
        ]
        validation_detail = (
            "; ".join(item.detail for item in scoped_records)
            if scoped_records
            else "no retained validation evidence for the selected platform/backend scope"
        )
        return WorkerRuntime(
            config=WorkerRuntimeConfig(
                coordinator_endpoint=configuration.coordinator_endpoint,
                listen_endpoint=configuration.endpoints.control_listen_endpoint,
                advertised_endpoint=configuration.endpoints.control_advertised_endpoint,
                backend=configuration.backend_selection.selected_backend,
                memory_limit_bytes=configuration.memory_budget.limit_bytes,
                identity_path=self.state.paths.node_identity,
                worker_id=self.worker_id(configuration),
                stage_runtime_enabled=True,
                data_listen_endpoint=configuration.endpoints.data_listen_endpoint,
                data_advertised_endpoint=configuration.endpoints.data_advertised_endpoint,
                device=configuration.backend_selection.selected_device,
                dtype=configuration.backend_selection.selected_dtype,
                model_cache_dir=self.state.paths.artifacts / "source-cache",
                artifact_storage_limit_bytes=configuration.storage_limit_bytes,
                allow_model_download=configuration.allow_model_download,
                trusted_coordinator_fingerprint=configuration.coordinator_fingerprint,
                worker_roles={WorkerRole.CONTIGUOUS_STAGE},
                service_mode=configuration.service_mode,
                platform_implementation_status=platform_identity.implementation_status,
                software_validation_status=software_status,
                physical_validation_status=physical_status,
                validation_evidence_ids=evidence_ids,
                latest_validation_unix_ns=max(timestamps) if timestamps else None,
                validation_detail=validation_detail,
            ),
            artifact_manager=self.artifact_manager,
        )

    def build_coordinator_runtime(
        self,
        *,
        cluster: ClusterMetadata,
        listen_endpoint: str,
        advertised_endpoint: str,
    ) -> CoordinatorRuntime:
        trust_path = self.state.paths.security / "trusted-workers.json"
        product_config = ProductCoordinatorConfig(
            coordinator_id=cluster.coordinator_id,
            trust_store_path=trust_path,
        )
        configuration = self.state.load_node_configuration()
        if self.artifact_manager is None:
            self.artifact_manager = ArtifactManager(
                state=self.state,
                node_id=cluster.coordinator_id,
                storage_limit_bytes=(
                    configuration.storage_limit_bytes if configuration is not None else 100 * _GIB
                ),
                clock_ns=self.clock_ns,
            )
        core = CoordinatorCore(
            product_config=product_config,
            state_directory=self.state.paths.coordinator_runtime_directory,
            coordinator_identity_path=self.state.paths.coordinator_identity,
        )
        assert core.coordinator_identity is not None
        control = PairingManager(
            state=self.state,
            trust_store=WorkerTrustStore(trust_path),
            coordinator_identity=core.coordinator_identity,
            cluster=cluster,
            clock_ns=self.clock_ns,
        )
        core.attach_cluster_control(control)
        network = NetworkProbeCoordinator(
            state=self.state,
            cluster=cluster,
            identity=core.coordinator_identity,
            maximum_bytes=product_config.network_probe_max_bytes,
            clock_ns=self.clock_ns,
        )
        core.network_probe_handler = network.handle
        core.artifact_operation_handler = ArtifactOperationCoordinator(self.artifact_manager).handle
        assert core.deployment_manager is not None
        core.deployment_manager.artifact_coordinator = TransportArtifactCoordinator(
            transport=core.deployment_manager.transport,
            manager=self.artifact_manager,
            identity=core.coordinator_identity,
            coordinator_id=cluster.coordinator_id,
            lease_seconds=product_config.deployment_lease_seconds,
        )
        return CoordinatorRuntime(
            core=core,
            listen_endpoint=listen_endpoint,
            advertised_endpoint=advertised_endpoint,
            service_mode="agent",
        )


__all__ = ["RuntimeManager"]
