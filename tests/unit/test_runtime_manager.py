from __future__ import annotations

from pathlib import Path

import pytest

from swarm_inference.cluster.models import (
    ClusterMetadata,
    VersionCompatibility,
    node_id_from_fingerprint,
)
from swarm_inference.cluster.runtime_manager import RuntimeManager
from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.config.models import Backend
from swarm_inference.exceptions import BackendIncompatibleError, ConfigurationError
from swarm_inference.platforms.base import BackendProbeResult, InterfaceAddress
from swarm_inference.security.identity import CoordinatorIdentity, WorkerIdentity


class _Platform:
    service_mode = "foreground"

    def __init__(self) -> None:
        self.probes = [
            BackendProbeResult(
                backend=Backend.TORCH_CUDA,
                device="cuda",
                detected=True,
                operational=False,
                reason="CUDA tensor probe failed",
            ),
            BackendProbeResult(
                backend=Backend.TORCH_CPU,
                device="cpu",
                detected=True,
                operational=True,
                reason="correct tensor probe passed",
                total_memory_bytes=16 * 1024**3,
                available_memory_bytes=8 * 1024**3,
                supported_dtypes=["float32", "bfloat16"],
            ),
        ]
        self.source = "192.168.1.20"
        self.addresses = [
            InterfaceAddress(
                interface="Ethernet",
                address="192.168.1.20",
                prefix_length=24,
                is_private=True,
                is_loopback=False,
                is_up=True,
                mtu=1500,
            ),
            InterfaceAddress(
                interface="VPN",
                address="10.8.0.4",
                prefix_length=24,
                is_private=True,
                is_loopback=False,
                is_up=True,
                mtu=1400,
            ),
        ]

    def accelerator_probes(self):
        return self.probes

    def interface_addresses(self):
        return self.addresses

    def routed_source_address(self, destination_endpoint: str) -> str:
        del destination_endpoint
        return self.source


def _cluster() -> ClusterMetadata:
    coordinator = CoordinatorIdentity.generate()
    return ClusterMetadata(
        cluster_id="cluster-runtime",
        name="runtime-test",
        coordinator_id=node_id_from_fingerprint(coordinator.public_key_fingerprint),
        coordinator_endpoint="192.168.1.10:50051",
        coordinator_public_key=coordinator.public_key_b64,
        coordinator_fingerprint=coordinator.public_key_fingerprint,
        created_at_unix_ns=1,
        runtime_compatibility=VersionCompatibility(
            minimum_runtime_version="0.1.0",
            maximum_runtime_version_exclusive="0.2.0",
        ),
    )


def test_backend_detection_falls_back_only_after_operational_probe(tmp_path: Path) -> None:
    manager = RuntimeManager(
        state=ClusterStateStore(tmp_path),
        platform=_Platform(),  # type: ignore[arg-type]
        port_available=lambda host, port: True,
    )
    report, selected = manager.select_backend()
    assert selected.backend == Backend.TORCH_CPU
    assert report.selected_dtype == "bfloat16"
    assert report.candidates[0].backend == Backend.TORCH_CUDA
    assert not report.candidates[0].operational

    with pytest.raises(BackendIncompatibleError, match="tensor probe failed"):
        manager.select_backend(override=Backend.TORCH_CUDA)


def test_memory_budget_defaults_and_overrides_are_bounded(tmp_path: Path) -> None:
    manager = RuntimeManager(
        state=ClusterStateStore(tmp_path),
        platform=_Platform(),  # type: ignore[arg-type]
    )
    probe = _Platform().probes[1]
    automatic = manager.calculate_memory_budget(probe)
    assert automatic.limit_bytes == int(8 * 1024**3 * 0.75)
    assert automatic.limit_bytes <= int(16 * 1024**3 * 0.75)
    explicit = manager.calculate_memory_budget(probe, explicit_percent=60)
    assert explicit.limit_bytes == int(8 * 1024**3 * 0.60)
    assert explicit.source == "explicit-percent"
    with pytest.raises(ConfigurationError, match="mutually exclusive"):
        manager.calculate_memory_budget(
            probe,
            explicit_bytes=1024,
            explicit_percent=50,
        )


def test_port_conflict_recovers_with_bounded_deterministic_scan(tmp_path: Path) -> None:
    calls = 0

    def available(host: str, port: int) -> bool:
        nonlocal calls
        del host, port
        calls += 1
        return calls > 1

    manager = RuntimeManager(
        state=ClusterStateStore(tmp_path),
        platform=_Platform(),  # type: ignore[arg-type]
        port_available=available,
    )
    endpoints = manager.select_endpoints(
        node_id="node-12345678",
        coordinator_endpoint="192.168.1.10:50051",
    )
    assert calls >= 3
    assert endpoints.control_advertised_endpoint != endpoints.data_advertised_endpoint
    assert endpoints.source_address == "192.168.1.20"


def test_endpoint_selection_rejects_loopback_wildcard_and_honours_multi_nic(tmp_path: Path) -> None:
    platform = _Platform()
    manager = RuntimeManager(
        state=ClusterStateStore(tmp_path),
        platform=platform,  # type: ignore[arg-type]
        port_available=lambda host, port: True,
    )
    with pytest.raises(ConfigurationError, match="loopback"):
        platform.source = "127.0.0.1"
        manager.select_endpoints(
            node_id="node-12345678",
            coordinator_endpoint="127.0.0.1:50051",
        )
    platform.source = "192.168.1.20"
    with pytest.raises(ConfigurationError, match="wildcard"):
        manager.select_endpoints(
            node_id="node-12345678",
            coordinator_endpoint="192.168.1.10:50051",
            control_override="0.0.0.0:51000",
        )
    selected = manager.select_endpoints(
        node_id="node-12345678",
        coordinator_endpoint="192.168.1.10:50051",
        interface_override="VPN",
    )
    assert selected.source_address == "10.8.0.4"
    assert selected.interface_name == "VPN"


def test_prepared_configuration_persists_every_automatic_choice(tmp_path: Path) -> None:
    state = ClusterStateStore(tmp_path)
    identity = WorkerIdentity.load_or_create(state.paths.node_identity)
    manager = RuntimeManager(
        state=state,
        platform=_Platform(),  # type: ignore[arg-type]
        port_available=lambda host, port: True,
    )
    configuration = manager.prepare_configuration(
        node_id=node_id_from_fingerprint(identity.public_key_fingerprint),
        cluster=_cluster(),
    )
    assert state.load_node_configuration() == configuration
    assert RuntimeManager.worker_id(configuration).endswith("/cpu-0")
    assert configuration.storage_limit_bytes > 0
