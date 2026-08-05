from __future__ import annotations

from pathlib import Path

import pytest

from swarm_inference.cluster.service_manager import ServiceManager
from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.platforms.base import ServiceDefinition, ServiceStatus


class _ServicePlatform:
    def __init__(self, path: Path, *, available: bool = True) -> None:
        self.path = path
        self.available = available

    def install_service(self, definition: ServiceDefinition) -> ServiceStatus:
        return ServiceStatus(
            service_name=definition.service_name,
            mode="systemd-user" if self.available else "unavailable",
            installed=self.available,
            running=self.available,
            detail="installed" if self.available else "use --foreground",
            log_path=self.path / "agent.log",
        )

    def uninstall_service(self, definition: ServiceDefinition) -> ServiceStatus:
        return ServiceStatus(
            service_name=definition.service_name,
            mode="systemd-user" if self.available else "unavailable",
            installed=False,
            running=False,
            detail="uninstalled",
            log_path=self.path / "agent.log",
        )

    start_service = install_service
    stop_service = uninstall_service

    def service_status(self, definition: ServiceDefinition) -> ServiceStatus:
        return self.install_service(definition)


def _definition(tmp_path: Path) -> ServiceDefinition:
    return ServiceDefinition(
        cluster_id="cluster-service",
        node_id="node-12345678",
        executable=tmp_path / "swarm",
        arguments=["node", "agent"],
    )


@pytest.mark.asyncio
async def test_service_manager_install_start_stop_uninstall_are_audited(tmp_path: Path) -> None:
    state = ClusterStateStore(tmp_path / "state")
    manager = ServiceManager(
        platform=_ServicePlatform(tmp_path),  # type: ignore[arg-type]
        state=state,
    )
    definition = _definition(tmp_path)
    assert (await manager.install(definition)).installed
    assert (await manager.start(definition)).running
    assert not (await manager.stop(definition)).running
    assert not (await manager.uninstall(definition)).installed
    audit = state.paths.audit_log.read_text(encoding="utf-8")
    assert "service_installed" in audit
    assert "service_started" in audit
    assert "service_stopped" in audit
    assert "service_uninstalled" in audit


@pytest.mark.asyncio
async def test_unsupported_service_manager_returns_foreground_remediation(tmp_path: Path) -> None:
    manager = ServiceManager(
        platform=_ServicePlatform(tmp_path, available=False),  # type: ignore[arg-type]
        state=ClusterStateStore(tmp_path / "state"),
    )
    status = await manager.install(_definition(tmp_path))
    assert not status.installed
    assert status.mode == "unavailable"
    assert "--foreground" in status.detail
