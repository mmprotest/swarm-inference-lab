"""Bounded node-agent facade over user services and owned firewall rules."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from uuid import uuid4

from swarm_inference.cluster.models import ClusterAuditEvent
from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.platforms.base import (
    FirewallRuleSpec,
    FirewallStatus,
    PlatformAdapter,
    ServiceDefinition,
    ServiceStatus,
)


class ServiceManager:
    """Never blocks the event loop or waits indefinitely for host tooling."""

    def __init__(
        self,
        *,
        platform: PlatformAdapter,
        state: ClusterStateStore,
        operation_timeout_seconds: float = 60.0,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        if operation_timeout_seconds <= 0 or operation_timeout_seconds > 300:
            raise ValueError("service operation timeout must be in (0, 300] seconds")
        self.platform = platform
        self.state = state
        self.operation_timeout_seconds = operation_timeout_seconds
        self.clock_ns = clock_ns

    async def _bounded(self, operation: Callable[[], object]) -> object:
        return await asyncio.wait_for(
            asyncio.to_thread(operation),
            timeout=self.operation_timeout_seconds,
        )

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

    async def _service_operation(
        self,
        definition: ServiceDefinition,
        *,
        event_success: str,
        event_failure: str,
        operation: Callable[[ServiceDefinition], ServiceStatus],
        succeeded: Callable[[ServiceStatus], bool],
    ) -> ServiceStatus:
        try:
            status = await self._bounded(lambda: operation(definition))
        except TimeoutError:
            detail = (
                f"service operation exceeded {self.operation_timeout_seconds:.1f} seconds; "
                "retry is safe after checking the platform service status"
            )
            self._audit(
                event_failure,
                cluster_id=definition.cluster_id,
                node_id=definition.node_id,
                detail=detail,
                category="execution",
            )
            raise TimeoutError(detail) from None
        assert isinstance(status, ServiceStatus)
        failed = not succeeded(status)
        self._audit(
            event_failure if failed else event_success,
            cluster_id=definition.cluster_id,
            node_id=definition.node_id,
            detail=status.detail,
            category="execution" if failed else None,
        )
        return status

    async def install(self, definition: ServiceDefinition) -> ServiceStatus:
        return await self._service_operation(
            definition,
            event_success="service_installed",
            event_failure="service_failed",
            operation=self.platform.install_service,
            succeeded=lambda status: status.installed,
        )

    async def uninstall(self, definition: ServiceDefinition) -> ServiceStatus:
        return await self._service_operation(
            definition,
            event_success="service_uninstalled",
            event_failure="service_failed",
            operation=self.platform.uninstall_service,
            succeeded=lambda status: not status.installed,
        )

    async def start(self, definition: ServiceDefinition) -> ServiceStatus:
        return await self._service_operation(
            definition,
            event_success="service_started",
            event_failure="service_failed",
            operation=self.platform.start_service,
            succeeded=lambda status: status.running,
        )

    async def stop(self, definition: ServiceDefinition) -> ServiceStatus:
        return await self._service_operation(
            definition,
            event_success="service_stopped",
            event_failure="service_failed",
            operation=self.platform.stop_service,
            succeeded=lambda status: not status.running,
        )

    async def status(self, definition: ServiceDefinition) -> ServiceStatus:
        value = await self._bounded(lambda: self.platform.service_status(definition))
        assert isinstance(value, ServiceStatus)
        return value

    async def _firewall_operation(
        self,
        specification: FirewallRuleSpec,
        *,
        success_event: str,
        operation: Callable[[FirewallRuleSpec], FirewallStatus],
    ) -> FirewallStatus:
        value = await self._bounded(lambda: operation(specification))
        assert isinstance(value, FirewallStatus)
        event_type = "firewall_blocked" if value.blocked else success_event
        self._audit(
            event_type,
            cluster_id=specification.cluster_id,
            node_id=specification.node_id,
            detail=value.detail,
            category="permission" if value.blocked else None,
        )
        return value

    async def configure_firewall(self, specification: FirewallRuleSpec) -> FirewallStatus:
        return await self._firewall_operation(
            specification,
            success_event="firewall_applied",
            operation=self.platform.configure_firewall,
        )

    async def firewall_status(self, specification: FirewallRuleSpec) -> FirewallStatus:
        value = await self._bounded(lambda: self.platform.firewall_status(specification))
        assert isinstance(value, FirewallStatus)
        return value

    async def remove_firewall(self, specification: FirewallRuleSpec) -> FirewallStatus:
        return await self._firewall_operation(
            specification,
            success_event="firewall_removed",
            operation=self.platform.remove_firewall,
        )


__all__ = ["ServiceManager"]
