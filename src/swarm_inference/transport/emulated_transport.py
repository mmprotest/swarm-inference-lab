"""Deterministic transport wrapper for tests that own a simulated clock."""

from __future__ import annotations

from collections.abc import Callable

from swarm_inference.protocol.messages import (
    Ack,
    ActivationRequest,
    ActivationResult,
    CancelRequest,
    DataPlaneAck,
    DataPlaneEnvelope,
    HealthResponse,
    RouteInstallRequest,
    StageAssignmentMessage,
)
from swarm_inference.transport.base import ActivationTransport


class EmulatedTransport:
    """Record deterministic delay metadata while delegating real tensor execution."""

    def __init__(
        self,
        inner: ActivationTransport,
        *,
        delay_for_payload_s: Callable[[int], float],
    ) -> None:
        self.inner = inner
        self.delay_for_payload_s = delay_for_payload_s
        self.injected_delays_s: list[float] = []

    async def assign(self, endpoint: str, assignment: StageAssignmentMessage) -> Ack:
        return await self.inner.assign(endpoint, assignment)

    async def execute(self, endpoint: str, request: ActivationRequest) -> ActivationResult:
        # This abstraction deliberately does not sleep. The owning deterministic
        # clock schedules completion using this recorded delay.
        self.injected_delays_s.append(self.delay_for_payload_s(len(request.tensor_payload)))
        return await self.inner.execute(endpoint, request)

    async def install_route(self, endpoint: str, request: RouteInstallRequest) -> Ack:
        return await self.inner.install_route(endpoint, request)

    async def dispatch(
        self,
        endpoint: str,
        request: DataPlaneEnvelope,
    ) -> DataPlaneAck:
        self.injected_delays_s.append(self.delay_for_payload_s(len(request.tensor_payload)))
        return await self.inner.dispatch(endpoint, request)

    async def cancel(self, endpoint: str, request: CancelRequest) -> Ack:
        return await self.inner.cancel(endpoint, request)

    async def health(self, endpoint: str) -> HealthResponse:
        return await self.inner.health(endpoint)

    async def close(self) -> None:
        await self.inner.close()
