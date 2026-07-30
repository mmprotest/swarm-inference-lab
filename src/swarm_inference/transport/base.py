"""Transport contract kept independent from worker execution."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

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


@runtime_checkable
class ActivationTransport(Protocol):
    async def assign(self, endpoint: str, assignment: StageAssignmentMessage) -> Ack: ...

    async def execute(self, endpoint: str, request: ActivationRequest) -> ActivationResult: ...

    async def install_route(self, endpoint: str, request: RouteInstallRequest) -> Ack: ...

    async def dispatch(self, endpoint: str, request: DataPlaneEnvelope) -> DataPlaneAck: ...

    async def cancel(self, endpoint: str, request: CancelRequest) -> Ack: ...

    async def health(self, endpoint: str) -> HealthResponse: ...

    async def close(self) -> None: ...
