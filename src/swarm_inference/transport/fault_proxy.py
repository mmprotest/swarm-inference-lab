"""Explicit loopback/physical fault injection around a real transport."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from swarm_inference.exceptions import IntegrityError, TransportError
from swarm_inference.protocol.messages import (
    Ack,
    ActivationRequest,
    ActivationResult,
    CancelRequest,
    HealthResponse,
    StageAssignmentMessage,
)
from swarm_inference.transport.base import ActivationTransport


@dataclass(slots=True)
class EndpointFault:
    delay_s: float = 0.0
    disconnected: bool = False
    corrupt_next_activation: bool = False
    timeout_next: bool = False


class FaultProxy:
    """Faults are observable and categorised separately from measured transport."""

    def __init__(self, inner: ActivationTransport) -> None:
        self.inner = inner
        self.faults: dict[str, EndpointFault] = {}
        self.events: list[dict[str, object]] = []

    def configure(self, endpoint: str, fault: EndpointFault) -> None:
        self.faults[endpoint] = fault

    async def assign(self, endpoint: str, assignment: StageAssignmentMessage) -> Ack:
        return await self.inner.assign(endpoint, assignment)

    async def execute(self, endpoint: str, request: ActivationRequest) -> ActivationResult:
        fault = self.faults.setdefault(endpoint, EndpointFault())
        if fault.disconnected:
            self.events.append({"type": "emulated_disconnection", "endpoint": endpoint})
            raise TransportError(f"fault proxy disconnected endpoint {endpoint}")
        if fault.timeout_next:
            fault.timeout_next = False
            self.events.append({"type": "emulated_timeout", "endpoint": endpoint})
            raise TransportError(f"fault proxy timed out endpoint {endpoint}")
        if fault.delay_s:
            self.events.append(
                {
                    "type": "emulated_wan_delay",
                    "endpoint": endpoint,
                    "delay_s": fault.delay_s,
                }
            )
            await asyncio.sleep(fault.delay_s)
        result = await self.inner.execute(endpoint, request)
        if fault.corrupt_next_activation:
            fault.corrupt_next_activation = False
            corrupted = bytearray(result.tensor_payload)
            if not corrupted:
                raise IntegrityError("cannot corrupt an empty activation")
            corrupted[-1] ^= 0x01
            result.tensor_payload = bytes(corrupted)
            self.events.append({"type": "emulated_corrupt_activation", "endpoint": endpoint})
        return result

    async def cancel(self, endpoint: str, request: CancelRequest) -> Ack:
        return await self.inner.cancel(endpoint, request)

    async def health(self, endpoint: str) -> HealthResponse:
        fault = self.faults.setdefault(endpoint, EndpointFault())
        if fault.disconnected:
            raise TransportError(f"fault proxy disconnected endpoint {endpoint}")
        return await self.inner.health(endpoint)

    async def close(self) -> None:
        await self.inner.close()
