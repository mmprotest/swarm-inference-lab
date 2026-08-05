"""Explicit diagnostic hooks for deterministic stage-ring transport faults.

No injector is installed by normal product construction.  These hooks exist so
tests and deliberately enabled diagnostics can interrupt the connection that is
actually carrying a matching frame instead of racing a listening socket.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Protocol

from swarm_inference.protocol.stage_ring import Operation, StageMessage


class StageRingFaultAction(StrEnum):
    CLOSE_ACTIVE_CONNECTION = "close_active_connection"


@dataclass(frozen=True, slots=True)
class FrameContext:
    topology_id: str
    route_generation: int
    session_id: str
    request_id: str
    operation: str
    source_stage: int
    destination_stage: int
    token_position: int
    sequence_number: int
    endpoint: str

    @classmethod
    def from_message(cls, message: StageMessage, *, endpoint: str) -> FrameContext:
        raw_generation = message.attributes.get("route_generation", 0)
        route_generation = (
            raw_generation
            if isinstance(raw_generation, int) and not isinstance(raw_generation, bool)
            else 0
        )
        return cls(
            topology_id=message.topology_id,
            route_generation=route_generation,
            session_id=message.session_id,
            request_id=message.request_id,
            operation=message.operation.name,
            source_stage=message.source_stage,
            destination_stage=message.destination_stage,
            token_position=message.token_position,
            sequence_number=message.sequence_number,
            endpoint=endpoint,
        )

    def as_dict(self) -> dict[str, str | int]:
        return asdict(self)


class StageRingFaultInjector(Protocol):
    """Optional asynchronous hooks around one live connection exchange."""

    async def before_send(self, context: FrameContext) -> StageRingFaultAction | None: ...

    async def after_send(self, context: FrameContext) -> StageRingFaultAction | None: ...

    async def before_receive(self, context: FrameContext) -> StageRingFaultAction | None: ...

    async def after_receive(self, context: FrameContext) -> StageRingFaultAction | None: ...

    async def fault_applied(
        self,
        context: FrameContext,
        action: StageRingFaultAction,
        *,
        active_connection: bool,
    ) -> None: ...


FaultEventSink = Callable[[dict[str, str | int | bool]], None]


@dataclass(slots=True)
class CloseConnectionBeforeSendInjector:
    """Close one matching active connection immediately before its frame write."""

    token_position: int
    operation: Operation = Operation.DECODE
    source_stage: int | None = None
    destination_stage: int | None = None
    request_id: str | None = None
    event_sink: FaultEventSink | None = None
    triggered: bool = field(default=False, init=False)
    events: list[dict[str, str | int | bool]] = field(default_factory=list, init=False)

    def _matches(self, context: FrameContext) -> bool:
        return (
            not self.triggered
            and context.operation == self.operation.name
            and context.token_position == self.token_position
            and (self.source_stage is None or context.source_stage == self.source_stage)
            and (
                self.destination_stage is None
                or context.destination_stage == self.destination_stage
            )
            and (self.request_id is None or context.request_id == self.request_id)
        )

    async def before_send(self, context: FrameContext) -> StageRingFaultAction | None:
        if not self._matches(context):
            return None
        # Reserve the one-shot boundary before yielding back to the transport.
        self.triggered = True
        return StageRingFaultAction.CLOSE_ACTIVE_CONNECTION

    async def after_send(self, context: FrameContext) -> StageRingFaultAction | None:
        return None

    async def before_receive(self, context: FrameContext) -> StageRingFaultAction | None:
        return None

    async def after_receive(self, context: FrameContext) -> StageRingFaultAction | None:
        return None

    async def fault_applied(
        self,
        context: FrameContext,
        action: StageRingFaultAction,
        *,
        active_connection: bool,
    ) -> None:
        event: dict[str, str | int | bool] = {
            "event_type": "stage_ring_fault_injected",
            "action": action.value,
            "active_connection": active_connection,
            **context.as_dict(),
        }
        self.events.append(event)
        if self.event_sink is not None:
            self.event_sink(dict(event))


__all__ = [
    "CloseConnectionBeforeSendInjector",
    "FaultEventSink",
    "FrameContext",
    "StageRingFaultAction",
    "StageRingFaultInjector",
]
