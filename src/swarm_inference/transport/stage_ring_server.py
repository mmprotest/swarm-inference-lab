"""Bounded persistent TCP server for canonical stage-ring messages."""

from __future__ import annotations

import asyncio
import socket
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import asdict, dataclass

from swarm_inference.exceptions import TransportError
from swarm_inference.host import format_endpoint, split_endpoint
from swarm_inference.protocol.stage_ring import (
    MAX_METADATA_BYTES,
    MAX_PAYLOAD_BYTES,
    Operation,
    StageMessage,
)
from swarm_inference.transport.stage_ring_connection import (
    read_stage_message,
    write_stage_message,
)

StageMessageHandler = Callable[[StageMessage], Awaitable[StageMessage]]


@dataclass(slots=True)
class StageRingServerMetrics:
    connections_accepted: int = 0
    active_connections: int = 0
    frames_received: int = 0
    frames_sent: int = 0
    reused_frames: int = 0
    malformed_frames: int = 0
    oversized_frames: int = 0
    backpressure_events: int = 0

    def snapshot(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class _InboundFrame:
    message: StageMessage
    future: asyncio.Future[StageMessage]


def _error_response(message: StageMessage, detail: str) -> StageMessage:
    return StageMessage(
        operation=Operation.ERROR,
        model_revision=message.model_revision,
        tokenizer_revision=message.tokenizer_revision,
        topology_id=message.topology_id,
        stage_id=message.stage_id,
        layer_start=message.layer_start,
        layer_end=message.layer_end,
        session_id=message.session_id,
        request_id=message.request_id,
        sequence_number=message.sequence_number,
        token_position=message.token_position,
        source_stage=message.destination_stage,
        destination_stage=message.source_stage,
        status="ERROR",
        attributes={"error": detail},
    )


class StageRingServer:
    """Serve many frames per TCP connection with bounded dispatch resources."""

    def __init__(
        self,
        *,
        handler: StageMessageHandler,
        queue_capacity: int = 256,
        dispatch_workers: int = 1,
        maximum_connections: int = 128,
        maximum_metadata_bytes: int = MAX_METADATA_BYTES,
        maximum_payload_bytes: int = MAX_PAYLOAD_BYTES,
        read_timeout_s: float = 30.0,
        write_timeout_s: float = 30.0,
        idle_timeout_s: float = 120.0,
        require_peer_authentication: bool | Callable[[], bool] = False,
    ) -> None:
        if min(queue_capacity, dispatch_workers, maximum_connections) <= 0:
            raise ValueError("stage-ring server bounds must be positive")
        if min(read_timeout_s, write_timeout_s, idle_timeout_s) <= 0:
            raise ValueError("stage-ring server timeouts must be positive")
        if not 0 < maximum_metadata_bytes <= MAX_METADATA_BYTES:
            raise ValueError("invalid stage-ring server metadata limit")
        if not 0 < maximum_payload_bytes <= MAX_PAYLOAD_BYTES:
            raise ValueError("invalid stage-ring server payload limit")
        self.handler = handler
        self.queue_capacity = queue_capacity
        self.dispatch_workers = dispatch_workers
        self.maximum_connections = maximum_connections
        self.maximum_metadata_bytes = maximum_metadata_bytes
        self.maximum_payload_bytes = maximum_payload_bytes
        self.read_timeout_s = read_timeout_s
        self.write_timeout_s = write_timeout_s
        self.idle_timeout_s = idle_timeout_s
        self.require_peer_authentication = require_peer_authentication
        self.metrics = StageRingServerMetrics()
        self._queue: asyncio.Queue[_InboundFrame] = asyncio.Queue(maxsize=queue_capacity)
        self._workers: list[asyncio.Task[None]] = []
        self._connections: set[asyncio.StreamWriter] = set()
        self._connection_tasks: set[asyncio.Task[None]] = set()
        self._server: asyncio.AbstractServer | None = None
        self.bound_endpoint: str | None = None

    async def start(self, endpoint: str) -> int:
        if self._server is not None:
            raise RuntimeError("stage-ring server is already started")
        host, port = split_endpoint(endpoint)
        self._workers = [
            asyncio.create_task(self._dispatch(), name=f"stage-ring-dispatch:{index}")
            for index in range(self.dispatch_workers)
        ]
        try:
            self._server = await asyncio.start_server(self._handle_connection, host, port)
        except BaseException:
            await self._stop_workers()
            raise
        server_sockets = self._server.sockets
        if not server_sockets:
            await self.stop()
            raise TransportError(f"could not bind stage-ring endpoint {endpoint}")
        bound = server_sockets[0].getsockname()
        bound_port = int(bound[1])
        self.bound_endpoint = format_endpoint(host, bound_port)
        return bound_port

    async def _dispatch(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                try:
                    response = await self.handler(item.message)
                except Exception as exc:
                    response = _error_response(
                        item.message,
                        f"{type(exc).__name__}: {exc}",
                    )
                if not item.future.done():
                    item.future.set_result(response)
            finally:
                self._queue.task_done()

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        connection_task = asyncio.current_task()
        if connection_task is not None:
            self._connection_tasks.add(connection_task)
        if len(self._connections) >= self.maximum_connections:
            writer.close()
            with suppress(OSError):
                await writer.wait_closed()
            if connection_task is not None:
                self._connection_tasks.discard(connection_task)
            return
        transport_socket = writer.get_extra_info("socket")
        if transport_socket is not None:
            transport_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._connections.add(writer)
        self.metrics.connections_accepted += 1
        self.metrics.active_connections += 1
        frames_on_connection = 0
        connection_role: str | None = None
        try:
            while True:
                try:
                    message = await read_stage_message(
                        reader,
                        read_timeout_s=(
                            self.idle_timeout_s if frames_on_connection else self.read_timeout_s
                        ),
                        maximum_metadata_bytes=self.maximum_metadata_bytes,
                        maximum_payload_bytes=self.maximum_payload_bytes,
                    )
                except asyncio.IncompleteReadError as exc:
                    if exc.partial:
                        self.metrics.malformed_frames += 1
                    break
                except TimeoutError:
                    break
                except ValueError as exc:
                    if "exceeds" in str(exc) or "oversized" in str(exc):
                        self.metrics.oversized_frames += 1
                    else:
                        self.metrics.malformed_frames += 1
                    break
                self.metrics.frames_received += 1
                peer_handshake = False
                authentication_required = (
                    self.require_peer_authentication()
                    if callable(self.require_peer_authentication)
                    else self.require_peer_authentication
                )
                if authentication_required:
                    if connection_role is None:
                        if message.source_stage == -1:
                            connection_role = "coordinator"
                        elif message.operation == Operation.HELLO and isinstance(
                            message.attributes.get("peer_handshake"), dict
                        ):
                            peer_handshake = True
                        else:
                            self.metrics.malformed_frames += 1
                            response = _error_response(
                                message,
                                "authenticated peer handshake required before stage frames",
                            )
                            await write_stage_message(
                                writer,
                                response,
                                write_timeout_s=self.write_timeout_s,
                            )
                            self.metrics.frames_sent += 1
                            break
                    elif (connection_role == "coordinator" and message.source_stage != -1) or (
                        connection_role == "peer" and message.source_stage < 0
                    ):
                        self.metrics.malformed_frames += 1
                        break
                if frames_on_connection:
                    self.metrics.reused_frames += 1
                frames_on_connection += 1
                loop = asyncio.get_running_loop()
                future: asyncio.Future[StageMessage] = loop.create_future()
                try:
                    self._queue.put_nowait(_InboundFrame(message=message, future=future))
                except asyncio.QueueFull:
                    self.metrics.backpressure_events += 1
                    response = _error_response(message, "stage-ring inbound queue is full")
                else:
                    response = await future
                await write_stage_message(
                    writer,
                    response,
                    write_timeout_s=self.write_timeout_s,
                )
                self.metrics.frames_sent += 1
                if peer_handshake:
                    if response.operation != Operation.HELLO or response.status != "OK":
                        break
                    connection_role = "peer"
        except (ConnectionError, OSError, TimeoutError):
            pass
        finally:
            self._connections.discard(writer)
            self.metrics.active_connections = max(0, self.metrics.active_connections - 1)
            writer.close()
            with suppress(OSError, TimeoutError):
                await asyncio.wait_for(writer.wait_closed(), timeout=self.write_timeout_s)
            if connection_task is not None:
                self._connection_tasks.discard(connection_task)

    async def _stop_workers(self) -> None:
        for task in self._workers:
            task.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def stop(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            server.close()
            await server.wait_closed()
        for writer in list(self._connections):
            writer.close()
        connection_tasks = list(self._connection_tasks)
        for task in connection_tasks:
            task.cancel()
        await asyncio.gather(*connection_tasks, return_exceptions=True)
        self._connections.clear()
        self._connection_tasks.clear()
        self.metrics.active_connections = 0
        await self._stop_workers()


__all__ = ["StageMessageHandler", "StageRingServer", "StageRingServerMetrics"]
