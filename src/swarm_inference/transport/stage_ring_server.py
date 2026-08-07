"""Bounded persistent TCP server for canonical stage-ring messages."""

from __future__ import annotations

import asyncio
import logging
import socket
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import asdict, dataclass, field

from swarm_inference.exceptions import TransportError
from swarm_inference.host import format_endpoint, split_endpoint
from swarm_inference.protocol.stage_ring import (
    MAX_METADATA_BYTES,
    MAX_PAYLOAD_BYTES,
    Operation,
    StageMessage,
)
from swarm_inference.security.tls import TlsServerConfig, require_tls_for_endpoint
from swarm_inference.transport.stage_ring_connection import (
    read_stage_message,
    write_stage_message,
)

StageMessageHandler = Callable[[StageMessage], Awaitable[StageMessage]]

LOGGER = logging.getLogger(__name__)


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
    connection_errors: int = 0
    shutdown_timeouts: int = 0

    def snapshot(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class _InboundFrame:
    message: StageMessage
    future: asyncio.Future[StageMessage]


@dataclass(slots=True, eq=False)
class _ActiveConnection:
    """Lifecycle record for one accepted stage-ring socket."""

    identity: int
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    task: asyncio.Task[None] | None
    topology_ids: set[str] = field(default_factory=set)
    route_generations: set[int] = field(default_factory=set)
    peer_identity: str | None = None
    session_ids: set[str] = field(default_factory=set)
    state: str = "accepted"
    close_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


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
        shutdown_timeout_s: float = 5.0,
        require_peer_authentication: bool | Callable[[], bool] = False,
        tls: TlsServerConfig | None = None,
        allow_plaintext_loopback: bool = True,
    ) -> None:
        if min(queue_capacity, dispatch_workers, maximum_connections) <= 0:
            raise ValueError("stage-ring server bounds must be positive")
        if min(read_timeout_s, write_timeout_s, idle_timeout_s, shutdown_timeout_s) <= 0:
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
        self.shutdown_timeout_s = shutdown_timeout_s
        self.require_peer_authentication = require_peer_authentication
        self.tls = tls
        self.allow_plaintext_loopback = allow_plaintext_loopback
        self.metrics = StageRingServerMetrics()
        self._queue: asyncio.Queue[_InboundFrame] = asyncio.Queue(maxsize=queue_capacity)
        self._dispatch_tasks: set[asyncio.Task[None]] = set()
        self._connections: dict[asyncio.StreamWriter, _ActiveConnection] = {}
        self._connection_tasks: set[asyncio.Task[None]] = set()
        self._server: asyncio.AbstractServer | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._stopping = False
        self._stopped = True
        self._next_connection_identity = 0
        self.bound_endpoint: str | None = None

    async def start(self, endpoint: str) -> int:
        async with self._lifecycle_lock:
            if self._server is not None or not self._stopped:
                raise RuntimeError("stage-ring server is already started")
            host, port = split_endpoint(endpoint)
            require_tls_for_endpoint(
                endpoint,
                tls_configured=self.tls is not None,
                allow_plaintext_loopback=self.allow_plaintext_loopback,
                transport_name="stage-ring server",
            )
            self._stopping = False
            self._stopped = False
            self.bound_endpoint = None
            self._dispatch_tasks = {
                asyncio.create_task(self._dispatch(), name=f"stage-ring-dispatch:{index}")
                for index in range(self.dispatch_workers)
            }
            try:
                server = await asyncio.start_server(
                    self._handle_connection,
                    host,
                    port,
                    ssl=self.tls.ssl_context() if self.tls is not None else None,
                    ssl_handshake_timeout=(self.read_timeout_s if self.tls is not None else None),
                )
                self._server = server
                server_sockets = server.sockets
                if not server_sockets:
                    raise TransportError(f"could not bind stage-ring endpoint {endpoint}")
            except BaseException:
                self._stopping = True
                partial_server = self._server
                self._server = None
                if partial_server is not None:
                    partial_server.close()
                await self._cancel_tasks(self._dispatch_tasks, owner="dispatch")
                self._drain_queue()
                if partial_server is not None:
                    with suppress(TimeoutError):
                        await asyncio.wait_for(
                            partial_server.wait_closed(),
                            timeout=self.shutdown_timeout_s,
                        )
                self._stopped = True
                raise
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
        if self._stopping or self._stopped or len(self._connections) >= self.maximum_connections:
            writer.close()
            with suppress(OSError, TimeoutError):
                await asyncio.wait_for(writer.wait_closed(), timeout=self.shutdown_timeout_s)
            return
        self._next_connection_identity += 1
        connection = _ActiveConnection(
            identity=self._next_connection_identity,
            reader=reader,
            writer=writer,
            task=connection_task,
        )
        if self.tls is not None:
            try:
                tls_object = writer.get_extra_info("ssl_object")
                peer_der = tls_object.getpeercert(binary_form=True) if tls_object else None
                connection.peer_identity = self.tls.validate_peer_der(peer_der)
            except TransportError:
                self.metrics.connection_errors += 1
                writer.close()
                with suppress(OSError, TimeoutError):
                    await asyncio.wait_for(writer.wait_closed(), timeout=self.shutdown_timeout_s)
                return
        if connection_task is not None:
            self._connection_tasks.add(connection_task)
        transport_socket = writer.get_extra_info("socket")
        if transport_socket is not None:
            transport_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._connections[writer] = connection
        self.metrics.connections_accepted += 1
        self.metrics.active_connections = len(self._connections)
        frames_on_connection = 0
        connection_role: str | None = None
        try:
            connection.state = "reading"
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
                connection.topology_ids.add(message.topology_id)
                connection.session_ids.add(message.session_id)
                route_generation = message.attributes.get("route_generation")
                if isinstance(route_generation, int):
                    connection.route_generations.add(route_generation)
                declared_peer = message.attributes.get("worker_id") or message.attributes.get(
                    "peer_id"
                )
                if isinstance(declared_peer, str) and declared_peer:
                    connection.peer_identity = declared_peer
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
                connection.state = "writing"
                await write_stage_message(
                    writer,
                    response,
                    write_timeout_s=self.write_timeout_s,
                )
                self.metrics.frames_sent += 1
                connection.state = "reading"
                if peer_handshake:
                    if response.operation != Operation.HELLO or response.status != "OK":
                        break
                    connection_role = "peer"
        except (ConnectionError, OSError, TimeoutError):
            pass
        except asyncio.CancelledError:
            raise
        except Exception:
            self.metrics.connection_errors += 1
            LOGGER.exception(
                "unexpected stage-ring connection failure",
                extra={"connection_identity": connection.identity},
            )
        finally:
            connection.state = "closing"
            self._connections.pop(writer, None)
            self.metrics.active_connections = len(self._connections)
            await self._close_connection(connection)
            connection.state = "closed"
            if connection_task is not None:
                self._connection_tasks.discard(connection_task)

    async def _close_connection(self, connection: _ActiveConnection) -> None:
        async with connection.close_lock:
            if connection.state == "closed":
                return
            connection.state = "closing"
            writer = connection.writer
            if not writer.is_closing():
                writer.close()
            try:
                await asyncio.wait_for(
                    writer.wait_closed(),
                    timeout=self.shutdown_timeout_s,
                )
            except TimeoutError:
                self.metrics.shutdown_timeouts += 1
                LOGGER.error(
                    "stage-ring writer closure timed out",
                    extra={
                        "connection_identity": connection.identity,
                        "timeout_s": self.shutdown_timeout_s,
                    },
                )
            except (ConnectionError, OSError):
                pass
            finally:
                connection.state = "closed"

    async def _close_active_connections(self) -> None:
        connections = tuple(self._connections.values())
        if connections:
            results = await asyncio.gather(
                *(self._close_connection(connection) for connection in connections),
                return_exceptions=True,
            )
            for connection, result in zip(connections, results, strict=True):
                if isinstance(result, BaseException) and not isinstance(
                    result, asyncio.CancelledError
                ):
                    self.metrics.connection_errors += 1
                    LOGGER.error(
                        "stage-ring connection cleanup failed",
                        exc_info=(type(result), result, result.__traceback__),
                        extra={"connection_identity": connection.identity},
                    )

    async def _cancel_tasks(
        self,
        tasks: set[asyncio.Task[None]],
        *,
        owner: str,
    ) -> None:
        current = asyncio.current_task()
        snapshot = [task for task in tuple(tasks) if task is not current]
        active = [task for task in snapshot if not task.done()]
        for task in active:
            task.cancel()
        if active:
            _done, pending = await asyncio.wait(active, timeout=self.shutdown_timeout_s)
            if pending:
                self.metrics.shutdown_timeouts += len(pending)
                LOGGER.error(
                    "stage-ring managed tasks did not stop before the shutdown deadline",
                    extra={
                        "owner": owner,
                        "pending_tasks": sorted(task.get_name() for task in pending),
                        "timeout_s": self.shutdown_timeout_s,
                    },
                )
                for task in pending:
                    task.cancel()
        completed = [task for task in snapshot if task.done()]
        if completed:
            results = await asyncio.gather(*completed, return_exceptions=True)
            for task, result in zip(completed, results, strict=True):
                if isinstance(result, BaseException) and not isinstance(
                    result, asyncio.CancelledError
                ):
                    LOGGER.error(
                        "stage-ring managed task failed",
                        exc_info=(type(result), result, result.__traceback__),
                        extra={"owner": owner, "task_name": task.get_name()},
                    )
        tasks.difference_update(task for task in tuple(tasks) if task.done())

    def _drain_queue(self) -> None:
        while not self._queue.empty():
            item = self._queue.get_nowait()
            if not item.future.done():
                item.future.cancel()
            self._queue.task_done()

    async def stop(self) -> None:
        async with self._lifecycle_lock:
            if self._stopped:
                return
            self._stopping = True
            server = self._server
            self._server = None
            self.bound_endpoint = None

            # Closing the listener prevents new accepts.  Active connections
            # remain explicitly owned below and are detached before the final
            # listener wait; this ordering is required by Python 3.12+.
            if server is not None:
                server.close()

            await self._close_active_connections()
            await self._cancel_tasks(self._connection_tasks, owner="connection")
            await self._cancel_tasks(self._dispatch_tasks, owner="dispatch")
            self._drain_queue()

            # A callback accepted just before server.close() may have entered
            # while the first snapshots were being cancelled.  Sweep once more
            # before waiting for listener closure.
            await self._close_active_connections()
            await self._cancel_tasks(self._connection_tasks, owner="connection")
            self._connections.clear()
            self.metrics.active_connections = 0

            if server is not None:
                try:
                    await asyncio.wait_for(
                        server.wait_closed(),
                        timeout=self.shutdown_timeout_s,
                    )
                except TimeoutError:
                    self.metrics.shutdown_timeouts += 1
                    LOGGER.error(
                        "stage-ring listener closure timed out",
                        extra={"timeout_s": self.shutdown_timeout_s},
                    )
            self._stopped = True


__all__ = ["StageMessageHandler", "StageRingServer", "StageRingServerMetrics"]
