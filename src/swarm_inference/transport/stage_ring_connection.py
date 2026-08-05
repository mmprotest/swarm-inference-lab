"""Persistent asynchronous TCP connections for product stage-ring frames."""

from __future__ import annotations

import asyncio
import socket
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass

from swarm_inference.exceptions import BackpressureError, IntegrityError, TransportError
from swarm_inference.host import is_wildcard_host, split_endpoint
from swarm_inference.protocol.stage_ring import (
    HEADER,
    MAX_METADATA_BYTES,
    MAX_PAYLOAD_BYTES,
    EncodedFrame,
    StageMessage,
    decode_message,
    encode_message,
    inspect_frame_header,
)

HandshakeFactory = Callable[[str], StageMessage | None]
HandshakeVerifier = Callable[[StageMessage, StageMessage], None]


async def read_stage_message(
    reader: asyncio.StreamReader,
    *,
    read_timeout_s: float,
    maximum_metadata_bytes: int = MAX_METADATA_BYTES,
    maximum_payload_bytes: int = MAX_PAYLOAD_BYTES,
) -> StageMessage:
    """Read one validated frame and reject oversized declarations up front."""

    header = await asyncio.wait_for(reader.readexactly(HEADER.size), timeout=read_timeout_s)
    inspected = inspect_frame_header(header)
    if inspected.metadata_length > maximum_metadata_bytes:
        raise ValueError("stage frame metadata exceeds the server limit")
    if inspected.payload_length > maximum_payload_bytes:
        raise ValueError("stage frame payload exceeds the server limit")
    body_length = inspected.metadata_length + inspected.payload_length
    body = await asyncio.wait_for(reader.readexactly(body_length), timeout=read_timeout_s)
    return decode_message(header + body)


async def write_encoded_frame(
    writer: asyncio.StreamWriter,
    encoded: EncodedFrame,
    *,
    write_timeout_s: float,
) -> None:
    """Write a complete frame through asyncio's partial-write-aware buffer."""

    writer.write(encoded.frame)
    await asyncio.wait_for(writer.drain(), timeout=write_timeout_s)


async def write_stage_message(
    writer: asyncio.StreamWriter,
    message: StageMessage,
    *,
    write_timeout_s: float,
) -> EncodedFrame:
    encoded = encode_message(message)
    await write_encoded_frame(writer, encoded, write_timeout_s=write_timeout_s)
    return encoded


@dataclass(slots=True)
class StageRingConnectionMetrics:
    connections_created: int = 0
    connection_reuses: int = 0
    reconnects: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    wire_bytes_sent: int = 0
    payload_bytes_sent: int = 0
    backpressure_events: int = 0
    failures: int = 0

    def snapshot(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class _OutboundFrame:
    message: StageMessage
    future: asyncio.Future[StageMessage]
    enqueued_ns: int


class _StageRingConnection:
    def __init__(
        self,
        *,
        endpoint: str,
        queue_capacity: int,
        connect_timeout_s: float,
        read_timeout_s: float,
        write_timeout_s: float,
        maximum_metadata_bytes: int,
        maximum_payload_bytes: int,
        metrics: StageRingConnectionMetrics,
        handshake_factory: HandshakeFactory | None,
        handshake_verifier: HandshakeVerifier | None,
    ) -> None:
        host, port = split_endpoint(endpoint)
        if is_wildcard_host(host) or port == 0:
            raise ValueError("outbound stage-ring endpoints must be concrete and non-zero")
        self.endpoint = endpoint
        self.host = host
        self.port = port
        self.connect_timeout_s = connect_timeout_s
        self.read_timeout_s = read_timeout_s
        self.write_timeout_s = write_timeout_s
        self.maximum_metadata_bytes = maximum_metadata_bytes
        self.maximum_payload_bytes = maximum_payload_bytes
        self.metrics = metrics
        self.handshake_factory = handshake_factory
        self.handshake_verifier = handshake_verifier
        self._queue: asyncio.Queue[_OutboundFrame] = asyncio.Queue(maxsize=queue_capacity)
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._runner: asyncio.Task[None] | None = None
        self._closed = False

    def start(self) -> None:
        if self._runner is None:
            self._runner = asyncio.create_task(self._run(), name=f"stage-ring:{self.endpoint}")

    async def enqueue(self, item: _OutboundFrame, *, timeout_s: float) -> None:
        if self._closed:
            raise TransportError(f"stage-ring connection to {self.endpoint} is closed")
        self.start()
        try:
            await asyncio.wait_for(self._queue.put(item), timeout=timeout_s)
        except TimeoutError as exc:
            self.metrics.backpressure_events += 1
            raise BackpressureError(
                f"stage-ring outbound queue to {self.endpoint} remained full"
            ) from exc

    async def _connect(self) -> None:
        if self._writer is not None and not self._writer.is_closing():
            self.metrics.connection_reuses += 1
            return
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(self.host, self.port), timeout=self.connect_timeout_s
        )
        transport_socket = writer.get_extra_info("socket")
        if transport_socket is not None:
            transport_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if self.metrics.connections_created:
            self.metrics.reconnects += 1
        self.metrics.connections_created += 1
        self._reader = reader
        self._writer = writer
        if self.handshake_factory is not None:
            request = self.handshake_factory(self.endpoint)
            if request is not None:
                encoded = await write_stage_message(
                    writer,
                    request,
                    write_timeout_s=self.write_timeout_s,
                )
                response = await read_stage_message(
                    reader,
                    read_timeout_s=self.read_timeout_s,
                    maximum_metadata_bytes=self.maximum_metadata_bytes,
                    maximum_payload_bytes=self.maximum_payload_bytes,
                )
                if self.handshake_verifier is None:
                    raise TransportError("stage-ring peer handshake verifier is missing")
                self.handshake_verifier(request, response)
                self.metrics.messages_sent += 1
                self.metrics.messages_received += 1
                self.metrics.wire_bytes_sent += encoded.wire_bytes
                self.metrics.payload_bytes_sent += encoded.payload_bytes

    async def _disconnect(self) -> None:
        writer = self._writer
        self._reader = None
        self._writer = None
        if writer is not None:
            writer.close()
            with suppress(OSError, TimeoutError):
                await asyncio.wait_for(writer.wait_closed(), timeout=self.write_timeout_s)

    async def _run(self) -> None:
        while not self._closed:
            item = await self._queue.get()
            try:
                if item.future.done():
                    continue
                await self._connect()
                assert self._reader is not None and self._writer is not None
                encoded = await write_stage_message(
                    self._writer,
                    item.message,
                    write_timeout_s=self.write_timeout_s,
                )
                self.metrics.messages_sent += 1
                self.metrics.wire_bytes_sent += encoded.wire_bytes
                self.metrics.payload_bytes_sent += encoded.payload_bytes
                response = await read_stage_message(
                    self._reader,
                    read_timeout_s=self.read_timeout_s,
                    maximum_metadata_bytes=self.maximum_metadata_bytes,
                    maximum_payload_bytes=self.maximum_payload_bytes,
                )
                self.metrics.messages_received += 1
                if not item.future.done():
                    item.future.set_result(response)
            except asyncio.CancelledError:
                if not item.future.done():
                    item.future.cancel()
                raise
            except (
                ConnectionError,
                OSError,
                TimeoutError,
                ValueError,
                IntegrityError,
                asyncio.IncompleteReadError,
            ) as exc:
                self.metrics.failures += 1
                await self._disconnect()
                if not item.future.done():
                    item.future.set_exception(
                        TransportError(f"stage-ring exchange with {self.endpoint} failed: {exc}")
                    )
            finally:
                self._queue.task_done()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._runner is not None:
            self._runner.cancel()
            with suppress(asyncio.CancelledError):
                await self._runner
            self._runner = None
        error = TransportError(f"stage-ring connection to {self.endpoint} closed")
        while not self._queue.empty():
            item = self._queue.get_nowait()
            if not item.future.done():
                item.future.set_exception(error)
            self._queue.task_done()
        await self._disconnect()


class StageRingConnectionPool:
    """Reuse one ordered TCP connection and bounded queue per next-stage peer."""

    def __init__(
        self,
        *,
        queue_capacity: int = 256,
        connect_timeout_s: float = 10.0,
        read_timeout_s: float = 120.0,
        write_timeout_s: float = 30.0,
        maximum_metadata_bytes: int = MAX_METADATA_BYTES,
        maximum_payload_bytes: int = MAX_PAYLOAD_BYTES,
        reconnect_attempts: int = 2,
        handshake_factory: HandshakeFactory | None = None,
        handshake_verifier: HandshakeVerifier | None = None,
    ) -> None:
        if queue_capacity <= 0 or reconnect_attempts <= 0:
            raise ValueError("stage-ring queue capacity and reconnect attempts must be positive")
        if min(connect_timeout_s, read_timeout_s, write_timeout_s) <= 0:
            raise ValueError("stage-ring timeouts must be positive")
        if not 0 < maximum_metadata_bytes <= MAX_METADATA_BYTES:
            raise ValueError("invalid stage-ring metadata limit")
        if not 0 < maximum_payload_bytes <= MAX_PAYLOAD_BYTES:
            raise ValueError("invalid stage-ring payload limit")
        self.queue_capacity = queue_capacity
        self.connect_timeout_s = connect_timeout_s
        self.read_timeout_s = read_timeout_s
        self.write_timeout_s = write_timeout_s
        self.maximum_metadata_bytes = maximum_metadata_bytes
        self.maximum_payload_bytes = maximum_payload_bytes
        self.reconnect_attempts = reconnect_attempts
        if (handshake_factory is None) != (handshake_verifier is None):
            raise ValueError(
                "stage-ring handshake factory and verifier must be configured together"
            )
        self.handshake_factory = handshake_factory
        self.handshake_verifier = handshake_verifier
        self.metrics = StageRingConnectionMetrics()
        self._connections: dict[str, _StageRingConnection] = {}
        self._closed = False

    def _connection(self, endpoint: str) -> _StageRingConnection:
        if self._closed:
            raise TransportError("stage-ring connection pool is closed")
        connection = self._connections.get(endpoint)
        if connection is None:
            connection = _StageRingConnection(
                endpoint=endpoint,
                queue_capacity=self.queue_capacity,
                connect_timeout_s=self.connect_timeout_s,
                read_timeout_s=self.read_timeout_s,
                write_timeout_s=self.write_timeout_s,
                maximum_metadata_bytes=self.maximum_metadata_bytes,
                maximum_payload_bytes=self.maximum_payload_bytes,
                metrics=self.metrics,
                handshake_factory=self.handshake_factory,
                handshake_verifier=self.handshake_verifier,
            )
            self._connections[endpoint] = connection
        return connection

    async def send(self, endpoint: str, message: StageMessage) -> StageMessage:
        last_error: TransportError | None = None
        for _ in range(self.reconnect_attempts):
            loop = asyncio.get_running_loop()
            future: asyncio.Future[StageMessage] = loop.create_future()
            await self._connection(endpoint).enqueue(
                _OutboundFrame(
                    message=message,
                    future=future,
                    enqueued_ns=time.monotonic_ns(),
                ),
                timeout_s=self.write_timeout_s,
            )
            try:
                return await asyncio.wait_for(future, timeout=self.read_timeout_s)
            except TransportError as exc:
                last_error = exc
            except TimeoutError:
                future.cancel()
                last_error = TransportError(
                    f"stage-ring exchange with {endpoint} exceeded the response timeout"
                )
                await self.remove(endpoint)
        assert last_error is not None
        raise last_error

    async def remove(self, endpoint: str) -> None:
        connection = self._connections.pop(endpoint, None)
        if connection is not None:
            await connection.close()

    def snapshot(self) -> dict[str, object]:
        return {
            **self.metrics.snapshot(),
            "active_connections": len(self._connections),
            "endpoints": sorted(self._connections),
            "queue_capacity": self.queue_capacity,
        }

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.gather(
            *(connection.close() for connection in self._connections.values()),
            return_exceptions=True,
        )
        self._connections.clear()


__all__ = [
    "HandshakeFactory",
    "HandshakeVerifier",
    "StageRingConnectionMetrics",
    "StageRingConnectionPool",
    "read_stage_message",
    "write_encoded_frame",
    "write_stage_message",
]
