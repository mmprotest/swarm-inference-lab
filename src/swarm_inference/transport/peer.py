"""Persistent multiplexed worker-to-worker gRPC data streams."""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from typing import Any

import grpc

from swarm_inference.exceptions import TransportError
from swarm_inference.protocol.messages import (
    Ack,
    DataPlaneAck,
    DataPlaneEnvelope,
    FinalResultMessage,
    parse_message,
    serialize_message,
)


@dataclass(slots=True)
class PeerConnectionMetrics:
    channels_created: int = 0
    streams_created: int = 0
    stream_reconnects: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    payload_bytes: int = 0
    control_bytes: int = 0
    queue_wait_ms: float = 0.0
    serialisation_time_ms: float = 0.0
    deserialisation_time_ms: float = 0.0
    network_transfer_time_ms: float = 0.0
    end_to_end_hop_time_ms: float = 0.0
    backpressure_events: int = 0
    duplicate_acks: int = 0

    def snapshot(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(slots=True)
class _OutboundItem:
    envelope: DataPlaneEnvelope
    enqueued_ns: int
    future: asyncio.Future[DataPlaneAck]


class _PeerConnection:
    def __init__(
        self,
        *,
        endpoint: str,
        queue_capacity: int,
        maximum_message_bytes: int,
        timeout_s: float,
        reconnect_attempts: int,
        reconnect_initial_backoff_ms: float,
        reconnect_max_backoff_ms: float,
        metrics: PeerConnectionMetrics,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_initial_backoff_ms = reconnect_initial_backoff_ms
        self.reconnect_max_backoff_ms = reconnect_max_backoff_ms
        self.metrics = metrics
        self._queue: asyncio.Queue[_OutboundItem] = asyncio.Queue(maxsize=queue_capacity)
        self._pending: dict[str, _OutboundItem] = {}
        self._channel = grpc.aio.insecure_channel(
            endpoint,
            options=[
                ("grpc.max_send_message_length", maximum_message_bytes),
                ("grpc.max_receive_message_length", maximum_message_bytes),
            ],
        )
        self.metrics.channels_created += 1
        self._runner: asyncio.Task[None] | None = None
        self._stream_call: Any | None = None
        self._closed = False
        self._started_once = False

    def start(self) -> None:
        if self._runner is None:
            self._runner = asyncio.create_task(
                self._run(),
                name=f"peer-stream:{self.endpoint}",
            )

    async def send(self, envelope: DataPlaneEnvelope) -> DataPlaneAck:
        if self._closed:
            raise TransportError(f"peer connection to {self.endpoint} is closed")
        self.start()
        last_error: BaseException | None = None
        for attempt in range(self.reconnect_attempts):
            loop = asyncio.get_running_loop()
            future: asyncio.Future[DataPlaneAck] = loop.create_future()
            item = _OutboundItem(
                envelope=envelope,
                enqueued_ns=time.perf_counter_ns(),
                future=future,
            )
            try:
                await asyncio.wait_for(self._queue.put(item), timeout=self.timeout_s)
            except TimeoutError:
                self.metrics.backpressure_events += 1
                return DataPlaneAck(
                    message_id=envelope.message_id,
                    status="backpressured",
                    detail=f"outbound queue to {self.endpoint} remained full",
                    accepted_timestamp_unix_ns=time.time_ns(),
                )
            try:
                ack = await asyncio.wait_for(future, timeout=self.timeout_s)
                if ack.status == "duplicate":
                    self.metrics.duplicate_acks += 1
                return ack
            except (TimeoutError, TransportError) as exc:
                last_error = exc
                if not future.done():
                    future.cancel()
                if attempt + 1 >= self.reconnect_attempts:
                    break
                backoff_ms = min(
                    self.reconnect_initial_backoff_ms * (2**attempt),
                    self.reconnect_max_backoff_ms,
                )
                await asyncio.sleep(backoff_ms / 1000)
        raise TransportError(
            f"peer send to {self.endpoint} failed after "
            f"{self.reconnect_attempts} attempts: {last_error}"
        )

    async def _run(self) -> None:
        reconnect_index = 0
        while not self._closed:

            async def requests() -> Any:
                while not self._closed:
                    item = await self._queue.get()
                    try:
                        if item.future.done():
                            continue
                        self.metrics.queue_wait_ms += (
                            time.perf_counter_ns() - item.enqueued_ns
                        ) / 1_000_000
                        serialisation_started = time.perf_counter_ns()
                        serialized = serialize_message(item.envelope)
                        self.metrics.serialisation_time_ms += (
                            time.perf_counter_ns() - serialisation_started
                        ) / 1_000_000
                        self.metrics.messages_sent += 1
                        self.metrics.payload_bytes += item.envelope.payload_length
                        self.metrics.control_bytes += max(
                            0, len(serialized) - item.envelope.payload_length
                        )
                        self._pending[item.envelope.message_id] = item
                        yield serialized
                    finally:
                        self._queue.task_done()

            call_factory = self._channel.stream_stream(
                "/swarm.v1.Worker/PeerStream",
                request_serializer=lambda value: value,
                response_deserializer=lambda value: value,
            )
            self._stream_call = call_factory(requests())
            self.metrics.streams_created += 1
            if self._started_once:
                self.metrics.stream_reconnects += 1
            self._started_once = True
            try:
                async for raw in self._stream_call:
                    deserialisation_started = time.perf_counter_ns()
                    ack = parse_message(raw, DataPlaneAck)
                    self.metrics.deserialisation_time_ms += (
                        time.perf_counter_ns() - deserialisation_started
                    ) / 1_000_000
                    self.metrics.control_bytes += len(raw)
                    item = self._pending.pop(ack.message_id, None)
                    if item is not None and not item.future.done():
                        elapsed_ms = (time.perf_counter_ns() - item.enqueued_ns) / 1_000_000
                        self.metrics.network_transfer_time_ms += elapsed_ms
                        self.metrics.end_to_end_hop_time_ms += elapsed_ms
                        item.future.set_result(ack)
                if not self._closed:
                    raise TransportError(f"peer stream to {self.endpoint} ended")
            except asyncio.CancelledError:
                if self._closed:
                    raise
                error = TransportError(f"peer stream to {self.endpoint} was interrupted")
                for item in list(self._pending.values()):
                    if not item.future.done():
                        item.future.set_exception(error)
                self._pending.clear()
                await asyncio.sleep(self.reconnect_initial_backoff_ms / 1000)
            except BaseException as exc:
                error = TransportError(f"peer stream to {self.endpoint} failed: {exc}")
                for item in list(self._pending.values()):
                    if not item.future.done():
                        item.future.set_exception(error)
                self._pending.clear()
                if self._closed:
                    break
                backoff_ms = min(
                    self.reconnect_initial_backoff_ms * (2**reconnect_index),
                    self.reconnect_max_backoff_ms,
                )
                reconnect_index = min(reconnect_index + 1, 30)
                await asyncio.sleep(backoff_ms / 1000)
            else:
                reconnect_index = 0
            finally:
                self._stream_call = None

    async def force_disconnect(self) -> None:
        if self._stream_call is not None:
            self._stream_call.cancel()
        await asyncio.sleep(0)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._stream_call is not None:
            self._stream_call.cancel()
        if self._runner is not None:
            self._runner.cancel()
            with suppress(asyncio.CancelledError):
                await self._runner
            self._runner = None
        error = TransportError(f"peer connection to {self.endpoint} closed")
        for item in list(self._pending.values()):
            if not item.future.done():
                item.future.set_exception(error)
        self._pending.clear()
        while not self._queue.empty():
            item = self._queue.get_nowait()
            if not item.future.done():
                item.future.set_exception(error)
            self._queue.task_done()
        await self._channel.close()


class PeerConnectionPool:
    """One bounded persistent multiplexed stream per ordered destination."""

    def __init__(
        self,
        *,
        queue_capacity: int = 1024,
        maximum_message_bytes: int = 4 * 1024 * 1024,
        timeout_s: float = 120.0,
        reconnect_attempts: int = 5,
        reconnect_initial_backoff_ms: float = 25.0,
        reconnect_max_backoff_ms: float = 1000.0,
    ) -> None:
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        self.queue_capacity = queue_capacity
        self.maximum_message_bytes = maximum_message_bytes
        self.timeout_s = timeout_s
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_initial_backoff_ms = reconnect_initial_backoff_ms
        self.reconnect_max_backoff_ms = reconnect_max_backoff_ms
        self.metrics = PeerConnectionMetrics()
        self._connections: dict[str, _PeerConnection] = {}
        self._closed = False

    def _connection(self, endpoint: str) -> _PeerConnection:
        if self._closed:
            raise TransportError("peer connection pool is closed")
        connection = self._connections.get(endpoint)
        if connection is None:
            connection = _PeerConnection(
                endpoint=endpoint,
                queue_capacity=self.queue_capacity,
                maximum_message_bytes=self.maximum_message_bytes,
                timeout_s=self.timeout_s,
                reconnect_attempts=self.reconnect_attempts,
                reconnect_initial_backoff_ms=self.reconnect_initial_backoff_ms,
                reconnect_max_backoff_ms=self.reconnect_max_backoff_ms,
                metrics=self.metrics,
            )
            self._connections[endpoint] = connection
        return connection

    async def send(self, endpoint: str, envelope: DataPlaneEnvelope) -> DataPlaneAck:
        return await self._connection(endpoint).send(envelope)

    def record_received(
        self,
        *,
        payload_bytes: int,
        control_bytes: int,
        deserialisation_ms: float,
        transfer_ms: float,
    ) -> None:
        self.metrics.messages_received += 1
        self.metrics.payload_bytes += max(0, payload_bytes)
        self.metrics.control_bytes += max(0, control_bytes)
        self.metrics.deserialisation_time_ms += max(0.0, deserialisation_ms)
        self.metrics.network_transfer_time_ms += max(0.0, transfer_ms)

    async def force_disconnect(self, endpoint: str) -> None:
        connection = self._connections.get(endpoint)
        if connection is not None:
            await connection.force_disconnect()

    def snapshot(self) -> dict[str, Any]:
        return {
            **self.metrics.snapshot(),
            "active_peer_pairs": len(self._connections),
            "peer_endpoints": sorted(self._connections),
            "outbound_queue_capacity": self.queue_capacity,
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


class FinalResultClient:
    """Persistent-channel client used only by final workers."""

    def __init__(
        self,
        endpoint: str,
        *,
        maximum_message_bytes: int = 4 * 1024 * 1024,
        timeout_s: float = 120.0,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.channel = grpc.aio.insecure_channel(
            endpoint,
            options=[
                ("grpc.max_send_message_length", maximum_message_bytes),
                ("grpc.max_receive_message_length", maximum_message_bytes),
            ],
        )

    async def send(self, message: FinalResultMessage) -> Ack:
        call = self.channel.unary_unary(
            "/swarm.v1.Coordinator/FinalResult",
            request_serializer=lambda value: value,
            response_deserializer=lambda value: value,
        )
        try:
            response = await call(serialize_message(message), timeout=self.timeout_s)
            return parse_message(response, Ack)
        except grpc.aio.AioRpcError as exc:
            raise TransportError(
                f"final-result send to {self.endpoint} failed ({exc.code().name}): {exc.details()}"
            ) from exc

    async def close(self) -> None:
        await self.channel.close()
