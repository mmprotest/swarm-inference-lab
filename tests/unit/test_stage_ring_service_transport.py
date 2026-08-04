from __future__ import annotations

import asyncio
import socket

import pytest

from swarm_inference.exceptions import BackpressureError, TransportError
from swarm_inference.protocol.stage_ring import (
    HEADER,
    MAGIC,
    STAGE_RING_PROTOCOL_VERSION,
    Operation,
    StageMessage,
    encode_message,
)
from swarm_inference.transport.stage_ring_connection import (
    StageRingConnectionPool,
    read_stage_message,
)
from swarm_inference.transport.stage_ring_server import StageRingServer


def message(sequence: int, *, payload: bytes = b"payload") -> StageMessage:
    return StageMessage(
        operation=Operation.HEALTH,
        model_revision="model",
        tokenizer_revision="tokenizer",
        topology_id="topology",
        stage_id=0,
        layer_start=0,
        layer_end=1,
        session_id="session",
        request_id=f"request-{sequence}",
        sequence_number=sequence,
        token_position=-1,
        source_stage=-1,
        destination_stage=0,
        payload=payload,
    )


async def echo(value: StageMessage) -> StageMessage:
    return StageMessage(
        operation=value.operation,
        model_revision=value.model_revision,
        tokenizer_revision=value.tokenizer_revision,
        topology_id=value.topology_id,
        stage_id=value.stage_id,
        layer_start=value.layer_start,
        layer_end=value.layer_end,
        session_id=value.session_id,
        request_id=value.request_id,
        sequence_number=value.sequence_number,
        token_position=value.token_position,
        source_stage=value.destination_stage,
        destination_stage=value.source_stage,
        payload=value.payload,
    )


@pytest.mark.asyncio
async def test_tcp_connection_is_persistent_reused_and_nodelay() -> None:
    server = StageRingServer(handler=echo, maximum_payload_bytes=1024)
    port = await server.start("127.0.0.1:0")
    pool = StageRingConnectionPool(
        queue_capacity=4,
        read_timeout_s=2,
        write_timeout_s=2,
        maximum_payload_bytes=1024,
    )
    endpoint = f"127.0.0.1:{port}"
    try:
        responses = [await pool.send(endpoint, message(index)) for index in range(10)]
        assert [item.request_id for item in responses] == [
            f"request-{index}" for index in range(10)
        ]
        assert pool.snapshot()["connections_created"] == 1
        assert server.metrics.connections_accepted == 1
        assert server.metrics.reused_frames == 9
        connection = pool._connection(endpoint)
        assert connection._writer is not None
        transport_socket = connection._writer.get_extra_info("socket")
        assert transport_socket.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY) == 1
    finally:
        await pool.close()
        await server.stop()


@pytest.mark.asyncio
async def test_async_reader_handles_one_byte_partial_reads() -> None:
    encoded = encode_message(message(0))
    reader = asyncio.StreamReader()

    async def feed() -> None:
        for value in encoded.frame:
            reader.feed_data(bytes([value]))
            await asyncio.sleep(0)
        reader.feed_eof()

    feeder = asyncio.create_task(feed())
    decoded = await read_stage_message(
        reader,
        read_timeout_s=2,
        maximum_payload_bytes=1024,
    )
    await feeder
    assert decoded == message(0)


@pytest.mark.asyncio
async def test_server_rejects_oversized_and_corrupt_frames_before_dispatch() -> None:
    server = StageRingServer(handler=echo, maximum_payload_bytes=32)
    port = await server.start("127.0.0.1:0")
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        oversized = HEADER.pack(
            MAGIC,
            STAGE_RING_PROTOCOL_VERSION,
            int(Operation.HEALTH),
            0,
            1,
            33,
            0,
            -1,
            -1,
            0,
            b"\0" * 32,
        )
        writer.write(oversized)
        await writer.drain()
        assert await asyncio.wait_for(reader.read(), timeout=2) == b""
        writer.close()
        await writer.wait_closed()
        assert server.metrics.oversized_frames == 1

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        corrupt = bytearray(encode_message(message(1)).frame)
        corrupt[-1] ^= 0xFF
        writer.write(corrupt)
        await writer.drain()
        assert await asyncio.wait_for(reader.read(), timeout=2) == b""
        writer.close()
        await writer.wait_closed()
        assert server.metrics.malformed_frames == 1
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_outbound_queue_applies_bounded_backpressure(monkeypatch) -> None:
    pool = StageRingConnectionPool(
        queue_capacity=1,
        read_timeout_s=0.05,
        write_timeout_s=0.01,
        reconnect_attempts=1,
    )
    connection = pool._connection("127.0.0.1:1")
    monkeypatch.setattr(connection, "start", lambda: None)
    first = asyncio.create_task(pool.send("127.0.0.1:1", message(0)))
    await asyncio.sleep(0)
    with pytest.raises(BackpressureError, match="queue"):
        await pool.send("127.0.0.1:1", message(1))
    await pool.close()
    with pytest.raises(TransportError, match="closed"):
        await first


@pytest.mark.asyncio
async def test_server_shutdown_cancels_inflight_connection_and_dispatch_tasks() -> None:
    entered = asyncio.Event()
    never_complete = asyncio.Event()

    async def blocked_handler(_: StageMessage) -> StageMessage:
        entered.set()
        await never_complete.wait()
        raise AssertionError("unreachable")

    server = StageRingServer(handler=blocked_handler)
    port = await server.start("127.0.0.1:0")
    pool = StageRingConnectionPool(read_timeout_s=2, write_timeout_s=2)
    pending = asyncio.create_task(pool.send(f"127.0.0.1:{port}", message(0)))
    await asyncio.wait_for(entered.wait(), timeout=1)
    await asyncio.wait_for(server.stop(), timeout=1)
    with pytest.raises(TransportError):
        await pending
    assert server.metrics.active_connections == 0
    assert not server._connection_tasks
    await pool.close()
