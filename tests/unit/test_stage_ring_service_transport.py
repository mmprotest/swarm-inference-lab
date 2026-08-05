from __future__ import annotations

import asyncio
import socket
from dataclasses import replace

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
from swarm_inference.transport.stage_ring_faults import CloseConnectionBeforeSendInjector
from swarm_inference.transport.stage_ring_server import StageRingServer


def message(
    sequence: int,
    *,
    payload: bytes = b"payload",
    operation: Operation = Operation.HEALTH,
    token_position: int = -1,
) -> StageMessage:
    return StageMessage(
        operation=operation,
        model_revision="model",
        tokenizer_revision="tokenizer",
        topology_id="topology",
        stage_id=0,
        layer_start=0,
        layer_end=1,
        session_id="session",
        request_id=f"request-{sequence}",
        sequence_number=sequence,
        token_position=token_position,
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
async def test_fault_injector_closes_the_matching_active_connection_and_evicts_it() -> None:
    server = StageRingServer(handler=echo, maximum_payload_bytes=1024)
    port = await server.start("127.0.0.1:0")
    endpoint = f"127.0.0.1:{port}"
    injector = CloseConnectionBeforeSendInjector(
        token_position=2,
        source_stage=0,
        destination_stage=1,
        request_id="request-2",
    )
    pool = StageRingConnectionPool(
        read_timeout_s=1,
        write_timeout_s=1,
        reconnect_attempts=1,
        fault_injector=injector,
    )
    try:
        # Establish and prove the connection before crossing the injected boundary.
        await pool.send(endpoint, message(0))
        assert pool.snapshot()["active_connections"] == 1
        injected = message(2, operation=Operation.DECODE, token_position=2)
        injected = replace(
            injected,
            source_stage=0,
            destination_stage=1,
            attributes={"route_generation": 7},
        )
        with pytest.raises(
            TransportError,
            match=r"route_generation=7.*request=request-2.*sequence=2",
        ):
            await pool.send(endpoint, injected)

        assert injector.triggered
        assert len(injector.events) == 1
        event = injector.events[0]
        assert event["event_type"] == "stage_ring_fault_injected"
        assert event["active_connection"] is True
        assert event["route_generation"] == 7
        assert event["token_position"] == 2
        snapshot = pool.snapshot()
        assert snapshot["fault_injections"] == 1
        assert snapshot["connection_evictions"] == 1
        assert snapshot["active_connections"] == 0

        # The listener and control-independent peer remain healthy, and the
        # one-shot injector does not poison the replacement connection.
        response = await pool.send(endpoint, message(3))
        assert response.request_id == "request-3"
        assert pool.snapshot()["connections_created"] == 2
    finally:
        await pool.close()
        await server.stop()


@pytest.mark.asyncio
async def test_listener_closure_is_not_counted_as_active_connection_injection() -> None:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                value = await read_stage_message(reader, read_timeout_s=1)
                response = await echo(value)
                writer.write(encode_message(response).frame)
                await writer.drain()
        except asyncio.IncompleteReadError:
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    listener = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = listener.sockets[0].getsockname()[1]
    endpoint = f"127.0.0.1:{port}"
    injector = CloseConnectionBeforeSendInjector(token_position=99)
    pool = StageRingConnectionPool(read_timeout_s=1, write_timeout_s=1, fault_injector=injector)
    try:
        assert (await pool.send(endpoint, message(0))).request_id == "request-0"
        listener.close()
        # Closing only the listener does not interrupt this established socket.
        assert (await pool.send(endpoint, message(1))).request_id == "request-1"
        assert not injector.triggered
        assert pool.snapshot()["fault_injections"] == 0
    finally:
        await pool.close()
        listener.close()
        await listener.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("abort", [False, True], ids=["eof", "reset"])
async def test_eof_and_reset_evict_failed_peer_connections(abort: bool) -> None:
    async def close_after_read(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await read_stage_message(reader, read_timeout_s=1)
        if abort:
            writer.transport.abort()
        else:
            writer.close()
            await writer.wait_closed()

    listener = await asyncio.start_server(close_after_read, "127.0.0.1", 0)
    port = listener.sockets[0].getsockname()[1]
    pool = StageRingConnectionPool(read_timeout_s=1, write_timeout_s=1, reconnect_attempts=1)
    try:
        with pytest.raises(TransportError, match="stage-ring exchange"):
            await pool.send(f"127.0.0.1:{port}", message(8))
        snapshot = pool.snapshot()
        assert snapshot["failures"] == 1
        assert snapshot["connection_evictions"] == 1
        assert snapshot["active_connections"] == 0
    finally:
        await pool.close()
        listener.close()
        await listener.wait_closed()


@pytest.mark.asyncio
async def test_pending_response_timeout_evicts_connection() -> None:
    release = asyncio.Event()

    async def never_respond(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await read_stage_message(reader, read_timeout_s=1)
            await release.wait()
        finally:
            writer.close()
            await writer.wait_closed()

    listener = await asyncio.start_server(never_respond, "127.0.0.1", 0)
    port = listener.sockets[0].getsockname()[1]
    pool = StageRingConnectionPool(
        read_timeout_s=0.05,
        write_timeout_s=1,
        reconnect_attempts=1,
    )
    try:
        with pytest.raises(
            TransportError,
            match=r"response timeout.*session=session.*request=request-9.*sequence=9",
        ):
            await pool.send(f"127.0.0.1:{port}", message(9))
        snapshot = pool.snapshot()
        assert snapshot["response_timeouts"] == 1
        assert snapshot["connection_evictions"] == 1
        assert snapshot["active_connections"] == 0
    finally:
        release.set()
        await pool.close()
        listener.close()
        await listener.wait_closed()


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
    assert not server._dispatch_tasks
    assert not server._connections
    await pool.close()


@pytest.mark.asyncio
async def test_server_stop_before_start_and_repeated_stop_are_safe() -> None:
    server = StageRingServer(handler=echo)

    await server.stop()
    await server.stop()

    assert server._stopped
    assert not server._connection_tasks
    assert not server._dispatch_tasks
    assert not server._connections


@pytest.mark.asyncio
async def test_concurrent_server_stop_calls_share_one_shutdown() -> None:
    server = StageRingServer(handler=echo)
    await server.start("127.0.0.1:0")

    await asyncio.gather(*(server.stop() for _ in range(8)))

    assert server._stopped
    assert server._server is None
    assert not server._connection_tasks
    assert not server._dispatch_tasks


@pytest.mark.asyncio
async def test_server_stop_closes_an_idle_accepted_connection() -> None:
    server = StageRingServer(handler=echo, shutdown_timeout_s=0.5)
    port = await server.start("127.0.0.1:0")
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    deadline = asyncio.get_running_loop().time() + 1
    while server.metrics.active_connections != 1:
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("server did not register the accepted connection")
        await asyncio.sleep(0)

    await server.stop()

    assert await asyncio.wait_for(reader.read(), timeout=1) == b""
    assert not server._connections
    assert not server._connection_tasks
    writer.close()
    await writer.wait_closed()


@pytest.mark.asyncio
async def test_partial_startup_failure_cleans_dispatch_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_start(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected bind failure")

    monkeypatch.setattr(asyncio, "start_server", fail_start)
    server = StageRingServer(handler=echo)

    with pytest.raises(OSError, match="injected bind failure"):
        await server.start("127.0.0.1:0")
    await server.stop()

    assert server._stopped
    assert server._server is None
    assert not server._dispatch_tasks
    assert not server._connection_tasks


@pytest.mark.asyncio
async def test_connection_task_exception_remains_visible_during_stop(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import swarm_inference.transport.stage_ring_server as server_module

    release = asyncio.Event()

    async def fail_read(*_args: object, **_kwargs: object) -> StageMessage:
        await release.wait()
        raise RuntimeError("injected connection task failure")

    monkeypatch.setattr(server_module, "read_stage_message", fail_read)
    server = StageRingServer(handler=echo, shutdown_timeout_s=0.5)
    port = await server.start("127.0.0.1:0")
    _reader, writer = await asyncio.open_connection("127.0.0.1", port)
    deadline = asyncio.get_running_loop().time() + 1
    while not server._connection_tasks:
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("server did not track the accepted connection task")
        await asyncio.sleep(0)

    release.set()
    await server.stop()

    assert server.metrics.connection_errors == 1
    assert "unexpected stage-ring connection failure" in caplog.text
    assert not server._connections
    assert not server._connection_tasks
    writer.close()


@pytest.mark.asyncio
async def test_writer_wait_closed_timeout_is_bounded_and_registry_is_cleared() -> None:
    import swarm_inference.transport.stage_ring_server as server_module

    class NeverClosingWriter:
        def __init__(self) -> None:
            self.closing = False
            self.close_count = 0

        def is_closing(self) -> bool:
            return self.closing

        def close(self) -> None:
            self.closing = True
            self.close_count += 1

        async def wait_closed(self) -> None:
            await asyncio.Event().wait()

    server = StageRingServer(handler=echo, shutdown_timeout_s=0.02)
    await server.start("127.0.0.1:0")
    writer = NeverClosingWriter()
    connection = server_module._ActiveConnection(
        identity=1,
        reader=asyncio.StreamReader(),
        writer=writer,
        task=None,
    )
    server._connections[writer] = connection  # type: ignore[index]
    server.metrics.active_connections = 1

    await asyncio.wait_for(server._close_connection(connection), timeout=1)
    await asyncio.wait_for(server.stop(), timeout=1)

    assert server.metrics.shutdown_timeouts >= 1
    assert writer.close_count == 1
    assert not server._connections
    assert not server._connection_tasks
    assert not server._dispatch_tasks


@pytest.mark.asyncio
async def test_stop_after_injected_reset_leaves_no_connection_or_managed_task() -> None:
    server = StageRingServer(handler=echo, maximum_payload_bytes=1024)
    port = await server.start("127.0.0.1:0")
    endpoint = f"127.0.0.1:{port}"
    injector = CloseConnectionBeforeSendInjector(token_position=4)
    pool = StageRingConnectionPool(
        read_timeout_s=1,
        write_timeout_s=1,
        reconnect_attempts=1,
        fault_injector=injector,
    )
    injected = replace(
        message(4, operation=Operation.DECODE, token_position=4),
        source_stage=0,
        destination_stage=1,
    )
    try:
        with pytest.raises(TransportError):
            await pool.send(endpoint, injected)
    finally:
        await pool.close()
        await server.stop()

    await asyncio.sleep(0)
    managed = {
        task.get_name()
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and task.get_name().startswith("stage-ring")
    }
    assert managed == set()
    assert not server._connections
    assert not server._connection_tasks
    assert not server._dispatch_tasks
