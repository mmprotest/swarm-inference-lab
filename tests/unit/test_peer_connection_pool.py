from __future__ import annotations

import asyncio
import time

import grpc
import pytest

from swarm_inference.config.models import OperationKind
from swarm_inference.exceptions import TransportError
from swarm_inference.protocol.checksums import sha256_bytes
from swarm_inference.protocol.messages import (
    ActivationMetadata,
    DataPlaneAck,
    DataPlaneEnvelope,
    parse_message,
    serialize_message,
)
from swarm_inference.transport.peer import PeerConnectionPool


def _envelope(message_id: str) -> DataPlaneEnvelope:
    payload = message_id.encode()
    return DataPlaneEnvelope(
        message_id=message_id,
        route_id="route",
        route_generation=1,
        request_id="request",
        stage_id=1,
        source_worker="worker-0",
        destination_worker="worker-1",
        token_position=0,
        operation=OperationKind.PREFILL,
        tensor_metadata=ActivationMetadata(
            request_id="request",
            tensor_id=message_id,
            stage_id=1,
            operation=OperationKind.PREFILL,
            token_position=0,
            sequence_length=1,
            cache_generation=0,
            model_id="synthetic",
            model_revision="v1",
        ),
        tensor_payload=payload,
        payload_length=len(payload),
        payload_checksum=sha256_bytes(payload),
        sequence_number=0,
        timestamp_unix_ns=time.time_ns(),
        signature="test",
    )


async def _start_peer_server() -> tuple[grpc.aio.Server, str]:
    seen: set[str] = set()

    async def peer_stream(iterator, _context):
        async for raw in iterator:
            envelope = parse_message(raw, DataPlaneEnvelope)
            duplicate = envelope.message_id in seen
            seen.add(envelope.message_id)
            yield serialize_message(
                DataPlaneAck(
                    message_id=envelope.message_id,
                    status="duplicate" if duplicate else "accepted",
                    detail="test",
                    accepted_timestamp_unix_ns=time.time_ns(),
                )
            )

    server = grpc.aio.server()
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                "swarm.v1.Worker",
                {
                    "PeerStream": grpc.stream_stream_rpc_method_handler(
                        peer_stream,
                        request_deserializer=lambda value: value,
                        response_serializer=lambda value: value,
                    )
                },
            ),
        )
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    return server, f"127.0.0.1:{port}"


@pytest.mark.asyncio
async def test_requests_multiplex_reconnect_and_shutdown_cleanly() -> None:
    server, endpoint = await _start_peer_server()
    pool = PeerConnectionPool(
        queue_capacity=8,
        timeout_s=2,
        reconnect_initial_backoff_ms=1,
        reconnect_max_backoff_ms=5,
    )
    try:
        acknowledgements = await asyncio.gather(
            *(pool.send(endpoint, _envelope(f"message-{index}")) for index in range(20))
        )
        assert all(ack.status == "accepted" for ack in acknowledgements)
        snapshot = pool.snapshot()
        assert snapshot["channels_created"] == 1
        assert snapshot["streams_created"] == 1
        assert snapshot["messages_sent"] == 20

        duplicate = await pool.send(endpoint, _envelope("message-0"))
        assert duplicate.status == "duplicate"
        assert pool.snapshot()["duplicate_acks"] == 1

        await pool.force_disconnect(endpoint)
        await asyncio.sleep(0.02)
        reconnected = await pool.send(endpoint, _envelope("after-reconnect"))
        assert reconnected.accepted
        snapshot = pool.snapshot()
        assert snapshot["streams_created"] >= 2
        assert snapshot["stream_reconnects"] >= 1
    finally:
        await asyncio.wait_for(pool.close(), timeout=2)
        await server.stop(0)


@pytest.mark.asyncio
async def test_outbound_queue_enforces_backpressure(monkeypatch) -> None:
    pool = PeerConnectionPool(
        queue_capacity=1,
        timeout_s=0.02,
        reconnect_attempts=1,
    )
    connection = pool._connection("127.0.0.1:1")
    monkeypatch.setattr(connection, "start", lambda: None)
    first = asyncio.create_task(pool.send("127.0.0.1:1", _envelope("first")))
    await asyncio.sleep(0)
    second = await pool.send("127.0.0.1:1", _envelope("second"))
    assert second.status == "backpressured"
    assert pool.snapshot()["backpressure_events"] >= 1
    with pytest.raises(TransportError):
        await first
    await pool.close()
