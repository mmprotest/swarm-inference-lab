from __future__ import annotations

import asyncio
import socket

import pytest

from swarm_inference.host import format_endpoint, split_endpoint
from swarm_inference.transport.tcp_meter import TcpMeteringProxy


async def _echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while payload := await reader.read(64 * 1024):
            writer.write(payload)
            await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()


async def _exchange(
    endpoint: str,
    payloads: tuple[bytes, ...],
    *,
    partial: bool = False,
) -> tuple[bytes, ...]:
    host, port = split_endpoint(endpoint)
    reader, writer = await asyncio.open_connection(host, port)
    results: list[bytes] = []
    try:
        for payload in payloads:
            if partial:
                chunk_size = 17 if len(payload) < 4096 else 4093
                for offset in range(0, len(payload), chunk_size):
                    writer.write(payload[offset : offset + chunk_size])
                    await writer.drain()
            else:
                writer.write(payload)
                await writer.drain()
            results.append(await reader.readexactly(len(payload)))
    finally:
        writer.close()
        await writer.wait_closed()
    return tuple(results)


@pytest.mark.asyncio
async def test_tcp_meter_exact_bytes_partial_writes_large_payload_and_reuse() -> None:
    server = await asyncio.start_server(_echo, "127.0.0.1", 0)
    upstream = format_endpoint("127.0.0.1", server.sockets[0].getsockname()[1])
    proxy = TcpMeteringProxy(
        listen_endpoint="127.0.0.1:0",
        upstream_endpoint=upstream,
        buffer_bytes=4096,
    )
    endpoint = await proxy.start()
    payloads = (b"partial-write" * 29, bytes(range(256)) * 4096)
    try:
        assert await _exchange(endpoint, payloads, partial=True) == payloads
        await asyncio.sleep(0)
        metrics = proxy.snapshot()
        expected = sum(map(len, payloads))
        assert metrics["bytes_sent"] == expected
        assert metrics["bytes_received"] == expected
        assert metrics["connection_count"] == 1
        assert int(metrics["transfer_count"]) > 2
        assert metrics["connection_failures"] == 0
    finally:
        await proxy.close()
        server.close()
        await server.wait_closed()
    assert proxy.snapshot()["active_connections"] == 0


@pytest.mark.asyncio
async def test_tcp_meter_multiple_workers_and_clean_shutdown() -> None:
    server = await asyncio.start_server(_echo, "127.0.0.1", 0)
    upstream = format_endpoint("127.0.0.1", server.sockets[0].getsockname()[1])
    proxy = TcpMeteringProxy(
        listen_endpoint="127.0.0.1:0",
        upstream_endpoint=upstream,
    )
    endpoint = await proxy.start()
    payloads = tuple(bytes([index]) * (1000 + index) for index in range(8))
    try:
        results = await asyncio.gather(*(_exchange(endpoint, (payload,)) for payload in payloads))
        assert tuple(item[0] for item in results) == payloads
        await asyncio.sleep(0)
        metrics = proxy.snapshot()
        assert metrics["connection_count"] == len(payloads)
        assert metrics["bytes_sent"] == sum(map(len, payloads))
        assert metrics["bytes_received"] == sum(map(len, payloads))
    finally:
        await proxy.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_tcp_meter_records_upstream_connection_failure() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as temporary:
        temporary.bind(("127.0.0.1", 0))
        unavailable_port = temporary.getsockname()[1]
    proxy = TcpMeteringProxy(
        listen_endpoint="127.0.0.1:0",
        upstream_endpoint=f"127.0.0.1:{unavailable_port}",
        connect_timeout_s=0.2,
    )
    endpoint = await proxy.start()
    host, port = split_endpoint(endpoint)
    try:
        reader, writer = await asyncio.open_connection(host, port)
        writer.write(b"unreachable")
        await writer.drain()
        assert await asyncio.wait_for(reader.read(), timeout=2) == b""
        writer.close()
        await writer.wait_closed()
        assert proxy.snapshot()["connection_failures"] == 1
    finally:
        await proxy.close()


@pytest.mark.asyncio
async def test_tcp_meter_ipv6_when_loopback_is_available() -> None:
    try:
        server = await asyncio.start_server(_echo, "::1", 0)
    except OSError:
        pytest.skip("IPv6 loopback is unavailable")
    upstream = format_endpoint("::1", server.sockets[0].getsockname()[1])
    proxy = TcpMeteringProxy(
        listen_endpoint="[::1]:0",
        upstream_endpoint=upstream,
    )
    try:
        endpoint = await proxy.start()
        payload = b"ipv6-transparent"
        assert await _exchange(endpoint, (payload,)) == (payload,)
    finally:
        await proxy.close()
        server.close()
        await server.wait_closed()
