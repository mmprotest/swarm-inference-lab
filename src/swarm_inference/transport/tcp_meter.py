"""Byte-transparent bounded TCP proxy with optional TLS on either side."""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from dataclasses import asdict, dataclass

from swarm_inference.exceptions import TransportError
from swarm_inference.host import format_endpoint, split_endpoint
from swarm_inference.security.tls import (
    TlsClientConfig,
    TlsServerConfig,
    require_tls_for_endpoint,
)


@dataclass(slots=True)
class TcpMeterMetrics:
    bytes_sent: int = 0
    bytes_received: int = 0
    connection_count: int = 0
    active_connections: int = 0
    transfer_count: int = 0
    connection_failures: int = 0
    started_monotonic_ns: int = 0

    def snapshot(self) -> dict[str, int | float | None]:
        duration = (
            (time.monotonic_ns() - self.started_monotonic_ns) / 1_000_000_000
            if self.started_monotonic_ns
            else None
        )
        return {**asdict(self), "runtime_duration_s": duration}


class TcpMeteringProxy:
    """Forward opaque bytes without decoding, mutation, or unbounded buffering.

    ``bytes_sent`` is client-to-upstream traffic and ``bytes_received`` is
    upstream-to-client traffic.  A transfer is one bounded socket read; it is
    deliberately not presented as a private-protocol message count.
    """

    def __init__(
        self,
        *,
        listen_endpoint: str,
        upstream_endpoint: str,
        inbound_tls: TlsServerConfig | None = None,
        outbound_tls: TlsClientConfig | None = None,
        buffer_bytes: int = 64 * 1024,
        connect_timeout_s: float = 10.0,
        shutdown_timeout_s: float = 5.0,
        allow_plaintext_loopback: bool = True,
    ) -> None:
        if not 1024 <= buffer_bytes <= 1024 * 1024:
            raise ValueError("TCP meter buffer must be between 1 KiB and 1 MiB")
        if connect_timeout_s <= 0 or shutdown_timeout_s <= 0:
            raise ValueError("TCP meter timeouts must be positive")
        self.listen_endpoint = listen_endpoint
        self.upstream_endpoint = upstream_endpoint
        self.inbound_tls = inbound_tls
        self.outbound_tls = outbound_tls
        self.buffer_bytes = buffer_bytes
        self.connect_timeout_s = connect_timeout_s
        self.shutdown_timeout_s = shutdown_timeout_s
        self.allow_plaintext_loopback = allow_plaintext_loopback
        self.metrics = TcpMeterMetrics()
        self.bound_endpoint: str | None = None
        self._server: asyncio.AbstractServer | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._writers: set[asyncio.StreamWriter] = set()
        self._stopping = False

    async def start(self) -> str:
        if self._server is not None:
            raise RuntimeError("TCP metering proxy is already started")
        listen_host, listen_port = split_endpoint(self.listen_endpoint)
        require_tls_for_endpoint(
            self.listen_endpoint,
            tls_configured=self.inbound_tls is not None,
            allow_plaintext_loopback=self.allow_plaintext_loopback,
            transport_name="TCP metering proxy listener",
        )
        require_tls_for_endpoint(
            self.upstream_endpoint,
            tls_configured=self.outbound_tls is not None,
            allow_plaintext_loopback=self.allow_plaintext_loopback,
            transport_name="TCP metering proxy upstream",
        )
        self._stopping = False
        self.metrics.started_monotonic_ns = time.monotonic_ns()
        self._server = await asyncio.start_server(
            self._accept,
            listen_host,
            listen_port,
            ssl=self.inbound_tls.ssl_context() if self.inbound_tls is not None else None,
        )
        sockets = self._server.sockets
        if not sockets:
            await self.close()
            raise TransportError(f"TCP metering proxy could not bind {self.listen_endpoint}")
        bound = sockets[0].getsockname()
        self.bound_endpoint = format_endpoint(listen_host, int(bound[1]))
        return self.bound_endpoint

    async def _pump(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        sent: bool,
    ) -> None:
        while payload := await reader.read(self.buffer_bytes):
            writer.write(payload)
            await writer.drain()
            if sent:
                self.metrics.bytes_sent += len(payload)
            else:
                self.metrics.bytes_received += len(payload)
            self.metrics.transfer_count += 1

    async def _accept(
        self,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)
        self._writers.add(client_writer)
        upstream_writer: asyncio.StreamWriter | None = None
        self.metrics.connection_count += 1
        self.metrics.active_connections += 1
        try:
            if self._stopping:
                return
            if self.inbound_tls is not None:
                tls_object = client_writer.get_extra_info("ssl_object")
                peer_der = tls_object.getpeercert(binary_form=True) if tls_object else None
                self.inbound_tls.validate_peer_der(peer_der)
            upstream_host, upstream_port = split_endpoint(self.upstream_endpoint)
            connected_reader, connected_writer = await asyncio.wait_for(
                asyncio.open_connection(
                    upstream_host,
                    upstream_port,
                    ssl=(
                        self.outbound_tls.ssl_context() if self.outbound_tls is not None else None
                    ),
                    server_hostname=(
                        self.outbound_tls.expected_server_name
                        if self.outbound_tls is not None
                        else None
                    ),
                ),
                timeout=self.connect_timeout_s,
            )
            upstream_reader = connected_reader
            upstream_writer = connected_writer
            self._writers.add(upstream_writer)
            if self.outbound_tls is not None:
                tls_object = upstream_writer.get_extra_info("ssl_object")
                peer_der = tls_object.getpeercert(binary_form=True) if tls_object else None
                self.outbound_tls.validate_peer_der(peer_der)
            forward = asyncio.create_task(
                self._pump(client_reader, upstream_writer, sent=True),
                name="tcp-meter-client-to-upstream",
            )
            reverse = asyncio.create_task(
                self._pump(upstream_reader, client_writer, sent=False),
                name="tcp-meter-upstream-to-client",
            )
            done, pending = await asyncio.wait(
                {forward, reverse}, return_when=asyncio.FIRST_COMPLETED
            )
            for pending_task in pending:
                pending_task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for completed in done:
                completed.result()
        except asyncio.CancelledError:
            raise
        except (OSError, TimeoutError, TransportError):
            self.metrics.connection_failures += 1
        finally:
            self.metrics.active_connections -= 1
            for writer in (client_writer, upstream_writer):
                if writer is None:
                    continue
                self._writers.discard(writer)
                writer.close()
                with suppress(OSError, TimeoutError):
                    await asyncio.wait_for(writer.wait_closed(), timeout=self.shutdown_timeout_s)
            if task is not None:
                self._tasks.discard(task)

    def snapshot(self) -> dict[str, int | float | None]:
        return self.metrics.snapshot()

    async def close(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        server = self._server
        self._server = None
        if server is not None:
            server.close()
            with suppress(TimeoutError):
                await asyncio.wait_for(server.wait_closed(), timeout=self.shutdown_timeout_s)
        for writer in tuple(self._writers):
            writer.close()
        for task in tuple(self._tasks):
            task.cancel()
        if self._tasks:
            with suppress(TimeoutError):
                await asyncio.wait_for(
                    asyncio.gather(*self._tasks, return_exceptions=True),
                    timeout=self.shutdown_timeout_s,
                )
        self._writers.clear()
        self._tasks.clear()
        self.metrics.active_connections = 0


__all__ = ["TcpMeterMetrics", "TcpMeteringProxy"]
