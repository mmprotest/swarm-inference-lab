"""Authenticated, bounded, directed peer-network measurement.

Probe tickets are signed by the pinned coordinator and authorize only one
source/destination worker pair, a finite payload set, and a short time window.
The direct probe socket is independent of deployed stage routes and never
carries inference data.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import statistics
import struct
import time
from collections import OrderedDict
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import TypeVar
from uuid import uuid4

from pydantic import ValidationError

from swarm_inference.cluster.models import (
    ClusterAuditEvent,
    ClusterMetadata,
    NetworkLinkMeasurement,
    NodeMetadata,
)
from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.config.models import StrictModel
from swarm_inference.exceptions import ConfigurationError, IntegrityError, TransportError
from swarm_inference.host import format_endpoint, split_endpoint
from swarm_inference.protocol.cluster import (
    DirectNetworkProbeAck,
    DirectNetworkProbeRequest,
    DirectNetworkProbeResponse,
    NetworkProbeControlRequest,
    NetworkProbeControlResponse,
    NetworkProbeTicket,
)
from swarm_inference.protocol.routes import BoundedNonceCache
from swarm_inference.security.identity import CoordinatorIdentity, WorkerIdentity
from swarm_inference.security.signatures import canonical_json_bytes, verify_signature
from swarm_inference.security.tls import (
    TlsClientConfig,
    TlsServerConfig,
    require_tls_for_endpoint,
)

NETWORK_PROBE_PROTOCOL_VERSION = 1
DEFAULT_NETWORK_MEASUREMENT_TTL_SECONDS = 900
DEFAULT_NETWORK_PROBE_MAX_BYTES = 16 * 1024 * 1024
DEFAULT_NETWORK_PROBE_TIMEOUT_SECONDS = 10.0
DEFAULT_NETWORK_PROBE_PAYLOAD_SIZES = (4096, 256 * 1024, 1024 * 1024)
_MAX_HEADER_BYTES = 64 * 1024
_HEADER = struct.Struct("!I")
_ModelT = TypeVar("_ModelT", bound=StrictModel)


def _signed_model_payload(value: StrictModel) -> bytes:
    return canonical_json_bytes(value.model_dump(mode="json", exclude={"signature"}))


def sign_probe_ticket(
    ticket: NetworkProbeTicket,
    identity: CoordinatorIdentity,
) -> NetworkProbeTicket:
    if ticket.coordinator_public_key != identity.public_key_b64:
        raise IntegrityError("network probe ticket coordinator key does not match signer")
    if ticket.coordinator_fingerprint != identity.public_key_fingerprint:
        raise IntegrityError("network probe ticket coordinator fingerprint does not match signer")
    return ticket.model_copy(update={"signature": identity.sign(_signed_model_payload(ticket))})


def verify_probe_ticket(
    ticket: NetworkProbeTicket,
    cluster: ClusterMetadata,
    *,
    now_unix_ns: int | None = None,
    future_tolerance_ns: int = 30_000_000_000,
) -> None:
    now = now_unix_ns or time.time_ns()
    if ticket.cluster_id != cluster.cluster_id:
        raise IntegrityError("network probe ticket belongs to a different cluster")
    if (
        ticket.coordinator_public_key != cluster.coordinator_public_key
        or ticket.coordinator_fingerprint != cluster.coordinator_fingerprint
    ):
        raise IntegrityError("network probe ticket does not match the pinned coordinator")
    if ticket.issued_at_unix_ns > now + future_tolerance_ns:
        raise IntegrityError("network probe ticket was issued too far in the future")
    if ticket.expires_at_unix_ns < now:
        raise IntegrityError("network probe ticket has expired")
    verify_signature(
        cluster.coordinator_public_key,
        _signed_model_payload(ticket),
        ticket.signature,
    )


def deterministic_probe_payload(seed: str, size: int) -> bytes:
    """Build reproducible bounded bytes without retaining a global random buffer."""

    if size <= 0 or size > DEFAULT_NETWORK_PROBE_MAX_BYTES:
        raise ValueError("network probe payload size is outside the bounded range")
    seed_bytes = seed.encode("utf-8")
    output = bytearray(size)
    offset = 0
    counter = 0
    while offset < size:
        block = hashlib.sha256(seed_bytes + counter.to_bytes(8, "big")).digest()
        copied = min(len(block), size - offset)
        output[offset : offset + copied] = block[:copied]
        offset += copied
        counter += 1
    return bytes(output)


async def _read_model(
    reader: asyncio.StreamReader,
    model: type[_ModelT],
    *,
    timeout_seconds: float,
) -> _ModelT:
    try:
        prefix = await asyncio.wait_for(
            reader.readexactly(_HEADER.size),
            timeout=timeout_seconds,
        )
        (size,) = _HEADER.unpack(prefix)
        if not 0 < size <= _MAX_HEADER_BYTES:
            raise IntegrityError("network probe metadata exceeds its bound")
        payload = await asyncio.wait_for(reader.readexactly(size), timeout=timeout_seconds)
        return model.model_validate_json(payload)
    except asyncio.IncompleteReadError as exc:
        raise TransportError("network probe peer closed an incomplete frame") from exc
    except ValidationError as exc:
        raise IntegrityError("network probe metadata failed strict validation") from exc


async def _write_model(
    writer: asyncio.StreamWriter,
    value: StrictModel,
    *,
    timeout_seconds: float,
) -> None:
    payload = value.model_dump_json().encode("utf-8")
    if len(payload) > _MAX_HEADER_BYTES:
        raise IntegrityError("network probe metadata exceeds its bound")
    writer.write(_HEADER.pack(len(payload)))
    writer.write(payload)
    await asyncio.wait_for(writer.drain(), timeout=timeout_seconds)


def _endpoint(value: object) -> str | None:
    if not isinstance(value, tuple) or len(value) < 2:
        return None
    return format_endpoint(str(value[0]), int(value[1]))


class DirectNetworkProbeServer:
    """Small authenticated TCP service owned by the persistent node agent."""

    def __init__(
        self,
        *,
        state: ClusterStateStore,
        cluster: ClusterMetadata,
        identity: WorkerIdentity,
        node_id: str,
        worker_id: str,
        maximum_bytes: int = DEFAULT_NETWORK_PROBE_MAX_BYTES,
        maximum_connections: int = 8,
        timeout_seconds: float = DEFAULT_NETWORK_PROBE_TIMEOUT_SECONDS,
        shutdown_timeout_seconds: float = 5.0,
        authentication_skew_seconds: float = 60.0,
        nonce_cache_capacity: int = 4096,
        clock_ns: Callable[[], int] = time.time_ns,
        tls_server: TlsServerConfig | None = None,
        allow_plaintext_loopback: bool = True,
    ) -> None:
        if not 1024 <= maximum_bytes <= 256 * 1024 * 1024:
            raise ValueError("network probe maximum bytes must be in [1 KiB, 256 MiB]")
        if not 1 <= maximum_connections <= 128:
            raise ValueError("network probe maximum connections must be in [1, 128]")
        if min(timeout_seconds, shutdown_timeout_seconds, authentication_skew_seconds) <= 0:
            raise ValueError("network probe timeouts must be positive")
        self.state = state
        self.cluster = cluster
        self.identity = identity
        self.node_id = node_id
        self.worker_id = worker_id
        self.maximum_bytes = maximum_bytes
        self.maximum_connections = maximum_connections
        self.timeout_seconds = timeout_seconds
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self.authentication_skew_ns = int(authentication_skew_seconds * 1_000_000_000)
        self.clock_ns = clock_ns
        self.tls_server = tls_server
        self.allow_plaintext_loopback = allow_plaintext_loopback
        self.nonce_cache = BoundedNonceCache(capacity=nonce_cache_capacity)
        self._server: asyncio.Server | None = None
        self._connections: set[asyncio.StreamWriter] = set()
        self._connection_tasks: set[asyncio.Task[None]] = set()
        self._lifecycle_lock = asyncio.Lock()
        self._closed = asyncio.Event()
        self.bound_endpoint: str | None = None

    async def start(self, endpoint: str) -> int:
        async with self._lifecycle_lock:
            if self._server is not None:
                existing_sockets = self._server.sockets
                if not existing_sockets:
                    raise TransportError("started network probe server has no bound socket")
                return int(existing_sockets[0].getsockname()[1])
            host, port = split_endpoint(endpoint)
            self._closed.clear()
            require_tls_for_endpoint(
                endpoint,
                tls_configured=self.tls_server is not None,
                allow_plaintext_loopback=self.allow_plaintext_loopback,
                transport_name="direct network probe",
            )
            server = await asyncio.start_server(
                self._handle_connection,
                host,
                port,
                ssl=self.tls_server.ssl_context() if self.tls_server is not None else None,
            )
            bound_sockets = server.sockets
            if not bound_sockets:
                server.close()
                await server.wait_closed()
                raise TransportError(f"could not bind network probe endpoint {endpoint}")
            self._server = server
            bound_port = int(bound_sockets[0].getsockname()[1])
            self.bound_endpoint = format_endpoint(host, bound_port)
            return bound_port

    async def wait(self) -> None:
        if self._server is None:
            return
        await self._closed.wait()

    async def stop(self) -> None:
        async with self._lifecycle_lock:
            server = self._server
            if server is None:
                self._closed.set()
                return
            self._server = None
            server.close()
            with suppress(TimeoutError):
                await asyncio.wait_for(
                    server.wait_closed(),
                    timeout=self.shutdown_timeout_seconds,
                )
            for writer in tuple(self._connections):
                writer.close()
            if self._connections:
                await asyncio.gather(
                    *(self._wait_writer_closed(writer) for writer in tuple(self._connections)),
                    return_exceptions=True,
                )
            tasks = [task for task in self._connection_tasks if task is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            if tasks:
                with suppress(TimeoutError):
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=self.shutdown_timeout_seconds,
                    )
            self._connections.clear()
            self._connection_tasks.clear()
            self.bound_endpoint = None
            self._closed.set()

    async def _wait_writer_closed(self, writer: asyncio.StreamWriter) -> None:
        with suppress(OSError, TimeoutError):
            await asyncio.wait_for(
                writer.wait_closed(),
                timeout=self.shutdown_timeout_seconds,
            )

    def _verify_request(self, request: DirectNetworkProbeRequest) -> None:
        ticket = request.ticket
        verify_probe_ticket(ticket, self.cluster, now_unix_ns=self.clock_ns())
        if ticket.destination_node_id != self.node_id:
            raise IntegrityError("network probe ticket targets a different node")
        if ticket.destination_worker_id != self.worker_id:
            raise IntegrityError("network probe ticket targets a different worker")
        if ticket.destination_public_key != self.identity.public_key_b64:
            raise IntegrityError("network probe ticket destination identity is not this node")
        if request.payload_size not in ticket.payload_sizes:
            raise IntegrityError("network probe payload size is not authorized")
        if request.sample_index >= ticket.sample_count:
            raise IntegrityError("network probe sample index is outside its bound")
        if request.payload_size > self.maximum_bytes:
            raise IntegrityError("network probe payload exceeds the server byte bound")
        if abs(self.clock_ns() - request.timestamp_unix_ns) > self.authentication_skew_ns:
            raise IntegrityError("network probe request timestamp is outside the allowed window")
        verify_signature(
            ticket.source_public_key,
            _signed_model_payload(request),
            request.signature,
        )
        self.nonce_cache.add(request.nonce)

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        task = asyncio.current_task()
        if len(self._connections) >= self.maximum_connections:
            writer.close()
            await self._wait_writer_closed(writer)
            return
        self._connections.add(writer)
        if task is not None:
            self._connection_tasks.add(task)
        try:
            tls_peer_fingerprint: str | None = None
            if self.tls_server is not None:
                tls_object = writer.get_extra_info("ssl_object")
                peer_der = tls_object.getpeercert(binary_form=True) if tls_object else None
                tls_peer_fingerprint = self.tls_server.validate_peer_der(peer_der)
            request = await _read_model(
                reader,
                DirectNetworkProbeRequest,
                timeout_seconds=self.timeout_seconds,
            )
            self._verify_request(request)
            if (
                tls_peer_fingerprint is not None
                and tls_peer_fingerprint != request.ticket.source_fingerprint
            ):
                raise IntegrityError(
                    "network probe TLS peer differs from the ticket source identity"
                )
            payload = await asyncio.wait_for(
                reader.readexactly(request.payload_size),
                timeout=self.timeout_seconds,
            )
            if hashlib.sha256(payload).hexdigest() != request.payload_sha256:
                raise IntegrityError("network probe upload checksum did not match")
            ack = DirectNetworkProbeAck(
                ticket_id=request.ticket.ticket_id,
                request_nonce=request.nonce,
                destination_node_id=self.node_id,
                destination_worker_id=self.worker_id,
                received_at_unix_ns=self.clock_ns(),
                signature="pending",
            )
            ack = ack.model_copy(
                update={"signature": self.identity.sign(_signed_model_payload(ack))}
            )
            await _write_model(writer, ack, timeout_seconds=self.timeout_seconds)
            response_payload = deterministic_probe_payload(
                request.response_seed,
                request.payload_size,
            )
            response = DirectNetworkProbeResponse(
                ticket_id=request.ticket.ticket_id,
                request_nonce=request.nonce,
                destination_node_id=self.node_id,
                destination_worker_id=self.worker_id,
                payload_size=len(response_payload),
                payload_sha256=hashlib.sha256(response_payload).hexdigest(),
                sent_at_unix_ns=self.clock_ns(),
                signature="pending",
            )
            response = response.model_copy(
                update={"signature": self.identity.sign(_signed_model_payload(response))}
            )
            await _write_model(writer, response, timeout_seconds=self.timeout_seconds)
            writer.write(response_payload)
            await asyncio.wait_for(writer.drain(), timeout=self.timeout_seconds)
        except (IntegrityError, TransportError, TimeoutError, asyncio.IncompleteReadError):
            # The closed socket is the only failure response.  It reveals neither
            # membership details nor which authentication check failed.
            pass
        finally:
            writer.close()
            await self._wait_writer_closed(writer)
            self._connections.discard(writer)
            if task is not None:
                self._connection_tasks.discard(task)


class DirectedNetworkMeasurer:
    """Measure one coordinator-authorized directed worker link."""

    def __init__(
        self,
        *,
        state: ClusterStateStore,
        cluster: ClusterMetadata,
        identity: WorkerIdentity,
        node_id: str,
        worker_id: str,
        source_interface: str | None = None,
        source_mtu: int | None = None,
        timeout_seconds: float = DEFAULT_NETWORK_PROBE_TIMEOUT_SECONDS,
        clock_ns: Callable[[], int] = time.time_ns,
        monotonic: Callable[[], float] = time.monotonic,
        random_bytes: Callable[[int], bytes] = os.urandom,
        tls_client: TlsClientConfig | None = None,
        allow_plaintext_loopback: bool = True,
    ) -> None:
        if timeout_seconds <= 0 or timeout_seconds > 60:
            raise ValueError("network measurement timeout must be in (0, 60] seconds")
        self.state = state
        self.cluster = cluster
        self.identity = identity
        self.node_id = node_id
        self.worker_id = worker_id
        self.source_interface = source_interface
        self.source_mtu = source_mtu
        self.timeout_seconds = timeout_seconds
        self.clock_ns = clock_ns
        self.monotonic = monotonic
        self.random_bytes = random_bytes
        self.tls_client = tls_client
        self.allow_plaintext_loopback = allow_plaintext_loopback

    def _verify_ticket_source(self, ticket: NetworkProbeTicket) -> None:
        verify_probe_ticket(ticket, self.cluster, now_unix_ns=self.clock_ns())
        if ticket.source_node_id != self.node_id or ticket.source_worker_id != self.worker_id:
            raise IntegrityError("network probe ticket source does not match this worker")
        if ticket.source_public_key != self.identity.public_key_b64:
            raise IntegrityError("network probe ticket source identity does not match this node")

    @staticmethod
    def _verify_destination_message(
        value: DirectNetworkProbeAck | DirectNetworkProbeResponse,
        ticket: NetworkProbeTicket,
        request_nonce: str,
    ) -> None:
        if (
            value.ticket_id != ticket.ticket_id
            or value.request_nonce != request_nonce
            or value.destination_node_id != ticket.destination_node_id
            or value.destination_worker_id != ticket.destination_worker_id
        ):
            raise IntegrityError("network probe response is not bound to its request")
        verify_signature(
            ticket.destination_public_key,
            _signed_model_payload(value),
            value.signature,
        )

    async def measure(self, ticket: NetworkProbeTicket) -> NetworkLinkMeasurement:
        self._verify_ticket_source(ticket)
        require_tls_for_endpoint(
            ticket.destination_endpoint,
            tls_configured=self.tls_client is not None,
            allow_plaintext_loopback=self.allow_plaintext_loopback,
            transport_name="direct network probe",
        )
        destination_host, destination_port = split_endpoint(ticket.destination_endpoint)
        upload_rates: list[float] = []
        download_rates: list[float] = []
        connect_timings_ms: list[float] = []
        transfer_timings_ms: list[float] = []
        source_endpoint: str | None = ticket.source_endpoint
        for payload_size in ticket.payload_sizes:
            for sample_index in range(ticket.sample_count):
                nonce = self.random_bytes(18).hex()
                response_seed = self.random_bytes(18).hex()
                payload = deterministic_probe_payload(nonce, payload_size)
                request = DirectNetworkProbeRequest(
                    ticket=ticket,
                    timestamp_unix_ns=self.clock_ns(),
                    nonce=nonce,
                    sample_index=sample_index,
                    payload_size=payload_size,
                    payload_sha256=hashlib.sha256(payload).hexdigest(),
                    response_seed=response_seed,
                    signature="pending",
                )
                request = request.model_copy(
                    update={"signature": self.identity.sign(_signed_model_payload(request))}
                )
                connect_started = self.monotonic()
                try:
                    reader, writer = await asyncio.wait_for(
                        asyncio.open_connection(
                            destination_host,
                            destination_port,
                            ssl=(
                                self.tls_client.ssl_context()
                                if self.tls_client is not None
                                else None
                            ),
                            server_hostname=(
                                self.tls_client.expected_server_name
                                if self.tls_client is not None
                                else None
                            ),
                        ),
                        timeout=self.timeout_seconds,
                    )
                except (OSError, TimeoutError) as exc:
                    raise TransportError(
                        f"could not reach peer probe endpoint {ticket.destination_endpoint}: {exc}"
                    ) from exc
                connect_timings_ms.append((self.monotonic() - connect_started) * 1000)
                if self.tls_client is not None:
                    tls_object = writer.get_extra_info("ssl_object")
                    peer_der = tls_object.getpeercert(binary_form=True) if tls_object else None
                    tls_peer_fingerprint = self.tls_client.validate_peer_der(peer_der)
                    if tls_peer_fingerprint != ticket.destination_fingerprint:
                        raise IntegrityError(
                            "network probe TLS peer differs from the ticket destination identity"
                        )
                source_endpoint = _endpoint(writer.get_extra_info("sockname")) or source_endpoint
                try:
                    transfer_started = self.monotonic()
                    upload_started = transfer_started
                    await _write_model(writer, request, timeout_seconds=self.timeout_seconds)
                    writer.write(payload)
                    await asyncio.wait_for(writer.drain(), timeout=self.timeout_seconds)
                    ack = await _read_model(
                        reader,
                        DirectNetworkProbeAck,
                        timeout_seconds=self.timeout_seconds,
                    )
                    upload_elapsed = max(self.monotonic() - upload_started, 1e-12)
                    self._verify_destination_message(ack, ticket, nonce)
                    download_started = self.monotonic()
                    response = await _read_model(
                        reader,
                        DirectNetworkProbeResponse,
                        timeout_seconds=self.timeout_seconds,
                    )
                    self._verify_destination_message(response, ticket, nonce)
                    response_payload = await asyncio.wait_for(
                        reader.readexactly(response.payload_size),
                        timeout=self.timeout_seconds,
                    )
                    download_elapsed = max(self.monotonic() - download_started, 1e-12)
                    if response.payload_size != payload_size:
                        raise IntegrityError("network probe response size changed")
                    if hashlib.sha256(response_payload).hexdigest() != response.payload_sha256:
                        raise IntegrityError("network probe download checksum did not match")
                    if response_payload != deterministic_probe_payload(response_seed, payload_size):
                        raise IntegrityError("network probe response payload was not deterministic")
                    transfer_timings_ms.append((self.monotonic() - transfer_started) * 1000)
                    upload_rates.append(payload_size / upload_elapsed)
                    download_rates.append(payload_size / download_elapsed)
                except (TimeoutError, asyncio.IncompleteReadError) as exc:
                    raise TransportError("peer network probe timed out or closed early") from exc
                finally:
                    writer.close()
                    with suppress(OSError, TimeoutError):
                        await asyncio.wait_for(
                            writer.wait_closed(),
                            timeout=self.timeout_seconds,
                        )
        round_trip_ms = statistics.median(connect_timings_ms)
        measurement = NetworkLinkMeasurement(
            source_worker_id=ticket.source_worker_id,
            destination_worker_id=ticket.destination_worker_id,
            source_node_id=ticket.source_node_id,
            destination_node_id=ticket.destination_node_id,
            measured_at_unix_ns=self.clock_ns(),
            round_trip_latency_ms=round_trip_ms,
            one_way_estimate_ms=round_trip_ms / 2,
            upload_bytes_per_s=statistics.median(upload_rates),
            download_bytes_per_s=statistics.median(download_rates),
            payload_sizes=list(ticket.payload_sizes),
            sample_count=len(transfer_timings_ms),
            p95_transfer_ms=_percentile(transfer_timings_ms, 95),
            jitter_ms=statistics.pstdev(connect_timings_ms),
            connection_stability=(len(transfer_timings_ms) / max(1, int(ticket.sample_count))),
            source_endpoint=source_endpoint,
            destination_endpoint=ticket.destination_endpoint,
            source_interface=self.source_interface,
            destination_interface=ticket.destination_interface,
            mtu=self.source_mtu,
            measured=True,
            probe_ticket_id=ticket.ticket_id,
            authentication_verified=True,
            payload_checksums_verified=True,
        )
        self.state.save_network_measurement(measurement)
        self.state.append_audit(
            ClusterAuditEvent(
                event_id=uuid4().hex,
                event_type="network_measurement_completed",
                timestamp_unix_ns=self.clock_ns(),
                cluster_id=self.cluster.cluster_id,
                node_id=self.node_id,
                worker_id=self.worker_id,
                detail=(
                    f"directed link {ticket.source_worker_id} -> "
                    f"{ticket.destination_worker_id} measured"
                ),
            )
        )
        return measurement


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile from no samples")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def network_measurement_is_fresh(
    measurement: NetworkLinkMeasurement,
    *,
    ttl_seconds: int = DEFAULT_NETWORK_MEASUREMENT_TTL_SECONDS,
    now_unix_ns: int | None = None,
) -> bool:
    if ttl_seconds <= 0:
        raise ValueError("network measurement TTL must be positive")
    now = now_unix_ns or time.time_ns()
    age = now - measurement.measured_at_unix_ns
    return 0 <= age <= ttl_seconds * 1_000_000_000


class NetworkMeasurementRepository:
    """Freshness-aware view over durable directed coordinator evidence."""

    def __init__(
        self,
        *,
        state: ClusterStateStore,
        ttl_seconds: int = DEFAULT_NETWORK_MEASUREMENT_TTL_SECONDS,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("network measurement TTL must be positive")
        self.state = state
        self.ttl_seconds = ttl_seconds
        self.clock_ns = clock_ns
        self._reported_expired: set[tuple[str, str, int]] = set()

    def get(
        self,
        source_worker_id: str,
        destination_worker_id: str,
        *,
        require_fresh: bool = True,
    ) -> NetworkLinkMeasurement | None:
        value = next(
            (
                item
                for item in self.state.load_network_measurements().measurements
                if item.source_worker_id == source_worker_id
                and item.destination_worker_id == destination_worker_id
            ),
            None,
        )
        if value is None:
            return None
        if require_fresh and not network_measurement_is_fresh(
            value,
            ttl_seconds=self.ttl_seconds,
            now_unix_ns=self.clock_ns(),
        ):
            return None
        return value

    def fresh(self) -> list[NetworkLinkMeasurement]:
        now = self.clock_ns()
        return [
            item
            for item in self.state.load_network_measurements().measurements
            if network_measurement_is_fresh(
                item,
                ttl_seconds=self.ttl_seconds,
                now_unix_ns=now,
            )
        ]

    def expire_stale(self, *, cluster_id: str | None = None) -> list[NetworkLinkMeasurement]:
        now = self.clock_ns()
        stale = [
            item
            for item in self.state.load_network_measurements().measurements
            if not network_measurement_is_fresh(
                item,
                ttl_seconds=self.ttl_seconds,
                now_unix_ns=now,
            )
        ]
        for item in stale:
            identity = (
                item.source_worker_id,
                item.destination_worker_id,
                item.measured_at_unix_ns,
            )
            if identity in self._reported_expired:
                continue
            self._reported_expired.add(identity)
            self.state.append_audit(
                ClusterAuditEvent(
                    event_id=uuid4().hex,
                    event_type="network_measurement_expired",
                    timestamp_unix_ns=now,
                    cluster_id=cluster_id,
                    worker_id=item.source_worker_id,
                    detail=f"directed link to {item.destination_worker_id} requires refresh",
                )
            )
        return stale


class NetworkProbeCoordinator:
    """Issue bounded tickets and persist verified measurement submissions."""

    def __init__(
        self,
        *,
        state: ClusterStateStore,
        cluster: ClusterMetadata,
        identity: CoordinatorIdentity,
        maximum_bytes: int = DEFAULT_NETWORK_PROBE_MAX_BYTES,
        ticket_ttl_seconds: int = 120,
        maximum_tickets: int = 4096,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        if not 1024 <= maximum_bytes <= 256 * 1024 * 1024:
            raise ValueError("network probe coordinator byte bound is invalid")
        if not 1 <= ticket_ttl_seconds <= 600:
            raise ValueError("network probe ticket TTL must be in [1, 600] seconds")
        if not 1 <= maximum_tickets <= 65_536:
            raise ValueError("network probe ticket cache bound is invalid")
        if identity.public_key_b64 != cluster.coordinator_public_key:
            raise IntegrityError("network probe signer is not the pinned cluster coordinator")
        self.state = state
        self.cluster = cluster
        self.identity = identity
        self.maximum_bytes = maximum_bytes
        self.ticket_ttl_ns = ticket_ttl_seconds * 1_000_000_000
        self.maximum_tickets = maximum_tickets
        self.clock_ns = clock_ns
        self._tickets: OrderedDict[str, NetworkProbeTicket] = OrderedDict()

    def _node_for_worker(self, worker_id: str) -> NodeMetadata:
        matches = [node for node in self.state.load_nodes().nodes if worker_id in node.worker_ids]
        if len(matches) != 1:
            raise ConfigurationError(
                f"worker {worker_id!r} is not owned by exactly one active cluster node"
            )
        node = matches[0]
        membership = self.state.membership(node.node_id)
        if membership is None or membership.status != "active" or node.revoked:
            raise IntegrityError(f"worker {worker_id!r} does not have active membership")
        if self.state.is_revoked_fingerprint(node.fingerprint):
            raise IntegrityError(f"worker {worker_id!r} belongs to a revoked node")
        return node

    def issue(
        self,
        request: NetworkProbeControlRequest,
        *,
        requesting_node_id: str,
    ) -> NetworkProbeTicket:
        if request.operation != "issue" or request.measurement is not None:
            raise ConfigurationError("network probe issue request has incompatible fields")
        source = self._node_for_worker(request.source_worker_id)
        destination = self._node_for_worker(request.destination_worker_id)
        if source.node_id != requesting_node_id:
            raise IntegrityError("a node may only request probes sourced from its own worker")
        if source.node_id == destination.node_id:
            raise ConfigurationError("direct network probes require distinct nodes")
        if destination.probe_endpoint is None:
            raise ConfigurationError(
                f"destination node {destination.node_id} has no advertised probe endpoint"
            )
        payload_sizes = sorted(set(request.payload_sizes or DEFAULT_NETWORK_PROBE_PAYLOAD_SIZES))
        if len(payload_sizes) > 8 or any(size > self.maximum_bytes for size in payload_sizes):
            raise ConfigurationError("network probe payload set exceeds configured bounds")
        selected_maximum = min(request.maximum_bytes, self.maximum_bytes)
        if 2 * sum(payload_sizes) * request.sample_count > selected_maximum:
            raise ConfigurationError("network probe samples exceed the total byte bound")
        now = self.clock_ns()
        unsigned = NetworkProbeTicket(
            ticket_id=uuid4().hex,
            cluster_id=self.cluster.cluster_id,
            source_node_id=source.node_id,
            source_worker_id=request.source_worker_id,
            source_public_key=source.public_key,
            source_fingerprint=source.fingerprint,
            source_endpoint=source.probe_endpoint,
            destination_node_id=destination.node_id,
            destination_worker_id=request.destination_worker_id,
            destination_public_key=destination.public_key,
            destination_fingerprint=destination.fingerprint,
            destination_endpoint=destination.probe_endpoint,
            issued_at_unix_ns=now,
            expires_at_unix_ns=now + self.ticket_ttl_ns,
            payload_sizes=payload_sizes,
            sample_count=request.sample_count,
            maximum_bytes=selected_maximum,
            coordinator_public_key=self.identity.public_key_b64,
            coordinator_fingerprint=self.identity.public_key_fingerprint,
            signature="pending",
        )
        ticket = sign_probe_ticket(unsigned, self.identity)
        self._tickets[ticket.ticket_id] = ticket
        while len(self._tickets) > self.maximum_tickets:
            self._tickets.popitem(last=False)
        return ticket

    def record(
        self,
        request: NetworkProbeControlRequest,
        *,
        requesting_node_id: str,
    ) -> NetworkLinkMeasurement:
        measurement = request.measurement
        if request.operation != "record" or measurement is None:
            raise ConfigurationError("network measurement record request has incompatible fields")
        if measurement.probe_ticket_id is None:
            raise IntegrityError("network measurement has no probe ticket evidence")
        ticket = self._tickets.get(measurement.probe_ticket_id)
        if ticket is None:
            raise IntegrityError("network measurement probe ticket is unknown or no longer active")
        verify_probe_ticket(ticket, self.cluster, now_unix_ns=self.clock_ns())
        if requesting_node_id != ticket.source_node_id:
            raise IntegrityError("a node may only submit its own directed measurement")
        if (
            request.source_worker_id != ticket.source_worker_id
            or request.destination_worker_id != ticket.destination_worker_id
            or measurement.source_worker_id != ticket.source_worker_id
            or measurement.destination_worker_id != ticket.destination_worker_id
            or measurement.payload_sizes != ticket.payload_sizes
            or measurement.sample_count != len(ticket.payload_sizes) * ticket.sample_count
            or not measurement.measured
            or not measurement.authentication_verified
            or not measurement.payload_checksums_verified
        ):
            raise IntegrityError("network measurement does not match its authorized ticket")
        if (
            not ticket.issued_at_unix_ns
            <= measurement.measured_at_unix_ns
            <= ticket.expires_at_unix_ns
        ):
            raise IntegrityError("network measurement timestamp is outside its ticket window")
        self.state.save_network_measurement(measurement)
        self._tickets.pop(ticket.ticket_id, None)
        self.state.append_audit(
            ClusterAuditEvent(
                event_id=uuid4().hex,
                event_type="network_measurement_completed",
                timestamp_unix_ns=self.clock_ns(),
                cluster_id=self.cluster.cluster_id,
                node_id=requesting_node_id,
                worker_id=measurement.source_worker_id,
                detail=f"directed link to {measurement.destination_worker_id} recorded",
            )
        )
        return measurement

    async def handle(
        self,
        request: NetworkProbeControlRequest,
    ) -> NetworkProbeControlResponse:
        try:
            if request.operation == "issue":
                ticket = self.issue(
                    request,
                    requesting_node_id=request.authentication.node_id,
                )
                return NetworkProbeControlResponse(accepted=True, ticket=ticket)
            measurement = self.record(
                request,
                requesting_node_id=request.authentication.node_id,
            )
            return NetworkProbeControlResponse(accepted=True, measurement=measurement)
        except (ConfigurationError, IntegrityError) as exc:
            return NetworkProbeControlResponse(accepted=False, detail=str(exc))


def default_probe_endpoint_path(state: ClusterStateStore) -> Path:
    """Stable discovery file location used by service diagnostics."""

    return state.paths.runtime / "network-probe-endpoint.json"


__all__ = [
    "DEFAULT_NETWORK_MEASUREMENT_TTL_SECONDS",
    "DEFAULT_NETWORK_PROBE_MAX_BYTES",
    "DEFAULT_NETWORK_PROBE_PAYLOAD_SIZES",
    "DEFAULT_NETWORK_PROBE_TIMEOUT_SECONDS",
    "NETWORK_PROBE_PROTOCOL_VERSION",
    "DirectNetworkProbeServer",
    "DirectedNetworkMeasurer",
    "NetworkMeasurementRepository",
    "NetworkProbeCoordinator",
    "deterministic_probe_payload",
    "network_measurement_is_fresh",
    "sign_probe_ticket",
    "verify_probe_ticket",
]
