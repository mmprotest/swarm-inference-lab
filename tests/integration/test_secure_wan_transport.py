from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import grpc
import pytest

from swarm_inference.exceptions import IntegrityError, TransportError
from swarm_inference.execution.expert import ExpertStore, deterministic_expert
from swarm_inference.host import format_endpoint, split_endpoint
from swarm_inference.protocol.messages import HealthResponse, serialize_message
from swarm_inference.protocol.stage_ring import Operation, StageMessage
from swarm_inference.security.identity import CoordinatorIdentity, WorkerIdentity
from swarm_inference.security.tls import (
    WORKER_TLS_NAME,
    TlsCertificatePaths,
    TlsClientConfig,
    TlsServerConfig,
    create_cluster_ca_certificate,
    generate_tls_private_key_pem,
    identity_tls_public_key_pem,
    issue_node_certificate,
    materialize_tls_identity,
    require_tls_for_endpoint,
    tls_public_key_pem,
    validate_certificate_binding,
)
from swarm_inference.transport.expert import ExpertTransportClient
from swarm_inference.transport.grpc_transport import GrpcTransport
from swarm_inference.transport.stage_ring_connection import StageRingConnectionPool
from swarm_inference.transport.stage_ring_server import StageRingServer
from swarm_inference.transport.tcp_meter import TcpMeteringProxy
from swarm_inference.worker.expert_service import ExpertWorkerRuntime, ExpertWorkerServer


def _material(
    root: Path,
    *,
    identity: WorkerIdentity,
    certificate: str,
    ca_certificate: str,
) -> TlsCertificatePaths:
    paths = TlsCertificatePaths(
        certificate=root / "certificate.pem",
        private_key=root / "private-key.pem",
        ca_certificate=root / "ca.pem",
    )
    materialize_tls_identity(
        identity=identity,
        certificate_pem=certificate,
        certificate_path=paths.certificate,
        private_key_path=paths.private_key,
    )
    paths.ca_certificate.write_text(ca_certificate, encoding="ascii")
    return paths


def _identities(
    tmp_path: Path,
) -> tuple[
    CoordinatorIdentity,
    WorkerIdentity,
    WorkerIdentity,
    str,
    TlsCertificatePaths,
    TlsCertificatePaths,
    TlsCertificatePaths,
]:
    coordinator = CoordinatorIdentity.generate()
    first = WorkerIdentity.generate()
    second = WorkerIdentity.generate()
    ca = create_cluster_ca_certificate(coordinator, cluster_id="cluster-secure-test")

    def worker_certificate(identity: WorkerIdentity) -> str:
        return issue_node_certificate(
            coordinator,
            ca_certificate_pem=ca,
            cluster_id="cluster-secure-test",
            node_public_key_b64=identity.public_key_b64,
            node_fingerprint=identity.public_key_fingerprint,
            node_tls_public_key_pem=identity_tls_public_key_pem(identity),
        )

    coordinator_paths = _material(
        tmp_path / "coordinator",
        identity=coordinator,
        certificate=ca,
        ca_certificate=ca,
    )
    first_paths = _material(
        tmp_path / "first",
        identity=first,
        certificate=worker_certificate(first),
        ca_certificate=ca,
    )
    second_paths = _material(
        tmp_path / "second",
        identity=second,
        certificate=worker_certificate(second),
        ca_certificate=ca,
    )
    return coordinator, first, second, ca, coordinator_paths, first_paths, second_paths


def _message(sequence: int) -> StageMessage:
    return StageMessage(
        operation=Operation.HEALTH,
        model_revision="model",
        tokenizer_revision="tokenizer",
        topology_id="secure-topology",
        stage_id=0,
        layer_start=0,
        layer_end=1,
        session_id="session",
        request_id=f"request-{sequence}",
        sequence_number=sequence,
        token_position=-1,
        source_stage=-1,
        destination_stage=0,
        payload=b"encrypted-activation",
    )


async def _echo(message: StageMessage) -> StageMessage:
    return replace(
        message,
        source_stage=message.destination_stage,
        destination_stage=message.source_stage,
    )


def test_certificate_identity_expiry_revocation_and_wrong_ca(tmp_path: Path) -> None:
    coordinator, first, _, ca, _, first_paths, _ = _identities(tmp_path)
    certificate = first_paths.certificate.read_text(encoding="ascii")
    assert (
        validate_certificate_binding(
            certificate,
            ca_certificate_pem=ca,
            cluster_id="cluster-secure-test",
            role="worker",
            expected_identity_fingerprint=first.public_key_fingerprint,
        )
        == first.public_key_fingerprint
    )
    with pytest.raises(IntegrityError, match="revoked"):
        validate_certificate_binding(
            certificate,
            ca_certificate_pem=ca,
            cluster_id="cluster-secure-test",
            role="worker",
            revoked_fingerprints=frozenset({first.public_key_fingerprint}),
        )
    expired = issue_node_certificate(
        coordinator,
        ca_certificate_pem=ca,
        cluster_id="cluster-secure-test",
        node_public_key_b64=first.public_key_b64,
        node_fingerprint=first.public_key_fingerprint,
        node_tls_public_key_pem=identity_tls_public_key_pem(first),
        now=datetime.now(UTC) - timedelta(days=3),
        lifetime_days=1,
    )
    with pytest.raises(IntegrityError, match="expired"):
        validate_certificate_binding(
            expired,
            ca_certificate_pem=ca,
            cluster_id="cluster-secure-test",
            role="worker",
        )
    attacker = CoordinatorIdentity.generate()
    attacker_ca = create_cluster_ca_certificate(attacker, cluster_id="cluster-secure-test")
    attacker_certificate = issue_node_certificate(
        attacker,
        ca_certificate_pem=attacker_ca,
        cluster_id="cluster-secure-test",
        node_public_key_b64=first.public_key_b64,
        node_fingerprint=first.public_key_fingerprint,
        node_tls_public_key_pem=identity_tls_public_key_pem(first),
    )
    with pytest.raises(IntegrityError, match=r"issuer|signature"):
        validate_certificate_binding(
            attacker_certificate,
            ca_certificate_pem=ca,
            cluster_id="cluster-secure-test",
            role="worker",
        )


def test_transport_key_rotates_without_changing_durable_identity(tmp_path: Path) -> None:
    coordinator = CoordinatorIdentity.generate()
    worker = WorkerIdentity.generate()
    ca = create_cluster_ca_certificate(coordinator, cluster_id="cluster-key-rotation")
    first_key = generate_tls_private_key_pem()
    second_key = generate_tls_private_key_pem()
    assert tls_public_key_pem(first_key) != tls_public_key_pem(second_key)

    certificates = [
        issue_node_certificate(
            coordinator,
            ca_certificate_pem=ca,
            cluster_id="cluster-key-rotation",
            node_public_key_b64=worker.public_key_b64,
            node_fingerprint=worker.public_key_fingerprint,
            node_tls_public_key_pem=tls_public_key_pem(private_key),
        )
        for private_key in (first_key, second_key)
    ]
    for certificate in certificates:
        assert (
            validate_certificate_binding(
                certificate,
                ca_certificate_pem=ca,
                cluster_id="cluster-key-rotation",
                role="worker",
                expected_identity_fingerprint=worker.public_key_fingerprint,
            )
            == worker.public_key_fingerprint
        )

    paths = TlsCertificatePaths(
        certificate=tmp_path / "rotated" / "certificate.pem",
        private_key=tmp_path / "rotated" / "private-key.pem",
        ca_certificate=tmp_path / "rotated" / "ca.pem",
    )
    materialize_tls_identity(
        identity=worker,
        certificate_pem=certificates[1],
        certificate_path=paths.certificate,
        private_key_path=paths.private_key,
        private_key_pem=second_key,
    )
    assert paths.private_key.read_bytes() == second_key
    assert tls_public_key_pem(paths.private_key.read_bytes()) == tls_public_key_pem(second_key)


@pytest.mark.asyncio
async def test_encrypted_mutually_authenticated_stage_path_reuses_connection(
    tmp_path: Path,
) -> None:
    _, first, second, _, _, first_paths, second_paths = _identities(tmp_path)
    server = StageRingServer(
        handler=_echo,
        tls=TlsServerConfig(
            first_paths,
            allowed_peer_fingerprints=frozenset({second.public_key_fingerprint}),
        ),
    )
    port = await server.start("127.0.0.1:0")
    client_tls = TlsClientConfig(
        second_paths,
        WORKER_TLS_NAME,
        expected_peer_fingerprint=first.public_key_fingerprint,
    )
    pool = StageRingConnectionPool(tls=client_tls)
    endpoint = f"127.0.0.1:{port}"
    try:
        assert (await pool.send(endpoint, _message(1))).payload == b"encrypted-activation"
        assert (await pool.send(endpoint, _message(2))).payload == b"encrypted-activation"
        assert pool.snapshot()["connections_created"] == 1
        assert server.metrics.reused_frames == 1
    finally:
        await pool.close()
        await server.stop()


@pytest.mark.asyncio
async def test_encrypted_mutually_authenticated_expert_control_path(
    tmp_path: Path,
) -> None:
    _, expert_identity, stage_identity, _, _, expert_paths, stage_paths = _identities(tmp_path)
    weights = deterministic_expert(latent_dimension=4, intermediate_dimension=8, seed=7)
    runtime = ExpertWorkerRuntime(
        worker_id="expert-worker",
        identity=expert_identity,
        model_id="test/olmoe",
        model_revision="revision",
        model_fingerprint="model-fingerprint",
        quantization_fingerprint="quantization-fingerprint",
        store=ExpertStore(
            owned={(0, 0)},
            loader=lambda _layer, _expert: weights,
            residency_budget_bytes=weights.byte_size,
            cache_budget_bytes=weights.byte_size,
        ),
    )
    server = ExpertWorkerServer(
        runtime,
        host="127.0.0.1",
        port=0,
        tls=TlsServerConfig(
            expert_paths,
            allowed_peer_fingerprints=frozenset({stage_identity.public_key_fingerprint}),
        ),
    )
    host, port = await server.start()
    endpoint = f"{host}:{port}"
    client = ExpertTransportClient(
        endpoint,
        tls=TlsClientConfig(
            stage_paths,
            WORKER_TLS_NAME,
            expected_peer_fingerprint=expert_identity.public_key_fingerprint,
        ),
    )
    try:
        response = await asyncio.to_thread(client.control, "status")
        assert response["status"]["worker_id"] == "expert-worker"
        reader, writer = await asyncio.open_connection(host, port)
        writer.write(b"plaintext is not a TLS handshake")
        await writer.drain()
        assert await asyncio.wait_for(reader.read(), timeout=2) == b""
        writer.close()
        await writer.wait_closed()
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_wrong_worker_certificate_and_plaintext_stage_connections_are_rejected(
    tmp_path: Path,
) -> None:
    _, expected, actual, _, _, expected_paths, actual_paths = _identities(tmp_path)
    server = StageRingServer(
        handler=_echo,
        tls=TlsServerConfig(actual_paths),
        read_timeout_s=0.5,
    )
    port = await server.start("127.0.0.1:0")
    endpoint = f"127.0.0.1:{port}"
    wrong_pin = TlsClientConfig(
        expected_paths,
        WORKER_TLS_NAME,
        expected_peer_fingerprint=expected.public_key_fingerprint,
    )
    pool = StageRingConnectionPool(
        tls=wrong_pin,
        reconnect_attempts=1,
        connect_timeout_s=1,
        read_timeout_s=1,
        write_timeout_s=1,
    )
    try:
        with pytest.raises(TransportError, match=r"certificate|connection"):
            await pool.send(endpoint, _message(1))
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"plaintext is not a TLS handshake")
        await writer.drain()
        assert await asyncio.wait_for(reader.read(), timeout=2) == b""
        writer.close()
        await writer.wait_closed()
        assert actual.public_key_fingerprint != expected.public_key_fingerprint
    finally:
        await pool.close()
        await server.stop()


@pytest.mark.asyncio
async def test_secure_control_path_and_address_independent_reconnect(tmp_path: Path) -> None:
    _, first, second, _, _, first_paths, second_paths = _identities(tmp_path)

    async def health(_: bytes, context: grpc.aio.ServicerContext) -> bytes:
        del context
        return serialize_message(
            HealthResponse(worker_id="secure-worker", healthy=True, queue_depth=0)
        )

    server = grpc.aio.server()
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                "swarm.v1.Worker",
                {
                    "Health": grpc.unary_unary_rpc_method_handler(
                        health,
                        request_deserializer=lambda value: value,
                        response_serializer=lambda value: value,
                    )
                },
            ),
        )
    )
    port = server.add_secure_port(
        "127.0.0.1:0",
        TlsServerConfig(
            first_paths,
            allowed_peer_fingerprints=frozenset({second.public_key_fingerprint}),
        ).grpc_credentials(),
    )
    await server.start()
    transport = GrpcTransport(
        tls=TlsClientConfig(
            second_paths,
            WORKER_TLS_NAME,
            expected_peer_fingerprint=first.public_key_fingerprint,
        )
    )
    try:
        response = await transport.health(f"127.0.0.1:{port}")
        assert response.healthy
    finally:
        await transport.close()
        await server.stop(0)

    # The certificate is pinned to a durable DNS identity, not an IP address
    # or port, so a reconnect at a changed advertised address remains valid.
    stage_server = StageRingServer(handler=_echo, tls=TlsServerConfig(first_paths))
    new_port = await stage_server.start("127.0.0.1:0")
    pool = StageRingConnectionPool(
        tls=TlsClientConfig(
            second_paths,
            WORKER_TLS_NAME,
            expected_peer_fingerprint=first.public_key_fingerprint,
        )
    )
    try:
        assert (await pool.send(f"localhost:{new_port}", _message(3))).request_id == "request-3"
    finally:
        await pool.close()
        await stage_server.stop()


@pytest.mark.asyncio
async def test_llamacpp_metering_proxy_enforces_inbound_identity_policy(
    tmp_path: Path,
) -> None:
    _, first, second, _, _, first_paths, second_paths = _identities(tmp_path)
    upstream_payloads: list[bytes] = []

    async def upstream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        payload = await reader.read(1024)
        upstream_payloads.append(payload)
        writer.write(payload)
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(upstream, "127.0.0.1", 0)
    upstream_endpoint = format_endpoint("127.0.0.1", server.sockets[0].getsockname()[1])
    client_tls = TlsClientConfig(
        second_paths,
        WORKER_TLS_NAME,
        expected_peer_fingerprint=first.public_key_fingerprint,
    )

    async def connect(proxy: TcpMeteringProxy) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        endpoint = await proxy.start()
        host, port = split_endpoint(endpoint)
        reader, writer = await asyncio.open_connection(
            host,
            port,
            ssl=client_tls.ssl_context(),
            server_hostname=client_tls.expected_server_name,
        )
        tls_object = writer.get_extra_info("ssl_object")
        peer_der = tls_object.getpeercert(binary_form=True) if tls_object else None
        client_tls.validate_peer_der(peer_der)
        return reader, writer

    accepted = TcpMeteringProxy(
        listen_endpoint="127.0.0.1:0",
        upstream_endpoint=upstream_endpoint,
        inbound_tls=TlsServerConfig(
            first_paths,
            allowed_peer_fingerprints=frozenset({second.public_key_fingerprint}),
        ),
    )
    rejected = TcpMeteringProxy(
        listen_endpoint="127.0.0.1:0",
        upstream_endpoint=upstream_endpoint,
        inbound_tls=TlsServerConfig(
            first_paths,
            allowed_peer_fingerprints=frozenset({first.public_key_fingerprint}),
        ),
    )
    try:
        reader, writer = await connect(accepted)
        writer.write(b"trusted-llama-rpc")
        await writer.drain()
        assert await reader.readexactly(17) == b"trusted-llama-rpc"
        writer.close()
        await writer.wait_closed()

        reader, writer = await connect(rejected)
        writer.write(b"rejected-llama-rpc")
        await writer.drain()
        assert await asyncio.wait_for(reader.read(), timeout=2) == b""
        writer.close()
        await writer.wait_closed()
        await asyncio.sleep(0)

        assert upstream_payloads == [b"trusted-llama-rpc"]
        assert accepted.snapshot()["connection_failures"] == 0
        assert rejected.snapshot()["connection_failures"] == 1
    finally:
        await accepted.close()
        await rejected.close()
        server.close()
        await server.wait_closed()


def test_non_loopback_plaintext_is_never_an_implicit_fallback() -> None:
    with pytest.raises(TransportError, match="refuses unauthenticated plaintext"):
        require_tls_for_endpoint(
            "203.0.113.10:50051",
            tls_configured=False,
            allow_plaintext_loopback=True,
            transport_name="test transport",
        )
