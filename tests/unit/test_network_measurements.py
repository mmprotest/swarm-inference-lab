from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path

import pytest

from swarm_inference.cluster.models import (
    ClusterMetadata,
    NetworkLinkMeasurement,
    NodeMembership,
    NodeMetadata,
    VersionCompatibility,
    node_id_from_fingerprint,
)
from swarm_inference.cluster.network import (
    DirectedNetworkMeasurer,
    DirectNetworkProbeServer,
    NetworkMeasurementRepository,
    NetworkProbeCoordinator,
    _read_model,
    _signed_model_payload,
    _write_model,
    deterministic_probe_payload,
    network_measurement_is_fresh,
    verify_probe_ticket,
)
from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.exceptions import IntegrityError, TransportError
from swarm_inference.protocol.cluster import (
    ClusterRequestAuthentication,
    DirectNetworkProbeAck,
    DirectNetworkProbeRequest,
    NetworkProbeControlRequest,
)
from swarm_inference.security.identity import CoordinatorIdentity, WorkerIdentity
from swarm_inference.security.tls import (
    WORKER_TLS_NAME,
    TlsCertificatePaths,
    TlsClientConfig,
    TlsServerConfig,
    create_cluster_ca_certificate,
    identity_tls_public_key_pem,
    issue_node_certificate,
    materialize_tls_identity,
)


def _tls_material(
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


def _metadata(
    identity: WorkerIdentity,
    *,
    worker_id: str,
    joined_at: int,
    probe_endpoint: str,
) -> NodeMetadata:
    node_id = node_id_from_fingerprint(identity.public_key_fingerprint)
    return NodeMetadata(
        node_id=node_id,
        public_key=identity.public_key_b64,
        fingerprint=identity.public_key_fingerprint,
        hostname=node_id,
        operating_system="test",
        architecture="test",
        agent_version="0.1.0",
        runtime_version="0.1.0",
        build_id="test",
        package_lock_hash="a" * 64,
        worker_ids=[worker_id],
        control_endpoint="127.0.0.1:1",
        data_endpoint="127.0.0.1:2",
        probe_endpoint=probe_endpoint,
        joined_at_unix_ns=joined_at,
        last_seen_at_unix_ns=joined_at,
    )


def _membership(
    cluster: ClusterMetadata,
    identity: WorkerIdentity,
    *,
    joined_at: int,
) -> NodeMembership:
    return NodeMembership(
        cluster_id=cluster.cluster_id,
        node_id=node_id_from_fingerprint(identity.public_key_fingerprint),
        node_public_key=identity.public_key_b64,
        node_fingerprint=identity.public_key_fingerprint,
        coordinator_public_key=cluster.coordinator_public_key,
        coordinator_fingerprint=cluster.coordinator_fingerprint,
        joined_at_unix_ns=joined_at,
    )


def _cluster(coordinator: CoordinatorIdentity, now: int) -> ClusterMetadata:
    return ClusterMetadata(
        cluster_id="cluster-network",
        name="network-test",
        coordinator_id="node-00000000",
        coordinator_endpoint="127.0.0.1:50051",
        coordinator_public_key=coordinator.public_key_b64,
        coordinator_fingerprint=coordinator.public_key_fingerprint,
        created_at_unix_ns=now,
        runtime_compatibility=VersionCompatibility(
            minimum_runtime_version="0.1.0",
            maximum_runtime_version_exclusive="0.2.0",
        ),
    )


def _authentication(node_id: str, now: int) -> ClusterRequestAuthentication:
    return ClusterRequestAuthentication(
        node_id=node_id,
        timestamp_unix_ns=now,
        nonce="unit-test",
        signature="verified-by-rpc-before-handler",
    )


@pytest.mark.asyncio
async def test_authenticated_directed_network_measurement_is_persisted(tmp_path: Path) -> None:
    now = time.time_ns()
    state = ClusterStateStore(tmp_path)
    coordinator_identity = CoordinatorIdentity.generate()
    source_identity = WorkerIdentity.generate()
    destination_identity = WorkerIdentity.generate()
    cluster = _cluster(coordinator_identity, now)
    ca_certificate = create_cluster_ca_certificate(
        coordinator_identity,
        cluster_id=cluster.cluster_id,
    )

    def issue(identity: WorkerIdentity) -> str:
        return issue_node_certificate(
            coordinator_identity,
            ca_certificate_pem=ca_certificate,
            cluster_id=cluster.cluster_id,
            node_public_key_b64=identity.public_key_b64,
            node_fingerprint=identity.public_key_fingerprint,
            node_tls_public_key_pem=identity_tls_public_key_pem(identity),
        )

    source_tls = _tls_material(
        tmp_path / "source-tls",
        identity=source_identity,
        certificate=issue(source_identity),
        ca_certificate=ca_certificate,
    )
    destination_tls = _tls_material(
        tmp_path / "destination-tls",
        identity=destination_identity,
        certificate=issue(destination_identity),
        ca_certificate=ca_certificate,
    )
    state.save_cluster(cluster)
    source_node_id = node_id_from_fingerprint(source_identity.public_key_fingerprint)
    destination_node_id = node_id_from_fingerprint(destination_identity.public_key_fingerprint)
    source_worker_id = f"{source_node_id}/cpu-0"
    destination_worker_id = f"{destination_node_id}/cpu-0"
    server = DirectNetworkProbeServer(
        state=state,
        cluster=cluster,
        identity=destination_identity,
        node_id=destination_node_id,
        worker_id=destination_worker_id,
        tls_server=TlsServerConfig(
            destination_tls,
            allowed_peer_fingerprints=frozenset({source_identity.public_key_fingerprint}),
        ),
    )
    await server.start("127.0.0.1:0")
    assert server.bound_endpoint is not None
    try:
        for membership in (
            _membership(cluster, source_identity, joined_at=now),
            _membership(cluster, destination_identity, joined_at=now),
        ):
            state.save_membership(membership)
        state.save_node(
            _metadata(
                source_identity,
                worker_id=source_worker_id,
                joined_at=now,
                probe_endpoint="127.0.0.1:3",
            )
        )
        state.save_node(
            _metadata(
                destination_identity,
                worker_id=destination_worker_id,
                joined_at=now,
                probe_endpoint=server.bound_endpoint,
            )
        )
        coordinator = NetworkProbeCoordinator(
            state=state,
            cluster=cluster,
            identity=coordinator_identity,
        )
        issue = NetworkProbeControlRequest(
            authentication=_authentication(source_node_id, now),
            source_worker_id=source_worker_id,
            destination_worker_id=destination_worker_id,
            payload_sizes=[4096],
            sample_count=2,
            maximum_bytes=32 * 1024,
            timeout_ms=5000,
        )
        ticket = coordinator.issue(issue, requesting_node_id=source_node_id)
        measurer = DirectedNetworkMeasurer(
            state=state,
            cluster=cluster,
            identity=source_identity,
            node_id=source_node_id,
            worker_id=source_worker_id,
            source_interface="Ethernet",
            source_mtu=1500,
            tls_client=TlsClientConfig(
                source_tls,
                WORKER_TLS_NAME,
                expected_peer_fingerprint=destination_identity.public_key_fingerprint,
            ),
        )
        measurement = await measurer.measure(ticket)

        assert measurement.source_worker_id == source_worker_id
        assert measurement.destination_worker_id == destination_worker_id
        assert measurement.sample_count == 2
        assert measurement.upload_bytes_per_s > 0
        assert measurement.download_bytes_per_s > 0
        assert measurement.authentication_verified
        assert measurement.payload_checksums_verified
        assert measurement.destination_endpoint == server.bound_endpoint
        assert state.load_network_measurements().measurements == [measurement]

        recorded = coordinator.record(
            NetworkProbeControlRequest(
                authentication=_authentication(source_node_id, time.time_ns()),
                operation="record",
                source_worker_id=source_worker_id,
                destination_worker_id=destination_worker_id,
                measurement=measurement,
            ),
            requesting_node_id=source_node_id,
        )
        assert recorded == measurement
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_probe_server_rejects_bad_upload_checksum(tmp_path: Path) -> None:
    now = time.time_ns()
    state = ClusterStateStore(tmp_path)
    coordinator_identity = CoordinatorIdentity.generate()
    source_identity = WorkerIdentity.generate()
    destination_identity = WorkerIdentity.generate()
    cluster = _cluster(coordinator_identity, now)
    source_node_id = node_id_from_fingerprint(source_identity.public_key_fingerprint)
    destination_node_id = node_id_from_fingerprint(destination_identity.public_key_fingerprint)
    source_worker_id = f"{source_node_id}/cpu-0"
    destination_worker_id = f"{destination_node_id}/cpu-0"
    server = DirectNetworkProbeServer(
        state=state,
        cluster=cluster,
        identity=destination_identity,
        node_id=destination_node_id,
        worker_id=destination_worker_id,
    )
    await server.start("127.0.0.1:0")
    assert server.bound_endpoint is not None
    try:
        for identity, worker_id, endpoint in (
            (source_identity, source_worker_id, "127.0.0.1:3"),
            (destination_identity, destination_worker_id, server.bound_endpoint),
        ):
            state.save_membership(_membership(cluster, identity, joined_at=now))
            state.save_node(
                _metadata(
                    identity,
                    worker_id=worker_id,
                    joined_at=now,
                    probe_endpoint=endpoint,
                )
            )
        coordinator = NetworkProbeCoordinator(
            state=state,
            cluster=cluster,
            identity=coordinator_identity,
        )
        ticket = coordinator.issue(
            NetworkProbeControlRequest(
                authentication=_authentication(source_node_id, now),
                source_worker_id=source_worker_id,
                destination_worker_id=destination_worker_id,
                payload_sizes=[4096],
                sample_count=1,
                maximum_bytes=8192,
                timeout_ms=5000,
            ),
            requesting_node_id=source_node_id,
        )
        payload = deterministic_probe_payload("payload", 4096)
        request = DirectNetworkProbeRequest(
            ticket=ticket,
            timestamp_unix_ns=time.time_ns(),
            nonce="bad-checksum",
            sample_index=0,
            payload_size=len(payload),
            payload_sha256=hashlib.sha256(b"different").hexdigest(),
            response_seed="response",
            signature="pending",
        )
        request = request.model_copy(
            update={"signature": source_identity.sign(_signed_model_payload(request))}
        )
        host, port_text = server.bound_endpoint.rsplit(":", 1)
        reader, writer = await asyncio.open_connection(host, int(port_text))
        try:
            await _write_model(writer, request, timeout_seconds=2)
            writer.write(payload)
            await writer.drain()
            with pytest.raises(TransportError, match="closed an incomplete frame"):
                await _read_model(reader, DirectNetworkProbeAck, timeout_seconds=2)
        finally:
            writer.close()
            await writer.wait_closed()
    finally:
        await server.stop()


def test_ticket_tampering_and_stale_measurement_are_rejected(tmp_path: Path) -> None:
    now = 2_000_000_000_000
    state = ClusterStateStore(tmp_path)
    coordinator_identity = CoordinatorIdentity.generate()
    source_identity = WorkerIdentity.generate()
    destination_identity = WorkerIdentity.generate()
    cluster = _cluster(coordinator_identity, now)
    source_node_id = node_id_from_fingerprint(source_identity.public_key_fingerprint)
    destination_node_id = node_id_from_fingerprint(destination_identity.public_key_fingerprint)
    source_worker_id = f"{source_node_id}/cpu-0"
    destination_worker_id = f"{destination_node_id}/cpu-0"
    for identity, worker_id, endpoint in (
        (source_identity, source_worker_id, "10.0.0.2:50001"),
        (destination_identity, destination_worker_id, "10.0.0.3:50002"),
    ):
        state.save_membership(_membership(cluster, identity, joined_at=now))
        state.save_node(
            _metadata(
                identity,
                worker_id=worker_id,
                joined_at=now,
                probe_endpoint=endpoint,
            )
        )
    coordinator = NetworkProbeCoordinator(
        state=state,
        cluster=cluster,
        identity=coordinator_identity,
        clock_ns=lambda: now,
    )
    ticket = coordinator.issue(
        NetworkProbeControlRequest(
            authentication=_authentication(source_node_id, now),
            source_worker_id=source_worker_id,
            destination_worker_id=destination_worker_id,
            payload_sizes=[4096],
            sample_count=1,
            maximum_bytes=8192,
            timeout_ms=1000,
        ),
        requesting_node_id=source_node_id,
    )
    tampered = ticket.model_copy(update={"destination_endpoint": "10.0.0.4:50002"})
    with pytest.raises(IntegrityError, match="signature"):
        verify_probe_ticket(tampered, cluster, now_unix_ns=now)

    measurement = NetworkLinkMeasurement(
        source_worker_id=source_worker_id,
        destination_worker_id=destination_worker_id,
        measured_at_unix_ns=now,
        round_trip_latency_ms=1,
        upload_bytes_per_s=1000,
        download_bytes_per_s=1000,
        payload_sizes=[4096],
        sample_count=1,
        p95_transfer_ms=2,
        measured=True,
    )
    state.save_network_measurement(measurement)
    assert network_measurement_is_fresh(
        measurement,
        ttl_seconds=900,
        now_unix_ns=now + 900_000_000_000,
    )
    repository = NetworkMeasurementRepository(
        state=state,
        ttl_seconds=900,
        clock_ns=lambda: now + 901_000_000_000,
    )
    assert repository.get(source_worker_id, destination_worker_id) is None
    assert repository.fresh() == []
    assert repository.expire_stale(cluster_id=cluster.cluster_id) == [measurement]
