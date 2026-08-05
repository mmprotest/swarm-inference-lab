from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

import swarm_inference.cluster.pairing as pairing_module
from swarm_inference.cluster.models import (
    ClusterMetadata,
    NodeMembership,
    NodeMetadata,
    VersionCompatibility,
    node_id_from_fingerprint,
)
from swarm_inference.cluster.pairing import (
    PairingClient,
    PairingInvitation,
    PairingManager,
    create_cluster_authentication,
)
from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.exceptions import PairingError, PairingExpiredError
from swarm_inference.protocol.cluster import ClusterRevokeRequest
from swarm_inference.security.identity import CoordinatorIdentity, WorkerIdentity
from swarm_inference.security.trust_store import WorkerTrustStore


class _Clock:
    def __init__(self) -> None:
        self.nanoseconds = 1_000_000_000
        self.seconds = 1.0

    def time_ns(self) -> int:
        return self.nanoseconds

    def monotonic(self) -> float:
        return self.seconds

    def advance(self, seconds: float) -> None:
        self.seconds += seconds
        self.nanoseconds += int(seconds * 1e9)


def _cluster(identity: CoordinatorIdentity) -> ClusterMetadata:
    return ClusterMetadata(
        cluster_id="cluster-pairing",
        name="pairing-test",
        coordinator_id=node_id_from_fingerprint(identity.public_key_fingerprint),
        coordinator_endpoint="10.0.0.1:50051",
        coordinator_public_key=identity.public_key_b64,
        coordinator_fingerprint=identity.public_key_fingerprint,
        created_at_unix_ns=1,
        runtime_compatibility=VersionCompatibility(
            minimum_runtime_version="0.1.0",
            maximum_runtime_version_exclusive="0.2.0",
        ),
    )


def _node(identity: WorkerIdentity, *, joined_at: int = 1) -> NodeMetadata:
    node_id = node_id_from_fingerprint(identity.public_key_fingerprint)
    return NodeMetadata(
        node_id=node_id,
        public_key=identity.public_key_b64,
        fingerprint=identity.public_key_fingerprint,
        hostname="joined-node",
        operating_system="Windows 11",
        architecture="AMD64",
        agent_version="0.1.0",
        runtime_version="0.1.0",
        build_id="test-build",
        package_lock_hash="3" * 64,
        worker_ids=[],
        joined_at_unix_ns=joined_at,
        last_seen_at_unix_ns=joined_at,
    )


def _manager(
    path: Path,
    coordinator: CoordinatorIdentity,
    clock: _Clock,
) -> PairingManager:
    state = ClusterStateStore(path, clock_ns=clock.time_ns)
    cluster = _cluster(coordinator)
    state.save_cluster(cluster)
    return PairingManager(
        state=state,
        trust_store=WorkerTrustStore(state.paths.security / "trusted-workers.json"),
        coordinator_identity=coordinator,
        cluster=cluster,
        clock_ns=clock.time_ns,
        monotonic=clock.monotonic,
    )


async def _join(
    manager: PairingManager,
    client_state: ClusterStateStore,
    identity: WorkerIdentity,
    uri: str,
) -> tuple[PairingClient, object]:
    client = PairingClient(state=client_state, identity=identity)
    result = await client.join(
        uri,
        node_metadata=_node(identity),
        hello_rpc=lambda hello: manager.begin(hello, source_address="10.0.0.2"),
        complete_rpc=lambda request: manager.complete(request, source_address="10.0.0.2"),
    )
    return client, result


@pytest.mark.asyncio
async def test_pairing_success_persists_trust_and_pins_cluster(tmp_path: Path) -> None:
    clock = _Clock()
    coordinator = CoordinatorIdentity.generate()
    manager = _manager(tmp_path / "coordinator", coordinator, clock)
    identity = WorkerIdentity.generate()
    client_state = ClusterStateStore(tmp_path / "node")
    invitation = await manager.create_session("10.0.0.1:50051")

    _, result = await _join(manager, client_state, identity, invitation.uri())
    node_id = node_id_from_fingerprint(identity.public_key_fingerprint)
    assert result.membership.node_id == node_id
    assert manager.trust_store.contains(identity.public_key_fingerprint)
    assert manager.state.membership(node_id).status == "active"
    assert client_state.load_cluster().coordinator_fingerprint == coordinator.public_key_fingerprint
    assert client_state.membership(node_id).coordinator_fingerprint == (
        coordinator.public_key_fingerprint
    )


@pytest.mark.asyncio
async def test_wrong_secret_is_rejected_without_trust(tmp_path: Path) -> None:
    clock = _Clock()
    coordinator = CoordinatorIdentity.generate()
    manager = _manager(tmp_path / "coordinator", coordinator, clock)
    identity = WorkerIdentity.generate()
    invitation = await manager.create_session("10.0.0.1:50051")
    wrong = PairingInvitation(
        coordinator_endpoint=invitation.coordinator_endpoint,
        session_id=invitation.session_id,
        pairing_secret=b"x" * 32,
        coordinator_ephemeral_public_key=invitation.coordinator_ephemeral_public_key,
    )

    with pytest.raises(PairingError, match="secret or encrypted transcript"):
        await _join(manager, ClusterStateStore(tmp_path / "node"), identity, wrong.uri())
    assert not manager.trust_store.contains(identity.public_key_fingerprint)


@pytest.mark.asyncio
async def test_expired_session_and_consumed_replay_are_rejected(tmp_path: Path) -> None:
    clock = _Clock()
    coordinator = CoordinatorIdentity.generate()
    manager = _manager(tmp_path / "coordinator", coordinator, clock)
    identity = WorkerIdentity.generate()
    expired = await manager.create_session("10.0.0.1:50051", ttl_seconds=1)
    clock.advance(2)
    with pytest.raises(PairingExpiredError):
        await _join(manager, ClusterStateStore(tmp_path / "expired-node"), identity, expired.uri())

    invitation = await manager.create_session("10.0.0.1:50051")
    captured_request = None

    async def complete(request):
        nonlocal captured_request
        captured_request = request
        return await manager.complete(request, source_address="10.0.0.2")

    client = PairingClient(
        state=ClusterStateStore(tmp_path / "node"),
        identity=identity,
    )
    await client.join(
        invitation.uri(),
        node_metadata=_node(identity),
        hello_rpc=lambda hello: manager.begin(hello, source_address="10.0.0.2"),
        complete_rpc=complete,
    )
    assert captured_request is not None
    with pytest.raises(PairingError, match="already consumed"):
        await manager.complete(captured_request, source_address="10.0.0.2")


@pytest.mark.asyncio
async def test_transcript_tampering_is_rejected(tmp_path: Path) -> None:
    clock = _Clock()
    coordinator = CoordinatorIdentity.generate()
    manager = _manager(tmp_path / "coordinator", coordinator, clock)
    identity = WorkerIdentity.generate()
    invitation = await manager.create_session("10.0.0.1:50051")

    async def tampered_hello(hello):
        challenge = await manager.begin(hello, source_address="10.0.0.2")
        return challenge.model_copy(update={"server_nonce": "A" * 24})

    client = PairingClient(state=ClusterStateStore(tmp_path / "node"), identity=identity)
    with pytest.raises(PairingError):
        await client.join(
            invitation.uri(),
            node_metadata=_node(identity),
            hello_rpc=tampered_hello,
            complete_rpc=lambda request: manager.complete(request, source_address="10.0.0.2"),
        )


@pytest.mark.asyncio
async def test_coordinator_signature_failure_is_rejected(tmp_path: Path) -> None:
    clock = _Clock()
    coordinator = CoordinatorIdentity.generate()
    manager = _manager(tmp_path / "coordinator", coordinator, clock)
    identity = WorkerIdentity.generate()
    invitation = await manager.create_session("10.0.0.1:50051")

    async def bad_signature(hello):
        challenge = await manager.begin(hello, source_address="10.0.0.2")
        live = manager._sessions[hello.session_id]
        handshake = live.handshake
        assert handshake is not None
        raw = pairing_module._decrypt_json(
            handshake.session_key,
            challenge.encrypted_payload,
            nonce=challenge.encryption_nonce,
            aad=pairing_module._aad(handshake.transcript, phase="challenge"),
        )
        raw["coordinator_signature"] = WorkerIdentity.generate().sign(b"wrong")
        ciphertext = pairing_module._encrypt_json(
            handshake.session_key,
            raw,
            nonce=pairing_module._unb64(challenge.encryption_nonce),
            aad=pairing_module._aad(handshake.transcript, phase="challenge"),
        )
        return challenge.model_copy(update={"encrypted_payload": ciphertext})

    client = PairingClient(state=ClusterStateStore(tmp_path / "node"), identity=identity)
    with pytest.raises(PairingError, match="coordinator identity signature"):
        await client.join(
            invitation.uri(),
            node_metadata=_node(identity),
            hello_rpc=bad_signature,
            complete_rpc=lambda request: manager.complete(request, source_address="10.0.0.2"),
        )


@pytest.mark.asyncio
async def test_node_signature_failure_is_rejected(tmp_path: Path) -> None:
    clock = _Clock()
    coordinator = CoordinatorIdentity.generate()
    manager = _manager(tmp_path / "coordinator", coordinator, clock)
    identity = WorkerIdentity.generate()
    invitation = await manager.create_session("10.0.0.1:50051")

    async def bad_completion(request):
        live = manager._sessions[request.session_id]
        handshake = live.handshake
        assert handshake is not None
        raw = pairing_module._decrypt_json(
            handshake.session_key,
            request.encrypted_payload,
            nonce=request.encryption_nonce,
            aad=pairing_module._aad(handshake.transcript, phase="node-completion"),
        )
        raw["node_signature"] = WorkerIdentity.generate().sign(b"wrong")
        ciphertext = pairing_module._encrypt_json(
            handshake.session_key,
            raw,
            nonce=pairing_module._unb64(request.encryption_nonce),
            aad=pairing_module._aad(handshake.transcript, phase="node-completion"),
        )
        return await manager.complete(
            request.model_copy(update={"encrypted_payload": ciphertext}),
            source_address="10.0.0.2",
        )

    client = PairingClient(state=ClusterStateStore(tmp_path / "node"), identity=identity)
    with pytest.raises(PairingError, match="node identity signature"):
        await client.join(
            invitation.uri(),
            node_metadata=_node(identity),
            hello_rpc=lambda hello: manager.begin(hello, source_address="10.0.0.2"),
            complete_rpc=bad_completion,
        )


@pytest.mark.asyncio
async def test_pairing_secret_is_redacted_from_repr_audit_and_status(tmp_path: Path) -> None:
    clock = _Clock()
    coordinator = CoordinatorIdentity.generate()
    manager = _manager(tmp_path / "coordinator", coordinator, clock)
    invitation = await manager.create_session("10.0.0.1:50051")
    secret_text = parse_qs(urlsplit(invitation.uri()).query)["secret"][0]
    assert secret_text not in repr(invitation)
    assert secret_text not in invitation.redacted_uri()
    assert secret_text not in manager.state.paths.audit_log.read_text(encoding="utf-8")
    persisted = manager.state.paths.pairing_sessions.read_text(encoding="utf-8")
    assert secret_text not in persisted
    assert "pairing_secret" not in persisted


@pytest.mark.asyncio
async def test_manual_trust_survives_pairing_and_revocation_blocks_node(tmp_path: Path) -> None:
    clock = _Clock()
    coordinator = CoordinatorIdentity.generate()
    manager = _manager(tmp_path / "coordinator", coordinator, clock)
    manual = WorkerIdentity.generate()
    manager.trust_store.trust(manual.public_key_fingerprint, label="manual")

    coordinator_node = _node(coordinator)
    coordinator_membership = NodeMembership(
        cluster_id=manager.cluster.cluster_id,
        node_id=coordinator_node.node_id,
        node_public_key=coordinator.public_key_b64,
        node_fingerprint=coordinator.public_key_fingerprint,
        coordinator_public_key=coordinator.public_key_b64,
        coordinator_fingerprint=coordinator.public_key_fingerprint,
        joined_at_unix_ns=clock.time_ns(),
    )
    manager.state.save_node(coordinator_node)
    manager.state.save_membership(coordinator_membership)

    joined = WorkerIdentity.generate()
    invitation = await manager.create_session("10.0.0.1:50051")
    await _join(manager, ClusterStateStore(tmp_path / "node"), joined, invitation.uri())
    target = node_id_from_fingerprint(joined.public_key_fingerprint)
    body = {"node_id": target, "reason": "test revocation"}
    auth = create_cluster_authentication(
        identity=coordinator,
        node_id=coordinator_node.node_id,
        action="cluster-revoke",
        body=body,
        timestamp_unix_ns=clock.time_ns(),
    )
    response = await manager.revoke(
        ClusterRevokeRequest(
            authentication=auth,
            node_id=target,
            reason="test revocation",
        )
    )
    assert response.revoked
    assert not manager.trust_store.contains(joined.public_key_fingerprint)
    assert manager.state.membership(target).status == "revoked"
    assert manager.trust_store.contains(manual.public_key_fingerprint)
