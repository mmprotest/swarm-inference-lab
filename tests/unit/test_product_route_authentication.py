from __future__ import annotations

import time
from dataclasses import dataclass

import pytest

from swarm_inference.exceptions import IntegrityError
from swarm_inference.model.partition import StageAssignment
from swarm_inference.protocol.routes import (
    BoundedNonceCache,
    PeerHandshake,
    RouteLeaseParticipant,
    SignedRouteLease,
    route_lease_hash,
    sign_peer_handshake,
    sign_route_lease,
    verify_peer_handshake,
    verify_route_lease,
    verify_worker_route_lease,
)
from swarm_inference.security.identity import CoordinatorIdentity, WorkerIdentity


def _assignment(stage_id: int) -> StageAssignment:
    return StageAssignment(
        stage_id=stage_id,
        layer_start=stage_id,
        layer_end=stage_id + 1,
        layer_ids=(stage_id,),
        weight_bytes=1024,
        estimated_compute_ns=1,
        measured_compute_ns=1,
        kv_cache_bytes_per_token=8,
        peak_temporary_bytes=64,
        activation_bytes=4,
        device="cpu",
        owns_embeddings=stage_id == 0,
        owns_final_norm=stage_id == 1,
        owns_output_projection=stage_id == 1,
    )


@dataclass(frozen=True)
class _RouteFixture:
    coordinator: CoordinatorIdentity
    workers: tuple[WorkerIdentity, WorkerIdentity]
    lease: SignedRouteLease


def _route_fixture(
    *,
    issued_unix_ns: int | None = None,
    expiry_unix_ns: int | None = None,
) -> _RouteFixture:
    now = time.time_ns()
    coordinator = CoordinatorIdentity.generate()
    workers = (WorkerIdentity.generate(), WorkerIdentity.generate())
    participants = [
        RouteLeaseParticipant(
            worker_id=f"worker-{stage_id}",
            worker_public_key=identity.public_key_b64,
            worker_public_key_fingerprint=identity.public_key_fingerprint,
            control_endpoint=f"127.0.0.1:{5000 + stage_id}",
            data_endpoint=f"127.0.0.1:{6000 + stage_id}",
            stage_id=stage_id,
            assignment=_assignment(stage_id),
            device="cpu",
            dtype="float32",
        )
        for stage_id, identity in enumerate(workers)
    ]
    unsigned = SignedRouteLease(
        topology_id="topology-auth",
        route_generation=3,
        model_id="test/olmoe",
        model_revision="model-commit",
        tokenizer_revision="tokenizer-commit",
        adapter_id="olmoe",
        dtype="float32",
        participants=participants,
        lease_issued_unix_ns=issued_unix_ns or now,
        lease_expiry_unix_ns=expiry_unix_ns or now + 60_000_000_000,
        nonce="route-nonce",
        coordinator_identity="coordinator-test",
        coordinator_public_key=coordinator.public_key_b64,
        coordinator_public_key_fingerprint=coordinator.public_key_fingerprint,
    )
    return _RouteFixture(
        coordinator=coordinator,
        workers=workers,
        lease=sign_route_lease(unsigned, coordinator),
    )


def _verify_worker(
    fixture: _RouteFixture,
    *,
    worker_id: str = "worker-0",
    control_endpoint: str = "127.0.0.1:5000",
    data_endpoint: str = "127.0.0.1:6000",
    model_revision: str = "model-commit",
    last_route_generation: int | None = None,
    nonce_cache: BoundedNonceCache | None = None,
) -> None:
    verify_worker_route_lease(
        fixture.lease,
        {"coordinator-test": fixture.coordinator.public_key_b64},
        worker_id=worker_id,
        worker_public_key=fixture.workers[0].public_key_b64,
        control_endpoint=control_endpoint,
        data_endpoint=data_endpoint,
        topology_id="topology-auth",
        route_generation=3,
        model_id="test/olmoe",
        model_revision=model_revision,
        tokenizer_revision="tokenizer-commit",
        assignment=_assignment(0),
        device="cpu",
        dtype="float32",
        last_route_generation=last_route_generation,
        nonce_cache=nonce_cache,
    )


def test_signed_route_creation_and_exact_worker_verification() -> None:
    fixture = _route_fixture()

    verify_route_lease(
        fixture.lease,
        {"coordinator-test": fixture.coordinator.public_key_b64},
    )
    _verify_worker(fixture)


def test_route_rejects_missing_or_invalid_signature() -> None:
    fixture = _route_fixture()
    trusted = {"coordinator-test": fixture.coordinator.public_key_b64}

    with pytest.raises(IntegrityError, match="signature is missing"):
        verify_route_lease(fixture.lease.model_copy(update={"signature": ""}), trusted)
    with pytest.raises(IntegrityError, match="invalid Ed25519 signature"):
        verify_route_lease(
            fixture.lease.model_copy(update={"model_revision": "tampered"}),
            trusted,
        )


def test_route_rejects_unknown_coordinator_and_expired_or_future_lease() -> None:
    fixture = _route_fixture()
    with pytest.raises(IntegrityError, match="unknown coordinator identity"):
        verify_route_lease(fixture.lease, {"another-coordinator": "unused"})

    now = time.time_ns()
    expired = _route_fixture(
        issued_unix_ns=now - 2_000_000_000,
        expiry_unix_ns=now - 1_000_000_000,
    )
    with pytest.raises(IntegrityError, match="expired"):
        verify_route_lease(
            expired.lease,
            {"coordinator-test": expired.coordinator.public_key_b64},
            now_unix_ns=now,
        )

    future = _route_fixture(
        issued_unix_ns=now + 31_000_000_000,
        expiry_unix_ns=now + 91_000_000_000,
    )
    with pytest.raises(IntegrityError, match="future-dated"):
        verify_route_lease(
            future.lease,
            {"coordinator-test": future.coordinator.public_key_b64},
            now_unix_ns=now,
            future_tolerance_ns=30_000_000_000,
        )


def test_route_rejects_stale_generation_and_nonce_replay() -> None:
    fixture = _route_fixture()
    with pytest.raises(IntegrityError, match="stale route generation"):
        _verify_worker(fixture, last_route_generation=3)

    cache = BoundedNonceCache(capacity=2)
    _verify_worker(fixture, nonce_cache=cache)
    with pytest.raises(IntegrityError, match="nonce was replayed"):
        _verify_worker(fixture, nonce_cache=cache)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"control_endpoint": "127.0.0.1:9999"}, "endpoint mismatch"),
        ({"data_endpoint": "127.0.0.1:9999"}, "endpoint mismatch"),
        ({"worker_id": "worker-1"}, "worker identity mismatch"),
        ({"model_revision": "wrong-revision"}, "model or topology identity mismatch"),
    ],
)
def test_route_rejects_identity_endpoint_and_model_mismatches(
    overrides: dict[str, str],
    error: str,
) -> None:
    fixture = _route_fixture()
    with pytest.raises(IntegrityError, match=error):
        _verify_worker(fixture, **overrides)


def _peer_handshake(fixture: _RouteFixture) -> PeerHandshake:
    unsigned = PeerHandshake(
        worker_id="worker-1",
        public_key_fingerprint=fixture.workers[1].public_key_fingerprint,
        topology_id=fixture.lease.topology_id,
        route_generation=fixture.lease.route_generation,
        stage_id=1,
        peer_stage_id=0,
        model_revision=fixture.lease.model_revision,
        nonce="peer-nonce",
        timestamp_unix_ns=time.time_ns(),
        route_lease_hash=route_lease_hash(fixture.lease),
    )
    return sign_peer_handshake(unsigned, fixture.workers[1])


def test_peer_handshake_authenticates_installed_peer_and_rejects_mismatch() -> None:
    fixture = _route_fixture()
    handshake = _peer_handshake(fixture)

    verify_peer_handshake(
        handshake,
        fixture.lease,
        expected_worker_id="worker-1",
        expected_stage_id=1,
        expected_peer_stage_id=0,
    )
    with pytest.raises(IntegrityError, match="identity mismatch"):
        verify_peer_handshake(
            handshake,
            fixture.lease,
            expected_worker_id="worker-0",
            expected_stage_id=0,
            expected_peer_stage_id=1,
        )


def test_peer_handshake_rejects_replayed_nonce_and_wrong_lease_hash() -> None:
    fixture = _route_fixture()
    handshake = _peer_handshake(fixture)
    cache = BoundedNonceCache(capacity=2)

    verify_peer_handshake(
        handshake,
        fixture.lease,
        expected_worker_id="worker-1",
        expected_stage_id=1,
        expected_peer_stage_id=0,
        nonce_cache=cache,
    )
    with pytest.raises(IntegrityError, match="nonce was replayed"):
        verify_peer_handshake(
            handshake,
            fixture.lease,
            expected_worker_id="worker-1",
            expected_stage_id=1,
            expected_peer_stage_id=0,
            nonce_cache=cache,
        )

    wrong_hash = sign_peer_handshake(
        handshake.model_copy(update={"route_lease_hash": "0" * 64, "signature": ""}),
        fixture.workers[1],
    )
    with pytest.raises(IntegrityError, match="route identity mismatch"):
        verify_peer_handshake(
            wrong_hash,
            fixture.lease,
            expected_worker_id="worker-1",
            expected_stage_id=1,
            expected_peer_stage_id=0,
        )
