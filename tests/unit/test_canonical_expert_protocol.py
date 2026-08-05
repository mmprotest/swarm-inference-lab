from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest
import torch

from swarm_inference.exceptions import IntegrityError
from swarm_inference.execution.expert import ExpertStore, deterministic_expert
from swarm_inference.execution.moe import WholeExpertRemoteBackend, WholeExpertTarget
from swarm_inference.protocol.expert import (
    ExpertPeerHandshake,
    ExpertProtocolVersion,
    ExpertRouteParticipant,
    SignedExpertRouteLease,
    TransportCodec,
    expert_route_lease_hash,
    negotiate_expert_protocol,
    sign_expert_peer_handshake,
    sign_expert_route_lease,
    verify_expert_peer_handshake,
    verify_expert_route_lease,
)
from swarm_inference.protocol.routes import BoundedNonceCache
from swarm_inference.security.identity import CoordinatorIdentity, WorkerIdentity
from swarm_inference.transport import expert as expert_transport
from swarm_inference.transport.expert import ExpertTransportClient, decode_array, encode_array
from swarm_inference.worker.expert_service import ExpertWorkerRuntime, ExpertWorkerServer


def _signed_lease(
    *,
    coordinator: CoordinatorIdentity,
    stage: WorkerIdentity,
    expert: WorkerIdentity,
    expert_endpoint: str = "127.0.0.1:50054",
    route_generation: int = 4,
    issued_ns: int | None = None,
    expiry_ns: int | None = None,
) -> SignedExpertRouteLease:
    now = time.time_ns()
    unsigned = SignedExpertRouteLease(
        topology_id="expert-topology",
        route_generation=route_generation,
        model_id="test/olmoe",
        model_revision="model-revision",
        model_fingerprint="model-fingerprint",
        quantization_fingerprint="quantization-fingerprint",
        participants=[
            ExpertRouteParticipant(
                worker_id="stage-worker",
                worker_public_key=stage.public_key_b64,
                worker_public_key_fingerprint=stage.public_key_fingerprint,
                endpoint="127.0.0.1:50053",
                roles=["contiguous-stage"],
                model_fingerprint="model-fingerprint",
                quantization_fingerprint="quantization-fingerprint",
            ),
            ExpertRouteParticipant(
                worker_id="expert-worker",
                worker_public_key=expert.public_key_b64,
                worker_public_key_fingerprint=expert.public_key_fingerprint,
                endpoint=expert_endpoint,
                roles=["whole-expert"],
                owned_experts={0: [0]},
                model_fingerprint="model-fingerprint",
                quantization_fingerprint="quantization-fingerprint",
            ),
        ],
        lease_issued_unix_ns=issued_ns or now,
        lease_expiry_unix_ns=expiry_ns or now + 60_000_000_000,
        nonce=f"route-{route_generation}",
        coordinator_identity="coordinator-test",
        coordinator_public_key=coordinator.public_key_b64,
        coordinator_public_key_fingerprint=coordinator.public_key_fingerprint,
    )
    return sign_expert_route_lease(unsigned, coordinator)


def _signed_handshake(
    *,
    lease: SignedExpertRouteLease,
    stage: WorkerIdentity,
    nonce: str = "peer-nonce",
) -> ExpertPeerHandshake:
    unsigned = ExpertPeerHandshake(
        protocol_versions=[ExpertProtocolVersion.V1],
        selected_version=ExpertProtocolVersion.V1,
        worker_id="stage-worker",
        public_key_fingerprint=stage.public_key_fingerprint,
        topology_id=lease.topology_id,
        route_generation=lease.route_generation,
        peer_worker_id="expert-worker",
        model_revision=lease.model_revision,
        quantization_fingerprint=lease.quantization_fingerprint,
        nonce=nonce,
        timestamp_unix_ns=time.time_ns(),
        route_lease_hash=expert_route_lease_hash(lease),
    )
    return sign_expert_peer_handshake(unsigned, stage)


def test_expert_protocol_negotiation_route_staleness_and_replay() -> None:
    coordinator = CoordinatorIdentity.generate()
    stage = WorkerIdentity.generate()
    expert = WorkerIdentity.generate()
    lease = _signed_lease(coordinator=coordinator, stage=stage, expert=expert)
    trusted = {"coordinator-test": coordinator.public_key_b64}
    assert negotiate_expert_protocol(["1.0"]) == ExpertProtocolVersion.V1
    verify_expert_route_lease(lease, trusted)
    with pytest.raises(IntegrityError, match="stale route generation"):
        verify_expert_route_lease(lease, trusted, last_route_generation=4)

    route_nonces = BoundedNonceCache(capacity=2)
    verify_expert_route_lease(lease, trusted, nonce_cache=route_nonces)
    with pytest.raises(IntegrityError, match="replayed"):
        verify_expert_route_lease(lease, trusted, nonce_cache=route_nonces)

    handshake = _signed_handshake(lease=lease, stage=stage)
    peer_nonces = BoundedNonceCache(capacity=2)
    assert (
        verify_expert_peer_handshake(
            handshake,
            lease,
            expected_worker_id="stage-worker",
            expected_peer_worker_id="expert-worker",
            nonce_cache=peer_nonces,
        )
        == ExpertProtocolVersion.V1
    )
    with pytest.raises(IntegrityError, match="replayed"):
        verify_expert_peer_handshake(
            handshake,
            lease,
            expected_worker_id="stage-worker",
            expected_peer_worker_id="expert-worker",
            nonce_cache=peer_nonces,
        )


def test_expert_protocol_rejects_expired_and_tampered_leases() -> None:
    coordinator = CoordinatorIdentity.generate()
    stage = WorkerIdentity.generate()
    expert = WorkerIdentity.generate()
    now = time.time_ns()
    expired = _signed_lease(
        coordinator=coordinator,
        stage=stage,
        expert=expert,
        issued_ns=now - 2_000_000,
        expiry_ns=now - 1_000_000,
    )
    with pytest.raises(IntegrityError, match="expired"):
        verify_expert_route_lease(
            expired,
            {"coordinator-test": coordinator.public_key_b64},
            now_unix_ns=now,
        )
    valid = _signed_lease(coordinator=coordinator, stage=stage, expert=expert)
    with pytest.raises(IntegrityError, match="invalid Ed25519 signature"):
        verify_expert_route_lease(
            valid.model_copy(update={"model_revision": "tampered"}),
            {"coordinator-test": coordinator.public_key_b64},
        )


def test_expert_codec_validates_geometry_and_global_frame_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = np.arange(8, dtype=np.float32).reshape(2, 4)
    encoded = encode_array(source, name="activation", codec=TransportCodec.RAW_FP32)
    with pytest.raises(ValueError, match="raw byte count"):
        decode_array(
            encoded.metadata.model_copy(update={"raw_bytes": encoded.metadata.raw_bytes + 4}),
            encoded.payload,
        )
    monkeypatch.setattr(expert_transport, "MAX_FRAME_BYTES", 16)
    with pytest.raises(ValueError, match="exceeds"):
        expert_transport.decode_packet(b"x" * 17)


@pytest.mark.asyncio
async def test_authenticated_tcp_whole_expert_result_is_verified_and_consumed() -> None:
    coordinator = CoordinatorIdentity.generate()
    stage_identity = WorkerIdentity.generate()
    expert_identity = WorkerIdentity.generate()
    weights = deterministic_expert(latent_dimension=4, intermediate_dimension=8, seed=44)
    store = ExpertStore(
        owned={(0, 0)},
        loader=lambda _layer, _expert: weights,
        residency_budget_bytes=weights.byte_size,
        cache_budget_bytes=weights.byte_size,
    )
    runtime = ExpertWorkerRuntime(
        worker_id="expert-worker",
        identity=expert_identity,
        model_id="test/olmoe",
        model_revision="model-revision",
        model_fingerprint="model-fingerprint",
        quantization_fingerprint="quantization-fingerprint",
        store=store,
        require_authenticated_routes=True,
        trusted_coordinators={"coordinator-test": coordinator.public_key_b64},
    )
    server = ExpertWorkerServer(runtime, host="127.0.0.1", port=0)
    host, port = await server.start()
    lease = _signed_lease(
        coordinator=coordinator,
        stage=stage_identity,
        expert=expert_identity,
        expert_endpoint=f"{host}:{port}",
    )
    runtime.install_route(lease)
    backend = WholeExpertRemoteBackend(
        targets={
            (0, 0): WholeExpertTarget(
                worker_id="expert-worker",
                client=ExpertTransportClient(f"{host}:{port}", timeout_s=5),
                expert_hash=weights.content_hash,
            )
        },
        model_id="test/olmoe",
        model_revision="model-revision",
        model_fingerprint="model-fingerprint",
        quantization_fingerprint="quantization-fingerprint",
        topology_id="expert-topology",
        route_generation=4,
    )
    backend.configure_route(lease, identity=stage_identity, worker_id="stage-worker")
    backend.open_session("session")
    try:
        call = {
            "session_id": "session",
            "request_id": "product-request",
            "token_position": 0,
            "layer_id": 0,
            "hidden_states": torch.ones((1, 1, 4), dtype=torch.float32),
            "router_logits": torch.ones((1, 1), dtype=torch.float32),
            "selected_experts": torch.zeros((1, 1), dtype=torch.long),
            "routing_weights": torch.ones((1, 1), dtype=torch.float32),
            "deadline_ns": time.time_ns() + 5_000_000_000,
        }
        result = await asyncio.to_thread(
            backend.execute_layer,
            **call,
        )
        duplicate = await asyncio.to_thread(
            backend.execute_layer,
            **call,
        )
        torch.testing.assert_close(duplicate.output, result.output, rtol=0, atol=0)
        assert result.events[0].event == "remote_whole_expert_result_consumed"
        assert result.events[0].worker_ids == ("expert-worker",)
        assert result.events[0].request_bytes > 0
        assert result.events[0].response_bytes > 0
        assert result.events[0].result_hash.startswith("sha256:")
        assert runtime.status()["remote_whole_expert_calls"] == 1
        assert runtime.status()["duplicate_requests"] == 1

        renewed_lease = _signed_lease(
            coordinator=coordinator,
            stage=stage_identity,
            expert=expert_identity,
            expert_endpoint=f"{host}:{port}",
            route_generation=5,
        )
        runtime.install_route(renewed_lease)
        backend.configure_route(
            renewed_lease,
            identity=stage_identity,
            worker_id="stage-worker",
        )
        renewed = await asyncio.to_thread(
            backend.execute_layer,
            **{
                **call,
                "request_id": "renewed-product-request",
                "token_position": 1,
                "deadline_ns": time.time_ns() + 5_000_000_000,
            },
        )
        assert renewed.events[0].token_position == 1
        assert backend.route_generation == 5
        assert runtime.status()["route_generation"] == 5
        assert runtime.status()["remote_whole_expert_calls"] == 2
        with pytest.raises(RuntimeError, match="missing peer authentication"):
            await asyncio.to_thread(
                ExpertTransportClient(f"{host}:{port}", timeout_s=5).control,
                "cancel_session",
                session_id="unauthenticated-session",
            )
        await asyncio.to_thread(backend.cancel_session, "session")
        assert "session" in runtime._cancelled_session_ids
    finally:
        backend.close()
        await server.close()
