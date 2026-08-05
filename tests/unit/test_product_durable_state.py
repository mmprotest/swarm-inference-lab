from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from swarm_inference.config.models import Backend, WorkerCapability
from swarm_inference.config.product import ProductCoordinatorConfig
from swarm_inference.coordinator.durable_state import DurableCoordinatorState
from swarm_inference.coordinator.registry import WorkerRegistry
from swarm_inference.coordinator.service import CoordinatorCore
from swarm_inference.exceptions import IntegrityError
from swarm_inference.model.partition import StageAssignment
from swarm_inference.protocol.messages import RegistrationRequest, SubmitRequest
from swarm_inference.protocol.product import (
    PlanWorkerAssignment,
    ProductRequestPhase,
    ProductRequestRecoveryState,
    SessionsRequest,
)
from swarm_inference.security.identity import WorkerIdentity
from swarm_inference.security.signatures import canonical_json_bytes


def _assignment() -> StageAssignment:
    return StageAssignment(
        stage_id=0,
        layer_start=0,
        layer_end=1,
        layer_ids=(0,),
        weight_bytes=1024,
        estimated_compute_ns=1,
        measured_compute_ns=1,
        kv_cache_bytes_per_token=8,
        peak_temporary_bytes=64,
        activation_bytes=4,
        device="cpu",
        owns_embeddings=True,
        owns_final_norm=True,
        owns_output_projection=True,
    )


def _capability(identity: WorkerIdentity) -> WorkerCapability:
    return WorkerCapability(
        worker_id="worker-durable",
        public_key=identity.public_key_b64,
        hostname="localhost",
        operating_system="test",
        architecture="test",
        backend=Backend.TORCH_CPU,
        cpu_model="test",
        logical_cpu_count=1,
        total_ram_bytes=1024**3,
        available_ram_bytes=1024**3,
        supported_dtypes=["float32"],
        upload_bandwidth_bytes_s=1_000_000,
        download_bandwidth_bytes_s=1_000_000,
        coordinator_latency_ms=0.1,
        memory_limit_bytes=1024**3,
        endpoint="127.0.0.1:5001",
        control_endpoint="127.0.0.1:5001",
        data_plane_endpoint="127.0.0.1:6001",
        device_identifier="cpu",
        stage_runtime_enabled=True,
        last_heartbeat=datetime.now(UTC),
    )


def _recovery_state(*, accepted: list[int]) -> ProductRequestRecoveryState:
    now = time.time_ns()
    assignment = PlanWorkerAssignment(
        stage_id=0,
        worker_id="worker-durable",
        control_endpoint="127.0.0.1:5001",
        data_endpoint="127.0.0.1:6001",
        device="cpu",
        effective_memory_bytes=1024**3,
        required_memory_bytes=2048,
        assignment=_assignment(),
    )
    return ProductRequestRecoveryState(
        request_id="request-durable",
        request_generation=1,
        session_id="session-before-restart",
        model_id="test/olmoe",
        model_revision="model-commit",
        tokenizer_revision="tokenizer-commit",
        topology_id="topology-durable",
        route_generation=1,
        prompt_token_ids=[10, 20],
        accepted_generated_token_ids=accepted,
        next_token_position=len(accepted),
        active_workers=["worker-durable"],
        stage_assignments=[assignment],
        last_healthy_checkpoint=len(accepted),
        status=ProductRequestPhase.RUNNING,
        started_unix_ns=now,
        updated_unix_ns=now,
    )


def _registration(
    capability: WorkerCapability,
    identity: WorkerIdentity,
    *,
    nonce: str,
) -> RegistrationRequest:
    payload = canonical_json_bytes(
        {
            "capability": capability.model_dump(mode="json"),
            "benchmark_nonce": nonce,
        }
    )
    return RegistrationRequest(
        capability=capability,
        benchmark_nonce=nonce,
        signature=identity.sign(payload),
    )


@pytest.mark.asyncio
async def test_coordinator_restart_reloads_validated_state_without_live_workers(
    tmp_path: Path,
) -> None:
    config = ProductCoordinatorConfig(
        worker_heartbeat_timeout_s=60,
        require_trusted_workers=False,
    )
    first = CoordinatorCore(product_config=config, state_directory=tmp_path)
    worker_identity = WorkerIdentity.generate()
    capability = _capability(worker_identity)
    nonce = "registration-nonce"
    try:
        response = await first.register(_registration(capability, worker_identity, nonce=nonce))
        assert response.accepted
        assert first.coordinator_identity is not None
        fingerprint = first.coordinator_identity.public_key_fingerprint
        assert first.durable_state is not None
        first.durable_state.append_replay_token(
            request_id="request-durable",
            request_generation=1,
            route_generation=1,
            token_position=0,
            token_id=21,
        )
        first.durable_state.save_request(_recovery_state(accepted=[21]))
    finally:
        await first.close()

    second = CoordinatorCore(product_config=config, state_directory=tmp_path)
    try:
        assert second.coordinator_identity is not None
        assert second.coordinator_identity.public_key_fingerprint == fingerprint
        assert second.registry.workers() == []
        assert second.durable_state is not None
        known = second.durable_state.known_workers()
        assert [item.worker_id for item in known] == ["worker-durable"]

        sessions = await second.product_sessions(SessionsRequest(include_terminal=True))
        assert len(sessions.sessions) == 1
        recovered = sessions.sessions[0]
        assert recovered.status == ProductRequestPhase.RECOVERABLE
        assert recovered.accepted_token_ids == [21]
        assert recovered.token_position == 1
        with pytest.raises(ValueError, match="duplicate or replayed request ID"):
            assert second.session_controller is not None
            second.session_controller.start(
                SubmitRequest(
                    request_id="request-durable",
                    prompt_token_ids=[10, 20],
                    max_new_tokens=1,
                    random_seed=1,
                    model_id="test/olmoe",
                    model_revision="model-commit",
                )
            )
        assert (tmp_path / "audit.jsonl").is_file()
        assert (tmp_path / "coordinator-identity.json").is_file()
    finally:
        await second.close()


def test_coordinator_restart_rejects_persisted_identity_metadata_mismatch(
    tmp_path: Path,
) -> None:
    original = CoordinatorCore(
        product_config=ProductCoordinatorConfig(
            coordinator_id="coordinator-original",
            require_trusted_workers=False,
        ),
        state_directory=tmp_path,
    )
    asyncio.run(original.close())

    with pytest.raises(IntegrityError, match="durable coordinator metadata"):
        CoordinatorCore(
            product_config=ProductCoordinatorConfig(
                coordinator_id="coordinator-different",
                require_trusted_workers=False,
            ),
            state_directory=tmp_path,
        )


def test_durable_worker_key_continuity_and_replay_log_mismatch_fail_closed(
    tmp_path: Path,
) -> None:
    durable = DurableCoordinatorState(tmp_path)
    first_identity = WorkerIdentity.generate()
    durable.save_worker(_capability(first_identity))
    with pytest.raises(IntegrityError, match="different identity"):
        durable.save_worker(_capability(WorkerIdentity.generate()))

    durable.save_request(_recovery_state(accepted=[21]))
    reloaded = durable.mark_restart_boundaries()["request-durable"]
    assert reloaded.status == ProductRequestPhase.FAILED
    assert "replay log" in (reloaded.last_error or "")


def test_failed_or_heartbeat_expired_worker_must_reregister_before_replacement() -> None:
    identity = WorkerIdentity.generate()
    capability = _capability(identity)
    registry = WorkerRegistry(heartbeat_timeout_s=1)
    registry.register(capability, benchmark_verified=True, now=10)
    assert [worker.worker_id for worker in registry.healthy_workers(now=10.5)] == ["worker-durable"]

    assert registry.expire(now=11.1) == ["worker-durable"]
    registry.heartbeat("worker-durable", queue_depth=0, assignments=[], now=11.2)
    assert registry.healthy_workers(now=11.2) == []

    registry.register(capability, benchmark_verified=True, now=12)
    registry.mark_unhealthy("worker-durable")
    registry.heartbeat("worker-durable", queue_depth=0, assignments=[], now=12.1)
    assert registry.healthy_workers(now=12.1) == []

    registry.register(capability, benchmark_verified=True, now=13)
    assert [worker.worker_id for worker in registry.healthy_workers(now=13)] == ["worker-durable"]


@pytest.mark.asyncio
async def test_product_worker_registration_requires_configured_identity_trust(
    tmp_path: Path,
) -> None:
    identity = WorkerIdentity.generate()
    capability = _capability(identity)
    rejected = CoordinatorCore(
        product_config=ProductCoordinatorConfig(
            require_trusted_workers=True,
            trusted_worker_fingerprints=["0" * 64],
        ),
        state_directory=tmp_path / "rejected",
    )
    try:
        with pytest.raises(IntegrityError, match="is not trusted"):
            await rejected.register(_registration(capability, identity, nonce="reject"))
    finally:
        await rejected.close()

    accepted = CoordinatorCore(
        product_config=ProductCoordinatorConfig(
            require_trusted_workers=True,
            trusted_worker_fingerprints=[identity.public_key_fingerprint],
        ),
        state_directory=tmp_path / "accepted",
    )
    try:
        response = await accepted.register(_registration(capability, identity, nonce="accept"))
        assert response.accepted
        assert response.coordinator_public_key_fingerprint is not None
        with pytest.raises(IntegrityError, match="nonce was replayed"):
            await accepted.register(_registration(capability, identity, nonce="accept"))
    finally:
        await accepted.close()
