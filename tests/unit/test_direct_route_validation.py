from __future__ import annotations

import os
import time

import pytest

from swarm_inference.config.models import (
    Backend,
    DataPlaneMode,
    OperationKind,
    QueueConfig,
    SyntheticModelConfig,
    WorkerCapability,
)
from swarm_inference.exceptions import RouteMessageError
from swarm_inference.model.synthetic import synthetic_activation
from swarm_inference.protocol.checksums import sha256_bytes
from swarm_inference.protocol.messages import (
    ActivationMetadata,
    DataPlaneEnvelope,
    RouteHop,
    RoutePlan,
    StageAssignmentMessage,
)
from swarm_inference.protocol.routes import (
    encode_route_key,
    sign_data_envelope,
    sign_route_plan,
)
from swarm_inference.protocol.tensor_codec import ActivationTensor, encode_tensor
from swarm_inference.security.identity import WorkerIdentity
from swarm_inference.simulation.model import build_synthetic_stages
from swarm_inference.worker.agent import WorkerAgent


def _capability(identity: WorkerIdentity) -> WorkerCapability:
    return WorkerCapability(
        worker_id="worker-0",
        public_key=identity.public_key_b64,
        hostname="localhost",
        operating_system="test",
        architecture="test",
        backend=Backend.SYNTHETIC,
        cpu_model="test",
        logical_cpu_count=1,
        total_ram_bytes=2**30,
        available_ram_bytes=2**30,
        supported_dtypes=["float32"],
        upload_bandwidth_bytes_s=1e9,
        download_bandwidth_bytes_s=1e9,
        coordinator_latency_ms=0.1,
        memory_limit_bytes=2**29,
        endpoint="127.0.0.1:50052",
    )


def _configured_agent() -> tuple[
    WorkerAgent,
    bytes,
    RoutePlan,
    DataPlaneEnvelope,
]:
    identity = WorkerIdentity.generate()
    agent = WorkerAgent(
        capability=_capability(identity),
        identity=identity,
        queue_config=QueueConfig(capacity=4),
    )
    model = SyntheticModelConfig(
        layer_count=1,
        stage_count=1,
        hidden_size=8,
        bytes_per_layer=1024,
    )
    stage = build_synthetic_stages(model)[0]
    key = os.urandom(32)
    assignment = StageAssignmentMessage(
        worker_id="worker-0",
        stage=stage,
        shard_path="synthetic://stage-0",
        shard_hash="stage-0",
        model_id="synthetic",
        model_revision="v1",
        synthetic_model=model,
        data_plane_mode=DataPlaneMode.DIRECT,
        coordinator_data_endpoint="127.0.0.1:1",
        route_signing_key=encode_route_key(key),
    )
    agent.assign_synthetic(config=model, stage=stage)
    agent.configure_data_plane(assignment)
    route = sign_route_plan(
        RoutePlan(
            route_id="route",
            route_generation=1,
            request_id="request",
            model_id="synthetic",
            model_revision="v1",
            assignments=[
                RouteHop(
                    stage_id=0,
                    worker_id="worker-0",
                    worker_data_endpoint="127.0.0.1:50052",
                    expected_shard_hash="stage-0",
                    expected_input_spec=stage.input_spec,
                    expected_output_spec=stage.output_spec,
                )
            ],
            route_lease_expiry_unix_ns=time.time_ns() + 60_000_000_000,
            workload_class="standard",
            cancellation_generation=0,
            integrity_policy="hmac-sha256+activation-sha256",
            final_result_destination="127.0.0.1:1",
        ),
        key,
    )
    agent.install_route(route)
    payload = encode_tensor(
        ActivationTensor(
            tensor_id="tensor",
            request_id="request",
            stage_id=0,
            token_position=0,
            sequence_length=1,
            array=synthetic_activation([1], hidden_size=8, dtype="float32"),
        )
    )
    metadata = ActivationMetadata(
        request_id="request",
        tensor_id="tensor",
        stage_id=0,
        operation=OperationKind.PREFILL,
        token_position=0,
        sequence_length=1,
        cache_generation=0,
        model_id="synthetic",
        model_revision="v1",
    )
    envelope = sign_data_envelope(
        DataPlaneEnvelope(
            message_id="message",
            route_id="route",
            route_generation=1,
            request_id="request",
            stage_id=0,
            source_worker="coordinator",
            destination_worker="worker-0",
            token_position=0,
            operation=OperationKind.PREFILL,
            tensor_metadata=metadata,
            tensor_payload=payload,
            payload_length=len(payload),
            payload_checksum=sha256_bytes(payload),
            sequence_number=0,
            timestamp_unix_ns=time.time_ns(),
        ),
        key,
    )
    return agent, key, route, envelope


@pytest.mark.asyncio
async def test_invalid_generation_checksum_order_unknown_and_cancel_are_rejected() -> None:
    agent, key, _, envelope = _configured_agent()
    try:
        stale = sign_data_envelope(
            envelope.model_copy(
                update={
                    "message_id": "stale",
                    "route_generation": 2,
                }
            ),
            key,
        )
        assert (await agent.handle_data_plane(stale)).status == "stale_route"

        checksum = sign_data_envelope(
            envelope.model_copy(
                update={
                    "message_id": "checksum",
                    "payload_checksum": "0" * 64,
                }
            ),
            key,
        )
        assert (await agent.handle_data_plane(checksum)).status == "invalid_checksum"

        wrong_sender = sign_data_envelope(
            envelope.model_copy(
                update={
                    "message_id": "order",
                    "source_worker": "unexpected-worker",
                }
            ),
            key,
        )
        assert (await agent.handle_data_plane(wrong_sender)).status == "invalid_stage_transition"

        unknown = sign_data_envelope(
            envelope.model_copy(
                update={
                    "message_id": "unknown",
                    "route_id": "unknown-route",
                }
            ),
            key,
        )
        assert (await agent.handle_data_plane(unknown)).status == "unknown_request"

        agent.cancel("request")
        cancelled = sign_data_envelope(
            envelope.model_copy(update={"message_id": "cancelled"}),
            key,
        )
        assert (await agent.handle_data_plane(cancelled)).status == "unknown_request"
    finally:
        await agent.stop()


@pytest.mark.asyncio
async def test_route_plan_rejects_skipped_stage_and_stale_install() -> None:
    agent, key, route, _ = _configured_agent()
    try:
        with pytest.raises(ValueError, match="ordered and contiguous"):
            RoutePlan(
                **{
                    **route.model_dump(mode="python", exclude={"assignments"}),
                    "assignments": [route.assignments[0].model_copy(update={"stage_id": 1})],
                }
            )

        newer = sign_route_plan(
            route.model_copy(update={"route_generation": 2, "signature": ""}),
            key,
        )
        agent.install_route(newer)
        with pytest.raises(RouteMessageError, match="older"):
            agent.install_route(route)
    finally:
        await agent.stop()
