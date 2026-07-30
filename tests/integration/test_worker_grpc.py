from __future__ import annotations

import numpy as np
import pytest

from swarm_inference.config.models import (
    Backend,
    OperationKind,
    QueueConfig,
    StageBenchmark,
    SyntheticModelConfig,
    WorkerCapability,
)
from swarm_inference.model.synthetic import synthetic_activation
from swarm_inference.protocol.messages import (
    ActivationMetadata,
    ActivationRequest,
    CancelRequest,
    StageAssignmentMessage,
)
from swarm_inference.protocol.tensor_codec import ActivationTensor, decode_tensor, encode_tensor
from swarm_inference.security.identity import WorkerIdentity
from swarm_inference.simulation.model import build_synthetic_stages
from swarm_inference.transport.grpc_transport import GrpcTransport, WorkerRpcServer
from swarm_inference.worker.agent import WorkerAgent


def capability(identity: WorkerIdentity) -> WorkerCapability:
    return WorkerCapability(
        worker_id="worker-test",
        public_key=identity.public_key_b64,
        hostname="localhost",
        operating_system="test",
        architecture="test",
        backend=Backend.SYNTHETIC,
        cpu_model="test",
        logical_cpu_count=1,
        total_ram_bytes=1024**3,
        available_ram_bytes=1024**3,
        supported_dtypes=["float32"],
        supported_quantisation_formats=[],
        stage_benchmarks=[
            StageBenchmark(
                worker_class="test",
                operation=OperationKind.DECODE,
                sequence_length=1,
                batch_size=1,
                mean_ms=1,
                p95_ms=1,
                samples=1,
                measured=True,
            )
        ],
        upload_bandwidth_bytes_s=1e9,
        download_bandwidth_bytes_s=1e9,
        coordinator_latency_ms=0.1,
        memory_limit_bytes=1024**3,
        endpoint="127.0.0.1:0",
    )


@pytest.mark.asyncio
async def test_worker_assignment_execution_chunking_health_and_cancel() -> None:
    identity = WorkerIdentity.generate()
    agent = WorkerAgent(
        capability=capability(identity),
        identity=identity,
        queue_config=QueueConfig(capacity=4),
    )
    server = WorkerRpcServer(agent=agent, maximum_message_bytes=4096)
    port = await server.start("127.0.0.1:0")
    endpoint = f"127.0.0.1:{port}"
    transport = GrpcTransport(maximum_message_bytes=4096)
    config = SyntheticModelConfig(
        layer_count=2,
        stage_count=1,
        hidden_size=2048,
        bytes_per_layer=1024,
    )
    stage = build_synthetic_stages(config)[0]
    try:
        ack = await transport.assign(
            endpoint,
            StageAssignmentMessage(
                worker_id="worker-test",
                stage=stage,
                shard_path="synthetic://test",
                shard_hash="synthetic",
                model_id="synthetic",
                model_revision="v1",
                synthetic_model=config,
            ),
        )
        assert ack.accepted
        source = synthetic_activation([1, 2, 3], hidden_size=2048, dtype="float32")
        encoded = encode_tensor(
            ActivationTensor(
                tensor_id="t",
                request_id="r",
                stage_id=0,
                token_position=0,
                sequence_length=3,
                array=source,
            )
        )
        assert len(encoded) > 4096
        result = await transport.execute(
            endpoint,
            ActivationRequest(
                metadata=ActivationMetadata(
                    request_id="r",
                    tensor_id="t",
                    stage_id=0,
                    operation=OperationKind.PREFILL,
                    token_position=0,
                    sequence_length=3,
                    cache_generation=0,
                    model_id="synthetic",
                    model_revision="v1",
                ),
                tensor_payload=encoded,
            ),
        )
        assert result.worker_id == "worker-test"
        assert not np.array_equal(decode_tensor(result.tensor_payload).array, source)
        health = await transport.health(endpoint)
        assert health.healthy
        assert health.loaded_stages == [0]
        cancel = await transport.cancel(
            endpoint,
            CancelRequest(request_id="r", model_revision="v1"),
        )
        assert cancel.accepted
        assert agent.shards.module(0).state_summary()["request_states"] == 0
    finally:
        await transport.close()
        await server.stop(0)
