from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import psutil
import pytest
import torch

from swarm_inference.config.models import Backend, QueueConfig, WorkerCapability
from swarm_inference.model.olmoe import inspect_olmoe_partition_metadata
from swarm_inference.model.partition import build_stage_plan
from swarm_inference.protocol.stage_ring import Operation, StageMessage
from swarm_inference.protocol.stage_worker import (
    CloseStageSessionRequest,
    InstallStageRouteRequest,
    LoadStageRequest,
    OpenStageSessionRequest,
    StageRouteEndpoint,
)
from swarm_inference.security.identity import WorkerIdentity
from swarm_inference.transport.grpc_transport import GrpcTransport
from swarm_inference.transport.stage_ring_connection import StageRingConnectionPool
from swarm_inference.transport.stage_tensor import pack_tensor, unpack_tensor
from swarm_inference.worker.agent import WorkerAgent
from swarm_inference.worker.stage_runtime import PersistentStageRuntime
from swarm_inference.worker.stage_service import PersistentStageWorkerService

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPOSITORY_ROOT / "artifacts" / "models" / "colibri" / "source-b89a7c4bc24f"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("SWARM_RUN_EXPERIMENT_011_GPU_TESTS") != "1"
        or not torch.cuda.is_available()
        or not MODEL_PATH.is_dir(),
        reason="set SWARM_RUN_EXPERIMENT_011_GPU_TESTS=1 for the real stage service test",
    ),
]


@pytest.mark.asyncio
async def test_real_olmoe_stage_stays_resident_across_service_sessions() -> None:
    device = f"cuda:{torch.cuda.current_device()}"
    model_revision = (
        (MODEL_PATH / ".cache" / "huggingface" / "download" / "config.json.metadata")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    tokenizer_revision = (
        "sha256:" + hashlib.sha256((MODEL_PATH / "tokenizer.json").read_bytes()).hexdigest()
    )
    metadata = inspect_olmoe_partition_metadata(
        MODEL_PATH,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
    )
    plan = build_stage_plan(
        MODEL_PATH,
        metadata=metadata,
        stage_count=2,
        method="equal",
        memory_limit_bytes=16_000_000_000,
        device=device,
    )
    owned = plan.assignments[1]
    identity = WorkerIdentity.generate()
    memory = psutil.virtual_memory()
    _, total_vram = torch.cuda.mem_get_info(torch.cuda.current_device())
    capability = WorkerCapability(
        worker_id="real-stage-worker",
        public_key=identity.public_key_b64,
        hostname="localhost",
        operating_system="test",
        architecture="test",
        backend=Backend.TORCH_CUDA,
        cpu_model="test",
        logical_cpu_count=1,
        total_ram_bytes=memory.total,
        available_ram_bytes=memory.available,
        gpu_model=torch.cuda.get_device_name(torch.cuda.current_device()),
        total_vram_bytes=total_vram,
        available_vram_bytes=torch.cuda.mem_get_info(torch.cuda.current_device())[0],
        supported_dtypes=["bfloat16"],
        upload_bandwidth_bytes_s=0,
        download_bandwidth_bytes_s=0,
        coordinator_latency_ms=0,
        memory_limit_bytes=16_000_000_000,
        endpoint="127.0.0.1:1",
        control_endpoint="127.0.0.1:1",
        data_plane_endpoint="127.0.0.1:1",
        device_identifier=device,
        stage_runtime_enabled=True,
    )
    agent = WorkerAgent(
        capability=capability,
        identity=identity,
        queue_config=QueueConfig(capacity=2),
    )
    runtime = PersistentStageRuntime(
        worker_id="real-stage-worker",
        device=device,
        dtype="bfloat16",
        memory_limit_bytes=16_000_000_000,
        maximum_sessions=2,
        capability=capability,
    )
    service = PersistentStageWorkerService(agent=agent, stage_runtime=runtime)
    grpc_client = GrpcTransport(timeout_s=300)
    data_client = StageRingConnectionPool(
        read_timeout_s=300,
        write_timeout_s=300,
    )
    control_port, data_port = await service.start(
        control_listen_endpoint="127.0.0.1:0",
        data_listen_endpoint="127.0.0.1:0",
    )
    assert data_port is not None
    control_endpoint = f"127.0.0.1:{control_port}"
    data_endpoint = f"127.0.0.1:{data_port}"
    load = LoadStageRequest(
        worker_id="real-stage-worker",
        request_id="load-real",
        model_id="colibri-olmoe",
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        topology_id=plan.topology_id,
        stage_count=2,
        assignment=owned,
        device=device,
        dtype="bfloat16",
        model_path=str(MODEL_PATH),
    )
    route = InstallStageRouteRequest(
        worker_id="real-stage-worker",
        request_id="route-real",
        model_id=load.model_id,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        topology_id=plan.topology_id,
        route_generation=1,
        assignment=owned,
        device=device,
        dtype="bfloat16",
        previous_stage=StageRouteEndpoint(
            worker_id="previous-worker",
            stage_id=0,
            data_endpoint="127.0.0.1:9",
            assignment=plan.assignments[0],
        ),
        next_stage=None,
        stage_count=2,
        lease_expiry_unix_ns=time.time_ns() + 600_000_000_000,
    )
    try:
        await grpc_client.load_stage(control_endpoint, load)
        await grpc_client.install_stage_route(control_endpoint, route)
        resident = runtime.loaded_executor
        assert resident is not None
        ownership = resident.ownership
        assert not ownership.owns_embeddings
        assert ownership.owns_output_projection
        for index in range(2):
            session_id = f"real-session-{index}"
            open_request = OpenStageSessionRequest(
                worker_id="real-stage-worker",
                request_id=f"open-{index}",
                model_id=load.model_id,
                model_revision=model_revision,
                tokenizer_revision=tokenizer_revision,
                topology_id=plan.topology_id,
                route_generation=1,
                stage_id=owned.stage_id,
                device=device,
                dtype="bfloat16",
                session_id=session_id,
            )
            await grpc_client.open_stage_session(control_endpoint, open_request)
            hidden = torch.zeros(
                (1, 1, metadata.hidden_size),
                dtype=torch.bfloat16,
            )
            packed = pack_tensor(hidden, requested_mode="none")
            result = await data_client.send(
                data_endpoint,
                StageMessage(
                    operation=Operation.PREFILL,
                    model_revision=model_revision,
                    tokenizer_revision=tokenizer_revision,
                    topology_id=plan.topology_id,
                    stage_id=owned.stage_id,
                    layer_start=owned.layer_start,
                    layer_end=owned.layer_end,
                    session_id=session_id,
                    request_id=f"execute-{index}",
                    sequence_number=0,
                    token_position=0,
                    source_stage=0,
                    destination_stage=owned.stage_id,
                    tensor_shape=packed.shape,
                    tensor_dtype=packed.dtype,
                    compression_mode=packed.compression_mode,
                    payload=packed.payload,
                    attributes={
                        "model_id": load.model_id,
                        "route_generation": 1,
                        "source_worker_id": "previous-worker",
                        "destination_worker_id": "real-stage-worker",
                        "cache_position_start": 0,
                        "tensor": packed.attributes(),
                    },
                ),
            )
            assert result.operation == Operation.TOKEN_RESULT
            token, _ = unpack_tensor(result.payload, dict(result.attributes["tensor"]))
            assert token.shape == (1,)
            await grpc_client.close_stage_session(
                control_endpoint,
                CloseStageSessionRequest(**open_request.model_dump()),
            )
            assert runtime.loaded_executor is resident
        assert runtime.load_count == 1
        assert data_client.snapshot()["connections_created"] == 1
    finally:
        await data_client.close()
        await grpc_client.close()
        await service.stop(0)
