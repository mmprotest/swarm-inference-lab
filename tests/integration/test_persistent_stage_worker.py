from __future__ import annotations

import os
import time

import pytest
import torch

from swarm_inference.config.models import Backend, QueueConfig, WorkerCapability
from swarm_inference.execution.interfaces import StageExecutionResult, WeightOwnership
from swarm_inference.model.partition import StageAssignment
from swarm_inference.protocol.stage_ring import Operation, StageMessage
from swarm_inference.protocol.stage_worker import (
    CloseStageSessionRequest,
    GetStageStatusRequest,
    InstallStageRouteRequest,
    LoadStageRequest,
    OpenStageSessionRequest,
)
from swarm_inference.security.identity import WorkerIdentity
from swarm_inference.transport.grpc_transport import GrpcTransport
from swarm_inference.transport.stage_ring_connection import StageRingConnectionPool
from swarm_inference.transport.stage_tensor import pack_tensor, unpack_tensor
from swarm_inference.worker.agent import WorkerAgent
from swarm_inference.worker.stage_runtime import PersistentStageRuntime
from swarm_inference.worker.stage_service import PersistentStageWorkerService


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
        activation_bytes=8,
        device="cpu",
        owns_embeddings=True,
        owns_final_norm=True,
        owns_output_projection=True,
    )


class _Executor:
    def __init__(self) -> None:
        self.sessions: dict[str, int] = {}
        self.closed = False
        self.ownership = WeightOwnership(
            stage_id=0,
            layer_start=0,
            layer_end=1,
            parameter_names=("model.layers.0.weight",),
            parameter_bytes=1024,
            parameter_count=1,
            owns_embeddings=True,
            owns_final_norm=True,
            owns_output_projection=True,
            ownership_hash="fake",
        )

    def open_session(self, session_id: str) -> None:
        self.sessions[session_id] = 0

    def execute_prefill(
        self,
        *,
        session_id: str,
        token_ids: torch.Tensor,
        cache_position_start: int,
    ) -> StageExecutionResult:
        assert self.sessions[session_id] == cache_position_start
        self.sessions[session_id] += int(token_ids.shape[1])
        hidden = token_ids.to(torch.float32).unsqueeze(-1)
        sampled = token_ids[:, -1].to(torch.int64)
        return StageExecutionResult(
            hidden_states=hidden,
            stage_boundary_hidden_states=hidden,
            router_logits=(),
            final_hidden_states=hidden,
            logits=None,
            sampled_token_ids=sampled,
            all_sampled_token_ids=sampled.unsqueeze(0),
            cache_sequence_length=self.sessions[session_id],
            compute_ns=1,
        )

    def execute_decode(
        self,
        *,
        session_id: str,
        hidden_states: torch.Tensor,
        cache_position_start: int,
    ) -> StageExecutionResult:
        return self.execute_prefill(
            session_id=session_id,
            token_ids=hidden_states.to(torch.int64),
            cache_position_start=cache_position_start,
        )

    def close_session(self, session_id: str) -> int:
        return self.cancel_session(session_id)

    def cancel_session(self, session_id: str) -> int:
        return self.sessions.pop(session_id) * 8

    def kv_cache_bytes(self, session_id: str) -> int:
        return self.sessions[session_id] * 8

    def close(self) -> None:
        self.sessions.clear()
        self.closed = True


def _capability(identity: WorkerIdentity) -> WorkerCapability:
    return WorkerCapability(
        worker_id="worker-stage",
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
        upload_bandwidth_bytes_s=0,
        download_bandwidth_bytes_s=0,
        coordinator_latency_ms=0,
        memory_limit_bytes=1024**3,
        endpoint="127.0.0.1:1",
        control_endpoint="127.0.0.1:1",
        data_plane_endpoint="127.0.0.1:1",
        device_identifier="cpu",
        stage_runtime_enabled=True,
    )


def _load() -> LoadStageRequest:
    return LoadStageRequest(
        worker_id="worker-stage",
        request_id="load",
        model_id="test/olmoe",
        model_revision="revision",
        tokenizer_revision="tokenizer",
        topology_id="topology",
        stage_count=1,
        assignment=_assignment(),
        device="cpu",
        dtype="float32",
        model_path="fake://model",
    )


def _route() -> InstallStageRouteRequest:
    return InstallStageRouteRequest(
        worker_id="worker-stage",
        request_id="route",
        model_id="test/olmoe",
        model_revision="revision",
        tokenizer_revision="tokenizer",
        topology_id="topology",
        route_generation=1,
        assignment=_assignment(),
        device="cpu",
        dtype="float32",
        previous_stage=None,
        next_stage=None,
        stage_count=1,
        lease_expiry_unix_ns=time.time_ns() + 60_000_000_000,
    )


def _session(session_id: str) -> OpenStageSessionRequest:
    return OpenStageSessionRequest(
        worker_id="worker-stage",
        request_id=f"open-{session_id}",
        model_id="test/olmoe",
        model_revision="revision",
        tokenizer_revision="tokenizer",
        topology_id="topology",
        route_generation=1,
        stage_id=0,
        device="cpu",
        dtype="float32",
        session_id=session_id,
    )


def _data(session_id: str, token_id: int) -> StageMessage:
    packed = pack_tensor(torch.tensor([[token_id]], dtype=torch.int64), requested_mode="none")
    return StageMessage(
        operation=Operation.PREFILL,
        model_revision="revision",
        tokenizer_revision="tokenizer",
        topology_id="topology",
        stage_id=0,
        layer_start=0,
        layer_end=1,
        session_id=session_id,
        request_id=f"execute-{session_id}",
        sequence_number=0,
        token_position=0,
        source_stage=-1,
        destination_stage=0,
        tensor_shape=packed.shape,
        tensor_dtype=packed.dtype,
        compression_mode=packed.compression_mode,
        payload=packed.payload,
        attributes={
            "model_id": "test/olmoe",
            "route_generation": 1,
            "source_worker_id": "coordinator",
            "destination_worker_id": "worker-stage",
            "cache_position_start": 0,
            "tensor": packed.attributes(),
        },
    )


@pytest.mark.asyncio
async def test_control_and_tcp_data_endpoints_share_one_persistent_runtime() -> None:
    identity = WorkerIdentity.generate()
    capability = _capability(identity)
    agent = WorkerAgent(
        capability=capability,
        identity=identity,
        queue_config=QueueConfig(capacity=4),
    )
    executor = _Executor()
    loads = 0

    def loader(_request, _path):
        nonlocal loads
        loads += 1
        return executor

    runtime = PersistentStageRuntime(
        worker_id="worker-stage",
        device="cpu",
        dtype="float32",
        memory_limit_bytes=1024**3,
        maximum_sessions=4,
        capability=capability,
        loader=loader,
    )
    service = PersistentStageWorkerService(agent=agent, stage_runtime=runtime)
    grpc_client = GrpcTransport(timeout_s=5)
    data_client = StageRingConnectionPool(read_timeout_s=5, write_timeout_s=5)
    control_port, data_port = await service.start(
        control_listen_endpoint="127.0.0.1:0",
        data_listen_endpoint="127.0.0.1:0",
    )
    assert data_port is not None
    control_endpoint = f"127.0.0.1:{control_port}"
    data_endpoint = f"127.0.0.1:{data_port}"
    try:
        loaded = await grpc_client.load_stage(control_endpoint, _load())
        assert loaded.accepted
        await grpc_client.install_stage_route(control_endpoint, _route())
        process_id = os.getpid()
        for index in range(3):
            session_id = f"session-{index}"
            await grpc_client.open_stage_session(control_endpoint, _session(session_id))
            result = await data_client.send(data_endpoint, _data(session_id, index + 1))
            token, _ = unpack_tensor(result.payload, dict(result.attributes["tensor"]))
            assert token.item() == index + 1
            await grpc_client.close_stage_session(
                control_endpoint,
                CloseStageSessionRequest(**_session(session_id).model_dump()),
            )
        status = await grpc_client.get_stage_status(
            control_endpoint,
            GetStageStatusRequest(worker_id="worker-stage", request_id="status"),
        )
        assert status.process_id == process_id
        assert status.loaded_stage is not None
        assert status.loaded_stage.load_count == 1
        assert status.sessions == []
        assert loads == 1
        assert data_client.snapshot()["connections_created"] == 1
        assert service.data_server is not None
        assert service.data_server.metrics.connections_accepted == 1
    finally:
        await data_client.close()
        await grpc_client.close()
        await service.stop(0)
    assert executor.closed
