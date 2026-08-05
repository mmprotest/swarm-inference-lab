from __future__ import annotations

import time

import pytest
import torch

from swarm_inference.execution.interfaces import StageExecutionResult, WeightOwnership
from swarm_inference.model.partition import StageAssignment
from swarm_inference.protocol.stage_ring import Operation, StageMessage
from swarm_inference.protocol.stage_worker import (
    CloseStageSessionRequest,
    InstallStageRouteRequest,
    LoadStageRequest,
    OpenStageSessionRequest,
    StageRouteEndpoint,
    VerifyStageRouteRequest,
)
from swarm_inference.transport.stage_ring_connection import StageRingConnectionPool
from swarm_inference.transport.stage_ring_server import StageRingServer
from swarm_inference.transport.stage_tensor import pack_tensor, unpack_tensor
from swarm_inference.worker.stage_runtime import PersistentStageRuntime


def _assignment(stage_id: int, stage_count: int) -> StageAssignment:
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
        activation_bytes=16,
        device="cpu",
        owns_embeddings=stage_id == 0,
        owns_final_norm=stage_id == stage_count - 1,
        owns_output_projection=stage_id == stage_count - 1,
    )


class _Executor:
    def __init__(self, assignment: StageAssignment) -> None:
        self.assignment = assignment
        self.positions: dict[str, int] = {}
        self.calls = 0
        self.ownership = WeightOwnership(
            stage_id=assignment.stage_id,
            layer_start=assignment.layer_start,
            layer_end=assignment.layer_end,
            parameter_names=(f"model.layers.{assignment.stage_id}.weight",),
            parameter_bytes=assignment.weight_bytes,
            parameter_count=1,
            owns_embeddings=assignment.owns_embeddings,
            owns_final_norm=assignment.owns_final_norm,
            owns_output_projection=assignment.owns_output_projection,
            ownership_hash=f"stage-{assignment.stage_id}",
        )

    def open_session(self, session_id: str) -> None:
        self.positions[session_id] = 0

    def _execute(
        self,
        *,
        session_id: str,
        tensor: torch.Tensor,
        cache_position_start: int,
    ) -> StageExecutionResult:
        assert self.positions[session_id] == cache_position_start
        sequence_length = int(tensor.shape[-2] if tensor.ndim >= 3 else tensor.shape[-1])
        self.positions[session_id] += sequence_length
        self.calls += 1
        hidden = tensor.to(torch.float32)
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(-1)
        final = self.assignment.owns_output_projection
        token = torch.tensor([42], dtype=torch.int64) if final else None
        return StageExecutionResult(
            hidden_states=hidden,
            stage_boundary_hidden_states=hidden,
            router_logits=(),
            final_hidden_states=hidden if final else None,
            logits=None,
            sampled_token_ids=token,
            all_sampled_token_ids=token.unsqueeze(0) if token is not None else None,
            cache_sequence_length=self.positions[session_id],
            compute_ns=1,
        )

    def execute_prefill(
        self,
        *,
        session_id: str,
        token_ids: torch.Tensor,
        cache_position_start: int,
    ) -> StageExecutionResult:
        return self._execute(
            session_id=session_id,
            tensor=token_ids,
            cache_position_start=cache_position_start,
        )

    def execute_decode(
        self,
        *,
        session_id: str,
        hidden_states: torch.Tensor,
        cache_position_start: int,
    ) -> StageExecutionResult:
        return self._execute(
            session_id=session_id,
            tensor=hidden_states,
            cache_position_start=cache_position_start,
        )

    def close_session(self, session_id: str) -> int:
        return self.cancel_session(session_id)

    def cancel_session(self, session_id: str) -> int:
        return self.positions.pop(session_id) * 8

    def kv_cache_bytes(self, session_id: str) -> int:
        return self.positions[session_id] * 8

    def close(self) -> None:
        self.positions.clear()


def _message(values: list[int], *, cache_position: int, sequence: int) -> StageMessage:
    packed = pack_tensor(torch.tensor([values], dtype=torch.int64), requested_mode="none")
    return StageMessage(
        operation=Operation.PREFILL if cache_position == 0 else Operation.DECODE,
        model_revision="model-revision",
        tokenizer_revision="tokenizer-revision",
        topology_id="n-stage-topology",
        stage_id=0,
        layer_start=0,
        layer_end=1,
        session_id="ring-session",
        request_id=f"execute-{sequence}",
        sequence_number=sequence,
        token_position=cache_position,
        source_stage=-1,
        destination_stage=0,
        tensor_shape=packed.shape,
        tensor_dtype=packed.dtype,
        compression_mode=packed.compression_mode,
        payload=packed.payload,
        attributes={
            "model_id": "test/olmoe",
            "route_generation": 1,
            "request_generation": 1,
            "replay_only": False,
            "source_worker_id": "coordinator",
            "destination_worker_id": "worker-0",
            "cache_position_start": cache_position,
            "tensor": packed.attributes(),
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stage_count", [3, 4, 8])
async def test_canonical_data_plane_runs_ordered_n_stage_ring(stage_count: int) -> None:
    assignments = [_assignment(stage_id, stage_count) for stage_id in range(stage_count)]
    executors = [_Executor(assignment) for assignment in assignments]
    runtimes = [
        PersistentStageRuntime(
            worker_id=f"worker-{stage_id}",
            device="cpu",
            dtype="float32",
            memory_limit_bytes=4096,
            maximum_sessions=2,
            loader=lambda request, _path, *, index=stage_id: executors[index],
        )
        for stage_id in range(stage_count)
    ]
    servers = [StageRingServer(handler=runtime.handle_message) for runtime in runtimes]
    ports: list[int] = []
    ingress = StageRingConnectionPool(read_timeout_s=2, write_timeout_s=2)
    try:
        for server in servers:
            ports.append(await server.start("127.0.0.1:0"))
        lease_expiry = time.time_ns() + 60_000_000_000
        for stage_id, runtime in enumerate(runtimes):
            await runtime.load_stage(
                LoadStageRequest(
                    worker_id=f"worker-{stage_id}",
                    request_id=f"load-{stage_id}",
                    model_id="test/olmoe",
                    model_revision="model-revision",
                    tokenizer_revision="tokenizer-revision",
                    topology_id="n-stage-topology",
                    stage_count=stage_count,
                    assignment=assignments[stage_id],
                    device="cpu",
                    dtype="float32",
                    model_path="fake://checkpoint",
                )
            )
        for stage_id, runtime in enumerate(runtimes):
            previous = (
                StageRouteEndpoint(
                    worker_id=f"worker-{stage_id - 1}",
                    stage_id=stage_id - 1,
                    data_endpoint=f"127.0.0.1:{ports[stage_id - 1]}",
                    assignment=assignments[stage_id - 1],
                )
                if stage_id > 0
                else None
            )
            following = (
                StageRouteEndpoint(
                    worker_id=f"worker-{stage_id + 1}",
                    stage_id=stage_id + 1,
                    data_endpoint=f"127.0.0.1:{ports[stage_id + 1]}",
                    assignment=assignments[stage_id + 1],
                )
                if stage_id < stage_count - 1
                else None
            )
            await runtime.install_route(
                InstallStageRouteRequest(
                    worker_id=f"worker-{stage_id}",
                    request_id=f"route-{stage_id}",
                    model_id="test/olmoe",
                    model_revision="model-revision",
                    tokenizer_revision="tokenizer-revision",
                    topology_id="n-stage-topology",
                    route_generation=1,
                    assignment=assignments[stage_id],
                    device="cpu",
                    dtype="float32",
                    previous_stage=previous,
                    next_stage=following,
                    stage_count=stage_count,
                    lease_expiry_unix_ns=lease_expiry,
                )
            )
            await runtime.open_session(
                OpenStageSessionRequest(
                    worker_id=f"worker-{stage_id}",
                    request_id=f"open-{stage_id}",
                    model_id="test/olmoe",
                    model_revision="model-revision",
                    tokenizer_revision="tokenizer-revision",
                    topology_id="n-stage-topology",
                    route_generation=1,
                    stage_id=stage_id,
                    device="cpu",
                    dtype="float32",
                    session_id="ring-session",
                    request_generation=1,
                )
            )
        for stage_id, runtime in enumerate(runtimes):
            await runtime.verify_route(
                VerifyStageRouteRequest(
                    worker_id=f"worker-{stage_id}",
                    request_id=f"verify-{stage_id}",
                    model_id="test/olmoe",
                    model_revision="model-revision",
                    tokenizer_revision="tokenizer-revision",
                    topology_id="n-stage-topology",
                    route_generation=1,
                    stage_id=stage_id,
                    device="cpu",
                    dtype="float32",
                )
            )
        endpoint = f"127.0.0.1:{ports[0]}"
        first = _message([2, 3], cache_position=0, sequence=0)
        second = _message([4], cache_position=2, sequence=1)
        first_response = await ingress.send(endpoint, first)
        second_response = await ingress.send(endpoint, second)

        assert first_response.operation == second_response.operation == Operation.TOKEN_RESULT
        first_token, _ = unpack_tensor(
            first_response.payload,
            dict(first_response.attributes["tensor"]),
        )
        second_token, _ = unpack_tensor(
            second_response.payload,
            dict(second_response.attributes["tensor"]),
        )
        assert int(first_token.item()) == int(second_token.item()) == 42
        assert [executor.calls for executor in executors] == [2] * stage_count
        assert all(
            runtime.connection_pool.snapshot()["connections_created"] == 1
            for runtime in runtimes[:-1]
        )
        assert ingress.snapshot()["connections_created"] == 1
        for stage_id, runtime in enumerate(runtimes):
            await runtime.close_session(
                CloseStageSessionRequest(
                    worker_id=f"worker-{stage_id}",
                    request_id=f"close-{stage_id}",
                    model_id="test/olmoe",
                    model_revision="model-revision",
                    tokenizer_revision="tokenizer-revision",
                    topology_id="n-stage-topology",
                    route_generation=1,
                    stage_id=stage_id,
                    device="cpu",
                    dtype="float32",
                    session_id="ring-session",
                    request_generation=1,
                )
            )
    finally:
        await ingress.close()
        for server, runtime in zip(reversed(servers), reversed(runtimes), strict=True):
            await server.stop()
            await runtime.close()
