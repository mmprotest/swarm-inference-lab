from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from swarm_inference.exceptions import BackpressureError, IntegrityError, MemoryLimitExceededError
from swarm_inference.execution.interfaces import StageExecutionResult, WeightOwnership
from swarm_inference.model.partition import StageAssignment
from swarm_inference.protocol.stage_ring import Operation, StageMessage
from swarm_inference.protocol.stage_worker import (
    CancelStageSessionRequest,
    CloseStageSessionRequest,
    DrainWorkerRequest,
    GetStageStatusRequest,
    InstallStageRouteRequest,
    LoadStageRequest,
    OpenStageSessionRequest,
    RemoveStageRouteRequest,
    StageRouteEndpoint,
    UnloadStageRequest,
)
from swarm_inference.transport.stage_ring_connection import StageRingConnectionPool
from swarm_inference.transport.stage_ring_server import StageRingServer
from swarm_inference.transport.stage_tensor import pack_tensor, unpack_tensor
from swarm_inference.worker.stage_runtime import PersistentStageRuntime


def assignment(*, weight_bytes: int = 1024) -> StageAssignment:
    return StageAssignment(
        stage_id=0,
        layer_start=0,
        layer_end=1,
        layer_ids=(0,),
        weight_bytes=weight_bytes,
        estimated_compute_ns=1,
        measured_compute_ns=1,
        kv_cache_bytes_per_token=16,
        peak_temporary_bytes=128,
        activation_bytes=16,
        device="cpu",
        owns_embeddings=True,
        owns_final_norm=True,
        owns_output_projection=True,
    )


class FakeStageExecutor:
    def __init__(self, owned: StageAssignment, *, parameter_bytes: int | None = None) -> None:
        self.assignment = owned
        self.sessions: dict[str, int] = {}
        self.values: dict[str, int] = {}
        self.closed = False
        self.ownership = WeightOwnership(
            stage_id=owned.stage_id,
            layer_start=owned.layer_start,
            layer_end=owned.layer_end,
            parameter_names=(
                "model.embed_tokens.weight",
                "model.layers.0.test_weight",
                "model.norm.weight",
                "lm_head.weight",
            ),
            parameter_bytes=parameter_bytes or owned.weight_bytes,
            parameter_count=4,
            owns_embeddings=owned.owns_embeddings,
            owns_final_norm=owned.owns_final_norm,
            owns_output_projection=owned.owns_output_projection,
            ownership_hash="fake-ownership",
        )

    def open_session(self, session_id: str) -> None:
        if self.closed:
            raise RuntimeError("executor closed")
        if session_id in self.sessions:
            raise ValueError("duplicate fake session")
        self.sessions[session_id] = 0
        self.values[session_id] = 0

    def _execute(
        self,
        *,
        session_id: str,
        tensor: torch.Tensor,
        cache_position_start: int,
    ) -> StageExecutionResult:
        assert self.sessions[session_id] == cache_position_start
        sequence_length = int(tensor.shape[-2] if tensor.ndim >= 3 else tensor.shape[-1])
        self.sessions[session_id] += sequence_length
        self.values[session_id] += int(tensor.to(torch.int64).sum().item())
        hidden = tensor.to(torch.float32).reshape(1, sequence_length, -1)
        token = torch.tensor([self.values[session_id]], dtype=torch.int64)
        return StageExecutionResult(
            hidden_states=hidden,
            stage_boundary_hidden_states=hidden,
            router_logits=(),
            final_hidden_states=hidden,
            logits=None,
            sampled_token_ids=token,
            all_sampled_token_ids=token.unsqueeze(0),
            cache_sequence_length=self.sessions[session_id],
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
        released = self.kv_cache_bytes(session_id)
        del self.sessions[session_id]
        del self.values[session_id]
        return released

    def cancel_session(self, session_id: str) -> int:
        return self.close_session(session_id)

    def kv_cache_bytes(self, session_id: str) -> int:
        return self.sessions[session_id] * 16

    def close(self) -> None:
        self.sessions.clear()
        self.values.clear()
        self.closed = True


class CountingLoader:
    def __init__(self, *, excess_parameter_bytes: bool = False) -> None:
        self.calls = 0
        self.executor: FakeStageExecutor | None = None
        self.excess_parameter_bytes = excess_parameter_bytes

    def __call__(self, request: LoadStageRequest, _path) -> FakeStageExecutor:
        self.calls += 1
        parameter_bytes = (
            request.assignment.weight_bytes + 1 if self.excess_parameter_bytes else None
        )
        self.executor = FakeStageExecutor(
            request.assignment,
            parameter_bytes=parameter_bytes,
        )
        return self.executor


class BlockingExecutor(FakeStageExecutor):
    def __init__(self, owned: StageAssignment) -> None:
        super().__init__(owned)
        self.started = threading.Event()
        self.release = threading.Event()

    def _execute(
        self,
        *,
        session_id: str,
        tensor: torch.Tensor,
        cache_position_start: int,
    ) -> StageExecutionResult:
        self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release blocking executor")
        return super()._execute(
            session_id=session_id,
            tensor=tensor,
            cache_position_start=cache_position_start,
        )


def load_request(*, request_id: str = "load", owned: StageAssignment | None = None):
    return LoadStageRequest(
        worker_id="worker-a",
        request_id=request_id,
        model_id="test/olmoe",
        model_revision="model-revision",
        tokenizer_revision="tokenizer-revision",
        topology_id="topology-a",
        stage_count=1,
        assignment=owned or assignment(),
        device="cpu",
        dtype="float32",
        model_path="fake://checkpoint",
    )


def route_request(*, generation: int = 1, replace_route: bool = False):
    return InstallStageRouteRequest(
        worker_id="worker-a",
        request_id=f"route-{generation}",
        model_id="test/olmoe",
        model_revision="model-revision",
        tokenizer_revision="tokenizer-revision",
        topology_id="topology-a",
        route_generation=generation,
        assignment=assignment(),
        device="cpu",
        dtype="float32",
        previous_stage=None,
        next_stage=None,
        stage_count=1,
        lease_expiry_unix_ns=time.time_ns() + 60_000_000_000,
        replace=replace_route,
    )


def session_request(session_id: str, *, request_id: str | None = None):
    return OpenStageSessionRequest(
        worker_id="worker-a",
        request_id=request_id or f"open-{session_id}",
        model_id="test/olmoe",
        model_revision="model-revision",
        tokenizer_revision="tokenizer-revision",
        topology_id="topology-a",
        route_generation=1,
        stage_id=0,
        device="cpu",
        dtype="float32",
        session_id=session_id,
    )


def data_message(
    session_id: str,
    values: list[int],
    *,
    cache_position: int,
    sequence: int,
) -> StageMessage:
    packed = pack_tensor(torch.tensor([values], dtype=torch.int64), requested_mode="none")
    return StageMessage(
        operation=Operation.PREFILL if cache_position == 0 else Operation.DECODE,
        model_revision="model-revision",
        tokenizer_revision="tokenizer-revision",
        topology_id="topology-a",
        stage_id=0,
        layer_start=0,
        layer_end=1,
        session_id=session_id,
        request_id=f"execute-{session_id}-{sequence}",
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
            "source_worker_id": "coordinator",
            "destination_worker_id": "worker-a",
            "cache_position_start": cache_position,
            "tensor": packed.attributes(),
        },
    )


async def loaded_runtime(*, maximum_sessions: int = 8):
    loader = CountingLoader()
    runtime = PersistentStageRuntime(
        worker_id="worker-a",
        device="cpu",
        dtype="float32",
        memory_limit_bytes=4096,
        maximum_sessions=maximum_sessions,
        loader=loader,
    )
    await runtime.load_stage(load_request())
    await runtime.install_route(route_request())
    return runtime, loader


@pytest.mark.asyncio
async def test_stage_is_loaded_once_across_one_hundred_sessions_and_pid_is_stable() -> None:
    runtime, loader = await loaded_runtime()
    process_id = os.getpid()
    try:
        retry = await runtime.load_stage(load_request(request_id="load-retry"))
        assert retry.idempotent
        for index in range(100):
            session_id = f"session-{index}"
            await runtime.open_session(session_request(session_id))
            response = await runtime.handle_message(
                data_message(session_id, [index + 1], cache_position=0, sequence=0)
            )
            assert response.operation == Operation.TOKEN_RESULT
            token, _ = unpack_tensor(response.payload, dict(response.attributes["tensor"]))
            assert token.item() == index + 1
            request = CloseStageSessionRequest(**session_request(session_id).model_dump())
            await runtime.close_session(request)
        status = await runtime.status(
            GetStageStatusRequest(worker_id="worker-a", request_id="status")
        )
        assert status.process_id == process_id
        assert status.loaded_stage is not None
        assert status.loaded_stage.load_count == 1
        assert status.sessions == []
        assert loader.calls == 1
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_interleaved_sessions_are_isolated_and_close_cancel_release_one_only() -> None:
    runtime, loader = await loaded_runtime()
    try:
        await runtime.open_session(session_request("a"))
        await runtime.open_session(session_request("b"))
        first_a = await runtime.handle_message(data_message("a", [1], cache_position=0, sequence=0))
        first_b = await runtime.handle_message(
            data_message("b", [10], cache_position=0, sequence=0)
        )
        second_a = await runtime.handle_message(
            data_message("a", [2], cache_position=1, sequence=1)
        )
        outputs = []
        for response in (first_a, first_b, second_a):
            tensor, _ = unpack_tensor(response.payload, dict(response.attributes["tensor"]))
            outputs.append(tensor.item())
        assert outputs == [1, 10, 3]

        closed = await runtime.close_session(
            CloseStageSessionRequest(**session_request("a").model_dump())
        )
        assert closed.released_kv_bytes == 32
        assert loader.executor is not None
        assert set(loader.executor.sessions) == {"b"}
        cancelled = await runtime.cancel_session(
            CancelStageSessionRequest(**session_request("b").model_dump())
        )
        assert cancelled.released_kv_bytes == 16
        assert loader.executor.sessions == {}
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_load_route_identity_memory_and_ownership_rejections() -> None:
    runtime, loader = await loaded_runtime()
    try:
        with pytest.raises(RuntimeError, match="incompatible"):
            await runtime.load_stage(
                load_request(request_id="wrong-model").model_copy(
                    update={"model_revision": "wrong"}
                )
            )
        with pytest.raises(RuntimeError, match="incompatible"):
            await runtime.load_stage(
                load_request(request_id="wrong-tokenizer").model_copy(
                    update={"tokenizer_revision": "wrong"}
                )
            )
        with pytest.raises(ValueError, match="replacement must be explicit"):
            await runtime.install_route(route_request(generation=2))
        replaced = await runtime.install_route(route_request(generation=2, replace_route=True))
        assert replaced.accepted
        with pytest.raises(ValueError, match="stale route"):
            await runtime.install_route(route_request(generation=1, replace_route=True))
        with pytest.raises(ValueError, match=r"identity|topology"):
            await runtime.open_session(
                session_request("unknown").model_copy(update={"topology_id": "other"})
            )
        remove = RemoveStageRouteRequest(
            worker_id="worker-a",
            request_id="remove",
            model_id="test/olmoe",
            model_revision="model-revision",
            tokenizer_revision="tokenizer-revision",
            topology_id="topology-a",
            route_generation=2,
            stage_id=0,
            device="cpu",
            dtype="float32",
        )
        assert (await runtime.remove_route(remove)).accepted
        assert (
            await runtime.remove_route(remove.model_copy(update={"request_id": "remove-again"}))
        ).idempotent
        with pytest.raises(ValueError, match="stale route"):
            await runtime.install_route(
                route_request(generation=1, replace_route=True).model_copy(
                    update={"request_id": "reinstall-stale"}
                )
            )
        assert loader.calls == 1
    finally:
        await runtime.close()

    memory_runtime = PersistentStageRuntime(
        worker_id="worker-a",
        device="cpu",
        dtype="float32",
        memory_limit_bytes=1024,
        maximum_sessions=1,
        loader=CountingLoader(),
    )
    with pytest.raises(MemoryLimitExceededError, match="estimated resident peak"):
        await memory_runtime.load_stage(
            load_request(owned=replace(assignment(), weight_bytes=1024))
        )
    await memory_runtime.close()

    ownership_loader = CountingLoader(excess_parameter_bytes=True)
    ownership_runtime = PersistentStageRuntime(
        worker_id="worker-a",
        device="cpu",
        dtype="float32",
        memory_limit_bytes=4096,
        maximum_sessions=1,
        loader=ownership_loader,
    )
    with pytest.raises(IntegrityError, match="more parameter bytes"):
        await ownership_runtime.load_stage(load_request())
    assert ownership_loader.executor is not None and ownership_loader.executor.closed
    await ownership_runtime.close()


@pytest.mark.asyncio
async def test_unload_is_idempotent_and_shutdown_releases_sessions_and_model() -> None:
    runtime, loader = await loaded_runtime()
    await runtime.open_session(session_request("open"))
    request = UnloadStageRequest(
        worker_id="worker-a",
        request_id="unload",
        model_id="test/olmoe",
        model_revision="model-revision",
        tokenizer_revision="tokenizer-revision",
        topology_id="topology-a",
        route_generation=1,
        stage_count=1,
        assignment=assignment(),
        device="cpu",
        dtype="float32",
        force=True,
    )
    unloaded = await runtime.unload_stage(request)
    assert unloaded.released_kv_bytes == 0
    assert loader.executor is not None and loader.executor.closed
    again = await runtime.unload_stage(request.model_copy(update={"request_id": "again"}))
    assert again.idempotent
    await runtime.close()


@pytest.mark.asyncio
async def test_maximum_session_count_is_enforced() -> None:
    runtime, _ = await loaded_runtime(maximum_sessions=1)
    try:
        await runtime.open_session(session_request("one"))
        with pytest.raises(ValueError, match="already open"):
            await runtime.open_session(session_request("one"))
        with pytest.raises(RuntimeError, match="maximum active"):
            await runtime.open_session(session_request("two"))
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_expired_route_still_allows_session_cleanup() -> None:
    runtime, loader = await loaded_runtime()
    await runtime.open_session(session_request("expired"))
    assert runtime._route is not None
    runtime._route.lease_expiry_unix_ns = time.time_ns() - 1
    try:
        with pytest.raises(ValueError, match="lease has expired"):
            await runtime.handle_message(data_message("expired", [1], cache_position=0, sequence=0))
        response = await runtime.cancel_session(
            CancelStageSessionRequest(**session_request("expired").model_dump())
        )
        assert response.accepted
        assert loader.executor is not None
        assert loader.executor.sessions == {}
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_closed_session_id_can_be_reopened_with_fresh_sequences() -> None:
    runtime, _ = await loaded_runtime()
    try:
        await runtime.open_session(session_request("reused"))
        first = await runtime.handle_message(
            data_message("reused", [1], cache_position=0, sequence=0)
        )
        assert first.sequence_number == 0
        await runtime.close_session(
            CloseStageSessionRequest(**session_request("reused").model_dump())
        )
        await runtime.open_session(session_request("reused", request_id="reopen"))
        second = await runtime.handle_message(
            data_message("reused", [2], cache_position=0, sequence=0)
        )
        assert second.sequence_number == 0
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_two_stage_ring_validates_and_reuses_both_direct_connections() -> None:
    stage_zero = replace(
        assignment(),
        owns_final_norm=False,
        owns_output_projection=False,
    )
    stage_one = replace(
        assignment(),
        stage_id=1,
        layer_start=1,
        layer_end=2,
        layer_ids=(1,),
        owns_embeddings=False,
    )
    loader_zero = CountingLoader()
    loader_one = CountingLoader()
    runtime_zero = PersistentStageRuntime(
        worker_id="worker-zero",
        device="cpu",
        dtype="float32",
        memory_limit_bytes=4096,
        maximum_sessions=2,
        loader=loader_zero,
    )
    runtime_one = PersistentStageRuntime(
        worker_id="worker-one",
        device="cpu",
        dtype="float32",
        memory_limit_bytes=4096,
        maximum_sessions=2,
        loader=loader_one,
    )
    await runtime_zero.load_stage(
        load_request().model_copy(
            update={
                "worker_id": "worker-zero",
                "request_id": "load-zero",
                "stage_count": 2,
                "assignment": stage_zero,
            }
        )
    )
    await runtime_one.load_stage(
        load_request().model_copy(
            update={
                "worker_id": "worker-one",
                "request_id": "load-one",
                "stage_count": 2,
                "assignment": stage_one,
            }
        )
    )
    server_zero = StageRingServer(handler=runtime_zero.handle_message)
    server_one = StageRingServer(handler=runtime_one.handle_message)
    port_zero = await server_zero.start("127.0.0.1:0")
    port_one = await server_one.start("127.0.0.1:0")
    lease = time.time_ns() + 60_000_000_000
    route_zero = InstallStageRouteRequest(
        worker_id="worker-zero",
        request_id="route-zero",
        model_id="test/olmoe",
        model_revision="model-revision",
        tokenizer_revision="tokenizer-revision",
        topology_id="topology-a",
        route_generation=1,
        assignment=stage_zero,
        device="cpu",
        dtype="float32",
        previous_stage=None,
        next_stage=StageRouteEndpoint(
            worker_id="worker-one",
            stage_id=1,
            data_endpoint=f"127.0.0.1:{port_one}",
            assignment=stage_one,
        ),
        stage_count=2,
        lease_expiry_unix_ns=lease,
    )
    route_one = InstallStageRouteRequest(
        worker_id="worker-one",
        request_id="route-one",
        model_id="test/olmoe",
        model_revision="model-revision",
        tokenizer_revision="tokenizer-revision",
        topology_id="topology-a",
        route_generation=1,
        assignment=stage_one,
        device="cpu",
        dtype="float32",
        previous_stage=StageRouteEndpoint(
            worker_id="worker-zero",
            stage_id=0,
            data_endpoint=f"127.0.0.1:{port_zero}",
            assignment=stage_zero,
        ),
        next_stage=None,
        stage_count=2,
        lease_expiry_unix_ns=lease,
    )
    client = StageRingConnectionPool(read_timeout_s=2, write_timeout_s=2)
    try:
        await runtime_zero.install_route(route_zero)
        await runtime_one.install_route(route_one)
        open_zero = session_request("ring").model_copy(
            update={"worker_id": "worker-zero", "request_id": "open-zero"}
        )
        open_one = session_request("ring").model_copy(
            update={"worker_id": "worker-one", "request_id": "open-one", "stage_id": 1}
        )
        await runtime_zero.open_session(open_zero)
        await runtime_one.open_session(open_one)
        endpoint = f"127.0.0.1:{port_zero}"
        first_message = data_message("ring", [2, 3], cache_position=0, sequence=0)
        first_message.attributes["destination_worker_id"] = "worker-zero"
        first = await client.send(
            endpoint,
            first_message,
        )
        second_message = data_message("ring", [4], cache_position=2, sequence=1)
        second_message.attributes["destination_worker_id"] = "worker-zero"
        second = await client.send(
            endpoint,
            second_message,
        )
        assert first.operation == second.operation == Operation.TOKEN_RESULT
        assert client.snapshot()["connections_created"] == 1
        assert runtime_zero.connection_pool.snapshot()["connections_created"] == 1
        assert server_zero.metrics.reused_frames == 1
        assert server_one.metrics.reused_frames == 1
        await runtime_zero.close_session(CloseStageSessionRequest(**open_zero.model_dump()))
        await runtime_one.close_session(CloseStageSessionRequest(**open_one.model_dump()))
    finally:
        await client.close()
        await server_zero.stop()
        await runtime_zero.close()
        await server_one.stop()
        await runtime_one.close()


@pytest.mark.asyncio
async def test_execution_queue_rejects_work_when_bounded_capacity_is_full() -> None:
    executor = BlockingExecutor(assignment())
    runtime = PersistentStageRuntime(
        worker_id="worker-a",
        device="cpu",
        dtype="float32",
        memory_limit_bytes=4096,
        maximum_sessions=3,
        execution_queue_capacity=1,
        loader=lambda _request, _path: executor,
    )
    await runtime.load_stage(load_request())
    await runtime.install_route(route_request())
    for session_id in ("one", "two", "three"):
        await runtime.open_session(session_request(session_id))
    first = asyncio.create_task(
        runtime.handle_message(data_message("one", [1], cache_position=0, sequence=0))
    )
    while not executor.started.is_set():
        await asyncio.sleep(0)
    second = asyncio.create_task(
        runtime.handle_message(data_message("two", [2], cache_position=0, sequence=0))
    )
    await asyncio.sleep(0)
    try:
        with pytest.raises(BackpressureError, match="execution queue"):
            await runtime.handle_message(data_message("three", [3], cache_position=0, sequence=0))
    finally:
        executor.release.set()
    await asyncio.gather(first, second)
    await runtime.close()


@pytest.mark.asyncio
async def test_local_checkpoint_model_and_tokenizer_revisions_are_verified(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model"
    metadata_path = model_path / ".cache" / "huggingface" / "download"
    metadata_path.mkdir(parents=True)
    (model_path / "config.json").write_text('{"model_type":"olmoe"}', encoding="utf-8")
    (model_path / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    (model_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (metadata_path / "config.json.metadata").write_text("actual-model\netag\n0\n", encoding="utf-8")
    (metadata_path / "tokenizer.json.metadata").write_text(
        "actual-tokenizer\netag\n0\n", encoding="utf-8"
    )
    runtime = PersistentStageRuntime(
        worker_id="worker-a",
        device="cpu",
        dtype="float32",
        memory_limit_bytes=4096,
        maximum_sessions=1,
    )
    base = load_request().model_copy(update={"model_path": str(model_path)})
    try:
        with pytest.raises(IntegrityError, match="model revision mismatch"):
            await runtime.load_stage(base)
        with pytest.raises(IntegrityError, match="tokenizer revision mismatch"):
            await runtime.load_stage(
                base.model_copy(
                    update={
                        "model_revision": "actual-model",
                        "tokenizer_revision": "wrong-tokenizer",
                    }
                )
            )
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_data_plane_rejects_duplicate_sequence_wrong_cache_and_topology() -> None:
    runtime, _ = await loaded_runtime()
    try:
        await runtime.open_session(session_request("strict"))
        first = data_message("strict", [1], cache_position=0, sequence=0)
        await runtime.handle_message(first)
        with pytest.raises(ValueError, match="duplicate"):
            await runtime.handle_message(first)
        with pytest.raises(ValueError, match="cache position"):
            await runtime.handle_message(data_message("strict", [2], cache_position=0, sequence=1))
        with pytest.raises(ValueError, match="topology"):
            await runtime.handle_message(
                replace(
                    data_message("strict", [2], cache_position=1, sequence=1),
                    topology_id="unknown",
                )
            )
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_drain_stops_admission_and_optionally_releases_active_sessions() -> None:
    runtime, loader = await loaded_runtime()
    try:
        await runtime.open_session(session_request("active"))
        response = await runtime.drain(
            DrainWorkerRequest(
                worker_id="worker-a",
                request_id="drain",
                cancel_active_sessions=True,
            )
        )
        assert response.accepted
        assert runtime.draining
        assert loader.executor is not None and loader.executor.sessions == {}
        with pytest.raises(RuntimeError, match="draining"):
            await runtime.open_session(session_request("new"))
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_slow_token_publisher_does_not_block_stage_execution() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def publisher(_publication) -> None:
        started.set()
        await release.wait()

    loader = CountingLoader()
    runtime = PersistentStageRuntime(
        worker_id="worker-a",
        device="cpu",
        dtype="float32",
        memory_limit_bytes=4096,
        maximum_sessions=1,
        loader=loader,
        token_publisher=publisher,
    )
    await runtime.load_stage(load_request())
    await runtime.install_route(route_request())
    await runtime.open_session(session_request("publish"))
    try:
        result = await asyncio.wait_for(
            runtime.handle_message(data_message("publish", [7], cache_position=0, sequence=0)),
            timeout=1,
        )
        assert result.operation == Operation.TOKEN_RESULT
        await asyncio.wait_for(started.wait(), timeout=1)
    finally:
        release.set()
        await runtime.close()
