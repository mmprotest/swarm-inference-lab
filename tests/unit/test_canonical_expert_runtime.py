from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from swarm_inference.coordinator.expert_planner import (
    ExpertStrategy,
    ExpertStrategyCandidate,
    ExpertUtilityInputs,
    ExpertUtilityPlanner,
)
from swarm_inference.execution.expert import (
    ExpertStore,
    deterministic_expert,
    execute_expert,
    reduce_partials,
    slice_expert_weights,
)
from swarm_inference.execution.microshard import (
    MicroshardRange,
    physical_microshard_ownership,
    reconstruct_microshard_result,
)
from swarm_inference.execution.moe import (
    HybridMoeBackend,
    LocalMoeBackend,
    MicroshardRemoteBackend,
    MicroshardTarget,
    WholeExpertRemoteBackend,
    WholeExpertTarget,
)
from swarm_inference.protocol.expert import (
    ExpertExecutionMode,
    ExpertExecutionRequest,
    ExpertResponseMode,
    ReductionMode,
    TransportCodec,
)
from swarm_inference.worker.expert_service import ExpertWorkerRuntime


def _request(
    *,
    request_id: str = "request-1",
    model_id: str = "model",
    expert_hash: str = "",
    execution_mode: ExpertExecutionMode = ExpertExecutionMode.WHOLE_EXPERT,
    hidden_start: int | None = None,
    hidden_end: int | None = None,
) -> ExpertExecutionRequest:
    return ExpertExecutionRequest(
        request_id=request_id,
        session_id="session-1",
        token_position=2,
        sequence_id=2,
        route_generation=1,
        topology_id="topology-1",
        model_id=model_id,
        model_revision="revision",
        model_fingerprint="model-fingerprint",
        quantization_fingerprint="quantization-fingerprint",
        layer_id=0,
        batch_rows=2,
        latent_dimension=4,
        expert_ids=[0],
        expert_hashes={0: expert_hash} if expert_hash else {},
        routing_weights=[1.0],
        top_k=1,
        response_mode=ExpertResponseMode.PER_WORKER_FAST,
        activations={},
        deadline_ns=time.time_ns() + 10_000_000_000,
        execution_mode=execution_mode,
        compression=TransportCodec.RAW_FP32,
        hidden_start=hidden_start,
        hidden_end=hidden_end,
        reduction_mode=ReductionMode.FIXED_ORDER_FP32,
    )


def test_exact_whole_expert_and_matched_microshard_union() -> None:
    weights = deterministic_expert(latent_dimension=4, intermediate_dimension=8, seed=1010)
    activation = np.arange(8, dtype=np.float32).reshape(2, 4) / np.float32(7)
    whole = execute_expert(activation, weights)
    first = slice_expert_weights(weights, hidden_start=0, hidden_end=4)
    second = slice_expert_weights(weights, hidden_start=4, hidden_end=8)
    partials = [
        (
            MicroshardRange(
                worker_id="worker-a",
                layer_id=0,
                expert_id=0,
                hidden_start=0,
                hidden_end=4,
                logical_intermediate_dimension=8,
                content_hash=first.content_hash,
            ),
            execute_expert(activation, first, hidden_start=0, hidden_end=4),
        ),
        (
            MicroshardRange(
                worker_id="worker-b",
                layer_id=0,
                expert_id=0,
                hidden_start=4,
                hidden_end=8,
                logical_intermediate_dimension=8,
                content_hash=second.content_hash,
            ),
            execute_expert(activation, second, hidden_start=4, hidden_end=8),
        ),
    ]
    ownership = physical_microshard_ownership([item[0] for item in partials])
    assert ownership["no_worker_owns_full_expert"] is True
    assert all(width < 8 for width in ownership["worker_hidden_units"].values())
    np.testing.assert_allclose(
        reconstruct_microshard_result(partials, mode=ReductionMode.FIXED_ORDER_FP32),
        whole,
        rtol=2e-6,
        atol=2e-8,
    )


def test_microshard_rejects_quantization_group_split_and_full_owner() -> None:
    base = deterministic_expert(latent_dimension=4, intermediate_dimension=8, seed=4)
    grouped = type(base)(
        up=base.up,
        gate=base.gate,
        down=base.down,
        content_hash=base.content_hash,
        scale_group_size=4,
    )
    with pytest.raises(ValueError, match="quantisation group"):
        slice_expert_weights(grouped, hidden_start=2, hidden_end=8)
    with pytest.raises(ValueError, match="full-expert owner"):
        physical_microshard_ownership(
            [
                MicroshardRange(
                    worker_id="worker-a",
                    layer_id=0,
                    expert_id=0,
                    hidden_start=0,
                    hidden_end=8,
                    logical_intermediate_dimension=8,
                    content_hash=base.content_hash,
                )
            ]
        )


def test_fixed_order_reduction_is_deterministic() -> None:
    partials = [
        ("worker-c", np.array([[1e-7, 3.0]], dtype=np.float32)),
        ("worker-a", np.array([[1e8, 1.0]], dtype=np.float32)),
        ("worker-b", np.array([[-1e8, 2.0]], dtype=np.float32)),
    ]
    expected = reduce_partials(partials, mode=ReductionMode.FIXED_ORDER_FP32)
    actual = reduce_partials(list(reversed(partials)), mode=ReductionMode.FIXED_ORDER_FP32)
    np.testing.assert_array_equal(actual, expected)


def _runtime(
    *,
    cache_budget_bytes: int = 10_000,
    roles: set[str] | None = None,
    owned_microshards: list[dict[str, Any]] | None = None,
    resident_weights: Any | None = None,
) -> tuple[ExpertWorkerRuntime, Any]:
    weights = resident_weights or deterministic_expert(
        latent_dimension=4, intermediate_dimension=8, seed=9
    )
    store = ExpertStore(
        owned={(0, 0)},
        loader=lambda _layer, _expert: weights,
        residency_budget_bytes=10_000,
        cache_budget_bytes=cache_budget_bytes,
    )
    from swarm_inference.security.identity import WorkerIdentity

    runtime = ExpertWorkerRuntime(
        worker_id="expert-worker",
        identity=WorkerIdentity.generate(),
        model_id="model",
        model_revision="revision",
        model_fingerprint="model-fingerprint",
        quantization_fingerprint="quantization-fingerprint",
        store=store,
        roles=roles or {"whole-expert"},
        owned_microshards=owned_microshards,
        maximum_queue_depth=2,
    )
    return runtime, weights


@pytest.mark.asyncio
async def test_worker_hash_identity_duplicate_and_cache_telemetry() -> None:
    runtime, weights = _runtime()
    activation = np.ones((2, 4), dtype=np.float32)
    request = _request(expert_hash=weights.content_hash)
    first, first_output = await runtime.execute(
        request, activation, bytes_received=activation.nbytes, decode_ns=4
    )
    duplicate, duplicate_output = await runtime.execute(
        request, activation, bytes_received=activation.nbytes, decode_ns=4
    )
    np.testing.assert_array_equal(duplicate_output, first_output)
    assert duplicate.request_id == first.request_id
    second_request = _request(request_id="request-2", expert_hash=weights.content_hash)
    await runtime.execute(second_request, activation, bytes_received=activation.nbytes, decode_ns=4)
    status = runtime.status()
    assert status["duplicate_requests"] == 1
    assert status["cache_misses"] == 1
    assert status["cache_hits"] == 1
    assert status["whole_expert_compute_ns"] == status["compute_ns"] > 0
    assert status["microshard_compute_ns"] == 0
    with pytest.raises(ValueError, match="model identity"):
        await runtime.execute(
            _request(request_id="bad-model", model_id="other"),
            activation,
            bytes_received=activation.nbytes,
            decode_ns=0,
        )
    with pytest.raises(ValueError, match="exact model fingerprint"):
        await runtime.execute(
            _request(request_id="missing-fingerprint").model_copy(update={"model_fingerprint": ""}),
            activation,
            bytes_received=activation.nbytes,
            decode_ns=0,
        )
    with pytest.raises(ValueError, match="content hash"):
        await runtime.execute(
            _request(request_id="bad-hash", expert_hash="sha256:wrong"),
            activation,
            bytes_received=activation.nbytes,
            decode_ns=0,
        )


@pytest.mark.asyncio
async def test_worker_deadline_cancellation_and_physical_microshard_role() -> None:
    runtime, _ = _runtime()
    activation = np.ones((2, 4), dtype=np.float32)
    expired = _request(request_id="expired").model_copy(update={"deadline_ns": time.time_ns() - 1})
    with pytest.raises(TimeoutError, match="deadline"):
        await runtime.execute(expired, activation, bytes_received=activation.nbytes, decode_ns=0)
    runtime.cancel_session("session-1")
    with pytest.raises(asyncio.CancelledError):
        await runtime.execute(
            _request(request_id="cancelled"),
            activation,
            bytes_received=activation.nbytes,
            decode_ns=0,
        )

    whole_weights = deterministic_expert(latent_dimension=4, intermediate_dimension=8, seed=9)
    sliced_weights = slice_expert_weights(whole_weights, hidden_start=0, hidden_end=4)
    micro, weights = _runtime(
        roles={"expert-microshard"},
        owned_microshards=[
            {
                "layer_id": 0,
                "expert_id": 0,
                "hidden_start": 0,
                "hidden_end": 4,
                "logical_intermediate_dimension": 8,
                "content_hash": sliced_weights.content_hash,
            }
        ],
        resident_weights=sliced_weights,
    )
    response, output = await micro.execute(
        _request(
            request_id="micro",
            expert_hash=weights.content_hash,
            execution_mode=ExpertExecutionMode.MICROSHARD,
            hidden_start=0,
            hidden_end=4,
        ),
        activation,
        bytes_received=activation.nbytes,
        decode_ns=0,
    )
    assert response.status == "ok"
    np.testing.assert_array_equal(
        output, execute_expert(activation, weights, hidden_start=0, hidden_end=4)
    )
    micro_status = micro.status()
    assert micro_status["microshard_compute_ns"] == micro_status["compute_ns"] > 0
    assert micro_status["whole_expert_compute_ns"] == 0


@pytest.mark.asyncio
async def test_worker_queue_is_bounded() -> None:
    started = threading.Event()
    release = threading.Event()
    weights = deterministic_expert(latent_dimension=4, intermediate_dimension=8, seed=10)

    def blocking_loader(_layer: int, _expert: int) -> Any:
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test loader release timed out")
        return weights

    store = ExpertStore(
        owned={(0, 0)},
        loader=blocking_loader,
        residency_budget_bytes=10_000,
        cache_budget_bytes=10_000,
    )
    from swarm_inference.security.identity import WorkerIdentity

    runtime = ExpertWorkerRuntime(
        worker_id="bounded-worker",
        identity=WorkerIdentity.generate(),
        model_id="model",
        model_revision="revision",
        model_fingerprint="model-fingerprint",
        quantization_fingerprint="quantization-fingerprint",
        store=store,
        maximum_queue_depth=1,
    )
    activation = np.ones((2, 4), dtype=np.float32)
    first = asyncio.create_task(
        runtime.execute(
            _request(request_id="queue-1"),
            activation,
            bytes_received=activation.nbytes,
            decode_ns=0,
        )
    )
    try:
        await asyncio.to_thread(started.wait, 5)
        with pytest.raises(OverflowError, match="queue is full"):
            await runtime.execute(
                _request(request_id="queue-2"),
                activation,
                bytes_received=activation.nbytes,
                decode_ns=0,
            )
    finally:
        release.set()
    await first
    assert runtime.status()["queue_rejections"] == 1


@pytest.mark.asyncio
async def test_concurrent_duplicate_request_executes_only_once() -> None:
    started = threading.Event()
    release = threading.Event()
    loads = 0
    weights = deterministic_expert(latent_dimension=4, intermediate_dimension=8, seed=11)

    def blocking_loader(_layer: int, _expert: int) -> Any:
        nonlocal loads
        loads += 1
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test loader release timed out")
        return weights

    store = ExpertStore(
        owned={(0, 0)},
        loader=blocking_loader,
        residency_budget_bytes=10_000,
        cache_budget_bytes=10_000,
    )
    from swarm_inference.security.identity import WorkerIdentity

    runtime = ExpertWorkerRuntime(
        worker_id="expert-worker",
        identity=WorkerIdentity.generate(),
        model_id="model",
        model_revision="revision",
        model_fingerprint="model-fingerprint",
        quantization_fingerprint="quantization-fingerprint",
        store=store,
        maximum_concurrent_requests=2,
    )
    request = _request(request_id="same-request", expert_hash=weights.content_hash)
    activation = np.ones((2, 4), dtype=np.float32)
    first = asyncio.create_task(
        runtime.execute(
            request,
            activation,
            bytes_received=activation.nbytes,
            decode_ns=0,
        )
    )
    await asyncio.to_thread(started.wait, 5)
    second = asyncio.create_task(
        runtime.execute(
            request,
            activation,
            bytes_received=activation.nbytes,
            decode_ns=0,
        )
    )
    release.set()
    first_result, second_result = await asyncio.gather(first, second)
    assert loads == 1
    assert runtime.status()["duplicate_requests"] == 1
    np.testing.assert_array_equal(first_result[1], second_result[1])


def test_microshard_worker_refuses_full_expert_residency() -> None:
    with pytest.raises(ValueError, match="cannot retain the full expert"):
        _runtime(
            roles={"expert-microshard"},
            owned_microshards=[
                {
                    "layer_id": 0,
                    "expert_id": 0,
                    "hidden_start": 0,
                    "hidden_end": 8,
                    "logical_intermediate_dimension": 8,
                    "content_hash": "sha256:not-used",
                }
            ],
        )


class _ExactClient:
    def __init__(self, weights: Any) -> None:
        self.weights = weights

    def execute(
        self,
        request: ExpertExecutionRequest,
        activation: np.ndarray,
        down_accumulators: np.ndarray | None = None,
    ) -> tuple[Any, np.ndarray, dict[str, int]]:
        output = execute_expert(
            activation,
            self.weights,
            hidden_start=request.hidden_start,
            hidden_end=request.hidden_end,
        )
        if request.response_mode == ExpertResponseMode.PER_EXPERT_EXACT:
            seed = (
                np.zeros((request.batch_rows, 1, request.latent_dimension), dtype=np.float32)
                if down_accumulators is None
                else np.ascontiguousarray(down_accumulators, dtype=np.float32).copy()
            )
            seed[:, 0, :] += output
            output = seed
        response = SimpleNamespace(
            status="ok",
            integrity=SimpleNamespace(
                model_fingerprint=request.model_fingerprint,
                expert_hashes=dict(request.expert_hashes),
            ),
        )
        return (
            response,
            output,
            {
                "request_bytes": int(activation.nbytes + 64),
                "response_bytes": int(output.nbytes + 64),
            },
        )


def _routing() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4) / 7
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    selected = torch.tensor([[0], [1]], dtype=torch.long)
    weights = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
    return hidden, logits, selected, weights


def test_remote_whole_and_microshard_backends_reconstruct_selected_experts() -> None:
    experts = {
        (0, 0): deterministic_expert(latent_dimension=4, intermediate_dimension=8, seed=20),
        (0, 1): deterministic_expert(latent_dimension=4, intermediate_dimension=8, seed=21),
    }
    whole = WholeExpertRemoteBackend(
        targets={
            key: WholeExpertTarget(
                worker_id=f"whole-{key[1]}",
                client=_ExactClient(value),
                expert_hash=value.content_hash,
            )
            for key, value in experts.items()
        },
        model_id="model",
        model_revision="revision",
        model_fingerprint="model-fingerprint",
        quantization_fingerprint="quantization-fingerprint",
        topology_id="topology",
        route_generation=1,
    )
    hidden, logits, selected, routing_weights = _routing()
    whole.open_session("session")
    whole_result = whole.execute_layer(
        session_id="session",
        request_id="request",
        token_position=0,
        layer_id=0,
        hidden_states=hidden,
        router_logits=logits,
        selected_experts=selected,
        routing_weights=routing_weights,
        deadline_ns=time.time_ns() + 5_000_000_000,
    )

    micro_targets: dict[tuple[int, int], list[MicroshardTarget]] = {}
    for key, value in experts.items():
        slices = [
            slice_expert_weights(value, hidden_start=0, hidden_end=4),
            slice_expert_weights(value, hidden_start=4, hidden_end=8),
        ]
        micro_targets[key] = [
            MicroshardTarget(
                ownership=MicroshardRange(
                    worker_id=f"micro-{key[1]}-{index}",
                    layer_id=0,
                    expert_id=key[1],
                    hidden_start=index * 4,
                    hidden_end=(index + 1) * 4,
                    logical_intermediate_dimension=8,
                    content_hash=item.content_hash,
                ),
                client=_ExactClient(item),
            )
            for index, item in enumerate(slices)
        ]
    micro = MicroshardRemoteBackend(
        targets=micro_targets,
        model_id="model",
        model_revision="revision",
        model_fingerprint="model-fingerprint",
        quantization_fingerprint="quantization-fingerprint",
        topology_id="topology",
        route_generation=1,
    )
    micro.open_session("session")
    micro_result = micro.execute_layer(
        session_id="session",
        request_id="request",
        token_position=0,
        layer_id=0,
        hidden_states=hidden,
        router_logits=logits,
        selected_experts=selected,
        routing_weights=routing_weights,
        deadline_ns=time.time_ns() + 5_000_000_000,
    )
    torch.testing.assert_close(micro_result.output, whole_result.output, rtol=2e-6, atol=2e-8)
    assert all(item.request_bytes > 0 and item.result_hash for item in whole_result.events)
    assert all(item.request_bytes > 0 and item.result_hash for item in micro_result.events)
    micro.close_session("session")
    micro.close()
    whole.close_session("session")
    whole.close()


class _BarrierExactClient(_ExactClient):
    def __init__(self, weights: Any, barrier: threading.Barrier) -> None:
        super().__init__(weights)
        self.barrier = barrier
        self.calls = 0
        self.response_modes: list[ExpertResponseMode] = []

    def execute(
        self,
        request: ExpertExecutionRequest,
        activation: np.ndarray,
        down_accumulators: np.ndarray | None = None,
    ) -> tuple[Any, np.ndarray, dict[str, int]]:
        self.calls += 1
        self.response_modes.append(request.response_mode)
        self.barrier.wait(timeout=2)
        return super().execute(request, activation, down_accumulators)


def test_microshard_fanout_is_parallel_and_reduction_is_hierarchical() -> None:
    weights = deterministic_expert(latent_dimension=4, intermediate_dimension=8, seed=1200)
    activation = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 7
    expected = execute_expert(activation.numpy(), weights)
    barrier = threading.Barrier(4)
    clients: list[_BarrierExactClient] = []
    targets: list[MicroshardTarget] = []
    for index in range(4):
        start, end = index * 2, (index + 1) * 2
        sliced = slice_expert_weights(weights, hidden_start=start, hidden_end=end)
        client = _BarrierExactClient(sliced, barrier)
        clients.append(client)
        targets.append(
            MicroshardTarget(
                ownership=MicroshardRange(
                    worker_id=f"parallel-{index}",
                    layer_id=0,
                    expert_id=0,
                    hidden_start=start,
                    hidden_end=end,
                    logical_intermediate_dimension=8,
                    content_hash=sliced.content_hash,
                ),
                client=client,
            )
        )
    backend = MicroshardRemoteBackend(
        targets={(0, 0): targets},
        model_id="model",
        model_revision="revision",
        model_fingerprint="model-fingerprint",
        quantization_fingerprint="quantization-fingerprint",
        topology_id="topology",
        route_generation=1,
        maximum_parallel_requests=4,
        reduction_branching_factor=2,
    )
    backend.open_session("session")
    try:
        output, event = backend.execute_expert_rows(
            session_id="session",
            request_id="request",
            token_position=0,
            layer_id=0,
            expert_id=0,
            activation=activation,
            deadline_ns=time.time_ns() + 5_000_000_000,
        )
        np.testing.assert_allclose(output.numpy(), expected, rtol=2e-6, atol=2e-8)
        assert [client.calls for client in clients] == [1, 1, 1, 1]
        assert all(
            client.response_modes == [ExpertResponseMode.PER_WORKER_FAST]
            for client in clients
        )
        assert event.total_messages == 8
        assert event.critical_path_messages == 2
        assert event.parallel_waits == 4
        assert event.serial_waits == 3
        assert event.fanout_depth == 1
        assert event.reduction_depth == 2
        assert event.critical_path_sync_rounds == 3
        assert event.scheduler_dispatch_ns > 0
        assert event.reduction_ns > 0
        assert event.root_dispatches == 4
        assert event.coordinator_waits == 0
        assert event.coordinator_sync_rounds == 0
        assert event.worker_sync_rounds == 3
        assert event.fanout_nodes == 4
        assert event.topology_construction_ns > 0
    finally:
        backend.close_session("session")
        backend.close()


class _FailingClient:
    def execute(self, _request: Any, _activation: Any) -> Any:
        raise TimeoutError("remote worker timed out")


@pytest.mark.parametrize(
    ("branching_factor", "expected_depth"),
    ((4, 5), (8, 4), (16, 3), (32, 2)),
)
def test_microshard_fanout_topology_bounds_root_dispatch_at_one_thousand(
    branching_factor: int,
    expected_depth: int,
) -> None:
    targets = [
        MicroshardTarget(
            ownership=MicroshardRange(
                worker_id=f"logical-{index:04d}",
                layer_id=0,
                expert_id=0,
                hidden_start=index,
                hidden_end=index + 1,
                logical_intermediate_dimension=1000,
                content_hash=f"shard-{index:04d}",
            ),
            client=_FailingClient(),
        )
        for index in range(1000)
    ]
    backend = MicroshardRemoteBackend(
        targets={(0, 0): targets},
        model_id="model",
        model_revision="revision",
        model_fingerprint="fingerprint",
        quantization_fingerprint="quant",
        topology_id="topology",
        route_generation=1,
        maximum_parallel_requests=8,
        fanout_branching_factor=branching_factor,
        reduction_branching_factor=8,
    )
    try:
        topology = backend._fanout_topologies[(0, 0)]
        assert len(topology.root_children) <= branching_factor
        assert topology.depth == expected_depth
        assert topology.node_count > 1000
        assert backend.topology_construction_ns > 0
    finally:
        backend.close()


def test_forced_remote_backend_refuses_local_fallback() -> None:
    whole = WholeExpertRemoteBackend(
        targets={(0, 0): WholeExpertTarget(worker_id="remote", client=_FailingClient())},
        model_id="model",
        model_revision="revision",
        model_fingerprint="fingerprint",
        quantization_fingerprint="quant",
        topology_id="topology",
        route_generation=1,
    )
    hybrid = HybridMoeBackend(
        local=LocalMoeBackend({(0, 0): nn.Identity()}),
        whole_remote=whole,
        placement={(0, 0): "whole-remote"},
        allow_local_fallback=True,
        require_remote=True,
    )
    hybrid.open_session("session")
    with pytest.raises(TimeoutError, match="timed out"):
        hybrid.execute_layer(
            session_id="session",
            request_id="request",
            token_position=0,
            layer_id=0,
            hidden_states=torch.ones((1, 1, 4)),
            router_logits=torch.ones((1, 1)),
            selected_experts=torch.zeros((1, 1), dtype=torch.long),
            routing_weights=torch.ones((1, 1)),
            deadline_ns=time.time_ns() + 5_000_000_000,
        )


def test_hybrid_fallback_permission_is_scoped_to_one_expert_placement() -> None:
    first = deterministic_expert(latent_dimension=4, intermediate_dimension=8, seed=30)
    whole = WholeExpertRemoteBackend(
        targets={
            (0, 0): WholeExpertTarget(
                worker_id="healthy",
                client=_ExactClient(first),
                expert_hash=first.content_hash,
            ),
            (0, 1): WholeExpertTarget(worker_id="failed", client=_FailingClient()),
        },
        model_id="model",
        model_revision="revision",
        model_fingerprint="fingerprint",
        quantization_fingerprint="quant",
        topology_id="topology",
        route_generation=1,
    )
    hybrid = HybridMoeBackend(
        local=LocalMoeBackend({(0, 0): nn.Identity(), (0, 1): nn.Identity()}),
        whole_remote=whole,
        placement={(0, 0): "whole-remote", (0, 1): "whole-remote"},
        fallback_placements={(0, 0)},
    )
    hybrid.open_session("session")
    with pytest.raises(TimeoutError, match="timed out"):
        hybrid.execute_layer(
            session_id="session",
            request_id="request",
            token_position=0,
            layer_id=0,
            hidden_states=torch.ones((1, 1, 4)),
            router_logits=torch.ones((1, 2)),
            selected_experts=torch.tensor([[0, 1]], dtype=torch.long),
            routing_weights=torch.tensor([[0.5, 0.5]]),
            deadline_ns=time.time_ns() + 5_000_000_000,
        )


def _planner_candidates(*, local_fits: bool, remote_utility_ms: float) -> list[Any]:
    remote_cost = 2.0 - remote_utility_ms
    return [
        ExpertStrategyCandidate(
            candidate_id="local",
            strategy=ExpertStrategy.LOCAL,
            memory_required_bytes=10,
            memory_available_bytes=10 if local_fits else 9,
        ),
        ExpertStrategyCandidate(
            candidate_id="remote",
            strategy=ExpertStrategy.WHOLE_REMOTE,
            worker_ids=["worker"],
            memory_required_bytes=4,
            memory_available_bytes=10,
            utility=ExpertUtilityInputs(
                measured_local_expert_ms=2.0,
                measured_remote_expert_ms=max(0.0, remote_cost),
                serialization_ms=0,
                network_transfer_ms=0,
                queue_delay_ms=0,
                reduction_ms=0,
                cache_hit_rate=1,
            ),
        ),
    ]


def test_planner_uses_positive_utility_capacity_and_local_preference() -> None:
    planner = ExpertUtilityPlanner()
    positive = planner.choose(
        stage_id=0,
        layer_id=0,
        expert_id=0,
        candidates=_planner_candidates(local_fits=True, remote_utility_ms=1.0),
    )
    assert positive.selected_strategy == ExpertStrategy.WHOLE_REMOTE
    negative = planner.choose(
        stage_id=0,
        layer_id=0,
        expert_id=0,
        candidates=_planner_candidates(local_fits=True, remote_utility_ms=-1.0),
    )
    assert negative.selected_strategy == ExpertStrategy.LOCAL
    assert any("non-positive" in reason for item in negative.rejected for reason in item.reasons)
    hybrid_negative = planner.choose(
        stage_id=0,
        layer_id=0,
        expert_id=0,
        candidates=_planner_candidates(local_fits=True, remote_utility_ms=-1.0),
        policy="hybrid",
    )
    assert hybrid_negative.selected_strategy == ExpertStrategy.LOCAL
    capacity = planner.choose(
        stage_id=0,
        layer_id=0,
        expert_id=0,
        candidates=_planner_candidates(local_fits=False, remote_utility_ms=-1.0),
    )
    assert capacity.selected_strategy == ExpertStrategy.WHOLE_REMOTE
    assert capacity.capacity_required


def test_planner_enforces_microshard_memory_per_physical_owner() -> None:
    planner = ExpertUtilityPlanner()
    utility = ExpertUtilityInputs(
        measured_local_expert_ms=2.0,
        measured_remote_expert_ms=1.0,
        serialization_ms=0,
        network_transfer_ms=0,
        queue_delay_ms=0,
        reduction_ms=0,
        cache_hit_rate=1,
    )
    feasible = ExpertStrategyCandidate(
        candidate_id="heterogeneous-microshards",
        strategy=ExpertStrategy.MICROSHARD_REMOTE,
        worker_ids=["large-owner", "small-owner"],
        utility=utility,
        memory_required_bytes=100,
        memory_available_bytes=150,
        worker_memory_required_bytes={"large-owner": 80, "small-owner": 20},
        worker_memory_available_bytes={"large-owner": 100, "small-owner": 50},
    )
    selected = planner.choose(
        stage_id=0,
        layer_id=0,
        expert_id=0,
        candidates=[feasible],
        policy="microshard-remote",
        require_remote=True,
    )
    assert selected.selected_candidate_id == feasible.candidate_id

    over_budget = feasible.model_copy(
        update={
            "candidate_id": "over-budget-microshards",
            "worker_memory_required_bytes": {"large-owner": 80, "small-owner": 60},
        }
    )
    decision = planner.choose(
        stage_id=0,
        layer_id=0,
        expert_id=0,
        candidates=[*_planner_candidates(local_fits=True, remote_utility_ms=-1.0), over_budget],
    )
    rejected = next(
        item for item in decision.rejected if item.candidate_id == over_budget.candidate_id
    )
    assert rejected.reasons == [
        "worker small-owner requires 60 bytes but only 50 bytes are available"
    ]
