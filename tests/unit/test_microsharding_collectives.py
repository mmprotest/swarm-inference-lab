from __future__ import annotations

import math

import pytest
import torch

from swarm_inference.config.models import TensorSpec
from swarm_inference.microsharding.collectives import SingleDeviceLogicalBackend
from swarm_inference.microsharding.projection import (
    NETWORK_PROFILES,
    CollectiveWork,
    EventDrivenProjector,
    NetworkProfile,
    collective_shape,
    estimate_collective,
    synchronous_group_decision,
    validate_projector,
)
from swarm_inference.microsharding.schemas import CollectivePlan
from swarm_inference.microsharding.sequence_parallel import sequence_parallel_rms_norm


def _plan(operation: str = "all_reduce_sum", *, timeout_ms: int = 1000) -> CollectivePlan:
    return CollectivePlan(
        collective_id=f"test-{operation}",
        operation=operation,
        group_id="g",
        rank_ids=["r0", "r1"],
        tensor_spec=TensorSpec(dtype="float32", shape=[2]),
        algorithm="ring",
        timeout_ms=timeout_ms,
    )


@pytest.mark.asyncio
async def test_single_device_collective_semantics() -> None:
    backend = SingleDeviceLogicalBackend(measure_cuda_time=False)
    values = {"r0": torch.tensor([1.0, 2.0]), "r1": torch.tensor([3.0, 4.0])}
    broadcast = await backend.broadcast(_plan("broadcast"), "r0", values["r0"])
    assert all(torch.equal(value, values["r0"]) for value in broadcast.values())
    reduced = await backend.all_reduce_sum(_plan(), values)
    assert all(torch.equal(value, torch.tensor([4.0, 6.0])) for value in reduced.values())
    gathered = await backend.all_gather(_plan("all_gather"), values)
    assert torch.equal(gathered["r0"], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    reduce_scatter = await backend.reduce_scatter_sum(
        _plan("reduce_scatter_sum"),
        {
            "r0": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "r1": torch.tensor([10.0, 20.0, 30.0, 40.0]),
        },
    )
    assert torch.equal(reduce_scatter["r0"], torch.tensor([11.0, 22.0]))
    assert torch.equal(reduce_scatter["r1"], torch.tensor([33.0, 44.0]))
    gathered_leader = await backend.gather_to_leader(
        _plan("gather_to_leader"), values, leader_rank="r1"
    )
    assert gathered_leader["r0"] is None
    assert torch.equal(gathered_leader["r1"], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    await backend.barrier(_plan("barrier"))
    assert {row["operation"] for row in backend.trace} >= {
        "broadcast",
        "all_reduce_sum",
        "all_gather",
        "reduce_scatter_sum",
        "gather_to_leader",
        "barrier",
    }


@pytest.mark.asyncio
async def test_all_to_all_and_distributed_argmax() -> None:
    backend = SingleDeviceLogicalBackend(measure_cuda_time=False)
    values = {
        "r0": {"r0": torch.tensor([0]), "r1": torch.tensor([1])},
        "r1": {"r0": torch.tensor([2]), "r1": torch.tensor([3])},
    }
    result = await backend.all_to_all(_plan("all_to_all"), values)
    assert result["r0"]["r0"].item() == 0
    assert result["r0"]["r1"].item() == 2
    maximum = await backend.distributed_argmax(
        _plan("distributed_argmax"),
        {
            "r0": (torch.tensor([5.0]), torch.tensor([20])),
            "r1": (torch.tensor([5.0]), torch.tensor([10])),
        },
    )
    assert maximum["r0"][0].item() == 5
    assert maximum["r1"][1].item() == 10


@pytest.mark.asyncio
async def test_collective_invalid_group_and_timeout() -> None:
    backend = SingleDeviceLogicalBackend(measure_cuda_time=False)
    with pytest.raises(ValueError, match="membership mismatch"):
        await backend.all_reduce_sum(_plan(), {"r0": torch.ones(2)})
    timeout_backend = SingleDeviceLogicalBackend(measure_cuda_time=False, simulated_delay_ms=2)
    with pytest.raises(TimeoutError):
        await timeout_backend.broadcast(_plan("broadcast", timeout_ms=1), "r0", torch.ones(2))


def test_collective_analytical_ring_and_tree_cases() -> None:
    assert collective_shape(
        operation="all_reduce_sum", algorithm="ring", rank_count=4, payload_bytes=400
    ) == (6, 100.0, 600.0, 2400.0)
    tree = collective_shape(
        operation="all_reduce_sum",
        algorithm="binary_tree",
        rank_count=4,
        payload_bytes=400,
    )
    assert tree == (4, 400.0, 800.0, 2400.0)
    two = estimate_collective(
        operation="all_reduce_sum",
        algorithm="ring",
        rank_count=2,
        payload_bytes=1_000_000,
        network=NetworkProfile("known", 1.0, 1000),
    )
    assert math.isclose(two.completion_time_ms, 10.0)
    assert validate_projector()["status"] == "PASS"


def test_event_projector_independent_shared_straggler_and_failure() -> None:
    projector = EventDrivenProjector(seed=17)
    work = CollectiveWork(
        collective_id="reduce",
        operation="all_reduce_sum",
        algorithm="recursive_doubling",
        payload_bytes=128,
        rank_ids=("r0", "r1", "r2", "r3"),
        phase="attention",
    )
    independent = projector.project_layer(
        layer_id=4,
        rank_compute_ms={"r0": 1, "r1": 2, "r2": 3, "r3": 4},
        collectives=[work],
        network=NETWORK_PROFILES["nvlink_class"],
        resource_mode="independent",
        straggler_delays_ms={"r2": 4},
        failure_rank="r1",
        rejoin_delay_ms=5,
    )
    shared = projector.project_layer(
        layer_id=4,
        rank_compute_ms={"r0": 1, "r1": 2, "r2": 3, "r3": 4},
        collectives=[],
        network=NETWORK_PROFILES["same_gpu_logical"],
        resource_mode="shared",
    )
    assert independent.compute_completion_time_ms == 7
    assert shared.completion_time_ms == 10
    event_types = {event.event_type for event in independent.events}
    assert {"rank_failure", "rank_rejoin", "collective_complete", "layer_complete"} <= event_types
    repeat = projector.project_layer(
        layer_id=4,
        rank_compute_ms={"r0": 1, "r1": 2, "r2": 3, "r3": 4},
        collectives=[work],
        network=NETWORK_PROFILES["nvlink_class"],
        resource_mode="independent",
        straggler_delays_ms={"r2": 4},
        failure_rank="r1",
        rejoin_delay_ms=5,
    )
    assert independent.completion_time_ms == repeat.completion_time_ms


def test_scheduler_rejects_non_positive_weak_rank() -> None:
    weak = synchronous_group_decision(
        existing_compute_ms=[1.0, 1.1],
        candidate_compute_ms=20.0,
        memory_feasibility_gain=0.1,
        added_collective_ms=1.0,
    )
    assert weak["marginal_benefit"] <= 0
    assert weak["join_synchronous_tensor_group"] is False
    assert "do not place" in weak["scheduler_rule"]


def test_sequence_parallel_prefill_is_exact_and_decode_disables_it() -> None:
    torch.manual_seed(3)
    hidden = torch.randn(2, 17, 8)
    weight = torch.randn(8)
    full = sequence_parallel_rms_norm(hidden, weight, degree=1, eps=1e-6)
    sharded = sequence_parallel_rms_norm(hidden, weight, degree=4, eps=1e-6)
    assert torch.equal(full.output, sharded.output)
    assert sharded.position_ranges == ((0, 5), (5, 9), (9, 13), (13, 17))
    assert sharded.collective_operations == ("all_gather",)
    decode = sequence_parallel_rms_norm(hidden[:, :1], weight, degree=4, eps=1e-6)
    assert decode.enabled is False
    assert "decode" in str(decode.disable_reason)
