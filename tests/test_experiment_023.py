from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from swarm_inference.experiments.experiment_019.checkpoint import balanced_range
from swarm_inference.experiments.experiment_022.model_graph import SHARD_BUFFER_BYTES
from swarm_inference.experiments.experiment_022.models import (
    Inventory,
    LayerAssignment,
    LayerSpec,
    LayerType,
    NetworkPeer,
    NodeCapability,
    PartitionKind,
    PlacementPlan,
    PlannerLevel,
)
from swarm_inference.experiments.experiment_023.baseline import frozen_placement_paths
from swarm_inference.experiments.experiment_023.correctness import (
    ReplicaAwareManifestK3Runner,
)
from swarm_inference.experiments.experiment_023.freeze import FROZEN_CONSTANTS
from swarm_inference.experiments.experiment_023.models import (
    Arm,
    E023Plan,
    ExpertGroupReplica,
    NetworkMode,
    canonical_group_reduce,
)
from swarm_inference.experiments.experiment_023.replica_memory import (
    abstract_node_cost,
    reconcile_plan_memory,
    standalone_whole_expert_group_memory,
)
from swarm_inference.experiments.experiment_023.replica_planner import (
    PLANNER_RANDOMIZATION,
    PlannerActionCandidate,
    PlanningProfile,
    PoolEnvelopeCandidate,
    PrimaryGroupCandidate,
    choose_best_action,
    clone_as_flex_pool,
    generate_alternate_assignment_candidates,
    retain_primary_group_candidates,
    select_flex_pool_envelope,
)
from swarm_inference.experiments.experiment_023.resource_calendar import (
    ResourceCalendar,
    ResourceCalendars,
    reserve_transfer,
)
from swarm_inference.experiments.experiment_023.routing import (
    GroupExecutionSpec,
    select_fork_join_routes,
)
from swarm_inference.experiments.experiment_023.serving_engine import (
    CONCURRENCY_LEVELS,
    TARGET_ROWS,
    measured_passes_per_slot,
)
from swarm_inference.experiments.experiment_023.serving_objective import (
    SLOScore,
    score_runs_under_latency_budget,
)


def _layer(layer_id: int = 89) -> LayerSpec:
    return LayerSpec(
        layer_id=layer_id,
        layer_type=LayerType.KDA,
        checkpoint_bytes=8_960_000,
        resident_bytes=17_920_000,
        component_bytes={"routed_expert": 8_000_000, "other": 960_000},
        tensor_count=100,
        attnres_snapshot=False,
    )


def _node(
    node_id: str,
    peer_ids: tuple[str, ...],
    *,
    memory: int = 100_000_000,
    cost: float = 1.0,
) -> NodeCapability:
    peers = {
        peer: NetworkPeer(peer, 1.0, 10.0)
        for peer in peer_ids
        if peer != node_id
    }
    return NodeCapability(
        node_id=node_id,
        accelerator_memory_bytes=memory,
        system_memory_bytes=memory * 2,
        compute_profile={"reference": 1.0},
        memory_bandwidth_profile={"reference": 1.0},
        supported_precisions=("MXFP4",),
        network_peers=peers,
        reliability=1.0,
        cost=cost,
        cached_shards=(),
        runtime_capabilities=("WHOLE_LAYER", "WHOLE_EXPERT"),
        locality_group="g0",
    )


def _inventory(*, n2_memory: int = 100_000_000) -> Inventory:
    node_ids = (*tuple(f"w{index}" for index in range(8)), "n2")
    return Inventory(
        inventory_id="fixture",
        family="fixture",
        seed=1,
        nodes=(
            _node("w0", node_ids, cost=2.0),
            _node("w1", node_ids, cost=3.0),
            *(_node(f"w{index}", node_ids, cost=0.0) for index in range(2, 8)),
            _node("n2", node_ids, memory=n2_memory, cost=5.0),
        ),
        evidence_class="SYNTHETIC",
        generator_version="test",
        scenario="test",
    )


def _base_plan(layer: LayerSpec | None = None) -> PlacementPlan:
    layer = layer or _layer()
    nodes = tuple(f"w{index}" for index in range(8))
    assignment = LayerAssignment(
        layer_id=layer.layer_id,
        partition_kind=PartitionKind.WHOLE_EXPERT,
        degree=8,
        node_ids=nodes,
        memory_by_node={node: 2_000_000 for node in nodes},
        checkpoint_bytes_by_node={node: 1_000_000 for node in nodes},
        coordinator_node_id="w0",
        candidate_id=f"layer-{layer.layer_id:02d}:WHOLE_EXPERT:p8",
    )
    return PlacementPlan(
        inventory_id="fixture",
        planner_level=PlannerLevel.B,
        chunk_rows=1,
        assignments=[assignment],
        endpoint_memory_by_node={"w0": 1_000_000},
        endpoint_checkpoint_bytes_by_node={"w0": 500_000},
        feasible=True,
    )


def _p8_plan() -> PlacementPlan:
    plan = _base_plan()
    plan.inventory_id = "fixture-p8"
    return plan


def test_replica_group_zero_and_seven_exclude_coordinator_fixed_bytes() -> None:
    layer = _layer()
    zero = standalone_whole_expert_group_memory(layer, 0)
    seven = standalone_whole_expert_group_memory(layer, 7)
    assert zero.checkpoint_bytes == 1_000_000
    assert seven.checkpoint_bytes == 1_000_000
    assert zero.resident_bytes == 2_000_000 + SHARD_BUFFER_BYTES
    assert seven.resident_bytes == 2_000_000 + SHARD_BUFFER_BYTES
    assert zero.persistent_state_bytes == seven.persistent_state_bytes == 0


def test_eight_p8_expert_ranges_are_disjoint_and_cover_all_experts() -> None:
    ranges = [balanced_range(896, 8, group) for group in range(8)]
    covered = [expert for owned in ranges for expert in range(owned.start, owned.stop)]
    assert covered == list(range(896))
    assert len(covered) == len(set(covered))


def test_only_stateless_whole_expert_p8_can_be_replicated() -> None:
    replica = ExpertGroupReplica(
        layer_id=89,
        logical_group_id=0,
        primary_node_id="w0",
        alternate_node_id="alt",
        checkpoint_bytes=1,
        resident_bytes=2,
    )
    assert replica.persistent_state_bytes == 0
    for kind in (
        PartitionKind.WHOLE_LAYER,
        PartitionKind.EXPERT_SHARD,
        PartitionKind.ATTENTION_PROJECTION_SHARD,
        PartitionKind.FULL_MIXED_STRIPE,
    ):
        with pytest.raises(ValueError, match="WHOLE_EXPERT:p8"):
            replica.validate_partition(kind, 8)


def test_maximum_two_copies_per_logical_group() -> None:
    base = _p8_plan()
    replica = ExpertGroupReplica(89, 0, "w0", "alt-a", 1, 2)
    with pytest.raises(ValueError, match="duplicate alternate"):
        E023Plan(
            "fixture-p8",
            "FLEX_POOL",
            base,
            (replica, ExpertGroupReplica(89, 0, "w0", "alt-b", 1, 2)),
        )


def test_primary_plus_replica_memory_reconciles_and_capacity_errors_are_detected() -> None:
    layer = _layer()
    memory = standalone_whole_expert_group_memory(layer, 0)
    base = _base_plan(layer)
    replica = ExpertGroupReplica(
        layer_id=89,
        logical_group_id=0,
        primary_node_id="w0",
        alternate_node_id="n2",
        checkpoint_bytes=memory.checkpoint_bytes,
        resident_bytes=memory.resident_bytes,
    )
    plan = E023Plan("fixture", "FLEX_POOL", base, (replica,))
    rows = reconcile_plan_memory(plan, _inventory())
    n2 = next(row for row in rows if row.node_id == "n2")
    assert n2.base_resident_bytes == 0
    assert n2.replica_resident_bytes == memory.resident_bytes
    assert n2.total_resident_bytes == memory.resident_bytes
    with pytest.raises(ValueError, match="capacity"):
        reconcile_plan_memory(plan, _inventory(n2_memory=memory.resident_bytes - 1))


def test_flex_free_cannot_activate_node_outside_u_strong() -> None:
    base = _base_plan()
    replica = ExpertGroupReplica(89, 0, "w0", "n2", 1, 2)
    with pytest.raises(ValueError, match="FLEX_FREE"):
        E023Plan(
            "fixture",
            "FLEX_FREE",
            base,
            (replica,),
            u_strong_used_nodes=tuple(f"w{index}" for index in range(8)),
        )


def test_node_cost_counted_once_and_replica_only_node_counted() -> None:
    base = _base_plan()
    no_replica = E023Plan("fixture", "U_STRONG", base)
    assert abstract_node_cost(no_replica, _inventory()) == 5.0
    replicas = (
        ExpertGroupReplica(89, 0, "w0", "n2", 1, 2),
        ExpertGroupReplica(89, 1, "w1", "n2", 1, 2),
    )
    with_replica = E023Plan("fixture", "FLEX_POOL", base, replicas)
    assert abstract_node_cost(with_replica, _inventory()) == 10.0


def test_physical_arrival_and_replica_choice_cannot_change_reduction_order() -> None:
    contributions = {
        group: np.full((2,), float(group + 1), dtype=np.float64)
        for group in reversed(range(8))
    }
    result, order = canonical_group_reduce(contributions)
    assert order == tuple(range(8))
    assert np.array_equal(result, np.array([36.0, 36.0]))
    primary = {group: value.copy() for group, value in contributions.items()}
    alternate = {group: value.copy() for group, value in contributions.items()}
    alternate[3] = primary[3].copy()
    primary_result, primary_order = canonical_group_reduce(primary)
    alternate_result, alternate_order = canonical_group_reduce(alternate)
    assert primary_order == alternate_order == tuple(range(8))
    assert np.array_equal(primary_result, alternate_result)


def test_interval_calendar_finds_gaps_and_rejects_overlap() -> None:
    calendar = ResourceCalendar("compute:w0")
    calendar.reserve(10.0, 10.0, {"id": "later"})
    assert calendar.earliest_start(0.0, 5.0) == 0.0
    calendar.reserve(0.0, 5.0, {"id": "first"})
    assert calendar.earliest_start(0.0, 6.0) == 20.0
    with pytest.raises(ValueError, match="overlap"):
        calendar.reserve(4.0, 2.0, {"id": "invalid"})


def test_shared_nic_transfer_reserves_link_tx_and_rx() -> None:
    inventory = _inventory()
    calendars = ResourceCalendars()
    first = reserve_transfer(
        calendars,
        inventory,
        NetworkMode.SHARED_NIC,
        source="w0",
        destination="w1",
        earliest_ms=0.0,
        payload_bytes=1_000_000,
        metadata={"id": "first"},
    )
    assert set(first.resource_ids) == {"link:w0->w1", "nic_tx:w0", "nic_rx:w1"}
    second = reserve_transfer(
        calendars,
        inventory,
        NetworkMode.SHARED_NIC,
        source="w0",
        destination="n2",
        earliest_ms=0.0,
        payload_bytes=1_000_000,
        metadata={"id": "second"},
    )
    assert second.start_ms >= first.finish_ms
    opposite = reserve_transfer(
        calendars,
        inventory,
        NetworkMode.SHARED_NIC,
        source="w1",
        destination="w0",
        earliest_ms=0.0,
        payload_bytes=1_000_000,
        metadata={"id": "opposite"},
    )
    assert opposite.start_ms == 0.0
    assert "nic_tx:w1" in opposite.resource_ids and "nic_rx:w0" in opposite.resource_ids


def test_legacy_transfer_reserves_only_directed_link() -> None:
    result = reserve_transfer(
        ResourceCalendars(),
        _inventory(),
        NetworkMode.LEGACY_DIRECTED_LINK,
        source="w0",
        destination="w1",
        earliest_ms=0.0,
        payload_bytes=1024,
        metadata={"id": "legacy"},
    )
    assert result.resource_ids == ("link:w0->w1",)


def test_routing_mask_is_deterministic_and_current_queue_changes_copy() -> None:
    inventory = _inventory()
    spec = GroupExecutionSpec(
        logical_group_id=0,
        coordinator_node_id="w0",
        primary_node_id="w1",
        alternate_node_id="n2",
        ready_ms=0.0,
        input_bytes=1024,
        output_bytes=1024,
        worker_protocol_ms=0.1,
        compute_ms_by_node={"w1": 1.0, "n2": 1.0},
    )
    empty = ResourceCalendars()
    first = select_fork_join_routes(
        (spec,), empty, inventory, NetworkMode.SHARED_NIC, commit=False
    )
    second = select_fork_join_routes(
        (spec,), empty, inventory, NetworkMode.SHARED_NIC, commit=False
    )
    assert first.mask == second.mask == 0
    queued = ResourceCalendars()
    queued.reserve(
        ("compute:w1",),
        earliest_ms=0.0,
        duration_ms=100.0,
        metadata={"id": "existing"},
    )
    changed = select_fork_join_routes(
        (spec,), queued, inventory, NetworkMode.SHARED_NIC, commit=False
    )
    assert changed.mask == 1
    assert changed.selected_nodes[0] == "n2"


def test_no_replica_routing_is_primary_only() -> None:
    spec = GroupExecutionSpec(
        logical_group_id=0,
        coordinator_node_id="w0",
        primary_node_id="w1",
        alternate_node_id=None,
        ready_ms=0.0,
        input_bytes=1024,
        output_bytes=1024,
        worker_protocol_ms=0.1,
        compute_ms_by_node={"w1": 1.0},
    )
    choice = select_fork_join_routes(
        (spec,), ResourceCalendars(), _inventory(), NetworkMode.SHARED_NIC
    )
    assert choice.mask == 0
    assert choice.selected_nodes == {0: "w1"}


def _action_candidate(
    *,
    objective: float,
    throughput: float,
    layer: int = 10,
    added_bytes: int = 100,
    arm: Arm = Arm.FLEX_POOL,
) -> PlannerActionCandidate:
    used = tuple(f"w{index}" for index in range(8))
    plan = E023Plan(
        "fixture",
        arm.value,
        _base_plan(),
        u_strong_used_nodes=used if arm is Arm.FLEX_FREE else (),
    )
    return PlannerActionCandidate(
        plan=plan,
        layer_id=layer,
        action_type="test",
        replica_count=2,
        primary_group_nodes=tuple(f"w{index}" for index in range(8)),
        replicated_logical_groups=(0, 1),
        alternate_nodes=((0, "n2"), (1, "n2")),
        added_checkpoint_bytes=50,
        added_resident_bytes=added_bytes,
        newly_activated_nodes=("n2",),
        expansion_score_before=1.0,
        expansion_score_after=1.25,
        target_rows_per_second=throughput,
        objective=objective,
        relative_gain=objective / 100.0 - 1.0,
        throughput_ratio=throughput / 100.0,
    )


def test_planner_gain_regression_limits_and_no_forced_action() -> None:
    assert choose_best_action(
        [_action_candidate(objective=100.49, throughput=100.0)],
        current_objective=100.0,
        current_throughput=100.0,
    ) is None
    assert choose_best_action(
        [_action_candidate(objective=110.0, throughput=98.99)],
        current_objective=100.0,
        current_throughput=100.0,
    ) is None
    winner = choose_best_action(
        [_action_candidate(objective=100.5000000001, throughput=99.0)],
        current_objective=100.0,
        current_throughput=100.0,
    )
    assert winner is not None


def test_planner_tie_breakers_are_deterministic_and_action_limit_is_six() -> None:
    candidates = [
        _action_candidate(objective=101.0, throughput=100.0, layer=11, added_bytes=100),
        _action_candidate(objective=101.0, throughput=100.0, layer=10, added_bytes=100),
        _action_candidate(objective=101.0, throughput=100.0, layer=9, added_bytes=101),
    ]
    winner = choose_best_action(
        candidates, current_objective=100.0, current_throughput=100.0
    )
    assert winner is candidates[1]
    assert FROZEN_CONSTANTS["planner_max_accepted_layer_actions"] == 6
    assert PLANNER_RANDOMIZATION == 0.0


def _slo_run(concurrency: int, throughput: float, p95_ms: float) -> SimpleNamespace:
    return SimpleNamespace(
        status="PASS",
        concurrency=concurrency,
        target_rows_per_second=throughput,
        p50_pass_latency_ms=0.8 * p95_ms,
        p95_pass_latency_ms=p95_ms,
    )


def _slo_fixture(
    *,
    c1: tuple[float, float],
    c8: tuple[float, float],
    c32: tuple[float, float],
) -> dict[int, SimpleNamespace]:
    return {
        1: _slo_run(1, *c1),
        8: _slo_run(8, *c8),
        32: _slo_run(32, *c32),
        64: _slo_run(64, c32[0] * 0.95, c32[1] * 1.5),
        128: _slo_run(128, c32[0] * 0.90, c32[1] * 2.0),
    }


def test_c32_improvement_cannot_override_primary_slo_cliff() -> None:
    current = score_runs_under_latency_budget(
        _slo_fixture(c1=(10.0, 100.0), c8=(50.0, 190.0), c32=(60.0, 300.0)),
        latency_budget_ms=200.0,
        abstract_cost=1.0,
    )
    candidate = score_runs_under_latency_budget(
        _slo_fixture(c1=(11.0, 100.0), c8=(50.2, 201.0), c32=(61.0, 295.0)),
        latency_budget_ms=200.0,
        abstract_cost=1.0,
    )
    action = _action_candidate(
        objective=candidate.rows_per_second_per_abstract_cost,
        throughput=candidate.target_rows_per_second,
    )
    action.c32_objective = 61.0
    action.c32_target_rows_per_second = 61.0
    assert action.c32_objective / 60.0 - 1.0 > 0.005
    assert action.c32_target_rows_per_second / 60.0 >= 0.99
    assert current.selected_concurrency == 8
    assert candidate.selected_concurrency == 1
    assert choose_best_action(
        [action],
        current_objective=current.rows_per_second_per_abstract_cost,
        current_throughput=current.target_rows_per_second,
        u_strong_throughput=current.target_rows_per_second,
    ) is None


def test_slo_positive_candidate_is_accepted_without_c32_gate() -> None:
    current = score_runs_under_latency_budget(
        _slo_fixture(c1=(10.0, 100.0), c8=(50.0, 190.0), c32=(60.0, 300.0)),
        latency_budget_ms=200.0,
        abstract_cost=1.0,
    )
    candidate = score_runs_under_latency_budget(
        _slo_fixture(c1=(10.0, 100.0), c8=(50.5, 195.0), c32=(59.0, 300.0)),
        latency_budget_ms=200.0,
        abstract_cost=1.0,
    )
    action = _action_candidate(
        objective=candidate.rows_per_second_per_abstract_cost,
        throughput=candidate.target_rows_per_second,
    )
    action.c32_objective = 59.0
    winner = choose_best_action(
        [action],
        current_objective=current.rows_per_second_per_abstract_cost,
        current_throughput=current.target_rows_per_second,
        u_strong_throughput=current.target_rows_per_second,
    )
    assert winner is action


@pytest.mark.parametrize("arm", (Arm.FLEX_FREE, Arm.FLEX_POOL))
def test_no_op_remains_feasible_when_true_objective_does_not_improve(
    arm: Arm,
) -> None:
    current = E023Plan(
        "fixture",
        arm.value,
        _base_plan(),
        u_strong_used_nodes=(
            tuple(f"w{index}" for index in range(8))
            if arm is Arm.FLEX_FREE
            else ()
        ),
    )
    candidate = _action_candidate(objective=100.4, throughput=100.0, arm=arm)
    winner = choose_best_action(
        [candidate],
        current_objective=100.0,
        current_throughput=100.0,
        u_strong_throughput=100.0,
    )
    final = current if winner is None else winner.plan
    assert final.canonical_sha256 == current.canonical_sha256


def _score(efficiency: float, *, throughput: float = 100.0) -> SLOScore:
    return SLOScore(
        latency_budget_ms=200.0,
        selected_concurrency=8,
        target_rows_per_second=throughput,
        p50_pass_latency_ms=150.0,
        p95_pass_latency_ms=190.0,
        abstract_node_cost=throughput / efficiency,
        rows_per_second_per_abstract_cost=efficiency,
    )


def test_flex_pool_envelope_contains_exact_flex_free_plan() -> None:
    free = E023Plan(
        "fixture",
        Arm.FLEX_FREE.value,
        _base_plan(),
        u_strong_used_nodes=tuple(f"w{index}" for index in range(8)),
    )
    cloned = clone_as_flex_pool(free, source="FLEX_FREE")
    assert cloned.canonical_sha256 == free.canonical_sha256
    assert cloned.metadata["seed_plan"] == "FLEX_FREE"


def test_flex_pool_envelope_cannot_score_below_flex_free() -> None:
    base = E023Plan("fixture", Arm.U_STRONG.value, _base_plan())
    candidates = tuple(
        PoolEnvelopeCandidate(source, clone_as_flex_pool(base, source=source), score)
        for source, score in (
            ("U_STRONG", _score(1.00)),
            ("FLEX_FREE", _score(1.10)),
            ("POOL-U", _score(1.04)),
            ("POOL-FREE", _score(1.08)),
        )
    )
    selected = select_flex_pool_envelope(
        candidates,
        u_strong_slo_target_rows_per_second=100.0,
    )
    assert selected.source == "FLEX_FREE"
    assert selected.score.rows_per_second_per_abstract_cost == 1.10


def test_pool_primary_pruning_retains_fastest_and_lowest_cost() -> None:
    groups = [
        PrimaryGroupCandidate(("expensive-a",), (1,), (1,), 1.0, resulting_abstract_node_cost=9.0),
        PrimaryGroupCandidate(("expensive-b",), (1,), (1,), 2.0, resulting_abstract_node_cost=8.0),
        PrimaryGroupCandidate(("paid",), (1,), (1,), 3.0, resulting_abstract_node_cost=5.0),
    ]
    retained = retain_primary_group_candidates(groups, selection_mode="FLEX_POOL")
    assert [row.node_ids for row in retained] == [("expensive-a",), ("paid",)]
    assert [row.candidate_source for row in retained] == [
        "FASTEST",
        "LOWEST_RESULTING_COST",
    ]


def test_pool_alternate_generation_retains_no_new_node_and_fastest() -> None:
    inventory = _inventory(n2_memory=1_000_000_000)
    layers = [_layer(0), _layer(1)]
    context = SimpleNamespace(
        model=SimpleNamespace(layers=layers),
        inventory=inventory,
        service=SimpleNamespace(service_ms=lambda *_args: 1.0),
    )
    profile = PlanningProfile(1.0, {}, {}, {}, {})
    base = _base_plan(layers[1])
    base.assignments.insert(0, _base_plan(layers[0]).assignments[0])
    plan = E023Plan("fixture", Arm.FLEX_POOL.value, base)
    maps, failures = generate_alternate_assignment_candidates(
        context,
        plan,
        profile,
        layer_id=1,
        replicated_groups=(0,),
        arm=Arm.FLEX_POOL,
        u_strong_used_nodes=set(),
    )
    assert not failures
    assert {row.alternate_assignment_source for row in maps} == {
        "NO_NEW_NODE",
        "UNRESTRICTED_FASTEST",
    }
    by_source = {row.alternate_assignment_source: row for row in maps}
    assert by_source["NO_NEW_NODE"].replicas[0].alternate_node_id != "n2"
    assert by_source["UNRESTRICTED_FASTEST"].replicas[0].alternate_node_id == "n2"


def test_cumulative_u_strong_throughput_guard_blocks_compounding_loss() -> None:
    action = _action_candidate(objective=101.0, throughput=98.9)
    assert 98.9 / 99.5 >= 0.99
    assert choose_best_action(
        [action],
        current_objective=100.0,
        current_throughput=99.5,
        u_strong_throughput=100.0,
    ) is None


def test_replica_correctness_override_changes_only_physical_destination() -> None:
    runner = object.__new__(ReplicaAwareManifestK3Runner)
    runner.replica_destinations = {(89, 0): "alternate"}
    runner.primary_dispatch_count = 0
    runner.alternate_dispatch_count = 0
    runner.forced_alternate_dispatch_count = 0
    runner.replica_dispatch_decisions = []
    alternate = runner.physical_expert_worker_id(
        layer=89,
        logical_group_id=0,
        chunk_index=0,
        logical_worker_id="primary",
    )
    primary = runner.physical_expert_worker_id(
        layer=89,
        logical_group_id=0,
        chunk_index=1,
        logical_worker_id="primary",
    )
    assert alternate == "alternate"
    assert primary == "primary"
    assert runner.forced_alternate_dispatch_count == 1
    assert runner.alternate_dispatch_count == 1
    assert runner.primary_dispatch_count == 1
    assert {
        row["logical_group_id"] for row in runner.replica_dispatch_decisions
    } == {0}


def test_all_frozen_a_to_e_manifests_are_considered() -> None:
    paths = frozen_placement_paths(Path.cwd(), "coarse-friendly-01")
    assert [path.stem for path in paths] == [
        "coarse-friendly-01-A",
        "coarse-friendly-01-B",
        "coarse-friendly-01-C",
        "coarse-friendly-01-D",
        "coarse-friendly-01-E",
    ]
    assert all(path.is_file() for path in paths)


def test_frozen_workload_ladder_and_exact_row_accounting() -> None:
    assert TARGET_ROWS == 17
    assert CONCURRENCY_LEVELS == (1, 8, 32, 64, 128)
    assert [measured_passes_per_slot(value) for value in CONCURRENCY_LEVELS] == [
        128,
        16,
        4,
        2,
        2,
    ]
