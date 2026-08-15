"""Deterministic unique-P8 and sparse-replica planners for E023."""

from __future__ import annotations

import itertools
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from swarm_inference.experiments.experiment_022.evaluator import (
    LATENT,
    ROUTE_METADATA_BYTES_PER_ROW,
    EndpointPolicy,
    target_chunks,
)
from swarm_inference.experiments.experiment_022.model_graph import candidate_memory
from swarm_inference.experiments.experiment_022.models import (
    Inventory,
    LayerAssignment,
    ModelGraph,
    NodeCapability,
    PartitionKind,
    PlacementPlan,
)
from swarm_inference.experiments.experiment_022.service import ResidentServiceModel

from .baseline import clone_placement
from .freeze import FROZEN_CONSTANTS
from .models import Arm, E023Plan, ExpertGroupReplica, NetworkMode, base_resident_bytes
from .replica_memory import abstract_node_cost, standalone_whole_expert_group_memory
from .serving_engine import ServingEngine, ServingRun

PLANNER_RANDOMIZATION = 0.0
_PROFILE_WORKER_ENGINE: ServingEngine | None = None
_PROFILE_WORKER_LEGACY_ENGINE: ServingEngine | None = None


def _initialize_profile_worker(
    model: ModelGraph,
    inventory: Inventory,
    service: ResidentServiceModel,
    endpoint: EndpointPolicy,
) -> None:
    global _PROFILE_WORKER_ENGINE
    global _PROFILE_WORKER_LEGACY_ENGINE
    _PROFILE_WORKER_ENGINE = ServingEngine(
        model,
        inventory,
        service,
        endpoint,
        network_mode=NetworkMode.SHARED_NIC,
    )
    _PROFILE_WORKER_LEGACY_ENGINE = ServingEngine(
        model,
        inventory,
        service,
        endpoint,
        network_mode=NetworkMode.LEGACY_DIRECTED_LINK,
    )


def _profile_worker(plan: E023Plan) -> tuple[str, PlanningProfile]:
    if _PROFILE_WORKER_ENGINE is None:
        raise RuntimeError("E023 planner profile worker was not initialized")
    run = _PROFILE_WORKER_ENGINE.run_closed_loop(
        plan, int(FROZEN_CONSTANTS["planning_concurrency"])
    )
    if run.status != "PASS":
        raise RuntimeError("MODEL_INVALID: C32 planning workload is incomplete")
    return plan.canonical_sha256, planning_profile(run)


def _serving_worker(
    request: tuple[E023Plan, NetworkMode, int],
) -> tuple[tuple[str, str, int], ServingRun]:
    plan, mode, concurrency = request
    engine = (
        _PROFILE_WORKER_ENGINE
        if mode is NetworkMode.SHARED_NIC
        else _PROFILE_WORKER_LEGACY_ENGINE
    )
    if engine is None:
        raise RuntimeError("E023 serving worker was not initialized")
    run = engine.run_closed_loop(plan, concurrency)
    return (plan.canonical_sha256, mode.value, concurrency), run


@dataclass(frozen=True, slots=True)
class PlanningProfile:
    target_rows_per_second: float
    layer_criticality: dict[int, float]
    group_criticality: dict[int, dict[int, float]]
    compute_queue_penalty_ms: dict[str, float]
    compute_busy_ms: dict[str, float]


@dataclass(slots=True)
class PlannerContext:
    model: ModelGraph
    inventory: Inventory
    service: ResidentServiceModel
    engine: ServingEngine
    profile_cache: dict[str, PlanningProfile] = field(default_factory=dict)
    serving_run_cache: dict[tuple[str, str, int], ServingRun] = field(
        default_factory=dict
    )
    _executor: ProcessPoolExecutor | None = field(
        default=None, init=False, repr=False
    )

    def enable_parallel_profiles(self, workers: int) -> None:
        if workers <= 1:
            return
        if self._executor is not None:
            raise RuntimeError("parallel planner profiles already enabled")
        self._executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_profile_worker,
            initargs=(
                self.model,
                self.inventory,
                self.service,
                self.engine.endpoint,
            ),
        )

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None

    def profiles(self, plans: list[E023Plan]) -> list[PlanningProfile]:
        missing: dict[str, E023Plan] = {}
        for plan in plans:
            if plan.canonical_sha256 not in self.profile_cache:
                missing.setdefault(plan.canonical_sha256, plan)
        if missing:
            values = list(missing.values())
            if self._executor is None:
                results = []
                for plan in values:
                    run = self.engine.run_closed_loop(
                        plan, int(FROZEN_CONSTANTS["planning_concurrency"])
                    )
                    if run.status != "PASS":
                        raise RuntimeError(
                            "MODEL_INVALID: C32 planning workload is incomplete"
                        )
                    results.append((plan.canonical_sha256, planning_profile(run)))
            else:
                results = list(self._executor.map(_profile_worker, values, chunksize=1))
            self.profile_cache.update(results)
        return [self.profile_cache[plan.canonical_sha256] for plan in plans]

    def profile(self, plan: E023Plan) -> PlanningProfile:
        return self.profiles([plan])[0]

    def serving_runs(
        self,
        requests: list[tuple[E023Plan, NetworkMode, int]],
    ) -> list[ServingRun]:
        missing: dict[tuple[str, str, int], tuple[E023Plan, NetworkMode, int]] = {}
        for plan, mode, concurrency in requests:
            key = (plan.canonical_sha256, mode.value, concurrency)
            if key not in self.serving_run_cache:
                missing.setdefault(key, (plan, mode, concurrency))
        if missing:
            values = list(missing.values())
            if self._executor is None:
                results = []
                for plan, mode, concurrency in values:
                    engine = self.engine
                    if mode is not NetworkMode.SHARED_NIC:
                        engine = ServingEngine(
                            self.model,
                            self.inventory,
                            self.service,
                            self.engine.endpoint,
                            network_mode=mode,
                        )
                    run = engine.run_closed_loop(plan, concurrency)
                    results.append(
                        ((plan.canonical_sha256, mode.value, concurrency), run)
                    )
            else:
                results = list(self._executor.map(_serving_worker, values, chunksize=1))
            self.serving_run_cache.update(results)
        return [
            self.serving_run_cache[(plan.canonical_sha256, mode.value, concurrency)]
            for plan, mode, concurrency in requests
        ]


@dataclass(frozen=True, slots=True)
class PrimaryGroupCandidate:
    node_ids: tuple[str, ...]
    resident_bytes: tuple[int, ...]
    checkpoint_bytes: tuple[int, ...]
    predicted_completion_ms: float


@dataclass(slots=True)
class PlannerActionCandidate:
    plan: E023Plan
    layer_id: int
    action_type: str
    replica_count: int
    primary_group_nodes: tuple[str, ...]
    replicated_logical_groups: tuple[int, ...]
    alternate_nodes: tuple[tuple[int, str], ...]
    added_checkpoint_bytes: int
    added_resident_bytes: int
    newly_activated_nodes: tuple[str, ...]
    expansion_score_before: float
    expansion_score_after: float
    target_rows_per_second: float
    objective: float
    relative_gain: float
    throughput_ratio: float


@dataclass(frozen=True, slots=True)
class _PrimarySearchState:
    layer_id: int
    was_whole_layer: bool
    primary: PrimaryGroupCandidate
    base: PlacementPlan
    preserved_replicas: tuple[ExpertGroupReplica, ...]
    existing_layer_replicas: tuple[ExpertGroupReplica, ...]
    primary_plan: E023Plan


@dataclass(frozen=True, slots=True)
class _ReplicaSearchState:
    plan: E023Plan
    layer_id: int
    action_type: str
    replica_count: int
    primary_group_nodes: tuple[str, ...]
    replicated_logical_groups: tuple[int, ...]
    replicas: tuple[ExpertGroupReplica, ...]
    added_checkpoint_bytes: int
    added_resident_bytes: int
    newly_activated_nodes: tuple[str, ...]
    expansion_score_before: float
    expansion_score_after: float


def planning_profile(run: ServingRun) -> PlanningProfile:
    eligible_keys = {
        (row.slot_id, row.pass_index)
        for row in run.passes
        if row.pass_index >= 2 and row.start_ms >= run.t0_ms
    }
    group: dict[int, dict[int, float]] = {}
    for pass_row in run.passes:
        if (pass_row.slot_id, pass_row.pass_index) not in eligible_keys:
            continue
        for fork in pass_row.fork_join_records:
            values = sorted(
                fork.group_arrival_ms.items(), key=lambda item: (item[1], item[0])
            )
            latest_value = values[-1][1]
            latest_groups = [
                group_id
                for group_id, arrival in values
                if abs(arrival - latest_value) <= 1e-9
            ]
            if len(latest_groups) != 1:
                continue
            delay = latest_value - values[-2][1]
            if delay <= 1e-9:
                continue
            layer_values = group.setdefault(
                fork.layer_id, {group_id: 0.0 for group_id in range(8)}
            )
            layer_values[latest_groups[0]] += delay
    queue_sum = run.compute_queue_wait_ms_by_node
    queue_count = run.compute_operation_count_by_node
    busy = run.compute_busy_ms_by_node
    return PlanningProfile(
        target_rows_per_second=run.target_rows_per_second,
        layer_criticality={
            layer_id: sum(values.values()) for layer_id, values in group.items()
        },
        group_criticality=group,
        compute_queue_penalty_ms={
            node_id: queue_sum[node_id] / queue_count[node_id]
            for node_id in queue_sum
        },
        compute_busy_ms=busy,
    )


def _runtime_supports_whole_expert(node: NodeCapability) -> bool:
    return node.available and "WHOLE_EXPERT" in node.runtime_capabilities


def _plan_resident_with_replicas(plan: E023Plan) -> dict[str, int]:
    result = base_resident_bytes(plan.base_plan)
    for replica in plan.replicas:
        result[replica.alternate_node_id] = (
            result.get(replica.alternate_node_id, 0) + replica.resident_bytes
        )
    return result


def _predicted_group_cost_ms(
    context: PlannerContext,
    *,
    layer_id: int,
    rows: int,
    coordinator: str,
    node_id: str,
    queue_penalty_ms: float,
) -> float:
    layer = context.model.layers[layer_id]
    if node_id == coordinator:
        return (
            context.service.service_ms(layer, "expert_whole_group", 8, rows)
            / context.inventory.node_map()[node_id].compute_multiplier
            + queue_penalty_ms
        )
    input_bytes = rows * LATENT * 4 + rows * ROUTE_METADATA_BYTES_PER_ROW
    output_bytes = rows * LATENT * 4
    nodes = context.inventory.node_map()
    return (
        nodes[coordinator].peer(node_id).transfer_ms(input_bytes)
        + context.service.service_ms(layer, "worker_protocol", 1, rows)
        + context.service.service_ms(layer, "expert_whole_group_remote", 8, rows)
        / nodes[node_id].compute_multiplier
        + nodes[node_id].peer(coordinator).transfer_ms(output_bytes)
        + queue_penalty_ms
    )


def _predicted_primary_completion(
    context: PlannerContext,
    plan: E023Plan,
    layer_id: int,
    group: tuple[str, ...],
    profile: PlanningProfile,
) -> float:
    coordinator = group[0]
    return sum(
        max(
            _predicted_group_cost_ms(
                context,
                layer_id=layer_id,
                rows=rows,
                coordinator=coordinator,
                node_id=node_id,
                queue_penalty_ms=profile.compute_queue_penalty_ms.get(node_id, 0.0),
            )
            for node_id in group
        )
        for rows in target_chunks(plan.base_plan.chunk_rows)
    )


def predicted_convertibility_criticality(
    context: PlannerContext,
    plan: E023Plan,
    layer_id: int,
    group: tuple[str, ...],
    profile: PlanningProfile,
) -> float:
    coordinator = group[0]
    total = 0.0
    for rows in target_chunks(plan.base_plan.chunk_rows):
        values = sorted(
            _predicted_group_cost_ms(
                context,
                layer_id=layer_id,
                rows=rows,
                coordinator=coordinator,
                node_id=node_id,
                queue_penalty_ms=profile.compute_queue_penalty_ms.get(node_id, 0.0),
            )
            for node_id in group
        )
        total += max(0.0, values[-1] - values[-2])
    return total


def generate_primary_groups(
    context: PlannerContext,
    plan: E023Plan,
    profile: PlanningProfile,
    *,
    layer_id: int,
    allowed_nodes: set[str] | None,
) -> tuple[PrimaryGroupCandidate, ...]:
    """Generate deterministic capability/memory-aware P8 primary layouts."""

    if layer_id == 0:
        return ()
    layer = context.model.layers[layer_id]
    if not all(
        context.service.supported_candidate(layer, PartitionKind.WHOLE_EXPERT, 8, rows)
        for rows in set(target_chunks(plan.base_plan.chunk_rows))
    ):
        return ()
    old = plan.base_plan.assignments[layer_id]
    resident, checkpoint = candidate_memory(layer, PartitionKind.WHOLE_EXPERT, 8)
    current_memory = _plan_resident_with_replicas(plan)
    for node_id, value in old.memory_by_node.items():
        current_memory[node_id] = current_memory.get(node_id, 0) - value
    nodes = [
        node
        for node in context.inventory.available_nodes
        if _runtime_supports_whole_expert(node)
        and (allowed_nodes is None or node.node_id in allowed_nodes)
    ]
    if len(nodes) < 8:
        return ()

    def feasible(node: NodeCapability, memory: int) -> bool:
        return (
            current_memory.get(node.node_id, 0) + memory
            <= node.accelerator_memory_bytes
        )

    by_locality: dict[str, list[NodeCapability]] = {}
    for node in nodes:
        by_locality.setdefault(node.locality_group, []).append(node)
    pools = [*by_locality.values(), nodes]
    raw: set[tuple[str, ...]] = set()
    for pool in pools:
        coordinators = [node for node in pool if feasible(node, resident[0])]
        coordinators.sort(
            key=lambda node: (
                profile.compute_busy_ms.get(node.node_id, 0.0),
                profile.compute_queue_penalty_ms.get(node.node_id, 0.0),
                -node.compute_multiplier,
                node.node_id,
            )
        )
        for coordinator in coordinators:
            others = [
                node
                for node in pool
                if node.node_id != coordinator.node_id
                and feasible(node, max(resident[1:]))
            ]
            others.sort(
                key=lambda node: (
                    _predicted_group_cost_ms(
                        context,
                        layer_id=layer_id,
                        rows=plan.base_plan.chunk_rows,
                        coordinator=coordinator.node_id,
                        node_id=node.node_id,
                        queue_penalty_ms=profile.compute_queue_penalty_ms.get(
                            node.node_id, 0.0
                        ),
                    ),
                    current_memory.get(node.node_id, 0),
                    node.node_id,
                )
            )
            if len(others) >= 7:
                raw.add(
                    (coordinator.node_id, *(node.node_id for node in others[:7]))
                )
            if len(others) >= 8:
                raw.add(
                    (coordinator.node_id, *(node.node_id for node in others[1:8]))
                )
    candidates = [
        PrimaryGroupCandidate(
            node_ids=group,
            resident_bytes=resident,
            checkpoint_bytes=checkpoint,
            predicted_completion_ms=_predicted_primary_completion(
                context, plan, layer_id, group, profile
            ),
        )
        for group in raw
    ]
    candidates.sort(key=lambda value: (value.predicted_completion_ms, value.node_ids))
    return tuple(
        candidates[: int(FROZEN_CONSTANTS["planner_primary_groups_per_layer"])]
    )


def make_p8_assignment(
    layer_id: int, candidate: PrimaryGroupCandidate
) -> LayerAssignment:
    return LayerAssignment(
        layer_id=layer_id,
        partition_kind=PartitionKind.WHOLE_EXPERT,
        degree=8,
        node_ids=candidate.node_ids,
        memory_by_node={
            node_id: value
            for node_id, value in zip(
                candidate.node_ids, candidate.resident_bytes, strict=True
            )
        },
        checkpoint_bytes_by_node={
            node_id: value
            for node_id, value in zip(
                candidate.node_ids, candidate.checkpoint_bytes, strict=True
            )
        },
        coordinator_node_id=candidate.node_ids[0],
        candidate_id=f"layer-{layer_id:02d}:WHOLE_EXPERT:p8",
    )


def replace_layer_assignment(
    plan: PlacementPlan,
    assignment: LayerAssignment,
) -> PlacementPlan:
    result = clone_placement(plan)
    result.assignments[assignment.layer_id] = assignment
    memory: dict[str, int] = dict(result.endpoint_memory_by_node)
    for row in result.assignments:
        for node_id, value in row.memory_by_node.items():
            memory[node_id] = memory.get(node_id, 0) + value
    result.memory_used_by_node = {node: value for node, value in memory.items() if value}
    result.used_nodes = tuple(sorted(result.memory_used_by_node))
    result.exact_tok_s_per_user = None
    result.critical_path_ms = None
    result.total_worker_compute_ms = None
    result.network_bytes = 0
    result.event_receipt = None
    result.objective_tuple = None
    return result


def sparse_flexibility_expansion_score(
    primary_nodes: tuple[str, ...],
    replicas: dict[int, str],
) -> float:
    if len(primary_nodes) != 8:
        raise ValueError("sparse flexibility expansion requires eight primary groups")
    minimum = math.inf
    groups = range(8)
    for size in range(1, 5):
        for subset in itertools.combinations(groups, size):
            nodes: set[str] = set()
            for group in subset:
                nodes.add(primary_nodes[group])
                alternate = replicas.get(group)
                if alternate is not None:
                    nodes.add(alternate)
            minimum = min(minimum, len(nodes) / size)
    return float(minimum)


def select_replica_groups(
    criticality: dict[int, float],
    target_count: int,
    existing_groups: set[int],
) -> tuple[int, ...]:
    if target_count not in (2, 4, 8) or len(existing_groups) > target_count:
        raise ValueError("invalid frozen replica target")
    selected = set(existing_groups)
    ranked = sorted(range(8), key=lambda group: (-criticality.get(group, 0.0), group))
    for group in ranked:
        if len(selected) == target_count:
            break
        selected.add(group)
    return tuple(sorted(selected))


def assign_alternates(
    context: PlannerContext,
    plan: E023Plan,
    profile: PlanningProfile,
    *,
    layer_id: int,
    replicated_groups: tuple[int, ...],
    allowed_nodes: set[str] | None,
) -> tuple[ExpertGroupReplica, ...] | None:
    assignment = plan.base_plan.assignments[layer_id]
    if (
        assignment.partition_kind is not PartitionKind.WHOLE_EXPERT
        or assignment.degree != 8
    ):
        raise ValueError("alternate assignment requires WHOLE_EXPERT:p8")
    current = {
        row.logical_group_id: row
        for row in plan.replicas
        if row.layer_id == layer_id
    }
    other_replicas = [row for row in plan.replicas if row.layer_id != layer_id]
    memory = base_resident_bytes(plan.base_plan)
    for replica in other_replicas:
        memory[replica.alternate_node_id] = (
            memory.get(replica.alternate_node_id, 0) + replica.resident_bytes
        )
    for replica in current.values():
        memory[replica.alternate_node_id] = (
            memory.get(replica.alternate_node_id, 0) + replica.resident_bytes
        )
    selected = dict(current)
    replica_nodes = {group: row.alternate_node_id for group, row in current.items()}
    layer = context.model.layers[layer_id]
    for group in replicated_groups:
        if group in selected:
            continue
        replica_memory = standalone_whole_expert_group_memory(layer, group)
        primary = assignment.node_ids[group]
        candidates: list[tuple[str, float, float, int]] = []
        for node in context.inventory.available_nodes:
            if (
                not _runtime_supports_whole_expert(node)
                or node.node_id == primary
                or (allowed_nodes is not None and node.node_id not in allowed_nodes)
                or memory.get(node.node_id, 0) + replica_memory.resident_bytes
                > node.accelerator_memory_bytes
            ):
                continue
            predicted = sum(
                _predicted_group_cost_ms(
                    context,
                    layer_id=layer_id,
                    rows=rows,
                    coordinator=assignment.coordinator_node_id,
                    node_id=node.node_id,
                    queue_penalty_ms=profile.compute_queue_penalty_ms.get(
                        node.node_id, 0.0
                    ),
                )
                for rows in target_chunks(plan.base_plan.chunk_rows)
            )
            expansion = sparse_flexibility_expansion_score(
                assignment.node_ids, {**replica_nodes, group: node.node_id}
            )
            candidates.append(
                (node.node_id, predicted, expansion, memory.get(node.node_id, 0))
            )
        if not candidates:
            return None
        best = candidates[0]
        for candidate in candidates[1:]:
            if candidate[1] < best[1] - 1e-9 or (abs(candidate[1] - best[1]) <= 1e-9 and (
                -candidate[2], candidate[3], candidate[0]
            ) < (-best[2], best[3], best[0])):
                best = candidate
        node_id = best[0]
        replica = ExpertGroupReplica(
            layer_id=layer_id,
            logical_group_id=group,
            primary_node_id=primary,
            alternate_node_id=node_id,
            checkpoint_bytes=replica_memory.checkpoint_bytes,
            resident_bytes=replica_memory.resident_bytes,
            persistent_state_bytes=0,
        )
        selected[group] = replica
        replica_nodes[group] = node_id
        memory[node_id] = memory.get(node_id, 0) + replica_memory.resident_bytes
    return tuple(selected[group] for group in sorted(selected))


def choose_best_action(
    candidates: list[PlannerActionCandidate],
    *,
    current_objective: float,
    current_throughput: float,
) -> PlannerActionCandidate | None:
    minimum_gain = float(FROZEN_CONSTANTS["planner_minimum_relative_gain"])
    throughput_floor = 1.0 - float(
        FROZEN_CONSTANTS["planner_local_throughput_regression_limit"]
    )
    eligible = [
        row
        for row in candidates
        if row.objective / current_objective - 1.0 >= minimum_gain - 1e-12
        and row.target_rows_per_second >= throughput_floor * current_throughput - 1e-12
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            -(row.objective / current_objective - 1.0),
            row.added_resident_bytes,
            len(row.newly_activated_nodes),
            row.layer_id,
            row.replica_count,
            row.primary_group_nodes,
            row.alternate_nodes,
        ),
    )


def _convertible_layers(
    context: PlannerContext,
    plan: E023Plan,
    profile: PlanningProfile,
    *,
    allowed_nodes: set[str] | None,
) -> tuple[list[int], dict[int, tuple[PrimaryGroupCandidate, ...]]]:
    groups: dict[int, tuple[PrimaryGroupCandidate, ...]] = {}
    ranked: list[tuple[float, int]] = []
    for assignment in plan.base_plan.assignments:
        if assignment.layer_id == 0 or assignment.partition_kind is not PartitionKind.WHOLE_LAYER:
            continue
        values = generate_primary_groups(
            context,
            plan,
            profile,
            layer_id=assignment.layer_id,
            allowed_nodes=allowed_nodes,
        )
        if not values:
            continue
        groups[assignment.layer_id] = values
        ranked.append(
            (
                max(
                    predicted_convertibility_criticality(
                        context,
                        plan,
                        assignment.layer_id,
                        value.node_ids,
                        profile,
                    )
                    for value in values
                ),
                assignment.layer_id,
            )
        )
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [layer for _, layer in ranked], groups


def run_unique_p8_refinement(
    context: PlannerContext,
    starting_plan: PlacementPlan,
) -> tuple[PlacementPlan, list[dict[str, Any]]]:
    current = E023Plan(starting_plan.inventory_id, Arm.U_STRONG.value, starting_plan)
    actions: list[dict[str, Any]] = []
    maximum = int(FROZEN_CONSTANTS["planner_max_accepted_layer_actions"])
    for iteration in range(1, maximum + 1):
        profile = context.profile(current)
        layers, groups = _convertible_layers(
            context, current, profile, allowed_nodes=None
        )
        layers = layers[: int(FROZEN_CONSTANTS["planner_candidate_layers"])]
        pending: list[tuple[int, tuple[str, ...], E023Plan]] = []
        for layer_id in layers:
            for group in groups[layer_id]:
                base = replace_layer_assignment(
                    current.base_plan, make_p8_assignment(layer_id, group)
                )
                candidate = E023Plan(base.inventory_id, Arm.U_STRONG.value, base)
                pending.append((layer_id, group.node_ids, candidate))
        candidate_profiles = context.profiles([row[2] for row in pending])
        candidates: list[tuple[float, int, tuple[str, ...], E023Plan, float]] = []
        for (layer_id, node_ids, candidate), candidate_profile in zip(
            pending, candidate_profiles, strict=True
        ):
            gain = (
                candidate_profile.target_rows_per_second
                / profile.target_rows_per_second
                - 1.0
            )
            candidates.append(
                (
                    gain,
                    layer_id,
                    node_ids,
                    candidate,
                    candidate_profile.target_rows_per_second,
                )
            )
        qualifying = [
            row
            for row in candidates
            if row[0]
            >= float(FROZEN_CONSTANTS["planner_minimum_relative_gain"]) - 1e-12
        ]
        if not qualifying:
            break
        winner = min(qualifying, key=lambda row: (-row[0], row[1], row[2]))
        current = winner[3]
        actions.append(
            {
                "inventory_id": starting_plan.inventory_id,
                "planner_iteration": iteration,
                "layer_id": winner[1],
                "action_type": "WHOLE_LAYER_TO_WHOLE_EXPERT_P8_UNIQUE",
                "primary_group_nodes": "|".join(winner[2]),
                "objective_before": profile.target_rows_per_second,
                "objective_after": winner[4],
                "relative_gain_percent": 100.0 * winner[0],
                "randomization": PLANNER_RANDOMIZATION,
                "accepted": True,
            }
        )
    return current.base_plan, actions


def run_flex_planner(
    context: PlannerContext,
    u_strong: PlacementPlan,
    *,
    arm: Arm,
) -> tuple[E023Plan, list[dict[str, Any]]]:
    if arm not in {Arm.FLEX_FREE, Arm.FLEX_POOL}:
        raise ValueError("FLEX planner arm must be FLEX_FREE or FLEX_POOL")
    u_nodes = tuple(sorted(base_resident_bytes(u_strong)))
    current = E023Plan(
        u_strong.inventory_id,
        arm.value,
        clone_placement(u_strong),
        (),
        u_strong_used_nodes=u_nodes if arm is Arm.FLEX_FREE else (),
    )
    rows: list[dict[str, Any]] = []
    maximum = int(FROZEN_CONSTANTS["planner_max_accepted_layer_actions"])
    allowed_nodes = set(u_nodes) if arm is Arm.FLEX_FREE else None
    for iteration in range(1, maximum + 1):
        profile = context.profile(current)
        current_cost = abstract_node_cost(current, context.inventory)
        current_objective = (
            profile.target_rows_per_second
            if arm is Arm.FLEX_FREE
            else profile.target_rows_per_second / current_cost
        )
        whole_layers, group_candidates = _convertible_layers(
            context, current, profile, allowed_nodes=allowed_nodes
        )
        p8_layers = [
            assignment.layer_id
            for assignment in current.base_plan.assignments
            if assignment.partition_kind is PartitionKind.WHOLE_EXPERT
            and assignment.degree == 8
            and assignment.layer_id != 0
        ]
        p8_layers.sort(
            key=lambda layer_id: (
                -profile.layer_criticality.get(layer_id, 0.0),
                layer_id,
            )
        )
        candidate_limit = int(FROZEN_CONSTANTS["planner_candidate_layers"])
        layer_candidates = p8_layers[:candidate_limit]
        for layer_id in whole_layers:
            if len(layer_candidates) == candidate_limit:
                break
            layer_candidates.append(layer_id)

        considered: list[PlannerActionCandidate] = []
        pending_rows: list[dict[str, Any]] = []
        primary_states: list[_PrimarySearchState] = []
        for layer_id in layer_candidates:
            assignment = current.base_plan.assignments[layer_id]
            if assignment.partition_kind is PartitionKind.WHOLE_LAYER:
                primary_layouts = group_candidates.get(layer_id, ())
            else:
                resident = tuple(
                    assignment.memory_by_node[node] for node in assignment.node_ids
                )
                checkpoint = tuple(
                    assignment.checkpoint_bytes_by_node[node]
                    for node in assignment.node_ids
                )
                primary_layouts = (
                    PrimaryGroupCandidate(
                        assignment.node_ids,
                        resident,
                        checkpoint,
                        0.0,
                    ),
                )
            for primary in primary_layouts:
                if assignment.partition_kind is PartitionKind.WHOLE_LAYER:
                    base = replace_layer_assignment(
                        current.base_plan, make_p8_assignment(layer_id, primary)
                    )
                    preserved = tuple(
                        replica for replica in current.replicas if replica.layer_id != layer_id
                    )
                else:
                    base = clone_placement(current.base_plan)
                    preserved = tuple(
                        replica for replica in current.replicas if replica.layer_id != layer_id
                    )
                existing_layer_replicas = tuple(
                    replica for replica in current.replicas if replica.layer_id == layer_id
                )
                primary_plan = E023Plan(
                    current.inventory_id,
                    arm.value,
                    base,
                    preserved,
                    u_strong_used_nodes=u_nodes if arm is Arm.FLEX_FREE else (),
                )
                primary_states.append(
                    _PrimarySearchState(
                        layer_id=layer_id,
                        was_whole_layer=(
                            assignment.partition_kind is PartitionKind.WHOLE_LAYER
                        ),
                        primary=primary,
                        base=base,
                        preserved_replicas=preserved,
                        existing_layer_replicas=existing_layer_replicas,
                        primary_plan=primary_plan,
                    )
                )

        primary_profiles = context.profiles(
            [state.primary_plan for state in primary_states]
        )
        replica_states: list[_ReplicaSearchState] = []
        for state, primary_profile in zip(
            primary_states, primary_profiles, strict=True
        ):
            layer_id = state.layer_id
            primary = state.primary
            existing_layer_replicas = state.existing_layer_replicas
            layer_criticality = primary_profile.group_criticality.get(
                layer_id, {group: 0.0 for group in range(8)}
            )
            existing_groups = {
                replica.logical_group_id for replica in existing_layer_replicas
            }
            expansion_before = sparse_flexibility_expansion_score(
                primary.node_ids,
                {
                    replica.logical_group_id: replica.alternate_node_id
                    for replica in existing_layer_replicas
                },
            )
            replica_seed_plan = E023Plan(
                current.inventory_id,
                arm.value,
                state.base,
                state.preserved_replicas + existing_layer_replicas,
                u_strong_used_nodes=u_nodes if arm is Arm.FLEX_FREE else (),
            )
            for target_count in FROZEN_CONSTANTS["replica_count_options"]:
                target = int(target_count)
                if len(existing_groups) >= target:
                    continue
                selected_groups = select_replica_groups(
                    layer_criticality, target, existing_groups
                )
                replicas = assign_alternates(
                    context,
                    replica_seed_plan,
                    primary_profile,
                    layer_id=layer_id,
                    replicated_groups=selected_groups,
                    allowed_nodes=allowed_nodes,
                )
                action_type = (
                    f"WHOLE_LAYER_TO_WHOLE_EXPERT_P8_PLUS_{target}_REPLICAS"
                    if state.was_whole_layer
                    else f"WHOLE_EXPERT_P8_ADD_TO_{target}_REPLICAS"
                )
                if replicas is None:
                    pending_rows.append(
                        {
                            "inventory_id": current.inventory_id,
                            "arm": arm.value,
                            "planner_iteration": iteration,
                            "layer_id": layer_id,
                            "action_type": action_type,
                            "replica_count": target,
                            "primary_group_nodes": "|".join(primary.node_ids),
                            "replicated_logical_groups": "|".join(
                                str(group) for group in selected_groups
                            ),
                            "alternate_nodes": "",
                            "added_checkpoint_bytes": 0,
                            "added_resident_bytes": 0,
                            "new_nodes_activated": 0,
                            "expansion_score_before": expansion_before,
                            "expansion_score_after": expansion_before,
                            "objective_before": current_objective,
                            "objective_after": "",
                            "relative_gain_percent": "",
                            "accepted": False,
                            "rejection_reason": "NO_FEASIBLE_ALTERNATE_ASSIGNMENT",
                        }
                    )
                    continue
                all_replicas = tuple(
                    replica
                    for replica in state.primary_plan.replicas
                    if replica.layer_id != layer_id
                ) + replicas
                candidate_plan = E023Plan(
                    current.inventory_id,
                    arm.value,
                    state.base,
                    all_replicas,
                    u_strong_used_nodes=u_nodes if arm is Arm.FLEX_FREE else (),
                )
                current_layer_replicas = [
                    replica
                    for replica in current.replicas
                    if replica.layer_id == layer_id
                ]
                added_checkpoint = sum(
                    replica.checkpoint_bytes for replica in replicas
                ) - sum(
                    replica.checkpoint_bytes for replica in current_layer_replicas
                )
                added_resident = sum(
                    replica.resident_bytes for replica in replicas
                ) - sum(
                    replica.resident_bytes for replica in current_layer_replicas
                )
                expansion_after = sparse_flexibility_expansion_score(
                    primary.node_ids,
                    {
                        replica.logical_group_id: replica.alternate_node_id
                        for replica in replicas
                    },
                )
                newly_used = tuple(
                    sorted(candidate_plan.used_nodes.difference(current.used_nodes))
                )
                replica_states.append(
                    _ReplicaSearchState(
                        plan=candidate_plan,
                        layer_id=layer_id,
                        action_type=action_type,
                        replica_count=target,
                        primary_group_nodes=primary.node_ids,
                        replicated_logical_groups=selected_groups,
                        replicas=replicas,
                        added_checkpoint_bytes=added_checkpoint,
                        added_resident_bytes=added_resident,
                        newly_activated_nodes=newly_used,
                        expansion_score_before=expansion_before,
                        expansion_score_after=expansion_after,
                    )
                )

        replica_profiles = context.profiles([state.plan for state in replica_states])
        for state, candidate_profile in zip(
            replica_states, replica_profiles, strict=True
        ):
            cost = abstract_node_cost(state.plan, context.inventory)
            objective = (
                candidate_profile.target_rows_per_second
                if arm is Arm.FLEX_FREE
                else candidate_profile.target_rows_per_second / cost
            )
            considered.append(
                PlannerActionCandidate(
                    plan=state.plan,
                    layer_id=state.layer_id,
                    action_type=state.action_type,
                    replica_count=state.replica_count,
                    primary_group_nodes=state.primary_group_nodes,
                    replicated_logical_groups=state.replicated_logical_groups,
                    alternate_nodes=tuple(
                        (replica.logical_group_id, replica.alternate_node_id)
                        for replica in state.replicas
                    ),
                    added_checkpoint_bytes=state.added_checkpoint_bytes,
                    added_resident_bytes=state.added_resident_bytes,
                    newly_activated_nodes=state.newly_activated_nodes,
                    expansion_score_before=state.expansion_score_before,
                    expansion_score_after=state.expansion_score_after,
                    target_rows_per_second=candidate_profile.target_rows_per_second,
                    objective=objective,
                    relative_gain=objective / current_objective - 1.0,
                    throughput_ratio=(
                        candidate_profile.target_rows_per_second
                        / profile.target_rows_per_second
                    ),
                )
            )
        winner = choose_best_action(
            considered,
            current_objective=current_objective,
            current_throughput=profile.target_rows_per_second,
        )
        for candidate in considered:
            gain = candidate.objective / current_objective - 1.0
            throughput_ok = candidate.throughput_ratio >= 1.0 - float(
                FROZEN_CONSTANTS["planner_local_throughput_regression_limit"]
            )
            gain_ok = (
                gain
                >= float(FROZEN_CONSTANTS["planner_minimum_relative_gain"]) - 1e-12
            )
            accepted = candidate is winner
            reason = (
                ""
                if accepted
                else "OBJECTIVE_GAIN_BELOW_0_5_PERCENT"
                if not gain_ok
                else "LOCAL_THROUGHPUT_REGRESSION_GT_1_PERCENT"
                if not throughput_ok
                else "QUALIFYING_BUT_NOT_BEST_DETERMINISTIC_ACTION"
            )
            pending_rows.append(
                {
                    "inventory_id": current.inventory_id,
                    "arm": arm.value,
                    "planner_iteration": iteration,
                    "layer_id": candidate.layer_id,
                    "action_type": candidate.action_type,
                    "replica_count": candidate.replica_count,
                    "primary_group_nodes": "|".join(candidate.primary_group_nodes),
                    "replicated_logical_groups": "|".join(
                        str(group) for group in candidate.replicated_logical_groups
                    ),
                    "alternate_nodes": "|".join(
                        f"{group}:{node}" for group, node in candidate.alternate_nodes
                    ),
                    "added_checkpoint_bytes": candidate.added_checkpoint_bytes,
                    "added_resident_bytes": candidate.added_resident_bytes,
                    "new_nodes_activated": len(candidate.newly_activated_nodes),
                    "expansion_score_before": candidate.expansion_score_before,
                    "expansion_score_after": candidate.expansion_score_after,
                    "objective_before": current_objective,
                    "objective_after": candidate.objective,
                    "relative_gain_percent": 100.0 * gain,
                    "accepted": accepted,
                    "rejection_reason": reason,
                }
            )
        rows.extend(pending_rows)
        if winner is None:
            break
        current = winner.plan
    current.planner_actions = tuple(row for row in rows if row["accepted"])
    return current, rows


__all__ = [
    "PLANNER_RANDOMIZATION",
    "PlannerActionCandidate",
    "PlannerContext",
    "PlanningProfile",
    "assign_alternates",
    "choose_best_action",
    "generate_primary_groups",
    "planning_profile",
    "replace_layer_assignment",
    "run_flex_planner",
    "run_unique_p8_refinement",
    "select_replica_groups",
    "sparse_flexibility_expansion_score",
]
