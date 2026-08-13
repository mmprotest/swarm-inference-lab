"""One shared capability-driven placement optimizer for every E022 arm."""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass
from typing import Any

from .evaluator import EndpointPolicy, PlacementEvaluator, common_endpoint_policy, target_chunks
from .model_graph import candidate_memory
from .models import (
    ALLOWED_BY_LEVEL,
    Inventory,
    LayerAssignment,
    ModelGraph,
    NodeCapability,
    PartitionKind,
    PlacementPlan,
    PlannerLevel,
)
from .service import ResidentServiceModel


@dataclass(frozen=True, slots=True)
class OptimizerConfiguration:
    proposal_budget: int = 96
    exact_evaluation_budget: int = 12
    restarts: int = 3
    candidate_groups_per_action: int = 5
    randomization: float = 0.12
    seed: int = 22022

    def __post_init__(self) -> None:
        if min(
            self.proposal_budget,
            self.exact_evaluation_budget,
            self.restarts,
            self.candidate_groups_per_action,
        ) <= 0:
            raise ValueError("optimizer budgets must be positive")


@dataclass(slots=True)
class _Construction:
    assignments: list[LayerAssignment]
    memory: dict[str, int]
    load: dict[str, float]
    approximate_network_ms: float
    previous_coordinator: str | None
    score: float = math.inf


@dataclass(frozen=True, slots=True)
class OptimizerResult:
    plan: PlacementPlan
    convergence: tuple[dict[str, Any], ...]
    elapsed_ms: float
    proposals: int
    exact_evaluations: int
    fallback_retained: bool


class SharedPlacementOptimizer:
    """Randomized greedy + exact re-ranking, parameterized only by action space.

    The algorithm, budgets, tie breakers, endpoint policy, and evaluator remain
    byte-for-byte the same between levels.  The `allowed` set is the sole arm
    distinction.  Adaptive calls may also receive A's plan as an explicit
    feasible incumbent; proposals are not deducted for that safety fallback.
    """

    def __init__(
        self,
        model: ModelGraph,
        service: ResidentServiceModel,
        configuration: OptimizerConfiguration | None = None,
    ) -> None:
        self.model = model
        self.service = service
        self.configuration = configuration or OptimizerConfiguration()

    def _candidate_supported(
        self,
        layer_id: int,
        kind: PartitionKind,
        degree: int,
        chunk_rows: int,
    ) -> bool:
        layer = self.model.layers[layer_id]
        if layer_id == 0:
            return kind is PartitionKind.WHOLE_LAYER and all(
                self.service.supported_candidate(layer, kind, degree, rows)
                for rows in set(target_chunks(chunk_rows))
            )
        return all(
            self.service.supported_candidate(layer, kind, degree, rows)
            for rows in set(target_chunks(chunk_rows))
        )

    @staticmethod
    def _runtime_supported(node: NodeCapability, kind: PartitionKind) -> bool:
        if kind is PartitionKind.WHOLE_LAYER:
            return "WHOLE_LAYER" in node.runtime_capabilities
        required = {
            PartitionKind.WHOLE_EXPERT: "WHOLE_EXPERT",
            PartitionKind.EXPERT_SHARD: "EXPERT_SHARD",
            PartitionKind.ATTENTION_PROJECTION_SHARD: "PROJECTION_SHARD",
            PartitionKind.FULL_MIXED_STRIPE: "EXPERT_SHARD",
        }[kind]
        return required in node.runtime_capabilities

    def _approximate_worker_ms(
        self,
        layer_id: int,
        kind: PartitionKind,
        degree: int,
        rows: int,
    ) -> float:
        layer = self.model.layers[layer_id]
        if kind is PartitionKind.WHOLE_LAYER:
            return self.service.service_ms(layer, "whole_layer", 1, rows)
        common = (
            2 * self.service.service_ms(layer, "attnres", 1, rows)
            + self.service.service_ms(layer, "router", 1, rows)
        )
        if kind in {
            PartitionKind.ATTENTION_PROJECTION_SHARD,
            PartitionKind.FULL_MIXED_STRIPE,
        }:
            attention = self.service.service_ms(layer, "attention_shard", degree, rows)
        else:
            attention = self.service.service_ms(layer, "attention_whole", 1, rows)
        if kind in {
            PartitionKind.WHOLE_EXPERT,
            PartitionKind.EXPERT_SHARD,
            PartitionKind.FULL_MIXED_STRIPE,
        }:
            expert_name = (
                "expert_stripe"
                if kind in {PartitionKind.EXPERT_SHARD, PartitionKind.FULL_MIXED_STRIPE}
                else "expert_whole_group"
            )
            latent_down = self.service.service_ms(
                layer,
                (
                    "latent_down"
                    if kind is PartitionKind.FULL_MIXED_STRIPE
                    else "latent_down_whole"
                ),
                degree if kind is PartitionKind.FULL_MIXED_STRIPE else 1,
                rows,
            )
            expert = latent_down + self.service.service_ms(
                layer, expert_name, degree, rows
            ) + self.service.service_ms(layer, "reduction", degree, rows)
        else:
            expert = (
                self.service.service_ms(layer, "latent_down_whole", 1, rows)
                + self.service.service_ms(layer, "expert_whole", 1, rows)
            )
        shared = (
            self.service.service_ms(layer, "shared_expert", degree, rows)
            + self.service.service_ms(layer, "reduction", degree, rows)
            if kind is PartitionKind.FULL_MIXED_STRIPE
            else self.service.service_ms(layer, "shared_expert_whole", 1, rows)
        )
        latent_up = (
            self.service.service_ms(layer, "latent_up", degree, rows)
            + self.service.service_ms(layer, "reduction", degree, rows)
            if kind is PartitionKind.FULL_MIXED_STRIPE
            else self.service.service_ms(layer, "latent_up_whole", 1, rows)
        )
        # Common work is charged to the coordinator while shard work is a
        # concrete worker ceiling, never ideal total_work / N.
        return common + attention + expert + shared + latent_up

    def _internal_network_ms(
        self,
        inventory: Inventory,
        nodes: tuple[str, ...],
        coordinator: str,
        rows: int,
        kind: PartitionKind,
    ) -> float:
        if len(nodes) == 1:
            return 0.0
        payload = rows * 7168 * 4
        rounds = 1
        if kind in {PartitionKind.EXPERT_SHARD, PartitionKind.WHOLE_EXPERT} or kind is PartitionKind.ATTENTION_PROJECTION_SHARD:
            rounds = 2
        elif kind is PartitionKind.FULL_MIXED_STRIPE:
            rounds = 5
        node_map = inventory.node_map()
        return rounds * max(
            node_map[coordinator].peer(node).transfer_ms(payload)
            for node in nodes
            if node != coordinator
        )

    def _candidate_groups(
        self,
        inventory: Inventory,
        construction: _Construction,
        *,
        layer_id: int,
        kind: PartitionKind,
        degree: int,
        chunk_rows: int,
        rng: random.Random,
    ) -> list[tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]]:
        layer = self.model.layers[layer_id]
        resident, checkpoint = candidate_memory(layer, kind, degree)
        nodes = [
            node
            for node in inventory.available_nodes
            if self._runtime_supported(node, kind)
        ]
        if len(nodes) < degree:
            return []

        # Worker 0 owns fixed/common state and therefore uses the first memory
        # requirement.  Candidate ordering is material, not an aggregate bin.
        def feasible(node: NodeCapability, memory: int) -> bool:
            return (
                construction.memory.get(node.node_id, 0) + memory
                <= node.accelerator_memory_bytes
            )

        groups: list[tuple[str, ...]] = []
        if degree == 1:
            eligible = [node for node in nodes if feasible(node, resident[0])]
            eligible.sort(
                key=lambda node: (
                    construction.load.get(node.node_id, 0.0)
                    / node.compute_multiplier,
                    -(
                        layer.resident_bytes
                        / max(node.accelerator_memory_bytes, 1)
                    ),
                    node.node_id,
                )
            )
            if eligible:
                limit = min(len(eligible), self.configuration.candidate_groups_per_action)
                groups.extend((node.node_id,) for node in eligible[:limit])
                if len(eligible) > limit:
                    for _ in range(min(2, len(eligible) - limit)):
                        groups.append((rng.choice(eligible[limit:]).node_id,))
        else:
            by_locality: dict[str, list[NodeCapability]] = {}
            for node in nodes:
                by_locality.setdefault(node.locality_group, []).append(node)
            pools = [*by_locality.values(), nodes]
            for pool in pools:
                # Select the coordinator first because it owns the fixed bytes.
                coordinators = [node for node in pool if feasible(node, resident[0])]
                coordinators.sort(
                    key=lambda node: (
                        construction.load.get(node.node_id, 0.0)
                        / node.compute_multiplier,
                        -node.compute_multiplier,
                        node.node_id,
                    )
                )
                for coordinator in coordinators[:2]:
                    others = [
                        node
                        for node in pool
                        if node.node_id != coordinator.node_id
                        and feasible(node, max(resident[1:], default=0))
                    ]
                    others.sort(
                        key=lambda node: (
                            construction.load.get(node.node_id, 0.0)
                            / node.compute_multiplier,
                            -node.compute_multiplier,
                            coordinator.peer(node.node_id).latency_ms,
                            node.node_id,
                        )
                    )
                    if len(others) >= degree - 1:
                        groups.append(
                            (
                                coordinator.node_id,
                                *(node.node_id for node in others[: degree - 1]),
                            )
                        )
            # A randomized feasible group makes deterministic restarts explore
            # alternatives without embedding a preferred topology.
            shuffled = list(nodes)
            rng.shuffle(shuffled)
            for coordinator in shuffled:
                if not feasible(coordinator, resident[0]):
                    continue
                others = [
                    node
                    for node in shuffled
                    if node.node_id != coordinator.node_id
                    and feasible(node, max(resident[1:], default=0))
                ]
                if len(others) >= degree - 1:
                    groups.append(
                        (
                            coordinator.node_id,
                            *(node.node_id for node in others[: degree - 1]),
                        )
                    )
                    break
        unique: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        for group in groups:
            if group not in seen:
                unique.append(group)
                seen.add(group)
        return [
            (group, resident, checkpoint)
            for group in unique[: self.configuration.candidate_groups_per_action]
        ]

    def _options(
        self,
        inventory: Inventory,
        construction: _Construction,
        *,
        layer_id: int,
        allowed: frozenset[PartitionKind],
        chunk_rows: int,
        rng: random.Random,
    ) -> list[tuple[float, LayerAssignment, dict[str, float]]]:
        options: list[tuple[float, LayerAssignment, dict[str, float]]] = []
        node_map = inventory.node_map()
        for kind in sorted(allowed, key=lambda value: value.value):
            degrees = (1,) if kind is PartitionKind.WHOLE_LAYER else (2, 4, 8, 16)
            for degree in degrees:
                if not self._candidate_supported(layer_id, kind, degree, chunk_rows):
                    continue
                groups = self._candidate_groups(
                    inventory,
                    construction,
                    layer_id=layer_id,
                    kind=kind,
                    degree=degree,
                    chunk_rows=chunk_rows,
                    rng=rng,
                )
                for group, resident, checkpoint in groups:
                    coordinator = group[0]
                    base = sum(
                        self._approximate_worker_ms(layer_id, kind, degree, rows)
                        for rows in target_chunks(chunk_rows)
                    )
                    worker_load = {
                        node_id: base / node_map[node_id].compute_multiplier
                        for node_id in group
                    }
                    # Coordinator common work is already included in `base`;
                    # non-coordinators are conservatively charged the same
                    # ceiling during proposal scoring. Exact DAG re-ranking
                    # later uses every measured phase separately.
                    peak = max(
                        construction.load.get(node_id, 0.0) + worker_load[node_id]
                        for node_id in group
                    )
                    internal_network = sum(
                        self._internal_network_ms(
                            inventory, group, coordinator, rows, kind
                        )
                        for rows in target_chunks(chunk_rows)
                    )
                    boundary = 0.0
                    if construction.previous_coordinator not in (None, coordinator):
                        boundary = node_map[construction.previous_coordinator].peer(
                            coordinator
                        ).transfer_ms(chunk_rows * 7168 * 4)
                    score = peak + internal_network + boundary
                    # Cache is a startup/migration tie-break only; it never
                    # changes steady-state service.
                    if f"layer-{layer_id:02d}" in node_map[coordinator].cached_shards:
                        score -= 1e-5
                    score *= 1.0 + rng.uniform(
                        -self.configuration.randomization,
                        self.configuration.randomization,
                    )
                    assignment = LayerAssignment(
                        layer_id=layer_id,
                        partition_kind=kind,
                        degree=degree,
                        node_ids=group,
                        memory_by_node={
                            node_id: value
                            for node_id, value in zip(group, resident, strict=True)
                        },
                        checkpoint_bytes_by_node={
                            node_id: value
                            for node_id, value in zip(group, checkpoint, strict=True)
                        },
                        coordinator_node_id=coordinator,
                        candidate_id=f"layer-{layer_id:02d}:{kind.value}:p{degree}",
                    )
                    options.append((score, assignment, worker_load))
        options.sort(key=lambda value: (value[0], value[1].candidate_id, value[1].node_ids))
        return options

    def _construct(
        self,
        inventory: Inventory,
        endpoint: EndpointPolicy,
        *,
        allowed: frozenset[PartitionKind],
        chunk_rows: int,
        rng: random.Random,
    ) -> _Construction | None:
        construction = _Construction(
            assignments=[],
            memory=dict(endpoint.memory_by_node),
            load={},
            approximate_network_ms=0.0,
            previous_coordinator=None,
        )
        for layer_id in range(len(self.model.layers)):
            options = self._options(
                inventory,
                construction,
                layer_id=layer_id,
                allowed=allowed,
                chunk_rows=chunk_rows,
                rng=rng,
            )
            if not options:
                return None
            # Favor the leading choices but allow restarts to escape local
            # minima. All choices have already passed hard byte constraints.
            window = options[: min(4, len(options))]
            rank = min(
                int(rng.random() ** 2 * len(window)),
                len(window) - 1,
            )
            score, assignment, worker_load = window[rank]
            for node_id, memory in assignment.memory_by_node.items():
                construction.memory[node_id] = construction.memory.get(node_id, 0) + memory
            for node_id, value in worker_load.items():
                construction.load[node_id] = construction.load.get(node_id, 0.0) + value
            construction.assignments.append(assignment)
            construction.approximate_network_ms += max(0.0, score)
            construction.previous_coordinator = assignment.coordinator_node_id
        construction.score = max(construction.load.values(), default=math.inf) + (
            construction.approximate_network_ms / max(len(self.model.layers), 1)
        )
        return construction

    @staticmethod
    def _clone_fallback(
        fallback: PlacementPlan,
        level: PlannerLevel,
    ) -> PlacementPlan:
        value = copy.deepcopy(fallback)
        value.planner_level = level
        value.optimizer_evaluations = 0
        return value

    def optimize(
        self,
        inventory: Inventory,
        level: PlannerLevel,
        *,
        fallback: PlacementPlan | None = None,
    ) -> OptimizerResult:
        started = time.perf_counter_ns()
        endpoint = common_endpoint_policy(self.model, inventory)
        if endpoint is None:
            plan = PlacementPlan(
                inventory_id=inventory.inventory_id,
                planner_level=level,
                chunk_rows=1,
                assignments=[],
                endpoint_memory_by_node={},
                endpoint_checkpoint_bytes_by_node={},
                feasible=False,
                infeasible_reason="identical endpoint policy cannot fit",
            )
            return OptimizerResult(plan, (), 0.0, 0, 0, False)
        allowed = ALLOWED_BY_LEVEL[level]
        evaluator = PlacementEvaluator(self.model, inventory, self.service, endpoint)
        candidates: list[tuple[float, int, _Construction]] = []
        convergence: list[dict[str, Any]] = []
        proposal = 0
        for restart in range(self.configuration.restarts):
            rng = random.Random(
                self.configuration.seed
                + inventory.seed * 1009
                + restart * 9176
            )
            per_restart = math.ceil(
                self.configuration.proposal_budget / self.configuration.restarts
            )
            for _ in range(per_restart):
                if proposal >= self.configuration.proposal_budget:
                    break
                chunk_rows = (1, 2, 4)[proposal % 3]
                construction = self._construct(
                    inventory,
                    endpoint,
                    allowed=allowed,
                    chunk_rows=chunk_rows,
                    rng=rng,
                )
                proposal += 1
                if construction is not None:
                    candidates.append((construction.score, chunk_rows, construction))
                convergence.append(
                    {
                        "inventory_id": inventory.inventory_id,
                        "planner_level": level.value,
                        "proposal": proposal,
                        "restart": restart,
                        "chunk_rows": chunk_rows,
                        "feasible_proposal": construction is not None,
                        "approximate_score": (
                            construction.score if construction is not None else ""
                        ),
                        "best_exact_tok_s": "",
                    }
                )
        candidates.sort(key=lambda value: value[0])
        unique: list[tuple[int, _Construction]] = []
        seen: set[tuple[tuple[str, tuple[str, ...]], ...]] = set()
        for _score, chunk_rows, construction in candidates:
            signature = tuple(
                (assignment.candidate_id, assignment.node_ids)
                for assignment in construction.assignments
            )
            if signature in seen:
                continue
            seen.add(signature)
            unique.append((chunk_rows, construction))
            if len(unique) >= self.configuration.exact_evaluation_budget:
                break

        best: PlacementPlan | None = None
        fallback_retained = False
        if fallback is not None and fallback.feasible:
            best = self._clone_fallback(fallback, level)
            # Re-evaluate after relabeling to prove the common evaluator gives
            # exactly the same objective for the contained A solution.
            evaluator.evaluate(best)
            fallback_retained = True
        exact_count = 0
        for chunk_rows, construction in unique:
            plan = PlacementPlan(
                inventory_id=inventory.inventory_id,
                planner_level=level,
                chunk_rows=chunk_rows,
                assignments=construction.assignments,
                endpoint_memory_by_node=dict(endpoint.memory_by_node),
                endpoint_checkpoint_bytes_by_node=dict(endpoint.checkpoint_bytes_by_node),
                feasible=True,
                memory_used_by_node={
                    node_id: value
                    for node_id, value in construction.memory.items()
                    if value > 0
                },
                optimizer_seed=self.configuration.seed,
                optimizer_evaluations=self.configuration.proposal_budget,
            )
            evaluator.evaluate(plan)
            exact_count += 1
            if best is None or (plan.objective_tuple or ()) > (best.objective_tuple or ()):
                best = plan
                fallback_retained = False
            convergence.append(
                {
                    "inventory_id": inventory.inventory_id,
                    "planner_level": level.value,
                    "proposal": proposal,
                    "restart": "exact",
                    "chunk_rows": chunk_rows,
                    "feasible_proposal": True,
                    "approximate_score": construction.score,
                    "best_exact_tok_s": best.exact_tok_s_per_user,
                }
            )
        if best is None:
            best = PlacementPlan(
                inventory_id=inventory.inventory_id,
                planner_level=level,
                chunk_rows=1,
                assignments=[],
                endpoint_memory_by_node=dict(endpoint.memory_by_node),
                endpoint_checkpoint_bytes_by_node=dict(endpoint.checkpoint_bytes_by_node),
                feasible=False,
                infeasible_reason="shared optimizer found no hard-memory-feasible placement",
                optimizer_seed=self.configuration.seed,
                optimizer_evaluations=self.configuration.proposal_budget,
            )
        if fallback is not None:
            best.verify_dominance_fallback(fallback)
        elapsed = (time.perf_counter_ns() - started) / 1e6
        return OptimizerResult(
            plan=best,
            convergence=tuple(convergence),
            elapsed_ms=elapsed,
            proposals=proposal,
            exact_evaluations=exact_count,
            fallback_retained=fallback_retained,
        )


__all__ = [
    "OptimizerConfiguration",
    "OptimizerResult",
    "SharedPlacementOptimizer",
]
