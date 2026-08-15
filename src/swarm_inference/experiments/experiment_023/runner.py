"""Deterministic baseline, FLEX planning, and serving evaluation for E023."""

from __future__ import annotations

import copy
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_022.completion_inputs import (
    load_frozen_inventories,
)
from swarm_inference.experiments.experiment_022.evaluator import common_endpoint_policy
from swarm_inference.experiments.experiment_022.io import (
    atomic_write_json,
    canonical_sha256,
    write_csv,
)
from swarm_inference.experiments.experiment_022.model_graph import build_model_graph
from swarm_inference.experiments.experiment_022.models import Inventory, ModelGraph
from swarm_inference.experiments.experiment_022.service import ResidentServiceModel

from .baseline import (
    UniqueEnvelopeCandidate,
    frozen_placement_paths,
    load_frozen_placement,
    repair_whole_layer_relocations,
    select_u_strong,
)
from .freeze import (
    CAPACITY_INVENTORIES,
    CONTROL_INVENTORIES,
    FROZEN_CONSTANTS,
    HEADLINE_INVENTORIES,
    validate_e023_freeze,
)
from .models import Arm, E023Plan, NetworkMode, base_resident_bytes
from .replica_memory import abstract_node_cost
from .replica_planner import (
    PlannerContext,
    run_flex_planner,
    run_unique_p8_refinement,
)
from .serving_engine import CONCURRENCY_LEVELS, ServingEngine, ServingRun

CHECKPOINT = Path("F:/models/Kimi-K3")
DEFAULT_PROFILE_WORKERS = min(12, os.cpu_count() or 1)
ARM_RESULT_FIELDS = (
    "inventory_id",
    "family",
    "cohort",
    "arm",
    "network_mode",
    "concurrency",
    "status",
    "target_passes_measured",
    "target_rows_measured",
    "measurement_window_ms",
    "target_rows_per_second",
    "p50_pass_latency_ms",
    "p95_pass_latency_ms",
    "abstract_node_cost",
    "rows_per_second_per_abstract_cost",
    "network_bytes",
    "network_bytes_per_target_row",
    "worker_compute_ms",
    "worker_compute_ms_per_target_row",
    "nodes_used",
    "replica_nodes",
    "replica_count",
    "replica_checkpoint_bytes",
    "replica_resident_bytes",
    "new_nodes_activated",
    "maximum_compute_utilization",
    "maximum_tx_utilization",
    "maximum_rx_utilization",
    "plan_sha256",
)


def cohort_for(inventory_id: str) -> str:
    if inventory_id in HEADLINE_INVENTORIES:
        return "headline"
    if inventory_id in CONTROL_INVENTORIES:
        return "negative_control"
    if inventory_id in CAPACITY_INVENTORIES:
        return "capacity_exploratory"
    raise ValueError(f"inventory {inventory_id} is outside the frozen E023 cohorts")


@dataclass(slots=True)
class ExperimentContext:
    repo: Path
    model: ModelGraph
    service: ResidentServiceModel
    inventories: dict[str, Inventory]

    @classmethod
    def load(cls, repo: Path) -> ExperimentContext:
        root = repo.resolve()
        validate_e023_freeze(root)
        if not CHECKPOINT.is_dir():
            raise RuntimeError(f"MODEL_INVALID: local Kimi K3 checkpoint missing: {CHECKPOINT}")
        inventories, _audit = load_frozen_inventories(root)
        model = build_model_graph(
            CHECKPOINT,
            whole_layer_service_csv=(
                root / "artifacts/experiment-018/physical/layer-service.csv"
            ),
        )
        service = ResidentServiceModel.from_csv(
            root
            / "artifacts/experiment-022/completion/validation/"
            "repaired-resident-service.csv"
        )
        return cls(
            repo=root,
            model=model,
            service=service,
            inventories={value.inventory_id: value for value in inventories},
        )

    def engines(
        self, inventory: Inventory
    ) -> tuple[ServingEngine, ServingEngine, PlannerContext]:
        endpoint = common_endpoint_policy(self.model, inventory)
        if endpoint is None:
            raise RuntimeError(
                f"MODEL_INVALID: common endpoint infeasible for {inventory.inventory_id}"
            )
        shared = ServingEngine(
            self.model,
            inventory,
            self.service,
            endpoint,
            network_mode=NetworkMode.SHARED_NIC,
        )
        legacy = ServingEngine(
            self.model,
            inventory,
            self.service,
            endpoint,
            network_mode=NetworkMode.LEGACY_DIRECTED_LINK,
        )
        return shared, legacy, PlannerContext(
            self.model, inventory, self.service, shared
        )


@dataclass(slots=True)
class BaselineResult:
    u_strong: E023Plan
    envelope_rows: list[dict[str, Any]]
    relocation_rows: list[dict[str, Any]]
    refinement_rows: list[dict[str, Any]]
    fastest_candidate: str


@dataclass(slots=True)
class InventoryResult:
    inventory_id: str
    plans: dict[str, E023Plan]
    baseline: BaselineResult
    planner_rows: list[dict[str, Any]]
    runs: dict[tuple[str, str, int], ServingRun]
    arm_rows: list[dict[str, Any]]
    saturation_rows: list[dict[str, Any]]
    routing_rows: list[dict[str, Any]]
    resource_rows: list[dict[str, Any]]
    network_rows: list[dict[str, Any]]
    memory_rows: list[dict[str, Any]]
    cost_rows: list[dict[str, Any]]


def _plan_key(plan: E023Plan) -> str:
    return plan.canonical_sha256


def construct_u_strong(
    context: ExperimentContext,
    inventory: Inventory,
    shared_engine: ServingEngine,
    planner_context: PlannerContext,
) -> BaselineResult:
    """Construct the frozen-envelope, repaired, uniquely refined U_STRONG."""

    endpoint = shared_engine.endpoint
    frozen: list[tuple[str, Any]] = []
    for level, path in zip(
        "ABCDE",
        frozen_placement_paths(context.repo, inventory.inventory_id),
        strict=True,
    ):
        plan = load_frozen_placement(path, inventory)
        if plan.feasible:
            frozen.append((f"FROZEN_{level}", plan))
    if not frozen:
        raise RuntimeError(
            f"MODEL_INVALID: no feasible frozen placement for {inventory.inventory_id}"
        )

    repair_cache: dict[str, tuple[Any, list[dict[str, Any]]]] = {}
    repaired: list[tuple[str, Any]] = []
    relocation_rows: list[dict[str, Any]] = []
    for frozen_name, plan in frozen:
        key = _plan_key(E023Plan(inventory.inventory_id, Arm.U_STRONG.value, plan))
        if key not in repair_cache:
            repair_cache[key] = repair_whole_layer_relocations(
                context.model,
                inventory,
                context.service,
                endpoint,
                plan,
                starting_candidate=frozen_name,
            )
        repaired_plan, cached_rows = repair_cache[key]
        name = f"REPAIRED_{frozen_name.removeprefix('FROZEN_')}"
        repaired.append((name, copy.deepcopy(repaired_plan)))
        for row in cached_rows:
            value = dict(row)
            value["starting_candidate"] = frozen_name
            relocation_rows.append(value)

    # U_P8_REFINE is singular.  Its deterministic seed is the repaired plan
    # with highest C32 throughput, matching the declared refinement objective.
    repaired_profiles = [
        (
            planner_context.profile(
                E023Plan(inventory.inventory_id, Arm.U_STRONG.value, plan)
            ).target_rows_per_second,
            E023Plan(inventory.inventory_id, Arm.U_STRONG.value, plan).canonical_sha256,
            name,
            plan,
        )
        for name, plan in repaired
    ]
    refinement_seed = min(
        repaired_profiles,
        key=lambda value: (-value[0], value[1], value[2]),
    )
    refined, refinement_rows = run_unique_p8_refinement(
        planner_context, copy.deepcopy(refinement_seed[3])
    )

    candidate_specs = [*frozen, *repaired, ("U_P8_REFINE", refined)]
    e023_candidates = [
        E023Plan(inventory.inventory_id, Arm.U_STRONG.value, plan)
        for _name, plan in candidate_specs
    ]
    requests = [
        (candidate, NetworkMode.SHARED_NIC, concurrency)
        for candidate in e023_candidates
        for concurrency in CONCURRENCY_LEVELS
    ]
    evaluated_runs = planner_context.serving_runs(requests)
    candidates: list[UniqueEnvelopeCandidate] = []
    offset = 0
    for (name, plan), e023 in zip(
        candidate_specs, e023_candidates, strict=True
    ):
        values = evaluated_runs[offset : offset + len(CONCURRENCY_LEVELS)]
        offset += len(CONCURRENCY_LEVELS)
        runs = dict(zip(CONCURRENCY_LEVELS, values, strict=True))
        if any(run.status != "PASS" for run in runs.values()):
            raise RuntimeError("MODEL_INVALID: unique baseline concurrency incomplete")
        peak = max(run.target_rows_per_second for run in values)
        peak_concurrency = min(
            run.concurrency
            for run in values
            if abs(run.target_rows_per_second - peak) <= 1e-12
        )
        candidates.append(
            UniqueEnvelopeCandidate(
                candidate_name=name,
                plan=plan,
                runs=runs,
                peak_target_rows_per_second=peak,
                peak_concurrency=peak_concurrency,
                abstract_node_cost=abstract_node_cost(e023, inventory),
                canonical_plan_sha256=e023.canonical_sha256,
            )
        )
    selected, fastest = select_u_strong(candidates)
    fastest_peak = max(row.peak_target_rows_per_second for row in candidates)
    floor = float(FROZEN_CONSTANTS["economic_fastest_throughput_floor"])
    envelope_rows = [
        {
            "inventory_id": inventory.inventory_id,
            "candidate_name": row.candidate_name,
            "feasible": True,
            "chunk_rows": row.plan.chunk_rows,
            "peak_target_rows_per_second": row.peak_target_rows_per_second,
            "peak_concurrency": row.peak_concurrency,
            "abstract_node_cost": row.abstract_node_cost,
            "peak_rows_per_second_per_abstract_cost": (
                row.peak_rows_per_second_per_abstract_cost
            ),
            "economic_floor_target_rows_per_second": floor * fastest_peak,
            "passes_economic_floor": (
                row.peak_target_rows_per_second >= floor * fastest_peak
            ),
            "canonical_plan_sha256": row.canonical_plan_sha256,
            "selected_u_fastest": row.candidate_name == fastest,
            "selected_u_strong": row.candidate_name == selected.candidate_name,
        }
        for row in candidates
    ]
    u_strong = E023Plan(
        inventory.inventory_id,
        Arm.U_STRONG.value,
        copy.deepcopy(selected.plan),
        metadata={
            "selected_from": selected.candidate_name,
            "u_fastest": fastest,
            "refinement_seed": refinement_seed[2],
            "planning_concurrency": 32,
        },
    )
    return BaselineResult(
        u_strong=u_strong,
        envelope_rows=envelope_rows,
        relocation_rows=relocation_rows,
        refinement_rows=refinement_rows,
        fastest_candidate=fastest,
    )


def derive_no_alt(plan: E023Plan, arm: Arm) -> E023Plan:
    if arm not in {Arm.FLEX_FREE_NO_ALT, Arm.FLEX_POOL_NO_ALT}:
        raise ValueError("no-alternative derivation requires a declared ablation arm")
    return E023Plan(
        plan.inventory_id,
        arm.value,
        copy.deepcopy(plan.base_plan),
        (),
        u_strong_used_nodes=plan.u_strong_used_nodes,
        metadata={
            "derived_from": plan.arm,
            "removed_only_alternate_residency": True,
            "source_plan_sha256": plan.canonical_sha256,
        },
    )


def _arm_result_row(
    inventory: Inventory,
    plan: E023Plan,
    run: ServingRun,
    u_strong_nodes: set[str],
) -> dict[str, Any]:
    cost = abstract_node_cost(plan, inventory)
    target_rows = run.target_rows_measured
    row = {
        "inventory_id": inventory.inventory_id,
        "family": inventory.family,
        "cohort": cohort_for(inventory.inventory_id),
        "arm": plan.arm,
        "network_mode": run.network_mode.value,
        "concurrency": run.concurrency,
        "status": run.status,
        "target_passes_measured": run.target_passes_measured,
        "target_rows_measured": target_rows,
        "measurement_window_ms": run.measurement_window_ms,
        "target_rows_per_second": run.target_rows_per_second,
        "p50_pass_latency_ms": run.p50_pass_latency_ms,
        "p95_pass_latency_ms": run.p95_pass_latency_ms,
        "abstract_node_cost": cost,
        "rows_per_second_per_abstract_cost": run.target_rows_per_second / cost,
        "network_bytes": run.network_bytes,
        "network_bytes_per_target_row": run.network_bytes / target_rows,
        "worker_compute_ms": run.worker_compute_ms,
        "worker_compute_ms_per_target_row": run.worker_compute_ms / target_rows,
        "nodes_used": len(plan.used_nodes),
        "replica_nodes": len({row.alternate_node_id for row in plan.replicas}),
        "replica_count": plan.replica_count,
        "replica_checkpoint_bytes": plan.replica_checkpoint_bytes,
        "replica_resident_bytes": plan.replica_resident_bytes,
        "new_nodes_activated": len(plan.used_nodes.difference(u_strong_nodes)),
        "maximum_compute_utilization": run.maximum_compute_utilization,
        "maximum_tx_utilization": run.maximum_tx_utilization,
        "maximum_rx_utilization": run.maximum_rx_utilization,
        "plan_sha256": plan.canonical_sha256,
    }
    if tuple(row) != ARM_RESULT_FIELDS:
        raise AssertionError("arm-results schema changed")
    return row


def _slo_selection(
    runs: dict[int, ServingRun], latency_budget_ms: float
) -> ServingRun | None:
    eligible = [
        run
        for run in runs.values()
        if run.status == "PASS" and run.p95_pass_latency_ms <= latency_budget_ms
    ]
    if not eligible:
        return None
    maximum = max(run.target_rows_per_second for run in eligible)
    near = [
        run
        for run in eligible
        if run.target_rows_per_second >= 0.995 * maximum
    ]
    return min(near, key=lambda run: run.concurrency)


def _saturation_rows(
    inventory: Inventory,
    plans: dict[str, E023Plan],
    runs: dict[tuple[str, str, int], ServingRun],
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for arm_name, plan in plans.items():
        for mode in (NetworkMode.SHARED_NIC, NetworkMode.LEGACY_DIRECTED_LINK):
            arm_runs = {
                concurrency: runs[(arm_name, mode.value, concurrency)]
                for concurrency in CONCURRENCY_LEVELS
                if (arm_name, mode.value, concurrency) in runs
            }
            if not arm_runs:
                continue
            u_c1 = runs[(Arm.U_STRONG.value, mode.value, 1)]
            peak = max(arm_runs.values(), key=lambda run: run.target_rows_per_second)
            row: dict[str, Any] = {
                "inventory_id": inventory.inventory_id,
                "family": inventory.family,
                "cohort": cohort_for(inventory.inventory_id),
                "arm": arm_name,
                "network_mode": mode.value,
                "peak_target_rows_per_second": peak.target_rows_per_second,
                "peak_concurrency": peak.concurrency,
                "abstract_node_cost": abstract_node_cost(plan, inventory),
                "plan_sha256": plan.canonical_sha256,
                "u_strong_c1_p95_pass_latency_ms": u_c1.p95_pass_latency_ms,
            }
            for multiplier in FROZEN_CONSTANTS["latency_sensitivity_multipliers"]:
                budget = float(multiplier) * u_c1.p95_pass_latency_ms
                selected = _slo_selection(arm_runs, budget)
                prefix = f"slo_{float(multiplier):.1f}x"
                row[f"{prefix}_latency_budget_ms"] = budget
                if selected is None:
                    row[f"{prefix}_target_rows_per_second"] = 0.0
                    row[f"{prefix}_concurrency"] = ""
                    row[f"{prefix}_p50_pass_latency_ms"] = ""
                    row[f"{prefix}_p95_pass_latency_ms"] = ""
                    row[f"{prefix}_rows_per_second_per_abstract_cost"] = 0.0
                    row[f"{prefix}_latency_eligible"] = False
                else:
                    row[f"{prefix}_target_rows_per_second"] = (
                        selected.target_rows_per_second
                    )
                    row[f"{prefix}_concurrency"] = selected.concurrency
                    row[f"{prefix}_p50_pass_latency_ms"] = (
                        selected.p50_pass_latency_ms
                    )
                    row[f"{prefix}_p95_pass_latency_ms"] = (
                        selected.p95_pass_latency_ms
                    )
                    row[f"{prefix}_rows_per_second_per_abstract_cost"] = (
                        selected.target_rows_per_second
                        / abstract_node_cost(plan, inventory)
                    )
                    row[f"{prefix}_latency_eligible"] = True
            values.append(row)
    return values


def _routing_rows(
    inventory: Inventory,
    plan: E023Plan,
    run: ServingRun,
) -> list[dict[str, Any]]:
    replicas = {(row.layer_id, row.logical_group_id): row for row in plan.replicas}
    if not replicas:
        return []
    eligible = {
        (row.slot_id, row.pass_index)
        for row in run.passes
        if row.pass_index >= 2 and row.start_ms >= run.t0_ms
    }
    counts = {key: [0, 0] for key in replicas}
    for pass_row in run.passes:
        if (pass_row.slot_id, pass_row.pass_index) not in eligible:
            continue
        for fork in pass_row.fork_join_records:
            assignment = plan.base_plan.assignments[fork.layer_id]
            for group, selected in fork.selected_nodes.items():
                key = (fork.layer_id, group)
                if key not in replicas:
                    continue
                counts[key][selected != assignment.node_ids[group]] += 1
    values: list[dict[str, Any]] = []
    for key, (primary, alternate) in sorted(counts.items()):
        replica = replicas[key]
        total = primary + alternate
        values.append(
            {
                "inventory_id": inventory.inventory_id,
                "arm": plan.arm,
                "network_mode": run.network_mode.value,
                "concurrency": run.concurrency,
                "layer_id": key[0],
                "logical_group_id": key[1],
                "primary_node": replica.primary_node_id,
                "alternate_node": replica.alternate_node_id,
                "selected_primary_count": primary,
                "selected_alternate_count": alternate,
                "alternate_selection_rate": alternate / total if total else 0.0,
            }
        )
    return values


def _resource_rows(
    inventory: Inventory,
    plan: E023Plan,
    run: ServingRun,
) -> list[dict[str, Any]]:
    base_memory = base_resident_bytes(plan.base_plan)
    replica_memory: dict[str, int] = {}
    for replica in plan.replicas:
        replica_memory[replica.alternate_node_id] = (
            replica_memory.get(replica.alternate_node_id, 0) + replica.resident_bytes
        )
    return [
        {
            "inventory_id": inventory.inventory_id,
            "arm": plan.arm,
            "network_mode": run.network_mode.value,
            "concurrency": run.concurrency,
            "node_id": node_id,
            "compute_busy_ms": run.compute_busy_ms_by_node.get(node_id, 0.0),
            "compute_queue_wait_ms": run.compute_queue_wait_ms_by_node.get(node_id, 0.0),
            "tx_busy_ms": run.tx_busy_ms_by_node.get(node_id, 0.0),
            "rx_busy_ms": run.rx_busy_ms_by_node.get(node_id, 0.0),
            "compute_utilization": run.compute_busy_ms_by_node.get(node_id, 0.0) / run.measurement_window_ms,
            "tx_utilization": run.tx_busy_ms_by_node.get(node_id, 0.0) / run.measurement_window_ms,
            "rx_utilization": run.rx_busy_ms_by_node.get(node_id, 0.0) / run.measurement_window_ms,
            "resident_model_bytes": base_memory.get(node_id, 0),
            "replica_resident_bytes": replica_memory.get(node_id, 0),
        }
        for node_id in sorted(plan.used_nodes)
    ]


def _network_row(
    inventory: Inventory,
    plan: E023Plan,
    run: ServingRun,
) -> dict[str, Any]:
    return {
        "inventory_id": inventory.inventory_id,
        "arm": plan.arm,
        "network_mode": run.network_mode.value,
        "concurrency": run.concurrency,
        "network_bytes": run.network_bytes,
        "transfer_count": run.transfer_count,
        "network_queue_wait_ms": run.network_queue_wait_ms,
        "maximum_tx_utilization": run.maximum_tx_utilization,
        "maximum_rx_utilization": run.maximum_rx_utilization,
        "maximum_directed_link_utilization": run.maximum_link_utilization,
        "target_rows_measured": run.target_rows_measured,
        "network_bytes_per_target_row": run.network_bytes / run.target_rows_measured,
    }


def reconcile_plan(
    model: ModelGraph,
    inventory: Inventory,
    plan: E023Plan,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    nodes = inventory.node_map()
    base = base_resident_bytes(plan.base_plan)
    replica: dict[str, int] = {}
    for value in plan.replicas:
        replica[value.alternate_node_id] = (
            replica.get(value.alternate_node_id, 0) + value.resident_bytes
        )
    memory_rows = []
    for node_id in sorted(nodes):
        total = base.get(node_id, 0) + replica.get(node_id, 0)
        capacity = nodes[node_id].accelerator_memory_bytes
        memory_rows.append(
            {
                "inventory_id": inventory.inventory_id,
                "arm": plan.arm,
                "node_id": node_id,
                "base_plan_resident_bytes": base.get(node_id, 0),
                "replica_resident_bytes": replica.get(node_id, 0),
                "total_resident_bytes": total,
                "accelerator_memory_bytes": capacity,
                "headroom_bytes": capacity - total,
                "within_capacity": total <= capacity,
                "replica_persistent_state_bytes": sum(
                    row.persistent_state_bytes
                    for row in plan.replicas
                    if row.alternate_node_id == node_id
                ),
            }
        )
    if not all(row["within_capacity"] for row in memory_rows):
        raise RuntimeError("MODEL_INVALID: final plan exceeds accelerator memory")
    if any(row["replica_persistent_state_bytes"] != 0 for row in memory_rows):
        raise RuntimeError("MODEL_INVALID: stateful model piece was replicated")
    expected_cost = sum(nodes[node].cost for node in sorted(plan.used_nodes))
    actual_cost = abstract_node_cost(plan, inventory)
    if actual_cost != expected_cost:
        raise RuntimeError("MODEL_INVALID: abstract cost reconciliation failed")
    cost_row = {
        "inventory_id": inventory.inventory_id,
        "arm": plan.arm,
        "nodes_used": len(plan.used_nodes),
        "replica_only_nodes": len(set(replica).difference(base)),
        "abstract_node_cost": actual_cost,
        "expected_abstract_node_cost": expected_cost,
        "difference": actual_cost - expected_cost,
        "node_counted_once": True,
        "status": "PASS",
        "unique_model_checkpoint_bytes": model.checkpoint_payload_bytes,
        "duplicate_replica_checkpoint_bytes": plan.replica_checkpoint_bytes,
        "total_resident_checkpoint_bytes": (
            model.checkpoint_payload_bytes + plan.replica_checkpoint_bytes
        ),
    }
    return memory_rows, cost_row


def evaluate_inventory(
    context: ExperimentContext,
    inventory_id: str,
    *,
    progress: Callable[[str], None] | None = None,
) -> InventoryResult:
    inventory = context.inventories[inventory_id]
    notify = progress if progress is not None else (lambda _message: None)
    shared, _legacy, planner_context = context.engines(inventory)
    planner_context.enable_parallel_profiles(DEFAULT_PROFILE_WORKERS)
    try:
        notify(f"{inventory_id}: constructing U_STRONG")
        baseline = construct_u_strong(context, inventory, shared, planner_context)
        u_strong = baseline.u_strong
        notify(f"{inventory_id}: planning FLEX_FREE")
        flex_free, free_rows = run_flex_planner(
            planner_context, u_strong.base_plan, arm=Arm.FLEX_FREE
        )
        notify(f"{inventory_id}: planning FLEX_POOL")
        flex_pool, pool_rows = run_flex_planner(
            planner_context, u_strong.base_plan, arm=Arm.FLEX_POOL
        )
        plans = {
            Arm.U_STRONG.value: u_strong,
            Arm.FLEX_FREE_NO_ALT.value: derive_no_alt(
                flex_free, Arm.FLEX_FREE_NO_ALT
            ),
            Arm.FLEX_FREE.value: flex_free,
            Arm.FLEX_POOL_NO_ALT.value: derive_no_alt(
                flex_pool, Arm.FLEX_POOL_NO_ALT
            ),
            Arm.FLEX_POOL.value: flex_pool,
        }
        notify(f"{inventory_id}: evaluating final concurrency ladder")
        request_keys: list[tuple[str, str, int]] = []
        requests: list[tuple[E023Plan, NetworkMode, int]] = []
        for arm_name, plan in plans.items():
            for concurrency in CONCURRENCY_LEVELS:
                request_keys.append(
                    (arm_name, NetworkMode.SHARED_NIC.value, concurrency)
                )
                requests.append((plan, NetworkMode.SHARED_NIC, concurrency))
        for arm_name in (Arm.U_STRONG.value, Arm.FLEX_POOL.value):
            plan = plans[arm_name]
            for concurrency in CONCURRENCY_LEVELS:
                request_keys.append(
                    (arm_name, NetworkMode.LEGACY_DIRECTED_LINK.value, concurrency)
                )
                requests.append((plan, NetworkMode.LEGACY_DIRECTED_LINK, concurrency))
        evaluated = planner_context.serving_runs(requests)
        runs = dict(zip(request_keys, evaluated, strict=True))
    finally:
        planner_context.close()
    planner_context.profile_cache.clear()
    planner_context.serving_run_cache.clear()
    expected = 5 * 5 + 2 * 5
    if len(runs) != expected or any(run.status != "PASS" for run in runs.values()):
        raise RuntimeError("MODEL_INVALID: required arm/concurrency row incomplete")

    u_nodes = u_strong.used_nodes
    arm_rows: list[dict[str, Any]] = []
    routing_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    network_rows: list[dict[str, Any]] = []
    for (arm_name, _mode, _concurrency), run in sorted(runs.items()):
        plan = plans[arm_name]
        arm_rows.append(_arm_result_row(inventory, plan, run, u_nodes))
        routing_rows.extend(_routing_rows(inventory, plan, run))
        resource_rows.extend(_resource_rows(inventory, plan, run))
        network_rows.append(_network_row(inventory, plan, run))
    saturation_rows = _saturation_rows(inventory, plans, runs)
    memory_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    for plan in plans.values():
        plan_memory, plan_cost = reconcile_plan(context.model, inventory, plan)
        memory_rows.extend(plan_memory)
        cost_rows.append(plan_cost)
    return InventoryResult(
        inventory_id=inventory_id,
        plans=plans,
        baseline=baseline,
        planner_rows=free_rows + pool_rows,
        runs=runs,
        arm_rows=arm_rows,
        saturation_rows=saturation_rows,
        routing_rows=routing_rows,
        resource_rows=resource_rows,
        network_rows=network_rows,
        memory_rows=memory_rows,
        cost_rows=cost_rows,
    )


@dataclass(slots=True)
class AttemptArtifacts:
    inventory_results: list[InventoryResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def write_attempt(
    context: ExperimentContext,
    attempt_root: Path,
    result: AttemptArtifacts,
) -> None:
    attempt_root.mkdir(parents=True, exist_ok=True)
    baseline_rows = [
        row
        for inventory in result.inventory_results
        for row in inventory.baseline.envelope_rows
    ]
    relocation_rows = [
        row
        for inventory in result.inventory_results
        for row in inventory.baseline.relocation_rows
    ]
    refinement_rows = [
        row
        for inventory in result.inventory_results
        for row in inventory.baseline.refinement_rows
    ]
    write_csv(attempt_root / "baseline/baseline-envelope.csv", baseline_rows)
    write_csv(attempt_root / "baseline/baseline-relocations.csv", relocation_rows)
    write_csv(
        attempt_root / "baseline/unique-refinement-actions.csv", refinement_rows
    )
    write_csv(
        attempt_root / "serving/replica-actions.csv",
        [row for item in result.inventory_results for row in item.planner_rows],
    )
    for name, attribute in (
        ("arm-results.csv", "arm_rows"),
        ("saturation-summary.csv", "saturation_rows"),
        ("replica-routing-summary.csv", "routing_rows"),
        ("resource-utilization.csv", "resource_rows"),
        ("network-summary.csv", "network_rows"),
    ):
        write_csv(
            attempt_root / "serving" / name,
            [
                row
                for item in result.inventory_results
                for row in getattr(item, attribute)
            ],
        )
    write_csv(
        attempt_root / "validation/memory-reconciliation.csv",
        [row for item in result.inventory_results for row in item.memory_rows],
    )
    write_csv(
        attempt_root / "validation/cost-reconciliation.csv",
        [row for item in result.inventory_results for row in item.cost_rows],
    )
    for item in result.inventory_results:
        inventory = context.inventories[item.inventory_id]
        for arm_name, plan in item.plans.items():
            atomic_write_json(
                attempt_root / "plans" / item.inventory_id / f"{arm_name}.json",
                plan.as_manifest(context.model, inventory),
            )
    atomic_write_json(
        attempt_root / "attempt-summary.json",
        {
            "schema_version": "experiment-023-deterministic-attempt-v1",
            "status": "PASS",
            "inventory_ids": [row.inventory_id for row in result.inventory_results],
            "inventory_count": len(result.inventory_results),
            "elapsed_seconds": result.elapsed_seconds,
            "thresholds_sha256": canonical_sha256(FROZEN_CONSTANTS),
        },
    )


def run_deterministic_attempt(
    repo: Path,
    attempt_root: Path,
    *,
    inventory_ids: Iterable[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> AttemptArtifacts:
    context = ExperimentContext.load(repo)
    ids = (
        tuple(inventory_ids)
        if inventory_ids is not None
        else tuple(context.inventories)
    )
    if any(value not in context.inventories for value in ids):
        raise ValueError("unknown inventory requested")
    started = time.perf_counter()
    result = AttemptArtifacts()
    for index, inventory_id in enumerate(ids, 1):
        if progress is not None:
            progress(f"[{index}/{len(ids)}] {inventory_id}: start")
        result.inventory_results.append(
            evaluate_inventory(context, inventory_id, progress=progress)
        )
        result.elapsed_seconds = time.perf_counter() - started
        write_attempt(context, attempt_root, result)
        if progress is not None:
            progress(
                f"[{index}/{len(ids)}] {inventory_id}: complete "
                f"({result.elapsed_seconds:.1f}s cumulative)"
            )
    return result


__all__ = [
    "AttemptArtifacts",
    "BaselineResult",
    "ExperimentContext",
    "InventoryResult",
    "construct_u_strong",
    "derive_no_alt",
    "evaluate_inventory",
    "run_deterministic_attempt",
    "write_attempt",
]
