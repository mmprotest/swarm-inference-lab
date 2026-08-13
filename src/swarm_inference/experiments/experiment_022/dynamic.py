"""Capability-driven join, slowdown, link-change, and loss tests for E022."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from .evaluator import EndpointPolicy, PlacementEvaluator
from .inventories import LINK_CLASSES
from .models import Inventory, NetworkPeer, NodeCapability, PlacementPlan, PlannerLevel
from .planner import SharedPlacementOptimizer


def _replace_node(inventory: Inventory, replacement: NodeCapability) -> Inventory:
    nodes = tuple(
        replacement if node.node_id == replacement.node_id else node
        for node in inventory.nodes
    )
    return replace(inventory, nodes=nodes)


def _join(inventory: Inventory, *, useful: bool) -> Inventory:
    identifier = "dynamic-useful" if useful else "dynamic-harmful"
    template = max(inventory.nodes, key=lambda node: node.accelerator_memory_bytes)
    group = template.locality_group if useful else "dynamic-remote"
    link_name = "fast" if useful else "slow"
    new_peers: dict[str, NetworkPeer] = {}
    updated: list[NodeCapability] = []
    for node in inventory.nodes:
        class_name = "fast" if useful and node.locality_group == group else link_name
        values = LINK_CLASSES[class_name]
        to_new = NetworkPeer(
            peer_id=identifier,
            latency_ms=values["latency_ms"],
            bandwidth_gbps=values["bandwidth_gbps"],
            software_overhead_ms=values["software_overhead_ms"],
            locality_class=class_name,
        )
        peers = dict(node.network_peers)
        peers[identifier] = to_new
        updated.append(replace(node, network_peers=peers))
        new_peers[node.node_id] = NetworkPeer(
            peer_id=node.node_id,
            latency_ms=values["latency_ms"],
            bandwidth_gbps=values["bandwidth_gbps"],
            software_overhead_ms=values["software_overhead_ms"],
            locality_class=class_name,
        )
    joined = NodeCapability(
        node_id=identifier,
        accelerator_memory_bytes=(
            max(16 * 1024**3, template.accelerator_memory_bytes // 2)
            if useful
            else max(node.accelerator_memory_bytes for node in inventory.nodes) // 4
        ),
        system_memory_bytes=template.system_memory_bytes,
        compute_profile={"reference": 1.0 if useful else 0.4},
        memory_bandwidth_profile={"reference": 1.0 if useful else 0.4},
        supported_precisions=template.supported_precisions,
        network_peers=new_peers,
        reliability=0.999 if useful else 0.97,
        cost=template.cost if useful else template.cost * 2,
        cached_shards=(
            tuple(f"layer-{layer:02d}" for layer in range(93))
            if useful
            else ()
        ),
        runtime_capabilities=template.runtime_capabilities,
        locality_group=group,
    )
    return replace(
        inventory,
        nodes=(*updated, joined),
        scenario=inventory.scenario + ("; useful node joined" if useful else "; harmful node joined"),
    )


def _slowdown(inventory: Inventory, plan: PlacementPlan) -> tuple[Inventory, str]:
    critical = max(
        plan.worker_utilization,
        key=lambda node: (plan.worker_utilization[node], node),
    )
    node = inventory.node_map()[critical]
    slower = replace(
        node,
        compute_profile={
            **node.compute_profile,
            "reference": node.compute_multiplier * 0.5,
        },
        memory_bandwidth_profile={
            **node.memory_bandwidth_profile,
            "reference": float(node.memory_bandwidth_profile["reference"]) * 0.5,
        },
    )
    return _replace_node(inventory, slower), critical


def _degrade_network(inventory: Inventory, plan: PlacementPlan) -> tuple[Inventory, str]:
    pair: tuple[str, str] | None = None
    for assignment in plan.assignments:
        if assignment.degree > 1:
            pair = (assignment.coordinator_node_id, assignment.node_ids[-1])
            break
    if pair is None:
        # The scenario still tests a concrete used coarse boundary.
        for left, right in zip(plan.assignments, plan.assignments[1:], strict=False):
            if left.coordinator_node_id != right.coordinator_node_id:
                pair = (left.coordinator_node_id, right.coordinator_node_id)
                break
    if pair is None:
        pair = (inventory.nodes[0].node_id, inventory.nodes[1].node_id)
    slow = LINK_CLASSES["slow"]
    updated: list[NodeCapability] = []
    for node in inventory.nodes:
        peers = dict(node.network_peers)
        if node.node_id == pair[0]:
            peers[pair[1]] = NetworkPeer(pair[1], slow["latency_ms"], slow["bandwidth_gbps"], slow["software_overhead_ms"], "slow")
        elif node.node_id == pair[1]:
            peers[pair[0]] = NetworkPeer(pair[0], slow["latency_ms"], slow["bandwidth_gbps"], slow["software_overhead_ms"], "slow")
        updated.append(replace(node, network_peers=peers))
    return replace(inventory, nodes=tuple(updated)), f"{pair[0]}<->{pair[1]}"


def _loss(inventory: Inventory, plan: PlacementPlan) -> tuple[Inventory, str]:
    lost = max(
        plan.worker_utilization,
        key=lambda node: (plan.worker_utilization[node], node),
    )
    remaining = [node for node in inventory.nodes if node.node_id != lost]
    updated = tuple(
        replace(
            node,
            network_peers={
                peer: value for peer, value in node.network_peers.items() if peer != lost
            },
        )
        for node in remaining
    )
    return replace(inventory, nodes=updated), lost


def _changes(before: PlacementPlan, after: PlacementPlan) -> dict[str, Any]:
    old = {assignment.layer_id: assignment for assignment in before.assignments}
    new = {assignment.layer_id: assignment for assignment in after.assignments}
    changed = [
        layer
        for layer in sorted(set(old) | set(new))
        if layer not in old
        or layer not in new
        or old[layer].candidate_id != new[layer].candidate_id
        or old[layer].node_ids != new[layer].node_ids
    ]
    migration = 0
    granularity = 0
    for layer in changed:
        if layer in new:
            migration += sum(new[layer].checkpoint_bytes_by_node.values())
        if layer in old and layer in new and old[layer].partition_kind != new[layer].partition_kind:
            granularity += 1
    for node, bytes_ in after.endpoint_checkpoint_bytes_by_node.items():
        if node not in before.endpoint_checkpoint_bytes_by_node:
            migration += int(bytes_)
    return {
        "placement_changes": len(changed),
        "changed_layers": changed,
        "migration_bytes": migration,
        "granularity_changes": granularity,
        "nodes_added": len(set(after.used_nodes) - set(before.used_nodes)),
        "nodes_removed": len(set(before.used_nodes) - set(after.used_nodes)),
        "endpoint_nodes_added": len(
            set(after.endpoint_memory_by_node) - set(before.endpoint_memory_by_node)
        ),
        "endpoint_nodes_removed": len(
            set(before.endpoint_memory_by_node) - set(after.endpoint_memory_by_node)
        ),
        "wavefront_chunk_changed": before.chunk_rows != after.chunk_rows,
    }


def _fixed_plan_on_changed_inventory(
    inventory: Inventory,
    initial: PlacementPlan,
    optimizer: SharedPlacementOptimizer,
) -> PlacementPlan | None:
    """Re-evaluate the old placement under changed concrete capabilities."""

    nodes = inventory.node_map()
    if any(node not in nodes or not nodes[node].available for node in initial.used_nodes):
        return None
    if any(
        int(memory) > nodes[node].accelerator_memory_bytes
        for node, memory in initial.memory_used_by_node.items()
    ):
        return None
    endpoint = EndpointPolicy(
        node_ids=tuple(initial.endpoint_memory_by_node),
        memory_by_node=dict(initial.endpoint_memory_by_node),
        checkpoint_bytes_by_node=dict(initial.endpoint_checkpoint_bytes_by_node),
        embedding_ms=0.08,
        final_norm_ms=0.04,
        lm_head_ms=0.55,
    )
    value = copy.deepcopy(initial)
    value.inventory_id = inventory.inventory_id
    PlacementEvaluator(
        optimizer.model,
        inventory,
        optimizer.service,
        endpoint,
    ).evaluate(value)
    return value


def run_dynamic_scenarios(
    inventory: Inventory,
    initial: PlacementPlan,
    optimizer: SharedPlacementOptimizer,
) -> dict[str, dict[str, Any]]:
    scenarios: dict[
        str,
        tuple[Inventory, str, PlacementPlan | None, Callable[[PlacementPlan], bool]],
    ] = {}
    scenarios["JOIN_USEFUL"] = (
        _join(inventory, useful=True),
        "dynamic-useful",
        initial,
        lambda plan: plan.feasible
        and (plan.exact_tok_s_per_user or 0) >= (initial.exact_tok_s_per_user or 0) * 0.99,
    )
    scenarios["JOIN_HARMFUL"] = (
        _join(inventory, useful=False),
        "dynamic-harmful",
        initial,
        lambda plan: plan.feasible
        and (plan.exact_tok_s_per_user or 0) >= (initial.exact_tok_s_per_user or 0) * 0.99,
    )
    slow_inventory, slow_node = _slowdown(inventory, initial)
    scenarios["SLOWDOWN"] = (slow_inventory, slow_node, initial, lambda plan: plan.feasible)
    network_inventory, link = _degrade_network(inventory, initial)
    scenarios["NETWORK_DEGRADATION"] = (
        network_inventory,
        link,
        initial,
        lambda plan: plan.feasible,
    )
    loss_inventory, lost = _loss(inventory, initial)
    scenarios["NODE_LOSS"] = (loss_inventory, lost, None, lambda plan: plan.feasible)

    output: dict[str, dict[str, Any]] = {}
    for name, (changed_inventory, subject, fallback, gate) in scenarios.items():
        started = time.perf_counter_ns()
        retained = (
            _fixed_plan_on_changed_inventory(changed_inventory, initial, optimizer)
            if fallback is not None
            else None
        )
        result = optimizer.optimize(changed_inventory, PlannerLevel.E)
        replanning_ms = (time.perf_counter_ns() - started) / 1e6
        plan = result.plan
        fallback_retained = False
        if retained is not None and (
            not plan.feasible
            or (retained.objective_tuple or ()) > (plan.objective_tuple or ())
        ):
            plan = retained
            fallback_retained = True
        changes = _changes(initial, plan) if plan.feasible else {
            "placement_changes": 0,
            "changed_layers": [],
            "migration_bytes": 0,
            "granularity_changes": 0,
            "nodes_added": 0,
            "nodes_removed": 0,
            "endpoint_nodes_added": 0,
            "endpoint_nodes_removed": 0,
            "wavefront_chunk_changed": False,
        }
        no_replan_tps = retained.exact_tok_s_per_user if retained is not None else None
        improvement_vs_no_replan = (
            (plan.exact_tok_s_per_user or 0) / (no_replan_tps or 1) - 1
            if plan.feasible and no_replan_tps is not None
            else None
        )
        change_count = (
            int(changes["placement_changes"])
            + int(changes["endpoint_nodes_added"])
            + int(changes["endpoint_nodes_removed"])
            + int(bool(changes["wavefront_chunk_changed"]))
        )
        threshold_replan_pass = (
            improvement_vs_no_replan is None
            or improvement_vs_no_replan <= 0.01
            or change_count > 0
        )
        used_subject = subject in plan.used_nodes if "<->" not in subject else None
        scenario_pass = gate(plan) and threshold_replan_pass
        if name == "JOIN_USEFUL":
            scenario_pass = (
                scenario_pass
                and "dynamic-useful" in plan.used_nodes
                and improvement_vs_no_replan is not None
                and improvement_vs_no_replan > 0
            )
        if name == "JOIN_HARMFUL":
            scenario_pass = scenario_pass and "dynamic-harmful" not in plan.used_nodes
        output[name] = {
            "inventory_id": inventory.inventory_id,
            "scenario": name,
            "changed_capability": subject,
            "status": "PASS" if scenario_pass else "FAIL",
            "feasible": plan.feasible,
            "before_tok_s": initial.exact_tok_s_per_user,
            "after_tok_s": plan.exact_tok_s_per_user,
            "throughput_change": (
                (plan.exact_tok_s_per_user or 0) / (initial.exact_tok_s_per_user or 1) - 1
                if plan.feasible
                else None
            ),
            "before_critical_path_ms": initial.critical_path_ms,
            "after_critical_path_ms": plan.critical_path_ms,
            "replanning_ms": replanning_ms,
            "optimizer_reported_ms": result.elapsed_ms,
            "old_placement_changed_conditions_tok_s": no_replan_tps,
            "improvement_vs_no_replan": improvement_vs_no_replan,
            "replan_threshold": 0.01,
            "improvement_over_threshold_triggered_new_plan": threshold_replan_pass,
            "fallback_retained": fallback_retained,
            "changed_resource_used_after": used_subject,
            "harmful_join_ignored": (
                name == "JOIN_HARMFUL" and "dynamic-harmful" not in plan.used_nodes
            ),
            "beneficial_join_non_regression": (
                name == "JOIN_USEFUL"
                and plan.feasible
                and (plan.exact_tok_s_per_user or 0)
                >= (initial.exact_tok_s_per_user or 0) * 0.99
            ),
            "beneficial_join_admitted_and_improved": (
                name == "JOIN_USEFUL"
                and "dynamic-useful" in plan.used_nodes
                and improvement_vs_no_replan is not None
                and improvement_vs_no_replan > 0
            ),
            "manual_topology_supplied": False,
            **changes,
        }
    return output


__all__ = ["run_dynamic_scenarios"]
