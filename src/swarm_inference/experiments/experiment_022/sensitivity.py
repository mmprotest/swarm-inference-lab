"""Network, memory, and compute heterogeneity sensitivity sweeps for E022."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .benchmark import plan_row
from .inventories import LINK_CLASSES
from .models import Inventory, NetworkPeer, NodeCapability, PlannerLevel
from .planner import SharedPlacementOptimizer


def _network(inventory: Inventory, class_name: str) -> Inventory:
    values = LINK_CLASSES[class_name]
    nodes = []
    for node in inventory.nodes:
        peers = {
            peer_id: NetworkPeer(
                peer_id=peer_id,
                latency_ms=values["latency_ms"],
                bandwidth_gbps=values["bandwidth_gbps"],
                software_overhead_ms=values["software_overhead_ms"],
                locality_class=class_name,
            )
            for peer_id in node.network_peers
        }
        nodes.append(replace(node, network_peers=peers))
    return replace(
        inventory,
        inventory_id=f"{inventory.inventory_id}:network-{class_name}",
        nodes=tuple(nodes),
        scenario=f"all links shaped as {class_name}",
    )


def _memory(inventory: Inventory, factor: float) -> Inventory:
    nodes = tuple(
        replace(
            node,
            accelerator_memory_bytes=max(
                1024**3, int(node.accelerator_memory_bytes * factor)
            ),
        )
        for node in inventory.nodes
    )
    return replace(
        inventory,
        inventory_id=f"{inventory.inventory_id}:memory-{factor:.2f}",
        nodes=nodes,
        scenario=f"accelerator memory x{factor:.2f}",
    )


def _compute(inventory: Inventory, pattern: tuple[float, ...], name: str) -> Inventory:
    nodes: list[NodeCapability] = []
    for index, node in enumerate(inventory.nodes):
        multiplier = pattern[index % len(pattern)]
        nodes.append(
            replace(
                node,
                compute_profile={"reference": multiplier},
                memory_bandwidth_profile={"reference": multiplier},
            )
        )
    return replace(
        inventory,
        inventory_id=f"{inventory.inventory_id}:compute-{name}",
        nodes=tuple(nodes),
        scenario=f"controlled compute pattern {name}",
    )


def run_sensitivities(
    inventory: Inventory,
    optimizer: SharedPlacementOptimizer,
) -> dict[str, list[dict[str, Any]]]:
    network_rows = []
    for class_name in ("fast", "medium", "regional", "slow"):
        candidate = _network(inventory, class_name)
        result = optimizer.optimize(candidate, PlannerLevel.E)
        row = plan_row(candidate, result)
        row["network_class"] = class_name
        row["latency_ms"] = LINK_CLASSES[class_name]["latency_ms"]
        row["bandwidth_gbps"] = LINK_CLASSES[class_name]["bandwidth_gbps"]
        network_rows.append(row)

    memory_rows = []
    for factor in (0.55, 0.70, 0.85, 1.0, 1.15):
        candidate = _memory(inventory, factor)
        whole = optimizer.optimize(candidate, PlannerLevel.A)
        adaptive = optimizer.optimize(candidate, PlannerLevel.E, fallback=whole.plan)
        for label, result in (("whole", whole), ("adaptive", adaptive)):
            row = plan_row(candidate, result)
            row["memory_factor"] = factor
            row["planner"] = label
            memory_rows.append(row)

    compute_rows = []
    patterns = {
        "uniform_reference": (1.0,),
        "mild": (1.0, 0.8),
        "strong": (1.0, 0.8, 0.6, 0.4),
        "mostly_slow": (0.8, 0.6, 0.4, 0.4),
    }
    for name, pattern in patterns.items():
        candidate = _compute(inventory, pattern, name)
        result = optimizer.optimize(candidate, PlannerLevel.E)
        row = plan_row(candidate, result)
        row["compute_scenario"] = name
        row["compute_pattern"] = ",".join(str(value) for value in pattern)
        compute_rows.append(row)
    return {
        "network-sensitivity": network_rows,
        "memory-sensitivity": memory_rows,
        "compute-sensitivity": compute_rows,
    }


__all__ = ["run_sensitivities"]
