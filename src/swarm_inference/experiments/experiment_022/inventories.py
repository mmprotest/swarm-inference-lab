"""Preregistered controlled heterogeneous resource inventories for E022."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from . import EVIDENCE_CLASS
from .io import atomic_write_json, canonical_sha256
from .models import Inventory, ModelGraph, NetworkPeer, NodeCapability

GENERATOR_VERSION = "experiment-022-inventory-generator-v1"
GIB = 1024**3

LINK_CLASSES: dict[str, dict[str, float]] = {
    "fast": {"latency_ms": 0.25, "bandwidth_gbps": 25.0, "software_overhead_ms": 0.04},
    "medium": {"latency_ms": 1.0, "bandwidth_gbps": 10.0, "software_overhead_ms": 0.05},
    "regional": {"latency_ms": 5.0, "bandwidth_gbps": 1.0, "software_overhead_ms": 0.08},
    "slow": {"latency_ms": 20.0, "bandwidth_gbps": 0.1, "software_overhead_ms": 0.12},
}

FAMILY_COUNTS = {
    "coarse-friendly": 3,
    "memory-fragmented": 6,
    "compute-heterogeneous": 6,
    "network-heterogeneous": 6,
    "full-mixed": 6,
}

SEEDS = {
    family: [22_000 + offset * 100 + index for index in range(count)]
    for offset, (family, count) in enumerate(FAMILY_COUNTS.items())
}


@dataclass(frozen=True, slots=True)
class _Draft:
    node_id: str
    memory_bytes: int
    compute_multiplier: float
    locality_group: str
    cost: float
    reliability: float
    cached_shards: tuple[str, ...]
    harmful: bool = False


def _round_mib(value: float) -> int:
    mib = 1024**2
    return math.ceil(value / mib) * mib


def memory_classes(model: ModelGraph) -> dict[str, int]:
    maximum = model.maximum_layer_resident_bytes
    return {
        "sub_layer": _round_mib(maximum * 0.57),
        "near_layer": _round_mib(maximum * 0.94),
        "one_layer_plus": _round_mib(maximum * 1.35),
        "multi_layer": _round_mib(maximum * 2.55),
        "large_multi_layer": _round_mib(maximum * 3.35),
    }


def generator_config(model: ModelGraph) -> dict[str, Any]:
    return {
        "schema_version": GENERATOR_VERSION,
        "created_before_planner_evaluation": True,
        "hypothesis_frozen": True,
        "families": FAMILY_COUNTS,
        "seeds": SEEDS,
        "memory_classes_bytes": memory_classes(model),
        "memory_class_basis": {
            "maximum_measured_or_derived_layer_resident_bytes": model.maximum_layer_resident_bytes,
            "sub_layer": "0.57 x maximum layer resident memory",
            "near_layer": "0.94 x maximum layer resident memory",
            "one_layer_plus": "1.35 x maximum layer resident memory",
            "multi_layer": "2.55 x maximum layer resident memory",
            "large_multi_layer": "3.35 x maximum layer resident memory",
        },
        "compute_multipliers": [1.0, 0.8, 0.6, 0.4],
        "compute_multiplier_rule": "slower-or-equal to local physical reference only",
        "link_classes": LINK_CLASSES,
        "evidence_class": EVIDENCE_CLASS,
        "whole_layer_feasibility_uses_actual_per_layer_resident_bytes": True,
        "endpoint_policy_identical": True,
        "external_resource_queries": 0,
        "gpu_rentals": 0,
    }


def _cached(rng: random.Random) -> tuple[str, ...]:
    if rng.random() >= 0.18:
        return ()
    start = rng.randrange(0, 93)
    return tuple(f"layer-{value:02d}" for value in range(start, min(93, start + rng.randint(1, 4))))


def _drafts_for_family(
    family: str,
    index: int,
    seed: int,
    classes: dict[str, int],
) -> tuple[list[_Draft], str, str]:
    rng = random.Random(seed)
    drafts: list[_Draft] = []

    def add(
        count: int,
        memory_class: str,
        multipliers: tuple[float, ...],
        groups: int,
        *,
        prefix: str,
        cost_base: float = 1.0,
        harmful: bool = False,
    ) -> None:
        for _ in range(count):
            number = len(drafts)
            multiplier = rng.choice(multipliers)
            drafts.append(
                _Draft(
                    node_id=f"{prefix}-{number:03d}",
                    memory_bytes=classes[memory_class],
                    compute_multiplier=multiplier,
                    locality_group=f"g{rng.randrange(groups):02d}",
                    cost=round(cost_base * (0.8 + 0.4 * rng.random()), 4),
                    reliability=round(0.97 + 0.029 * rng.random(), 6),
                    cached_shards=_cached(rng),
                    harmful=harmful,
                )
            )

    if family == "coarse-friendly":
        add(48, "multi_layer", (1.0, 0.8), 1, prefix="coarse")
        scenario = "whole-layer-capable uniform memory; unnecessary collectives face medium links"
        link_policy = "coarse"
    elif family == "memory-fragmented":
        if index % 2 == 0:
            add(48, "multi_layer", (1.0, 0.8, 0.6), 3, prefix="frag-whole")
            add(36 + index * 2, "sub_layer", (1.0, 0.8, 0.6), 3, prefix="frag-small", cost_base=0.55)
            scenario = "A: whole-layer feasible with substantial sub-layer-sized stranded memory"
        else:
            add(24, "multi_layer", (0.8, 0.6), 3, prefix="frag-whole")
            add(76 + index * 2, "sub_layer", (1.0, 0.8, 0.6), 3, prefix="frag-small", cost_base=0.55)
            scenario = "B: whole-layer infeasible although total distributed memory is sufficient"
        link_policy = "locality"
    elif family == "compute-heterogeneous":
        add(16, "multi_layer", (0.4,), 3, prefix="compute-slow", cost_base=0.7)
        add(32, "multi_layer", (0.8, 0.6), 3, prefix="compute-main")
        add(28 + index, "sub_layer", (1.0, 0.8), 3, prefix="compute-spare", cost_base=0.6)
        scenario = "whole-layer feasible; slow high-memory nodes coexist with faster sub-layer-capable nodes"
        link_policy = "locality"
    elif family == "network-heterogeneous":
        add(48, "multi_layer", (1.0, 0.8), 4, prefix="network-main")
        add(20 + index, "sub_layer", (1.0, 0.8, 0.6), 4, prefix="network-small", cost_base=0.6)
        scenario = "four locality groups with fast, medium, regional, and slow peer classes"
        link_policy = "network"
    elif family == "full-mixed":
        if index < 3:
            add(34, "multi_layer", (0.8, 0.6, 0.4), 4, prefix="mixed-multi")
            add(27, "one_layer_plus", (1.0, 0.8, 0.6), 4, prefix="mixed-one")
            add(32, "sub_layer", (1.0, 0.8, 0.6, 0.4), 4, prefix="mixed-small", cost_base=0.55)
            feasibility = "whole-layer feasible"
        else:
            add(24, "multi_layer", (0.8, 0.6, 0.4), 4, prefix="mixed-multi")
            add(31, "one_layer_plus", (1.0, 0.8, 0.6), 4, prefix="mixed-one")
            add(62, "sub_layer", (1.0, 0.8, 0.6, 0.4), 4, prefix="mixed-small", cost_base=0.55)
            feasibility = "whole-layer intentionally capacity-challenged"
        add(5, "near_layer", (0.4,), 1, prefix="mixed-harmful", cost_base=1.4, harmful=True)
        scenario = f"{feasibility}; mixed memory/compute/network plus explicitly ignorable weak nodes"
        link_policy = "network"
    else:
        raise ValueError(f"unknown inventory family {family}")
    return drafts, scenario, link_policy


def _link_class(left: _Draft, right: _Draft, policy: str) -> str:
    if left.harmful or right.harmful:
        return "slow"
    if policy == "coarse":
        return "medium"
    if left.locality_group == right.locality_group:
        return "fast"
    if policy == "locality":
        return "medium"
    left_index = int(left.locality_group[1:])
    right_index = int(right.locality_group[1:])
    distance = abs(left_index - right_index)
    if distance <= 1:
        return "medium"
    if distance == 2:
        return "regional"
    return "slow"


def _materialize_nodes(drafts: list[_Draft], link_policy: str) -> tuple[NodeCapability, ...]:
    nodes: list[NodeCapability] = []
    for draft in drafts:
        peers: dict[str, NetworkPeer] = {}
        for other in drafts:
            if other.node_id == draft.node_id:
                continue
            class_name = _link_class(draft, other, link_policy)
            link = LINK_CLASSES[class_name]
            peers[other.node_id] = NetworkPeer(
                peer_id=other.node_id,
                latency_ms=link["latency_ms"],
                bandwidth_gbps=link["bandwidth_gbps"],
                software_overhead_ms=link["software_overhead_ms"],
                locality_class=class_name,
            )
        nodes.append(
            NodeCapability(
                node_id=draft.node_id,
                accelerator_memory_bytes=draft.memory_bytes,
                system_memory_bytes=max(2 * draft.memory_bytes, 16 * GIB),
                compute_profile={"reference": draft.compute_multiplier},
                memory_bandwidth_profile={"reference": draft.compute_multiplier},
                supported_precisions=("mxfp4", "int8", "fp32"),
                network_peers=peers,
                reliability=draft.reliability,
                cost=draft.cost,
                cached_shards=draft.cached_shards,
                runtime_capabilities=(
                    "WHOLE_LAYER",
                    "WHOLE_EXPERT",
                    "EXPERT_SHARD",
                    "KDA_SHARD",
                    "MLA_SHARD",
                    "PROJECTION_SHARD",
                    "REDUCTION",
                ),
                locality_group=draft.locality_group,
            )
        )
    return tuple(nodes)


def generate_inventory_suite(model: ModelGraph) -> tuple[dict[str, Any], list[Inventory]]:
    config = generator_config(model)
    classes = memory_classes(model)
    inventories: list[Inventory] = []
    for family, count in FAMILY_COUNTS.items():
        for index in range(count):
            seed = SEEDS[family][index]
            drafts, scenario, policy = _drafts_for_family(
                family, index, seed, classes
            )
            inventory_id = f"{family}-{index + 1:02d}"
            inventories.append(
                Inventory(
                    inventory_id=inventory_id,
                    family=family,
                    seed=seed,
                    nodes=_materialize_nodes(drafts, policy),
                    evidence_class=EVIDENCE_CLASS,
                    generator_version=GENERATOR_VERSION,
                    scenario=scenario,
                )
            )
    if len(inventories) != 27:
        raise RuntimeError("preregistered E022 suite must contain exactly 27 inventories")
    suite_rows = []
    for inventory in inventories:
        value = inventory.as_dict()
        suite_rows.append(
            {
                **value,
                "inventory_sha256": canonical_sha256(value),
            }
        )
    suite = {
        "schema_version": "experiment-022-inventory-suite-v1",
        "frozen_before_planner_comparison": True,
        "generator_config_sha256": canonical_sha256(config),
        "inventory_count": len(suite_rows),
        "family_counts": FAMILY_COUNTS,
        "inventories": suite_rows,
    }
    suite["suite_sha256"] = canonical_sha256(suite)
    return config, inventories


def materialize_inventory_suite(
    artifact_root: Path,
    model: ModelGraph,
) -> tuple[dict[str, Any], list[Inventory]]:
    config, inventories = generate_inventory_suite(model)
    inventory_root = artifact_root / "inventories"
    atomic_write_json(inventory_root / "generator-config.json", config)
    atomic_write_json(
        inventory_root / "seeds.json",
        {
            "schema_version": "experiment-022-inventory-seeds-v1",
            "seeds": SEEDS,
            "seed_count": sum(len(values) for values in SEEDS.values()),
        },
    )
    suite_rows = []
    for inventory in inventories:
        value = inventory.as_dict()
        value["inventory_sha256"] = canonical_sha256(value)
        suite_rows.append(value)
        atomic_write_json(
            inventory_root
            / inventory.family
            / f"{inventory.inventory_id}.json",
            value,
        )
    suite = {
        "schema_version": "experiment-022-inventory-suite-v1",
        "frozen_before_planner_comparison": True,
        "generator_config_sha256": canonical_sha256(config),
        "inventory_count": len(suite_rows),
        "family_counts": FAMILY_COUNTS,
        "inventories": suite_rows,
    }
    suite["suite_sha256"] = canonical_sha256(suite)
    atomic_write_json(inventory_root / "inventory-suite.json", suite)
    return suite, inventories


def replace_nodes(inventory: Inventory, nodes: list[NodeCapability], scenario: str) -> Inventory:
    """Create a dynamic inventory while rebuilding complete peer maps elsewhere."""

    return replace(inventory, nodes=tuple(nodes), scenario=scenario)


__all__ = [
    "FAMILY_COUNTS",
    "GENERATOR_VERSION",
    "LINK_CLASSES",
    "SEEDS",
    "generate_inventory_suite",
    "generator_config",
    "materialize_inventory_suite",
    "memory_classes",
]
