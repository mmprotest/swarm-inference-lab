"""Frozen commodity-pool definitions.

Pool construction is intentionally unreachable after a failed Phase 0 gate.
The constants remain importable so the invalid result can prove that changing
network size or memory cannot repair an empty layer-0 candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import CommodityScenario

COMPUTE_MULTIPLIERS = (1.0, 0.8, 0.6, 0.4)


@dataclass(frozen=True, slots=True)
class LinkDefinition:
    latency_ms: float
    bandwidth_gbps: float
    software_overhead_ms: float


GOOD_LINK = LinkDefinition(1.0, 10.0, 0.05)
REGIONAL_WITHIN = LinkDefinition(1.0, 10.0, 0.05)
REGIONAL_ACROSS = LinkDefinition(5.0, 1.0, 0.08)
WAN_WITHIN = LinkDefinition(5.0, 1.0, 0.08)
WAN_ACROSS = LinkDefinition(20.0, 0.1, 0.12)


def compute_multiplier(node_index: int) -> float:
    if node_index < 0:
        raise ValueError("node index must be non-negative")
    return COMPUTE_MULTIPLIERS[node_index % 4]


def group_id(node_index: int) -> int:
    if node_index < 0:
        raise ValueError("node index must be non-negative")
    return node_index % 8


def link_definition(
    scenario: CommodityScenario, source_index: int, destination_index: int
) -> LinkDefinition:
    if source_index == destination_index:
        return LinkDefinition(0.0, 1_000_000.0, 0.0)
    if scenario is CommodityScenario.COMMODITY_GOOD:
        return GOOD_LINK
    same_group = group_id(source_index) == group_id(destination_index)
    if scenario is CommodityScenario.COMMODITY_REGIONAL:
        return REGIONAL_WITHIN if same_group else REGIONAL_ACROSS
    if scenario is CommodityScenario.COMMODITY_WAN:
        return WAN_WITHIN if same_group else WAN_ACROSS
    raise ValueError(f"unknown commodity scenario: {scenario}")


__all__ = [
    "COMPUTE_MULTIPLIERS",
    "GOOD_LINK",
    "REGIONAL_ACROSS",
    "REGIONAL_WITHIN",
    "WAN_ACROSS",
    "WAN_WITHIN",
    "LinkDefinition",
    "compute_multiplier",
    "group_id",
    "link_definition",
]
