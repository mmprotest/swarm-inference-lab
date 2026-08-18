"""Typed records shared by Experiment 024."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class Verdict(StrEnum):
    MODEL_INVALID = "MODEL_INVALID"
    YES_SWARM_WEDGE = "YES_SWARM_WEDGE"
    CONDITIONAL_SWARM_WEDGE = "CONDITIONAL_SWARM_WEDGE"
    MECHANISM_ONLY = "MECHANISM_ONLY"
    NO_WEDGE = "NO_WEDGE"


class StageAArm(StrEnum):
    A_CURRENT = "A_CURRENT"
    B_RETAIN_HIDDEN = "B_RETAIN_HIDDEN"
    C_SLICE_LATENT = "C_SLICE_LATENT"
    D_FUSE_OUTPUT = "D_FUSE_OUTPUT"


class Architecture(StrEnum):
    CONCENTRATED_FAST = "CONCENTRATED_FAST"
    SWARM_CURRENT_OPT = "SWARM_CURRENT_OPT"
    SWARM_D_OPT = "SWARM_D_OPT"
    CURRENT_ON_D_PLACEMENT = "CURRENT_ON_D_PLACEMENT"


class CommodityScenario(StrEnum):
    COMMODITY_GOOD = "COMMODITY_GOOD"
    COMMODITY_REGIONAL = "COMMODITY_REGIONAL"
    COMMODITY_WAN = "COMMODITY_WAN"


@dataclass(frozen=True, slots=True)
class CommercialMetrics:
    performance_retention: float
    swarm_cost_per_m: float
    api_cost_ratio: float
    performance_cost_leverage: float
    api_discount_percent: float
    gross_profit_per_m_if_sold_at_15: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "Architecture",
    "CommercialMetrics",
    "CommodityScenario",
    "StageAArm",
    "Verdict",
]
