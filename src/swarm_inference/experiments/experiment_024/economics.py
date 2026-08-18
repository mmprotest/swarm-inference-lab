"""Authoritative Experiment 024 output-token economics."""

from __future__ import annotations

import math

from .freeze import KIMI_OUTPUT_API_PRICE_USD_PER_M
from .models import CommercialMetrics, Verdict


def cost_per_million(hourly_cost_usd: float, output_tokens_per_second: float) -> float:
    if hourly_cost_usd < 0:
        raise ValueError("hourly cost cannot be negative")
    if output_tokens_per_second < 0:
        raise ValueError("throughput cannot be negative")
    if output_tokens_per_second == 0:
        return math.inf
    return hourly_cost_usd * 1_000_000 / (output_tokens_per_second * 3600)


def commercial_metrics(
    *,
    swarm_slo_output_tokens_per_second: float,
    concentrated_fast_slo_output_tokens_per_second: float,
    swarm_hourly_cost_usd: float,
) -> CommercialMetrics:
    if concentrated_fast_slo_output_tokens_per_second <= 0:
        raise ValueError("reference SLO throughput must be positive")
    performance = (
        swarm_slo_output_tokens_per_second
        / concentrated_fast_slo_output_tokens_per_second
    )
    cost = cost_per_million(
        swarm_hourly_cost_usd, swarm_slo_output_tokens_per_second
    )
    ratio = cost / KIMI_OUTPUT_API_PRICE_USD_PER_M
    leverage = performance / ratio if ratio > 0 else math.inf
    return CommercialMetrics(
        performance_retention=performance,
        swarm_cost_per_m=cost,
        api_cost_ratio=ratio,
        performance_cost_leverage=leverage,
        api_discount_percent=100 * (1 - ratio),
        gross_profit_per_m_if_sold_at_15=KIMI_OUTPUT_API_PRICE_USD_PER_M - cost,
    )


def max_total_spend_per_hour(
    target_price_usd_per_m: float, output_tokens_per_second: float
) -> float:
    if target_price_usd_per_m < 0 or output_tokens_per_second < 0:
        raise ValueError("target price and throughput must be non-negative")
    return target_price_usd_per_m * output_tokens_per_second * 3600 / 1_000_000


def max_uniform_payout_per_active_node_hour(
    target_price_usd_per_m: float,
    output_tokens_per_second: float,
    active_node_count: int,
) -> float:
    if active_node_count <= 0:
        raise ValueError("active node count must be positive")
    return max_total_spend_per_hour(
        target_price_usd_per_m, output_tokens_per_second
    ) / active_node_count


def max_compute_weighted_payout_per_hour(
    target_price_usd_per_m: float,
    output_tokens_per_second: float,
    active_compute_equivalents: float,
) -> float:
    if active_compute_equivalents <= 0:
        raise ValueError("active compute equivalents must be positive")
    return max_total_spend_per_hour(
        target_price_usd_per_m, output_tokens_per_second
    ) / active_compute_equivalents


def scenario_wedge_pass(
    *,
    cost_per_m: float,
    slo_feasible: bool,
    whole_layer_incapable_compute_share: float,
    global_correctness_pass: bool,
) -> bool:
    return (
        slo_feasible
        and cost_per_m < KIMI_OUTPUT_API_PRICE_USD_PER_M
        and whole_layer_incapable_compute_share >= 0.95
        and global_correctness_pass
    )


def mechanical_verdict(
    *,
    mandatory_validity_failure: bool,
    scenario_wedge_count: int,
    mechanism_only_pass: bool,
) -> Verdict:
    if mandatory_validity_failure:
        return Verdict.MODEL_INVALID
    if scenario_wedge_count >= 2:
        return Verdict.YES_SWARM_WEDGE
    if scenario_wedge_count == 1:
        return Verdict.CONDITIONAL_SWARM_WEDGE
    if mechanism_only_pass:
        return Verdict.MECHANISM_ONLY
    return Verdict.NO_WEDGE


__all__ = [
    "commercial_metrics",
    "cost_per_million",
    "max_compute_weighted_payout_per_hour",
    "max_total_spend_per_hour",
    "max_uniform_payout_per_active_node_hour",
    "mechanical_verdict",
    "scenario_wedge_pass",
]
