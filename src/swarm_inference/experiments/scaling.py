"""Scaling metrics with explicit homogeneous and capacity-normalised definitions."""

from __future__ import annotations

from collections.abc import Iterable


def throughput_gain(throughput: float, baseline_throughput: float) -> float:
    if baseline_throughput <= 0:
        return 0.0
    return throughput / baseline_throughput


def marginal_throughput(before: float, after: float) -> float:
    return after - before


def homogeneous_scaling_efficiency(
    *,
    throughput: float,
    baseline_throughput: float,
    node_count: int,
    baseline_node_count: int,
) -> float:
    if baseline_throughput <= 0 or baseline_node_count <= 0 or node_count <= 0:
        return 0.0
    node_gain = node_count / baseline_node_count
    return throughput_gain(throughput, baseline_throughput) / node_gain


def capacity_normalised_efficiency(
    observed_aggregate_throughput: float,
    predicted_ideal_throughput: float,
) -> float:
    if predicted_ideal_throughput <= 0:
        return 0.0
    return observed_aggregate_throughput / predicted_ideal_throughput


def predicted_ideal_throughput(stage_service_capacities: Iterable[float]) -> float:
    capacities = list(stage_service_capacities)
    if not capacities or any(capacity < 0 for capacity in capacities):
        return 0.0
    return min(capacities)
