from __future__ import annotations

from swarm_inference.experiments.scaling import (
    capacity_normalised_efficiency,
    homogeneous_scaling_efficiency,
    marginal_throughput,
    predicted_ideal_throughput,
    throughput_gain,
)


def test_scaling_metric_calculations() -> None:
    assert throughput_gain(32, 16) == 2
    assert marginal_throughput(16, 28) == 12
    assert (
        homogeneous_scaling_efficiency(
            throughput=28,
            baseline_throughput=16,
            node_count=8,
            baseline_node_count=4,
        )
        == 0.875
    )
    assert capacity_normalised_efficiency(18, 20) == 0.9
    assert predicted_ideal_throughput([30, 20, 40]) == 20


def test_zero_baselines_are_explicit_zero() -> None:
    assert throughput_gain(1, 0) == 0
    assert capacity_normalised_efficiency(1, 0) == 0
