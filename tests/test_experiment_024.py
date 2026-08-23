"""Focused scientific invariants for Experiment 024."""

from __future__ import annotations

import math

import pytest

from swarm_inference.experiments.experiment_024.commodity_pool import (
    compute_multiplier,
    group_id,
)
from swarm_inference.experiments.experiment_024.communication_bound import (
    D_LOWER_BOUND_RATIO,
    LOWER_BOUND_BYTES,
    assert_lower_bound,
)
from swarm_inference.experiments.experiment_024.composer import validate_composer
from swarm_inference.experiments.experiment_024.decode_serving_engine import (
    DecodeServingEngine,
    microbatch_sizes,
    token_latency_quantiles,
)
from swarm_inference.experiments.experiment_024.decode_step import (
    DecodeStepTaskBuilder,
)
from swarm_inference.experiments.experiment_024.economics import (
    commercial_metrics,
    cost_per_million,
    mechanical_verdict,
    scenario_wedge_pass,
    whole_layer_incapable_compute_shares,
)
from swarm_inference.experiments.experiment_024.geometry import (
    A_BYTES,
    B_BYTES,
    C_BYTES,
    D_BYTES,
    GEOMETRY,
    H_BYTES,
    assert_frozen_geometry,
)
from swarm_inference.experiments.experiment_024.models import (
    CommodityScenario,
    StageAArm,
    Verdict,
)
from swarm_inference.experiments.experiment_024.stage_a import run_stage_a_cell


def test_frozen_communication_geometry() -> None:
    assert_frozen_geometry()
    assert (H_BYTES, A_BYTES, B_BYTES, C_BYTES, D_BYTES) == (
        28_672,
        1_004_416,
        803_712,
        715_904,
        515_200,
    )


def test_fixed_placement_lower_bound() -> None:
    assert_lower_bound()
    assert LOWER_BOUND_BYTES == 502_712
    assert pytest.approx(1.0248412610003341) == D_LOWER_BOUND_RATIO


def test_latent_down_receives_complete_hidden_after_routes() -> None:
    validate_composer()


@pytest.mark.parametrize(
    ("tokens_per_second", "hourly_cost", "expected"),
    (
        (1.0, 1.0, 277.7777777778),
        (100.0, 4.0, 11.1111111111),
        (200.0, 4.0, 5.5555555556),
    ),
)
def test_economic_sanity(
    tokens_per_second: float, hourly_cost: float, expected: float
) -> None:
    assert cost_per_million(hourly_cost, tokens_per_second) == pytest.approx(expected)


def test_zero_throughput_costs_infinity() -> None:
    assert math.isinf(cost_per_million(1.0, 0.0))


def test_cheap_but_slow_is_not_rejected_by_performance_threshold() -> None:
    assert scenario_wedge_pass(
        cost_per_m=0.05 * 15,
        slo_feasible=True,
        layer_zero_uses_exact_whole_candidate=True,
        p8_layer_count=92,
        no_whole_layer_execution_on_layers_1_92=True,
        whole_layer_only_commodity_model_feasible=False,
        p8_required_whole_layer_incapable_compute_share=0.95,
        global_correctness_pass=True,
    )


def test_fast_but_expensive_does_not_pass() -> None:
    assert not scenario_wedge_pass(
        cost_per_m=1.10 * 15,
        slo_feasible=True,
        layer_zero_uses_exact_whole_candidate=True,
        p8_layer_count=92,
        no_whole_layer_execution_on_layers_1_92=True,
        whole_layer_only_commodity_model_feasible=False,
        p8_required_whole_layer_incapable_compute_share=1.0,
        global_correctness_pass=True,
    )


def test_api_parity_is_not_strictly_cheaper() -> None:
    assert not scenario_wedge_pass(
        cost_per_m=15.0,
        slo_feasible=True,
        layer_zero_uses_exact_whole_candidate=True,
        p8_layer_count=92,
        no_whole_layer_execution_on_layers_1_92=True,
        whole_layer_only_commodity_model_feasible=False,
        p8_required_whole_layer_incapable_compute_share=1.0,
        global_correctness_pass=True,
    )
    assert scenario_wedge_pass(
        cost_per_m=14.99,
        slo_feasible=True,
        layer_zero_uses_exact_whole_candidate=True,
        p8_layer_count=92,
        no_whole_layer_execution_on_layers_1_92=True,
        whole_layer_only_commodity_model_feasible=False,
        p8_required_whole_layer_incapable_compute_share=1.0,
        global_correctness_pass=True,
    )


def test_example_frontier_leverage_is_not_a_threshold() -> None:
    metrics = commercial_metrics(
        swarm_slo_output_tokens_per_second=70,
        concentrated_fast_slo_output_tokens_per_second=100,
        swarm_hourly_cost_usd=0.2 * 15 * 70 * 3600 / 1_000_000,
    )
    assert metrics.performance_retention == pytest.approx(0.70)
    assert metrics.api_cost_ratio == pytest.approx(0.20)
    assert metrics.performance_cost_leverage == pytest.approx(3.5)


def test_model_invalid_has_first_verdict_precedence() -> None:
    assert (
        mechanical_verdict(
            mandatory_validity_failure=True,
            scenario_wedge_count=3,
            mechanism_only_pass=True,
        )
        is Verdict.MODEL_INVALID
    )


def test_mechanism_only_remains_valid_without_a_commercial_wedge() -> None:
    assert (
        mechanical_verdict(
            mandatory_validity_failure=False,
            scenario_wedge_count=0,
            mechanism_only_pass=True,
        )
        is Verdict.MECHANISM_ONLY
    )


def test_decode_unit_and_closed_loop_measurement() -> None:
    builder = DecodeStepTaskBuilder()
    assert builder.build(1).generated_output_tokens == 1
    assert builder.build(2).generated_output_tokens == 2
    assert builder.build(4).generated_output_tokens == 4
    assert microbatch_sizes(1) == (1,)
    assert microbatch_sizes(4) == (4,)
    assert microbatch_sizes(16) == (4, 4, 4, 4)
    assert microbatch_sizes(64) == (4,) * 16
    assert microbatch_sizes(128) == (4,) * 32
    for concurrency in (1, 4, 16, 64, 128):
        plan = DecodeServingEngine().measurement_plan(concurrency)
        assert plan.minimum_measured_output_tokens >= 256


def test_token_latency_repeats_per_batch_row() -> None:
    p50, p95 = token_latency_quantiles(((10.0, 1), (20.0, 4)))
    expected = [10.0, 20.0, 20.0, 20.0, 20.0]
    assert p50 == pytest.approx(20.0)
    assert p95 == pytest.approx(19.999999999999996)
    assert len(expected) == 5


def test_commodity_heterogeneity_and_locality_are_deterministic() -> None:
    assert tuple(compute_multiplier(index) for index in range(8)) == (
        1.0,
        0.8,
        0.6,
        0.4,
        1.0,
        0.8,
        0.6,
        0.4,
    )
    assert tuple(group_id(index) for index in range(10)) == (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        0,
        1,
    )


def test_p8_incapable_share_excludes_dense_layer_zero_from_denominator() -> None:
    overall, p8_required = whole_layer_incapable_compute_shares(
        (
            {
                "layer": 0,
                "compute_ms": 10.0,
                "worker_memory_bytes": 10,
                "whole_layer_resident_bytes": 5,
            },
            {
                "layer": 1,
                "compute_ms": 90.0,
                "worker_memory_bytes": 10,
                "whole_layer_resident_bytes": 20,
            },
        )
    )
    assert overall == pytest.approx(0.9)
    assert p8_required == pytest.approx(1.0)


def test_stage_a_task_graph_preserves_exact_frozen_moe_bytes() -> None:
    class StubService:
        def __init__(self) -> None:
            self.layer_type_by_id = {layer: "KDA" for layer in range(93)}

        @staticmethod
        def service_ms(layer: int, operation: str, degree: int, rows: int) -> float:
            del layer, operation, degree, rows
            return 0.01

    for arm in StageAArm:
        row = run_stage_a_cell(
            service=StubService(),  # type: ignore[arg-type]
            scenario=CommodityScenario.COMMODITY_REGIONAL,
            layer=89,
            rows=1,
            concurrency=1,
            arm=arm,
        )
        assert row["moe_network_bytes_per_row"] == GEOMETRY[arm].bytes_per_row
