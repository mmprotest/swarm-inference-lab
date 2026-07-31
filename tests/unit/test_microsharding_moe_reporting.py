from __future__ import annotations

from pathlib import Path

import pytest
import torch

from swarm_inference.experiments.microsharding import (
    _repeat_median_stability,
    classify_microsharding_overall,
)
from swarm_inference.microsharding.moe import (
    ExpertCacheProfile,
    ExpertParallelMoE,
    ReplicatedMoEReference,
    TinyMoEConfig,
    adversarial_routing,
    deterministic_moe_state,
    expert_ownership,
    project_expert_cache,
)
from swarm_inference.microsharding.reporting import REQUIRED_CHARTS, generate_microsharding_charts


@pytest.mark.parametrize("expert_count", [8, 16, 32])
@pytest.mark.parametrize("top_k", [2, 4])
@pytest.mark.parametrize("ep_degree", [1, 2, 4, 8])
@pytest.mark.parametrize("etp_degree", [1, 2, 4])
def test_deterministic_moe_partition_matrix_matches_reference(
    expert_count: int,
    top_k: int,
    ep_degree: int,
    etp_degree: int,
) -> None:
    config = TinyMoEConfig(
        hidden_size=8,
        expert_intermediate_size=16,
        num_experts=expert_count,
        top_k=top_k,
        shared_expert_intermediate_size=8,
    )
    state = deterministic_moe_state(config)
    torch.manual_seed(6006)
    hidden = torch.randn(2, 3, 8)
    expected = ReplicatedMoEReference(config, state)(hidden)
    actual_model = ExpertParallelMoE(
        config,
        state,
        expert_parallel_degree=ep_degree,
        expert_tensor_parallel_degree=etp_degree,
    )
    actual = actual_model(hidden)
    assert torch.equal(actual.selected_experts, expected.selected_experts)
    assert torch.allclose(actual.routing_weights, expected.routing_weights, atol=1e-7, rtol=1e-7)
    assert torch.allclose(actual.output, expected.output, atol=2e-6, rtol=2e-6)
    memory = actual_model.memory_report()
    assert memory["status"] == "PASS"
    if ep_degree > 1:
        assert all(not rank["owns_all_experts"] for rank in memory["ranks"])
    if etp_degree > 1:
        assert all(not rank["owns_complete_expert_matrix"] for rank in memory["ranks"])


@pytest.mark.parametrize(
    "distribution",
    [
        "uniform",
        "highly_skewed",
        "one_hot_expert",
        "alternating_experts",
        "all_selected_experts_on_one_rank",
        "maximum_rank_fanout",
    ],
)
def test_adversarial_expert_dispatch_and_return(distribution: str) -> None:
    config = TinyMoEConfig(
        hidden_size=8,
        expert_intermediate_size=16,
        num_experts=8,
        top_k=2,
        shared_expert_intermediate_size=8,
    )
    state = deterministic_moe_state(config)
    hidden = torch.randn(1, 6, 8)
    ownership = expert_ownership(
        num_experts=8,
        expert_parallel_degree=4,
        strategy="contiguous",
    )
    override = adversarial_routing(
        distribution,
        token_count=6,
        top_k=2,
        num_experts=8,
        ownership_by_rank=ownership,
    )
    expected = ReplicatedMoEReference(config, state)(hidden, routing_override=override)
    model = ExpertParallelMoE(
        config,
        state,
        expert_parallel_degree=4,
        expert_tensor_parallel_degree=2,
    )
    actual = model(hidden, routing_override=override)
    assert torch.equal(actual.selected_experts, override)
    assert torch.allclose(actual.output, expected.output, atol=2e-6, rtol=2e-6)
    assert [item["operation"] for item in actual.metrics["collectives"]] == [
        "all_to_all",
        "all_to_all",
    ]
    assert actual.metrics["dispatch_bytes"] > 0
    assert actual.metrics["return_bytes"] > 0


def test_load_balanced_sparse_trace_assigns_every_rank() -> None:
    ownership = expert_ownership(
        num_experts=32,
        expert_parallel_degree=32,
        strategy="load_balanced_from_trace",
        routing_counts={0: 100, 1: 50},
    )
    assert len(ownership) == 32
    assert all(len(experts) == 1 for experts in ownership.values())
    assert set().union(*(set(experts) for experts in ownership.values())) == set(range(32))


def test_single_rank_adversary_falls_back_when_top_k_exceeds_local_experts() -> None:
    ownership = {rank: [rank] for rank in range(8)}
    routing = adversarial_routing(
        "all_selected_experts_on_one_rank",
        token_count=3,
        top_k=4,
        num_experts=8,
        ownership_by_rank=ownership,
    )
    assert routing.shape == (3, 4)
    assert all(len(set(row)) == 4 for row in routing.tolist())
    assert routing.unique().numel() == 4


def test_expert_ownership_strategies_and_empty_rank_rejection() -> None:
    contiguous = expert_ownership(num_experts=8, expert_parallel_degree=4, strategy="contiguous")
    round_robin = expert_ownership(num_experts=8, expert_parallel_degree=4, strategy="round_robin")
    balanced = expert_ownership(
        num_experts=8,
        expert_parallel_degree=4,
        strategy="load_balanced_from_trace",
        routing_counts={0: 100, 1: 90, 2: 2, 3: 2, 4: 1, 5: 1, 6: 1, 7: 1},
    )
    assert contiguous[0] == [0, 1]
    assert round_robin[0] == [0, 4]
    assert set().union(*map(set, balanced.values())) == set(range(8))
    with pytest.raises(ValueError):
        expert_ownership(num_experts=4, expert_parallel_degree=5, strategy="contiguous")


@pytest.mark.parametrize(
    "policy", ["LRU", "LFU", "routing-prediction-prefetch", "hot-expert-pinning"]
)
def test_on_demand_expert_cache_projection(policy: str) -> None:
    result = project_expert_cache(
        [[0, 1], [0, 2], [0, 1], [3, 2]],
        expert_bytes={index: 100 for index in range(4)},
        profile=ExpertCacheProfile(
            expert_cache_capacity_bytes=200,
            local_storage_bandwidth_mbps=1000,
            peer_transfer_bandwidth_mbps=500,
            peer_transfer_latency_ms=1,
            expert_load_time_ms=0.2,
        ),
        policy=policy,  # type: ignore[arg-type]
    )
    assert 0 <= result["expert_cache_hit_rate"] <= 1
    assert result["bytes_loaded"] > 0
    assert result["minimum_node_memory_bytes"] == 200
    assert result["classification"] == "independent_rank_projection"


def _passing_statuses() -> dict[str, str]:
    return {
        "experiment_integrity_status": "PASS",
        "dense_partition_status": "PASS",
        "dense_tensor_shard_status": "PASS",
        "dense_layer_correctness_status": "PASS",
        "dense_token_identity_status": "PASS",
        "kv_partition_status": "PASS",
        "vocabulary_parallel_status": "PASS",
        "collective_semantics_status": "PASS",
        "collective_projection_status": "PASS",
        "hybrid_pipeline_tensor_status": "PASS",
        "heterogeneous_rank_status": "PASS",
        "deterministic_moe_status": "PASS",
        "expert_projection_status": "PASS",
        "more_partitions_than_layers_status": "PASS",
        "real_moe_layer_status": "PASS",
        "k3_projection_status": "PASS",
    }


def test_reporting_full_pass_and_blocked_real_moe_partial_pass() -> None:
    statuses = _passing_statuses()
    assert classify_microsharding_overall(statuses) == "PASS"
    statuses["real_moe_layer_status"] = "BLOCKED"
    assert classify_microsharding_overall(statuses) == "PARTIAL_PASS"
    statuses["dense_token_identity_status"] = "FAIL"
    assert classify_microsharding_overall(statuses) == "FAIL"
    statuses = _passing_statuses()
    statuses["experiment_integrity_status"] = "FAIL"
    assert classify_microsharding_overall(statuses) == "FAIL"


def test_repeat_median_stability_enforces_configured_cv() -> None:
    stable = [
        {
            "tensor_parallel_degree": 2,
            "workload": "decode",
            "rank": 0,
            "repeat": repeat,
            "total_compute_ms": value,
        }
        for repeat, value in enumerate((1.0, 1.01, 0.99, 1.0, 1.0))
    ]
    result = _repeat_median_stability(stable, [], maximum_cv=0.10)
    assert result["status"] == "PASS"
    assert result["maximum_repeat_median_cv"] < 0.10

    unstable = [
        {
            "tensor_parallel_degree": 2,
            "workload": "decode",
            "rank": 0,
            "repeat": repeat,
            "total_compute_ms": value,
        }
        for repeat, value in enumerate((1.0, 2.0, 1.0, 2.0, 1.0))
    ]
    result = _repeat_median_stability(unstable, [], maximum_cv=0.10)
    assert result["status"] == "FAIL"


def test_reporting_generates_every_required_chart(tmp_path: Path) -> None:
    generate_microsharding_charts(
        tmp_path,
        memory=[],
        correctness=[],
        boundaries=[],
        kv=[],
        collective_metrics=[],
        projections=[],
        break_even=[],
        hybrid=[],
        heterogeneous=[],
        compression=[],
        moe=[],
        expert_projection=[],
        expert_cache=[],
        k3_plans=[],
    )
    assert {path.name for path in tmp_path.glob("*.png")} == set(REQUIRED_CHARTS)
