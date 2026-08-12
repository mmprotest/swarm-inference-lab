from __future__ import annotations

import numpy as np
import pytest

from swarm_inference.execution.dcp import (
    combine_attention_partials,
    dcp_partial_payload_bytes,
    full_attention,
    shard_attention,
)
from swarm_inference.execution.verification import (
    VerificationBlock,
    build_grouped_expert_plan,
    deterministic_scatter_reduce,
    expert_reuse_statistics,
    partition_certified_batch,
    scatter_grouped_expert_outputs,
)


def test_verification_block_includes_the_target_bonus_row() -> None:
    block = VerificationBlock(
        session_id="request-1",
        cache_position_start=11,
        candidate_count=7,
    )

    assert block.row_count == 8
    assert block.positions == tuple(range(11, 19))


@pytest.mark.parametrize("candidates", [1, 2, 4, 7, 12, 16])
def test_grouped_plan_covers_required_block_edge_sizes(candidates: int) -> None:
    rows = candidates + 1
    routes = np.stack([np.roll(np.arange(16, dtype=np.int32), row % 16) for row in range(rows)])

    plan = build_grouped_expert_plan(
        routes,
        supported_batch_sizes=(1, 2, 4, 8, 16),
    )

    assert plan.rows == rows
    assert plan.total_assignments == rows * 16
    assert plan.unique_experts == 16
    assert sum(group.count for group in plan.groups) == rows * 16
    assert all(sum(group.native_chunks) == group.count for group in plan.groups)


def test_grouped_assignment_order_is_deterministic() -> None:
    routes = np.array([[4, 1, 3], [3, 4, 2]], dtype=np.int32)

    first = build_grouped_expert_plan(routes, supported_batch_sizes=(1, 2, 4))
    second = build_grouped_expert_plan(routes.copy(), supported_batch_sizes=(4, 2, 1))

    assert first == second
    assert [item.expert_id for item in first.groups] == [1, 2, 3, 4]
    assert [(item.expert_id, item.row, item.slot) for item in first.assignments] == [
        (1, 0, 1),
        (2, 1, 2),
        (3, 0, 2),
        (3, 1, 0),
        (4, 0, 0),
        (4, 1, 1),
    ]


def test_grouped_plan_materializes_a_one_shot_supported_size_iterable() -> None:
    routes = np.array([[1, 2], [2, 3]], dtype=np.int32)

    plan = build_grouped_expert_plan(
        routes,
        supported_batch_sizes=(size for size in (1, 2, 4)),
    )

    assert [group.native_chunks for group in plan.groups] == [(1,), (2,), (1,)]


def test_scatter_and_reduction_restore_router_slot_order() -> None:
    routes = np.array([[7, 2], [2, 9]], dtype=np.int32)
    weights = np.array([[0.25, 0.75], [0.6, 0.4]], dtype=np.float32)
    plan = build_grouped_expert_plan(routes, supported_batch_sizes=(1, 2))
    grouped = np.stack(
        [
            np.array([item.row * 10 + item.slot, item.expert_id], dtype=np.float32)
            for item in plan.assignments
        ]
    )

    restored = scatter_grouped_expert_outputs(grouped, plan)
    actual = deterministic_scatter_reduce(grouped, weights, plan)
    expected = np.zeros((2, 2), dtype=np.float32)
    for row in range(2):
        for slot in range(2):
            expected[row] += restored[row, slot] * weights[row, slot]

    np.testing.assert_array_equal(actual, expected)


def test_batch_one_grouped_plan_is_token_major_equivalent() -> None:
    routes = np.array([[8, 3, 5]], dtype=np.int32)
    plan = build_grouped_expert_plan(routes, supported_batch_sizes=(1, 2, 4))
    grouped = np.stack(
        [np.full(4, assignment.slot + 1, dtype=np.float32) for assignment in plan.assignments]
    )
    weights = np.array([[0.2, 0.3, 0.5]], dtype=np.float32)

    actual = deterministic_scatter_reduce(grouped, weights, plan)
    expected = sum(np.full(4, slot + 1, dtype=np.float32) * weights[0, slot] for slot in range(3))

    np.testing.assert_array_equal(actual[0], expected)


def test_expert_reuse_reports_union_and_adjacent_overlap() -> None:
    routes = np.array([[1, 2, 3], [2, 3, 4], [2, 4, 5]], dtype=np.int32)

    result = expert_reuse_statistics(routes)

    assert result["total_assignments"] == 9
    assert result["unique_experts"] == 5
    assert result["unique_experts_per_assignment"] == pytest.approx(5 / 9)
    assert result["mean_assignments_per_touched_expert"] == pytest.approx(9 / 5)
    assert result["mean_adjacent_position_overlap"] == 2.0


def test_invalid_or_duplicate_routes_fail_closed() -> None:
    with pytest.raises(ValueError, match="unique experts"):
        build_grouped_expert_plan([[1, 1]])
    with pytest.raises(TypeError, match="integers"):
        build_grouped_expert_plan([[1.0, 2.0]])
    with pytest.raises(ValueError, match="include one"):
        partition_certified_batch(7, (2, 4))


@pytest.mark.parametrize("degree", [1, 2, 4, 8])
def test_exact_dcp_reduction_matches_full_attention_for_uneven_shards(degree: int) -> None:
    generator = np.random.default_rng(16016)
    scores = generator.normal(size=(2, 3, 37)).astype(np.float32)
    values = generator.normal(size=(2, 3, 37, 5)).astype(np.float32)
    reference = full_attention(scores, values)
    partials = [
        shard_attention(score_shard, value_shard, shard_index=index)
        for index, (score_shard, value_shard) in enumerate(
            zip(
                np.array_split(scores, degree, axis=-1),
                np.array_split(values, degree, axis=-2),
                strict=True,
            )
        )
    ]

    forward = combine_attention_partials(partials)
    reversed_input = combine_attention_partials(list(reversed(partials)))

    np.testing.assert_allclose(forward, reference, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(forward, reversed_input)


def test_dcp_rejects_duplicate_shards_and_reports_payload() -> None:
    scores = np.array([[[0.0, 1.0]]], dtype=np.float32)
    values = np.ones((1, 1, 2, 4), dtype=np.float32)
    partial = shard_attention(scores, values, shard_index=0)

    with pytest.raises(ValueError, match="indices must be unique"):
        combine_attention_partials([partial, partial])
    assert dcp_partial_payload_bytes(query_rows=8) == 8 * 64 * 130 * 4
