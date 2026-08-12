from __future__ import annotations

import numpy as np
import pytest

from swarm_inference.execution.kda_verification import (
    AffineKDAFactor,
    factorized_kda_block,
    kda_state_traffic_model,
    replay_kda_state,
    serial_kda_block,
    token_factors,
)

BLOCK_SIZES = (1, 2, 4, 7, 12, 16)


def _fixture(
    rows: int,
    *,
    seed: int = 17,
    key_dimension: int = 8,
    value_dimension: int = 6,
    dtype: np.dtype | type[np.floating] = np.float64,
):
    rng = np.random.default_rng(seed)
    state = rng.normal(size=(key_dimension, value_dimension)).astype(dtype)
    alphas = np.exp(-rng.uniform(0.01, 1.0, size=(rows, key_dimension))).astype(dtype)
    keys = rng.normal(size=(rows, key_dimension)).astype(dtype)
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    values = rng.normal(size=(rows, value_dimension)).astype(dtype)
    betas = (1.0 / (1.0 + np.exp(-rng.normal(size=rows)))).astype(dtype)
    queries = rng.normal(size=(rows, key_dimension)).astype(dtype)
    return state, token_factors(alphas, keys, values, betas), queries


@pytest.mark.parametrize("rows", BLOCK_SIZES)
def test_factorized_block_matches_exact_serial_for_required_windows(rows: int) -> None:
    state, factors, queries = _fixture(rows)
    serial_output, serial_state, _ = serial_kda_block(state, factors, queries)
    factor_output, factor_state, prefix = factorized_kda_block(state, factors, queries)

    np.testing.assert_allclose(factor_output, serial_output, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(factor_state, serial_state, rtol=2e-13, atol=2e-13)
    assert prefix.transition_rank == rows
    assert prefix.additive_rank == rows


@pytest.mark.parametrize("rows", BLOCK_SIZES)
def test_accepted_state_advancement_reconstructs_selected_prefix(rows: int) -> None:
    state, factors, queries = _fixture(rows)
    for accepted in {0, 1, rows // 2, rows}:
        _, expected, _ = serial_kda_block(state, factors[:accepted], queries[:accepted])
        _, actual, prefix = factorized_kda_block(state, factors, queries, accepted_tokens=accepted)
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
        assert prefix.transition_rank == accepted


@pytest.mark.parametrize("split", (1, 2, 4, 7, 12, 16))
def test_factor_prefix_composition_matches_direct_application(split: int) -> None:
    state, factors, _ = _fixture(16)
    prefix = AffineKDAFactor.identity(state.shape[0], state.shape[1])
    for factor in factors[:split]:
        prefix = prefix.then(factor.expand())
    direct = state.copy()
    for factor in factors[:split]:
        direct = factor.expand().apply(direct)
    np.testing.assert_allclose(prefix.apply(state), direct, rtol=2e-13, atol=2e-13)


def test_continuing_state_and_replay_match_uninterrupted_serial() -> None:
    state, factors, queries = _fixture(16)
    _, state7, _ = serial_kda_block(state, factors[:7], queries[:7])
    tail_output, final_state, _ = serial_kda_block(state7, factors[7:], queries[7:])
    all_output, all_state, _ = serial_kda_block(state, factors, queries)

    replayed = replay_kda_state(state, factors, accepted_tokens=7)
    factor_tail, factor_final, _ = factorized_kda_block(replayed, factors[7:], queries[7:])
    np.testing.assert_allclose(replayed, state7, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(factor_tail, tail_output, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(factor_final, final_state, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(final_state, all_state, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(tail_output, all_output[7:], rtol=2e-13, atol=2e-13)


def test_empty_block_preserves_state_without_aliasing() -> None:
    state, _, _ = _fixture(1)
    output, advanced, prefix = factorized_kda_block(
        state,
        (),
        np.empty((0, state.shape[0]), dtype=state.dtype),
        accepted_tokens=0,
    )
    assert output.shape == (0, state.shape[1])
    np.testing.assert_array_equal(advanced, state)
    assert not np.shares_memory(advanced, state)
    assert prefix.transition_rank == 0


def test_inactive_padded_state_slots_are_untouched() -> None:
    active_state, factors, queries = _fixture(7)
    padded = np.full((12, active_state.shape[1]), 1234.5, dtype=np.float64)
    padded[: active_state.shape[0]] = active_state
    _, advanced, _ = factorized_kda_block(padded[: active_state.shape[0]], factors, queries)
    result = padded.copy()
    result[: active_state.shape[0]] = advanced
    np.testing.assert_array_equal(result[active_state.shape[0] :], 1234.5)


def test_repeated_invocation_has_no_factor_or_state_aliasing() -> None:
    state, factors, queries = _fixture(7)
    output1, state1, prefix1 = factorized_kda_block(state, factors, queries)
    output2, state2, prefix2 = factorized_kda_block(state, factors, queries)
    np.testing.assert_array_equal(output1, output2)
    np.testing.assert_array_equal(state1, state2)
    assert prefix1 is not prefix2
    state1[0, 0] += 1.0
    assert state1[0, 0] != state2[0, 0]


def test_float32_reassociation_meets_exact_experiment_tolerance() -> None:
    state, factors, queries = _fixture(16, dtype=np.float32)
    serial_output, serial_state, _ = serial_kda_block(state, factors, queries)
    factor_output, factor_state, _ = factorized_kda_block(state, factors, queries)
    output_relative_l2 = np.linalg.norm(factor_output - serial_output) / np.linalg.norm(
        serial_output
    )
    state_relative_l2 = np.linalg.norm(factor_state - serial_state) / np.linalg.norm(serial_state)
    assert output_relative_l2 <= 2e-5
    assert state_relative_l2 <= 2e-5


def test_full_shape_state_traffic_model() -> None:
    traffic = kda_state_traffic_model(block_tokens=7)
    assert traffic["full_state_bytes"] == 96 * 128 * 128 * 4
    assert traffic["compact_token_factor_bytes"] == 96 * 7 * 385 * 4
    assert traffic["serial_logical_state_read_bytes"] == 14 * 96 * 128 * 128 * 4
    assert traffic["accepted_state_copy_bytes"] == 0
    assert traffic["accepted_state_materializations_factorized"] == 1
    assert traffic["snapshot_compression_ratio"] > 6.0


@pytest.mark.parametrize("accepted", (-1, 2))
def test_rejects_accepted_positions_outside_edge_block(accepted: int) -> None:
    state, factors, queries = _fixture(1)
    with pytest.raises(ValueError, match="outside"):
        factorized_kda_block(state, factors, queries, accepted_tokens=accepted)
