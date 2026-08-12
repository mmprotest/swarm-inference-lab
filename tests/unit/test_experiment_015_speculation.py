from __future__ import annotations

import numpy as np
import pytest

from swarm_inference.experiments.experiment_015.contracts import EconomicsConfig
from swarm_inference.experiments.experiment_015.speculation import (
    RequestState,
    SpeculativeSession,
    greedy_acceptance,
    stochastic_acceptance,
)


def test_configurable_economics_reproduce_preregistered_targets() -> None:
    economics = EconomicsConfig()
    assert economics.break_even_tok_s_per_paid_gpu == pytest.approx(2.7777777778)
    assert economics.margin_tok_s_per_paid_gpu == pytest.approx(5.5555555556)


@pytest.mark.parametrize(
    ("draft", "target", "bonus", "accepted", "replacement", "rejection"),
    [
        ([1, 2, 3], [1, 2, 3], 4, (1, 2, 3), 4, None),
        ([1, 9, 3], [1, 2, 3], 4, (1,), 2, 1),
        ([9, 2, 3], [1, 2, 3], 4, (), 1, 0),
        ([9], [1], None, (), 1, 0),
    ],
)
def test_greedy_acceptance_paths(
    draft: list[int],
    target: list[int],
    bonus: int | None,
    accepted: tuple[int, ...],
    replacement: int | None,
    rejection: int | None,
) -> None:
    result = greedy_acceptance(draft, target, bonus_token=bonus)
    assert result.accepted_draft_tokens == accepted
    assert result.replacement_token == replacement
    assert result.first_rejection_index == rejection


def test_eos_stops_commit_inside_draft_block() -> None:
    result = greedy_acceptance([1, 7, 3], [1, 7, 3], bonus_token=4, eos_token_id=7)
    assert result.output_tokens == (1, 7)
    assert result.eos_reached
    assert not result.full_block_accepted


def test_stochastic_full_and_rejection_paths() -> None:
    q = np.array([[0.8, 0.2], [0.2, 0.8]])
    p = np.array([[0.8, 0.2], [0.1, 0.9], [0.3, 0.7]])
    full = stochastic_acceptance([0, 1], q, p, [0.1, 0.1, 0.2])
    assert full.accepted_draft_tokens == (0, 1)
    assert full.full_block_accepted
    assert full.replacement_token == 0
    assert full.target_rows_consumed == 3
    rejected = stochastic_acceptance([0, 0], q, p, [0.1, 0.9, 0.2])
    assert rejected.first_rejection_index == 1
    assert rejected.replacement_token == 1


def test_transaction_commit_rollback_cancel_and_isolation() -> None:
    original = RequestState(
        position=5,
        token_ids=[10],
        kda={1: {"state": np.ones((2, 2), dtype=np.float32)}},
        mla={3: {"latent": np.ones((2, 2), dtype=np.float32)}},
        attnres={"blocks": np.ones((2, 2), dtype=np.float32)},
    )
    session = SpeculativeSession(original)
    baseline = session.committed.fingerprint()
    working = session.begin()
    working.kda[1]["state"] *= 9
    working.mla[3]["latent"] *= 8
    working.attnres["blocks"] *= 7
    assert session.rollback().fingerprint() == baseline

    working = session.begin()
    working.kda[1]["state"] *= 3
    session.stage_verified_prefix()
    working.kda[1]["state"] *= 2
    session.stage_verified_prefix()
    working.kda[1]["state"] *= 2
    session.stage_verified_prefix()
    committed = session.commit(greedy_acceptance([11, 12], [11, 12], bonus_token=13))
    assert committed.position == 8
    assert committed.token_ids == [10, 11, 12, 13]
    assert float(committed.kda[1]["state"][0, 0]) == 12.0
    session.cancel()
    assert session.committed.cancelled


def test_partial_acceptance_commits_exact_prefix_state() -> None:
    session = SpeculativeSession(
        RequestState(
            position=0,
            kda={1: {"state": np.ones(1, dtype=np.float32)}},
        )
    )
    working = session.begin()
    for value in (2.0, 3.0, 4.0):
        working.kda[1]["state"][:] = value
        session.stage_verified_prefix()
    result = greedy_acceptance([10, 99, 30], [10, 20, 30], bonus_token=40)
    committed = session.commit(result)
    assert committed.token_ids == [10, 20]
    assert committed.position == 2
    assert float(committed.kda[1]["state"][0]) == 3.0


def test_multiple_request_sessions_never_share_state() -> None:
    first = SpeculativeSession(
        RequestState(position=0, kda={1: {"state": np.ones(1, dtype=np.float32)}})
    )
    second = SpeculativeSession(
        RequestState(position=7, kda={1: {"state": np.full(1, 9, dtype=np.float32)}})
    )
    first_working = first.begin()
    first_working.kda[1]["state"][:] = 5
    first.rollback()
    assert float(second.committed.kda[1]["state"][0]) == 9.0
    assert second.committed.position == 7
