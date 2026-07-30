from __future__ import annotations

import pytest

from swarm_inference.config.models import OperationKind
from swarm_inference.coordinator.replay_log import ReplayLog
from swarm_inference.exceptions import IntegrityError, ReplayUnavailableError
from swarm_inference.security.reputation import ReputationBook


def test_reputation_quarantines_below_threshold() -> None:
    book = ReputationBook(disagreement_penalty=0.3, quarantine_threshold=0.5)
    assert book.record_disagreement("bad") == pytest.approx(0.7)
    assert not book.is_quarantined("bad")
    assert book.record_disagreement("bad") == pytest.approx(0.4)
    assert book.is_quarantined("bad")


def test_replay_log_orders_inputs() -> None:
    log = ReplayLog(maximum_bytes=100)
    for position in [2, 0, 1]:
        log.append(
            request_id="r",
            model_revision="v",
            stage_id=0,
            cache_generation=0,
            token_position=position,
            operation=OperationKind.PREFILL if position == 0 else OperationKind.DECODE,
            payload=bytes([position]),
            recorded_monotonic_ns=position,
        )
    entries = log.entries_for(
        request_id="r",
        model_revision="v",
        stage_id=0,
        cache_generation=0,
    )
    assert [entry.token_position for entry in entries] == [0, 1, 2]


def test_replay_gap_and_duplicate_are_rejected() -> None:
    log = ReplayLog(maximum_bytes=100)
    for position in [0, 2]:
        log.append(
            request_id="r",
            model_revision="v",
            stage_id=0,
            cache_generation=0,
            token_position=position,
            operation=OperationKind.DECODE,
            payload=b"x",
            recorded_monotonic_ns=position,
        )
    with pytest.raises(ReplayUnavailableError, match="gap"):
        log.entries_for(
            request_id="r",
            model_revision="v",
            stage_id=0,
            cache_generation=0,
        )
    with pytest.raises(IntegrityError, match="duplicate"):
        log.append(
            request_id="r",
            model_revision="v",
            stage_id=0,
            cache_generation=0,
            token_position=0,
            operation=OperationKind.DECODE,
            payload=b"x",
            recorded_monotonic_ns=3,
        )
