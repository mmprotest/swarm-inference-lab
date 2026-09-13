from __future__ import annotations

from swarm_inference.experiments.experiment_027.runner import ngram_draft, token_hash


def test_ngram_draft_uses_prior_continuation() -> None:
    history = [1, 2, 3, 4, 1, 2, 3]
    assert ngram_draft(history, 3) == [4, 1, 2]


def test_ngram_draft_does_not_invent_unobserved_tokens() -> None:
    assert ngram_draft([1, 2, 3, 4], 8) == []


def test_token_hash_is_stable() -> None:
    assert token_hash([1, 2, 3]) == token_hash((1, 2, 3))
