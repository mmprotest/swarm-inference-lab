"""Exact speculative drafting primitives for the stage pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass


class DraftProvider(ABC):
    name: str

    @abstractmethod
    def propose(self, history: Sequence[int], *, depth: int) -> list[int]:
        """Return at most ``depth`` non-authoritative candidate token IDs."""


class PromptLookupDraftProvider(DraftProvider):
    """Draft from the longest repeated suffix in prompt plus accepted history."""

    name = "prompt_lookup"

    def __init__(self, *, minimum_match: int = 2, maximum_match: int = 64) -> None:
        if minimum_match < 1 or maximum_match < minimum_match:
            raise ValueError("invalid prompt-lookup match limits")
        self.minimum_match = minimum_match
        self.maximum_match = maximum_match

    def propose(self, history: Sequence[int], *, depth: int) -> list[int]:
        if depth < 1:
            raise ValueError("draft depth must be positive")
        values = list(int(token) for token in history)
        if len(values) <= self.minimum_match:
            return []
        maximum = min(self.maximum_match, len(values) - 1)
        for match_length in range(maximum, self.minimum_match - 1, -1):
            suffix = values[-match_length:]
            latest_start = len(values) - match_length
            for start in range(latest_start - 1, -1, -1):
                if values[start : start + match_length] != suffix:
                    continue
                continuation_start = start + match_length
                continuation = values[continuation_start : continuation_start + depth]
                if continuation:
                    return continuation
        return []


@dataclass(frozen=True, slots=True)
class VerificationResult:
    proposed_tokens: tuple[int, ...]
    authoritative_tokens: tuple[int, ...]
    accepted_tokens: tuple[int, ...]
    rejected_tokens: tuple[int, ...]
    first_rejection_index: int | None

    @property
    def acceptance_rate(self) -> float:
        return (
            len(self.accepted_tokens) / len(self.proposed_tokens) if self.proposed_tokens else 0.0
        )


def verify_greedy_candidates(
    proposed_tokens: Sequence[int], authoritative_tokens: Sequence[int]
) -> VerificationResult:
    proposed = tuple(int(token) for token in proposed_tokens)
    authoritative = tuple(int(token) for token in authoritative_tokens)
    if len(authoritative) < len(proposed):
        raise ValueError("authoritative result does not cover every proposed position")
    accepted_count = 0
    for proposed_token, target_token in zip(proposed, authoritative, strict=False):
        if proposed_token != target_token:
            break
        accepted_count += 1
    return VerificationResult(
        proposed_tokens=proposed,
        authoritative_tokens=authoritative[: len(proposed)],
        accepted_tokens=proposed[:accepted_count],
        rejected_tokens=proposed[accepted_count:],
        first_rejection_index=accepted_count if accepted_count < len(proposed) else None,
    )
