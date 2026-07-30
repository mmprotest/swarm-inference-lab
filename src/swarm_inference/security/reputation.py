"""Bounded worker reputation and quarantine state."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ReputationRecord:
    score: float = 1.0
    agreements: int = 0
    disagreements: int = 0
    quarantined: bool = False


class ReputationBook:
    def __init__(
        self,
        *,
        disagreement_penalty: float = 0.25,
        agreement_reward: float = 0.01,
        quarantine_threshold: float = 0.5,
    ) -> None:
        if not 0 <= disagreement_penalty <= 1:
            raise ValueError("disagreement_penalty must be in [0, 1]")
        if not 0 <= agreement_reward <= 1:
            raise ValueError("agreement_reward must be in [0, 1]")
        if not 0 <= quarantine_threshold <= 1:
            raise ValueError("quarantine_threshold must be in [0, 1]")
        self.disagreement_penalty = disagreement_penalty
        self.agreement_reward = agreement_reward
        self.quarantine_threshold = quarantine_threshold
        self._records: dict[str, ReputationRecord] = {}

    def record(self, worker_id: str) -> ReputationRecord:
        return self._records.setdefault(worker_id, ReputationRecord())

    def record_agreement(self, worker_id: str) -> float:
        record = self.record(worker_id)
        record.agreements += 1
        record.score = min(1.0, record.score + self.agreement_reward)
        return record.score

    def record_disagreement(self, worker_id: str) -> float:
        record = self.record(worker_id)
        record.disagreements += 1
        record.score = max(0.0, record.score - self.disagreement_penalty)
        if record.score < self.quarantine_threshold:
            record.quarantined = True
        return record.score

    def is_quarantined(self, worker_id: str) -> bool:
        return self.record(worker_id).quarantined

    def score(self, worker_id: str) -> float:
        return self.record(worker_id).score

    def evidence(self) -> dict[str, dict[str, object]]:
        return {
            worker_id: {
                "score": record.score,
                "agreements": record.agreements,
                "disagreements": record.disagreements,
                "quarantined": record.quarantined,
            }
            for worker_id, record in sorted(self._records.items())
        }
