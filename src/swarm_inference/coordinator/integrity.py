"""Coordinator-side audit comparison and reputation actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from swarm_inference.config.models import IntegrityConfig
from swarm_inference.security.reputation import ReputationBook


@dataclass(frozen=True, slots=True)
class AuditOutcome:
    agreed: bool
    primary_worker_id: str
    audit_worker_id: str
    maximum_absolute_error: float
    quarantined_workers: tuple[str, ...]


class IntegrityCoordinator:
    def __init__(
        self,
        config: IntegrityConfig,
        reputation: ReputationBook | None = None,
    ) -> None:
        self.config = config
        self.reputation = reputation or ReputationBook(
            disagreement_penalty=config.disagreement_penalty,
            agreement_reward=config.agreement_reward,
            quarantine_threshold=config.quarantine_threshold,
        )

    def compare(
        self,
        *,
        primary_worker_id: str,
        audit_worker_id: str,
        primary: np.ndarray,
        audit: np.ndarray,
        exact: bool,
    ) -> AuditOutcome:
        same_shape = primary.shape == audit.shape
        if exact:
            agreed = same_shape and bool(np.array_equal(primary, audit))
        else:
            agreed = same_shape and bool(
                np.allclose(
                    primary,
                    audit,
                    atol=self.config.real_model_atol,
                    rtol=self.config.real_model_rtol,
                )
            )
        maximum_error = (
            float(np.max(np.abs(primary.astype(np.float64) - audit.astype(np.float64))))
            if same_shape and primary.size
            else float("inf")
        )
        if agreed:
            self.reputation.record_agreement(primary_worker_id)
            self.reputation.record_agreement(audit_worker_id)
        else:
            self.reputation.record_disagreement(primary_worker_id)
            self.reputation.record_disagreement(audit_worker_id)
        quarantined = tuple(
            worker_id
            for worker_id in (primary_worker_id, audit_worker_id)
            if self.reputation.is_quarantined(worker_id)
        )
        return AuditOutcome(
            agreed=agreed,
            primary_worker_id=primary_worker_id,
            audit_worker_id=audit_worker_id,
            maximum_absolute_error=maximum_error,
            quarantined_workers=quarantined,
        )
