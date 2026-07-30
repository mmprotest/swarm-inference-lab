from __future__ import annotations

import numpy as np

from swarm_inference.config.models import IntegrityConfig
from swarm_inference.coordinator.integrity import IntegrityCoordinator


def test_audit_disagreement_penalises_and_quarantines() -> None:
    coordinator = IntegrityCoordinator(
        IntegrityConfig(
            audit_fraction=1,
            disagreement_penalty=0.6,
            quarantine_threshold=0.5,
        )
    )
    outcome = coordinator.compare(
        primary_worker_id="bad",
        audit_worker_id="trusted",
        primary=np.array([1, 2, 3], dtype=np.float32),
        audit=np.array([1, 2, 4], dtype=np.float32),
        exact=True,
    )
    assert not outcome.agreed
    assert "bad" in outcome.quarantined_workers
    assert outcome.maximum_absolute_error == 1


def test_real_model_audit_uses_tolerance() -> None:
    coordinator = IntegrityCoordinator(IntegrityConfig(real_model_atol=1e-3))
    outcome = coordinator.compare(
        primary_worker_id="a",
        audit_worker_id="b",
        primary=np.array([1.0], dtype=np.float32),
        audit=np.array([1.0001], dtype=np.float32),
        exact=False,
    )
    assert outcome.agreed
