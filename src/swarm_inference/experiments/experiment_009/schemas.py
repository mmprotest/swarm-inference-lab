"""Evidence and verdict schemas for Experiment 009."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from swarm_inference.config.models import StrictModel


class EvidenceClass(StrEnum):
    MEASURED = "MEASURED"
    FIXTURE = "FIXTURE"
    PROJECTED = "PROJECTED"


class GateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_EVALUATED = "NOT_EVALUATED"


class Experiment009Verdict(StrEnum):
    PASS_STRONG = "PASS_STRONG"
    PASS_INTEGRATION = "PASS_INTEGRATION"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"


class GateResult(StrictModel):
    gate_id: int = Field(ge=1, le=10)
    name: str
    status: GateStatus
    evidence_class: EvidenceClass | None = None
    reasons: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


def overall_verdict(
    gates: list[GateResult],
    *,
    full_run: bool,
    valid_colibri_execution: bool,
    reverse_confirmed_gain: float | None,
) -> Experiment009Verdict:
    if not valid_colibri_execution:
        return Experiment009Verdict.FAIL
    if not full_run:
        return Experiment009Verdict.PARTIAL
    passed = {gate.gate_id for gate in gates if gate.status == GateStatus.PASS}
    if passed != set(range(1, 11)):
        return Experiment009Verdict.PARTIAL
    if reverse_confirmed_gain is not None and reverse_confirmed_gain >= 0.03:
        return Experiment009Verdict.PASS_STRONG
    return Experiment009Verdict.PASS_INTEGRATION
