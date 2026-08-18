"""Validity gates for Experiment 024."""

from __future__ import annotations

from pathlib import Path

from .freeze import Phase0Audit, audit_immutable_inputs


class ModelInvalidError(RuntimeError):
    """Raised when mandatory evidence cannot support an E024 conclusion."""


def require_phase0(repo_root: Path) -> Phase0Audit:
    audit = audit_immutable_inputs(repo_root)
    if audit.status != "PASS":
        raise ModelInvalidError(audit.reason or "Experiment 024 Phase 0 failed")
    return audit


__all__ = ["ModelInvalidError", "require_phase0"]
