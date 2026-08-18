"""Experiment 024 phase runner with mandatory fail-closed behavior."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .finalize import finalize_invalid
from .freeze import audit_immutable_inputs


def run_phase0(repo_root: Path, *, pre_ruff_findings: int = 725) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    audit = audit_immutable_inputs(repo_root)
    if audit.status != "PASS":
        return finalize_invalid(
            repo_root,
            audit,
            started_at_utc=started_at,
            elapsed_seconds=time.perf_counter() - started,
            pre_ruff_findings=pre_ruff_findings,
        )
    raise RuntimeError("Phase 0 passed; the full E024 pipeline is not yet frozen")


__all__ = ["run_phase0"]
