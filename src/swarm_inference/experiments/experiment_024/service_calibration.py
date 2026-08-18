"""Fresh-service calibration entry gate.

Calibration is deliberately not entered after a mandatory immutable-input
failure.  This prevents GPU work from being mistaken for a valid commercial
model when a complete Stage B placement cannot exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .correctness import require_phase0
from .freeze import (
    CALIBRATION_ITERATIONS,
    CALIBRATION_WARMUP,
    FUSION_ITERATIONS,
    FUSION_WARMUP,
)


@dataclass(frozen=True, slots=True)
class CalibrationProtocol:
    calibration_layers: tuple[int, int] = (45, 47)
    heldout_layers: tuple[int, int] = (89, 91)
    rows: tuple[int, int, int] = (1, 2, 4)
    warmup: int = CALIBRATION_WARMUP
    iterations: int = CALIBRATION_ITERATIONS
    fusion_warmup: int = FUSION_WARMUP
    fusion_iterations: int = FUSION_ITERATIONS


def calibration_preflight(repo_root: Path) -> CalibrationProtocol:
    require_phase0(repo_root)
    return CalibrationProtocol()


__all__ = ["CalibrationProtocol", "calibration_preflight"]
