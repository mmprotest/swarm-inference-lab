"""Stage A execution preflight and frozen workload sizes."""

from __future__ import annotations

import math
from pathlib import Path

from .correctness import require_phase0
from .freeze import STAGE_A_CONCURRENCY


def measured_blocks_per_slot(concurrency: int) -> int:
    if concurrency not in STAGE_A_CONCURRENCY:
        raise ValueError("concurrency is outside the frozen Stage A ladder")
    return max(4, math.ceil(256 / concurrency))


class BlockEngine:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def preflight(self) -> None:
        require_phase0(self.repo_root)


__all__ = ["BlockEngine", "measured_blocks_per_slot"]
