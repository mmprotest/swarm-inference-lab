"""Frozen fixed-placement uncoded communication lower bound."""

from __future__ import annotations

from .freeze import COMMUNICATION_LOWER_BOUND_RATIO_MAX, REMOTE_WORKERS
from .geometry import D_BYTES, H_BYTES, L_BYTES, R_BYTES

LOWER_BOUND_BYTES = (
    REMOTE_WORKERS * H_BYTES
    + REMOTE_WORKERS * R_BYTES
    + REMOTE_WORKERS * L_BYTES
    + 56
    + REMOTE_WORKERS * H_BYTES
)
D_LOWER_BOUND_RATIO = D_BYTES / LOWER_BOUND_BYTES
LOWER_BOUND_DESCRIPTION = (
    "fixed-placement uncoded communication payload lower bound under the "
    "frozen P8 decomposition."
)


def assert_lower_bound() -> None:
    if LOWER_BOUND_BYTES != 502_712:
        raise RuntimeError("frozen E024 lower-bound bytes changed")
    if D_LOWER_BOUND_RATIO > COMMUNICATION_LOWER_BOUND_RATIO_MAX:
        raise RuntimeError("D exceeds the frozen communication lower-bound ratio")


__all__ = [
    "D_LOWER_BOUND_RATIO",
    "LOWER_BOUND_BYTES",
    "LOWER_BOUND_DESCRIPTION",
    "assert_lower_bound",
]
