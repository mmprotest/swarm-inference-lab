"""Exact A/B/C/D communication geometry for frozen P8 ownership."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .freeze import (
    FLOAT_BYTES,
    HIDDEN,
    LATENT,
    REMOTE_WORKERS,
    ROUTE_ID_BYTES,
    ROUTE_WEIGHT_BYTES,
    TOPK,
)
from .models import StageAArm

H_BYTES = HIDDEN * FLOAT_BYTES
L_BYTES = LATENT * FLOAT_BYTES
R_BYTES = TOPK * (ROUTE_ID_BYTES + ROUTE_WEIGHT_BYTES)
L_SLICE_BYTES = (LATENT // 8) * FLOAT_BYTES

A_BYTES = REMOTE_WORKERS * (
    (H_BYTES + R_BYTES) + L_BYTES + H_BYTES + H_BYTES + L_BYTES + H_BYTES
)
B_BYTES = REMOTE_WORKERS * (
    H_BYTES + R_BYTES + L_BYTES + H_BYTES + L_BYTES + H_BYTES
)
C_BYTES = REMOTE_WORKERS * (
    H_BYTES + R_BYTES + L_BYTES + H_BYTES + L_SLICE_BYTES + H_BYTES
)
D_BYTES = REMOTE_WORKERS * (
    H_BYTES + R_BYTES + L_BYTES + L_SLICE_BYTES + H_BYTES
)


@dataclass(frozen=True, slots=True)
class CommunicationGeometry:
    arm: StageAArm
    bytes_per_row: int
    messages_per_row: int

    def as_dict(self) -> dict[str, int | str]:
        value = asdict(self)
        value["arm"] = self.arm.value
        return value


GEOMETRY = {
    StageAArm.A_CURRENT: CommunicationGeometry(StageAArm.A_CURRENT, A_BYTES, 42),
    StageAArm.B_RETAIN_HIDDEN: CommunicationGeometry(
        StageAArm.B_RETAIN_HIDDEN, B_BYTES, 42
    ),
    StageAArm.C_SLICE_LATENT: CommunicationGeometry(
        StageAArm.C_SLICE_LATENT, C_BYTES, 42
    ),
    StageAArm.D_FUSE_OUTPUT: CommunicationGeometry(
        StageAArm.D_FUSE_OUTPUT, D_BYTES, 35
    ),
}


def geometry_for(arm: StageAArm, rows: int = 1) -> CommunicationGeometry:
    if rows not in (1, 2, 4):
        raise ValueError("physically supported rows are 1, 2, and 4")
    base = GEOMETRY[arm]
    return CommunicationGeometry(arm, base.bytes_per_row * rows, base.messages_per_row)


def assert_frozen_geometry() -> None:
    expected = {
        StageAArm.A_CURRENT: (1_004_416, 42),
        StageAArm.B_RETAIN_HIDDEN: (803_712, 42),
        StageAArm.C_SLICE_LATENT: (715_904, 42),
        StageAArm.D_FUSE_OUTPUT: (515_200, 35),
    }
    actual = {
        arm: (value.bytes_per_row, value.messages_per_row)
        for arm, value in GEOMETRY.items()
    }
    if actual != expected:
        raise RuntimeError(f"frozen E024 geometry changed: {actual!r}")


__all__ = [
    "A_BYTES",
    "B_BYTES",
    "C_BYTES",
    "D_BYTES",
    "GEOMETRY",
    "H_BYTES",
    "L_BYTES",
    "L_SLICE_BYTES",
    "R_BYTES",
    "CommunicationGeometry",
    "assert_frozen_geometry",
    "geometry_for",
]
