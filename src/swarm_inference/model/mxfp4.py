"""Production value object for compressed-tensors MXFP4 weights."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

MXFP4_GROUP_SIZE = 32


@dataclass(frozen=True, slots=True)
class MXFP4Tensor:
    """Official compressed-tensors mxfp4-pack-quantized row layout."""

    packed: np.ndarray
    scales: np.ndarray
    output_dimension: int
    input_dimension: int

    def __post_init__(self) -> None:
        packed = np.asarray(self.packed)
        scales = np.asarray(self.scales)
        if self.input_dimension % MXFP4_GROUP_SIZE:
            raise ValueError("MXFP4 input dimension must align to 32-value scale groups")
        if packed.dtype != np.uint8 or packed.shape != (
            self.output_dimension,
            self.input_dimension // 2,
        ):
            raise ValueError("MXFP4 packed tensor must be uint8 [O, I/2]")
        if scales.dtype != np.uint8 or scales.shape != (
            self.output_dimension,
            self.input_dimension // MXFP4_GROUP_SIZE,
        ):
            raise ValueError("MXFP4 scales must be UE8M0 uint8 [O, I/32]")
        if np.any((scales == 0) | (scales == 255)):
            raise ValueError("MXFP4 weights require finite non-denormal UE8M0 scales")

    @property
    def byte_size(self) -> int:
        return int(self.packed.nbytes + self.scales.nbytes)

    @property
    def content_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"mxfp4-e2m1-low-even-ue8m0-g32")
        digest.update(np.ascontiguousarray(self.packed).tobytes())
        digest.update(np.ascontiguousarray(self.scales).tobytes())
        return "sha256:" + digest.hexdigest()


__all__ = ["MXFP4_GROUP_SIZE", "MXFP4Tensor"]
