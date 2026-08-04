"""Deprecated compatibility exports for canonical stage tensor transport.

New code must import :mod:`swarm_inference.transport.stage_tensor` directly.
"""

from swarm_inference.transport.stage_tensor import (
    SUPPORTED_DTYPES,
    AdaptiveTransportInputs,
    PackedTensor,
    RequestedCompressionMode,
    pack_tensor,
    tensor_raw_bytes,
    unpack_tensor,
)

__all__ = [
    "SUPPORTED_DTYPES",
    "AdaptiveTransportInputs",
    "PackedTensor",
    "RequestedCompressionMode",
    "pack_tensor",
    "tensor_raw_bytes",
    "unpack_tensor",
]
