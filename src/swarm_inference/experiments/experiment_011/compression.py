"""Deprecated compatibility exports for lossless transport compression.

New code must import :mod:`swarm_inference.transport.compression` directly.
"""

from swarm_inference.transport.compression import (
    AdaptiveCompressionController,
    CompressionDecision,
    CompressionMode,
    CompressionResult,
    DecompressionResult,
    byte_shuffle,
    byte_unshuffle,
    compress_lossless,
    decompress_lossless,
)

__all__ = [
    "AdaptiveCompressionController",
    "CompressionDecision",
    "CompressionMode",
    "CompressionResult",
    "DecompressionResult",
    "byte_shuffle",
    "byte_unshuffle",
    "compress_lossless",
    "decompress_lossless",
]
