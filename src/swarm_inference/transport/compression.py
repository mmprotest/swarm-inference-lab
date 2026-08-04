"""Bitwise-lossless byte compression with measured adaptive selection."""

from __future__ import annotations

import hashlib
import time
import zlib
from dataclasses import asdict, dataclass
from typing import Literal

CompressionMode = Literal["none", "byte_shuffle_fast_codec"]


def byte_shuffle(payload: bytes, element_width: int) -> bytes:
    """Group equal byte lanes from fixed-width elements."""

    if element_width <= 0:
        raise ValueError("element width must be positive")
    if len(payload) % element_width:
        raise ValueError("payload length must be divisible by element width")
    view = memoryview(payload)
    return b"".join(bytes(view[lane::element_width]) for lane in range(element_width))


def byte_unshuffle(payload: bytes, element_width: int) -> bytes:
    """Reverse :func:`byte_shuffle` exactly."""

    if element_width <= 0:
        raise ValueError("element width must be positive")
    if len(payload) % element_width:
        raise ValueError("payload length must be divisible by element width")
    element_count = len(payload) // element_width
    restored = bytearray(len(payload))
    source = memoryview(payload)
    for lane in range(element_width):
        start = lane * element_count
        restored[lane::element_width] = source[start : start + element_count]
    return bytes(restored)


@dataclass(frozen=True, slots=True)
class CompressionResult:
    mode: CompressionMode
    payload: bytes
    raw_bytes: int
    encoded_bytes: int
    encode_ns: int
    checksum: str
    element_width: int
    codec: str
    codec_version: str

    @property
    def ratio(self) -> float:
        return self.raw_bytes / self.encoded_bytes if self.encoded_bytes else 1.0


def compress_lossless(
    payload: bytes, *, mode: CompressionMode, element_width: int
) -> CompressionResult:
    """Compress bytes losslessly, or preserve them verbatim when disabled."""

    if not isinstance(payload, bytes):
        raise TypeError("compression payload must be bytes")
    if element_width <= 0:
        raise ValueError("element width must be positive")
    if len(payload) % element_width:
        raise ValueError("payload length must be divisible by element width")
    started = time.perf_counter_ns()
    if mode == "none":
        encoded = payload
        codec = "none"
    elif mode == "byte_shuffle_fast_codec":
        encoded = zlib.compress(byte_shuffle(payload, element_width), level=1)
        codec = "zlib-level-1"
    else:
        raise ValueError(f"unsupported compression mode {mode!r}")
    return CompressionResult(
        mode=mode,
        payload=encoded,
        raw_bytes=len(payload),
        encoded_bytes=len(encoded),
        encode_ns=time.perf_counter_ns() - started,
        checksum=hashlib.sha256(payload).hexdigest(),
        element_width=element_width,
        codec=codec,
        codec_version=zlib.ZLIB_VERSION,
    )


@dataclass(frozen=True, slots=True)
class DecompressionResult:
    payload: bytes
    decode_ns: int


def decompress_lossless(result: CompressionResult) -> DecompressionResult:
    """Decode and verify a lossless compression result."""

    if result.raw_bytes < 0 or result.encoded_bytes < 0:
        raise ValueError("compression byte counts cannot be negative")
    if result.encoded_bytes != len(result.payload):
        raise ValueError("compressed payload length does not match its metadata")
    if result.element_width <= 0 or result.raw_bytes % result.element_width:
        raise ValueError("invalid compression element geometry")
    started = time.perf_counter_ns()
    if result.mode == "none":
        if result.codec != "none":
            raise ValueError("uncompressed payload declares a compressed codec")
        restored = result.payload
    elif result.mode == "byte_shuffle_fast_codec":
        if result.codec != "zlib-level-1":
            raise ValueError("compressed payload declares the wrong codec")
        try:
            shuffled = zlib.decompress(result.payload)
        except zlib.error as exc:
            raise ValueError(f"invalid compressed payload: {exc}") from exc
        restored = byte_unshuffle(shuffled, result.element_width)
    else:
        raise ValueError(f"unsupported compression mode {result.mode!r}")
    if len(restored) != result.raw_bytes:
        raise ValueError("decompressed activation has the wrong byte length")
    if hashlib.sha256(restored).hexdigest() != result.checksum:
        raise ValueError("decompressed activation checksum mismatch")
    return DecompressionResult(payload=restored, decode_ns=time.perf_counter_ns() - started)


@dataclass(frozen=True, slots=True)
class CompressionDecision:
    selected_mode: CompressionMode
    raw_payload_bytes: int
    compressed_payload_bytes: int
    bandwidth_bps: float | None
    rtt_ms: float
    queue_delay_ms: float
    encode_ns: int
    decode_ns: int
    raw_transfer_ns: int
    compressed_transfer_ns: int
    predicted_net_saving_ns: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class AdaptiveCompressionController:
    """Select compression only when measured expected latency is lower."""

    def __init__(self, *, minimum_saving_ns: int = 50_000) -> None:
        if minimum_saving_ns < 0:
            raise ValueError("minimum compression saving cannot be negative")
        self.minimum_saving_ns = minimum_saving_ns

    def decide(
        self,
        *,
        raw_payload_bytes: int,
        compressed_payload_bytes: int,
        encode_ns: int,
        decode_ns: int,
        bandwidth_bps: float | None,
        rtt_ms: float,
        queue_delay_ms: float,
    ) -> CompressionDecision:
        if raw_payload_bytes < 0 or compressed_payload_bytes < 0:
            raise ValueError("payload byte counts cannot be negative")
        if encode_ns < 0 or decode_ns < 0:
            raise ValueError("codec timings cannot be negative")
        if bandwidth_bps is not None and bandwidth_bps <= 0:
            raise ValueError("measured bandwidth must be positive when supplied")
        if rtt_ms < 0 or queue_delay_ms < 0:
            raise ValueError("latency inputs cannot be negative")
        bytes_per_second = bandwidth_bps / 8.0 if bandwidth_bps is not None else float("inf")
        shared_latency_ns = int((rtt_ms / 2.0 + queue_delay_ms) * 1e6)
        raw_serialisation_ns = (
            0
            if bytes_per_second == float("inf")
            else int(raw_payload_bytes / bytes_per_second * 1e9)
        )
        compressed_serialisation_ns = (
            0
            if bytes_per_second == float("inf")
            else int(compressed_payload_bytes / bytes_per_second * 1e9)
        )
        raw_transfer_ns = shared_latency_ns + raw_serialisation_ns
        compressed_transfer_ns = (
            shared_latency_ns + compressed_serialisation_ns + encode_ns + decode_ns
        )
        saving = raw_transfer_ns - compressed_transfer_ns
        selected = saving > self.minimum_saving_ns and compressed_payload_bytes < raw_payload_bytes
        return CompressionDecision(
            selected_mode="byte_shuffle_fast_codec" if selected else "none",
            raw_payload_bytes=raw_payload_bytes,
            compressed_payload_bytes=compressed_payload_bytes,
            bandwidth_bps=bandwidth_bps,
            rtt_ms=rtt_ms,
            queue_delay_ms=queue_delay_ms,
            encode_ns=encode_ns,
            decode_ns=decode_ns,
            raw_transfer_ns=raw_transfer_ns,
            compressed_transfer_ns=compressed_transfer_ns,
            predicted_net_saving_ns=saving,
            reason=(
                "measured transfer saving exceeds codec cost"
                if selected
                else "codec cost is not lower than measured transfer saving"
            ),
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
