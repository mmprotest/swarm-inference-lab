"""Canonical raw-tensor transport for stage-ring payloads.

Unlike :mod:`swarm_inference.protocol.tensor_codec`, this codec does not build
an envelope.  It returns raw or losslessly compressed tensor bytes plus
semantic metadata for a surrounding stage-ring message.
"""

from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import torch

from swarm_inference.transport.compression import (
    AdaptiveCompressionController,
    CompressionDecision,
    CompressionMode,
    CompressionResult,
    compress_lossless,
    decompress_lossless,
)

RequestedCompressionMode = Literal["none", "byte_shuffle_fast_codec", "adaptive"]

_DTYPE_NAMES = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.uint8: "uint8",
}
_NAME_DTYPES = {name: dtype for dtype, name in _DTYPE_NAMES.items()}
_ELEMENT_WIDTHS = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "int64": 8,
    "int32": 4,
    "uint8": 1,
}
SUPPORTED_DTYPES = frozenset(_NAME_DTYPES)


@dataclass(frozen=True, slots=True)
class AdaptiveTransportInputs:
    bandwidth_bps: float | None
    rtt_ms: float
    queue_delay_ms: float


@dataclass(frozen=True, slots=True)
class PackedTensor:
    payload: bytes
    shape: tuple[int, ...]
    dtype: str
    byte_order: str
    compression_mode: CompressionMode
    raw_bytes: int
    encoded_bytes: int
    raw_checksum: str
    encode_ns: int
    decode_trial_ns: int
    codec: str
    codec_version: str
    compression_decision: CompressionDecision | None

    def attributes(self) -> dict[str, Any]:
        """Return JSON-native semantic metadata, excluding the raw payload."""

        value = asdict(self)
        value.pop("payload")
        value["shape"] = list(self.shape)
        value["compression_decision"] = (
            self.compression_decision.to_dict() if self.compression_decision is not None else None
        )
        return value


def _canonical_little_endian(raw: bytes, element_width: int) -> bytes:
    if sys.byteorder == "little" or element_width == 1:
        return raw
    restored = bytearray(len(raw))
    for offset in range(0, len(raw), element_width):
        restored[offset : offset + element_width] = raw[offset : offset + element_width][::-1]
    return bytes(restored)


def tensor_raw_bytes(tensor: torch.Tensor) -> bytes:
    """Return detached, row-major, little-endian bytes for any tensor view."""

    source = tensor.detach()
    dtype_name = _DTYPE_NAMES.get(source.dtype)
    if dtype_name is None:
        raise ValueError(f"unsupported tensor dtype {source.dtype}")
    # A fresh CPU allocation canonicalises non-contiguous tensors and singleton
    # views whose inherited stride is not byte-viewable by PyTorch.
    canonical = torch.empty(tuple(source.shape), dtype=source.dtype, device="cpu")
    canonical.copy_(source)
    raw = canonical.view(torch.uint8).numpy().tobytes(order="C")
    return _canonical_little_endian(raw, _ELEMENT_WIDTHS[dtype_name])


def pack_tensor(
    tensor: torch.Tensor,
    *,
    requested_mode: RequestedCompressionMode | str,
    adaptive_inputs: AdaptiveTransportInputs | None = None,
    controller: AdaptiveCompressionController | None = None,
) -> PackedTensor:
    """Pack a tensor without enabling compression implicitly."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("stage tensor must be a torch.Tensor")
    contiguous = tensor.detach().contiguous()
    dtype_name = _DTYPE_NAMES.get(contiguous.dtype)
    if dtype_name is None:
        raise ValueError(f"unsupported tensor dtype {contiguous.dtype}")
    raw = tensor_raw_bytes(contiguous)
    width = _ELEMENT_WIDTHS[dtype_name]
    decision: CompressionDecision | None = None
    decode_trial_ns = 0
    if requested_mode == "adaptive":
        if adaptive_inputs is None:
            raise ValueError("adaptive compression requires measured network inputs")
        candidate = compress_lossless(raw, mode="byte_shuffle_fast_codec", element_width=width)
        trial = decompress_lossless(candidate)
        decode_trial_ns = trial.decode_ns
        if trial.payload != raw:
            raise ValueError("adaptive compression trial was not bitwise lossless")
        selector = controller or AdaptiveCompressionController()
        decision = selector.decide(
            raw_payload_bytes=len(raw),
            compressed_payload_bytes=len(candidate.payload),
            encode_ns=candidate.encode_ns,
            decode_ns=trial.decode_ns,
            bandwidth_bps=adaptive_inputs.bandwidth_bps,
            rtt_ms=adaptive_inputs.rtt_ms,
            queue_delay_ms=adaptive_inputs.queue_delay_ms,
        )
        selected_mode = decision.selected_mode
        compressed = (
            candidate
            if selected_mode == "byte_shuffle_fast_codec"
            else compress_lossless(raw, mode="none", element_width=width)
        )
    else:
        if requested_mode not in {"none", "byte_shuffle_fast_codec"}:
            raise ValueError(f"unsupported requested compression mode {requested_mode!r}")
        selected_mode = cast(CompressionMode, requested_mode)
        compressed = compress_lossless(raw, mode=selected_mode, element_width=width)
    return PackedTensor(
        payload=compressed.payload,
        shape=tuple(int(value) for value in contiguous.shape),
        dtype=dtype_name,
        byte_order="little",
        compression_mode=compressed.mode,
        raw_bytes=compressed.raw_bytes,
        encoded_bytes=compressed.encoded_bytes,
        raw_checksum=compressed.checksum,
        encode_ns=compressed.encode_ns,
        decode_trial_ns=decode_trial_ns,
        codec=compressed.codec,
        codec_version=compressed.codec_version,
        compression_decision=decision,
    )


def _metadata_int(metadata: dict[str, Any], key: str) -> int:
    try:
        value = metadata[key]
    except KeyError as exc:
        raise ValueError(f"tensor metadata is missing {key!r}") from exc
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"tensor metadata {key!r} must be an integer")
    return value


def _metadata_shape(metadata: dict[str, Any]) -> tuple[int, ...]:
    try:
        value = metadata["shape"]
    except KeyError as exc:
        raise ValueError("tensor metadata is missing 'shape'") from exc
    if not isinstance(value, (list, tuple)):
        raise ValueError("tensor shape metadata must be an array")
    shape: list[int] = []
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise ValueError("tensor dimensions must be integers")
        if dimension < 0:
            raise ValueError("tensor dimensions cannot be negative")
        shape.append(dimension)
    return tuple(shape)


def unpack_tensor(payload: bytes, metadata: dict[str, Any]) -> tuple[torch.Tensor, int]:
    """Validate metadata, restore raw bytes, and reconstruct an owned tensor."""

    if not isinstance(payload, bytes):
        raise TypeError("stage tensor payload must be bytes")
    if not isinstance(metadata, dict):
        raise ValueError("tensor metadata must be an object")
    try:
        dtype_name = metadata["dtype"]
    except KeyError as exc:
        raise ValueError("tensor metadata is missing 'dtype'") from exc
    if not isinstance(dtype_name, str) or dtype_name not in _NAME_DTYPES:
        raise ValueError(f"unsupported wire tensor dtype {dtype_name!r}")
    if metadata.get("byte_order") != "little":
        raise ValueError("only explicit little-endian tensor payloads are accepted")
    shape = _metadata_shape(metadata)
    element_count = math.prod(shape)
    width = _ELEMENT_WIDTHS[dtype_name]
    raw_bytes = _metadata_int(metadata, "raw_bytes")
    encoded_bytes = _metadata_int(metadata, "encoded_bytes")
    encode_ns = _metadata_int(metadata, "encode_ns")
    if raw_bytes < 0 or encoded_bytes < 0 or encode_ns < 0:
        raise ValueError("tensor byte counts and timings cannot be negative")
    if raw_bytes != element_count * width:
        raise ValueError("tensor shape and raw byte length disagree")
    if encoded_bytes != len(payload):
        raise ValueError("tensor encoded byte length disagrees with its payload")
    mode = metadata.get("compression_mode")
    if mode not in {"none", "byte_shuffle_fast_codec"}:
        raise ValueError("unsupported tensor compression mode")
    checksum = metadata.get("raw_checksum")
    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum)
    ):
        raise ValueError("tensor raw checksum is malformed")
    codec = metadata.get("codec")
    codec_version = metadata.get("codec_version")
    if not isinstance(codec, str) or not isinstance(codec_version, str):
        raise ValueError("tensor codec metadata is malformed")
    compressed = CompressionResult(
        mode=cast(CompressionMode, mode),
        payload=payload,
        raw_bytes=raw_bytes,
        encoded_bytes=encoded_bytes,
        encode_ns=encode_ns,
        checksum=checksum,
        element_width=width,
        codec=codec,
        codec_version=codec_version,
    )
    restored = decompress_lossless(compressed)
    if hashlib.sha256(restored.payload).hexdigest() != checksum:
        raise ValueError("restored tensor failed its raw checksum")
    if not restored.payload:
        return torch.empty(shape, dtype=_NAME_DTYPES[dtype_name]), restored.decode_ns
    # bytearray gives torch a writable buffer; clone detaches the result from it.
    byte_tensor = torch.frombuffer(bytearray(restored.payload), dtype=torch.uint8)
    tensor = byte_tensor.view(_NAME_DTYPES[dtype_name]).reshape(shape).clone()
    return tensor, restored.decode_ns


__all__ = [
    "SUPPORTED_DTYPES",
    "AdaptiveTransportInputs",
    "PackedTensor",
    "RequestedCompressionMode",
    "pack_tensor",
    "tensor_raw_bytes",
    "unpack_tensor",
]
