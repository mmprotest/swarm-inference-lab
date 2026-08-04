"""Raw tensor-byte transport with optional reversible byte-shuffle compression."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import torch

from swarm_inference.experiments.experiment_011.compression import (
    AdaptiveCompressionController,
    CompressionDecision,
    CompressionMode,
    CompressionResult,
    compress_lossless,
    decompress_lossless,
)

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
        value = asdict(self)
        value.pop("payload")
        decision = value.get("compression_decision")
        if self.compression_decision is not None:
            value["compression_decision"] = self.compression_decision.to_dict()
        elif decision is None:
            value["compression_decision"] = None
        return value


def tensor_raw_bytes(tensor: torch.Tensor) -> bytes:
    source = tensor.detach()
    if source.dtype not in _DTYPE_NAMES:
        raise ValueError(f"unsupported tensor dtype {source.dtype}")
    # PyTorch may report singleton views as contiguous even when their sole
    # stride is inherited from a larger tensor (for example logits[:, -1]).
    # Reinterpreting such an int64 view as bytes is rejected because its last
    # stride is not one.  Copy into newly allocated row-major CPU storage so
    # the wire representation is canonical for every shape and dtype.
    canonical = torch.empty(tuple(source.shape), dtype=source.dtype, device="cpu")
    canonical.copy_(source)
    return canonical.view(torch.uint8).numpy().tobytes(order="C")


def pack_tensor(
    tensor: torch.Tensor,
    *,
    requested_mode: CompressionMode | str,
    adaptive_inputs: AdaptiveTransportInputs | None = None,
    controller: AdaptiveCompressionController | None = None,
) -> PackedTensor:
    contiguous = tensor.detach().contiguous()
    dtype_name = _DTYPE_NAMES.get(contiguous.dtype)
    if dtype_name is None:
        raise ValueError(f"unsupported tensor dtype {contiguous.dtype}")
    raw = tensor_raw_bytes(contiguous)
    width = _ELEMENT_WIDTHS[dtype_name]
    decision = None
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
        selected_mode: CompressionMode = decision.selected_mode
        compressed = (
            candidate
            if selected_mode == "byte_shuffle_fast_codec"
            else compress_lossless(raw, mode="none", element_width=width)
        )
    else:
        selected_mode = str(requested_mode)
        if selected_mode not in {"none", "byte_shuffle_fast_codec"}:
            raise ValueError(f"unsupported requested compression mode {requested_mode!r}")
        compressed = compress_lossless(raw, mode=selected_mode, element_width=width)  # type: ignore[arg-type]
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


def unpack_tensor(payload: bytes, metadata: dict[str, Any]) -> tuple[torch.Tensor, int]:
    dtype_name = str(metadata["dtype"])
    if dtype_name not in _NAME_DTYPES:
        raise ValueError(f"unsupported wire tensor dtype {dtype_name!r}")
    if str(metadata.get("byte_order")) != "little":
        raise ValueError("only explicit little-endian tensor payloads are accepted")
    shape = tuple(int(value) for value in metadata["shape"])
    element_count = 1
    for dimension in shape:
        if dimension < 0:
            raise ValueError("tensor dimensions cannot be negative")
        element_count *= dimension
    width = _ELEMENT_WIDTHS[dtype_name]
    raw_bytes = int(metadata["raw_bytes"])
    if raw_bytes != element_count * width:
        raise ValueError("tensor shape and raw byte length disagree")
    mode = str(metadata["compression_mode"])
    if mode not in {"none", "byte_shuffle_fast_codec"}:
        raise ValueError("unsupported tensor compression mode")
    compressed = CompressionResult(
        mode=mode,  # type: ignore[arg-type]
        payload=payload,
        raw_bytes=raw_bytes,
        encoded_bytes=len(payload),
        encode_ns=int(metadata.get("encode_ns", 0)),
        checksum=str(metadata["raw_checksum"]),
        element_width=width,
        codec=str(metadata.get("codec", "none")),
        codec_version=str(metadata.get("codec_version", "unknown")),
    )
    restored = decompress_lossless(compressed)
    if hashlib.sha256(restored.payload).hexdigest() != str(metadata["raw_checksum"]):
        raise ValueError("restored tensor failed its raw checksum")
    # bytearray gives torch a writable, owned buffer; clone detaches the tensor
    # from that temporary storage before this function returns.
    byte_tensor = torch.frombuffer(bytearray(restored.payload), dtype=torch.uint8)
    tensor = byte_tensor.view(_NAME_DTYPES[dtype_name]).reshape(shape).clone()
    return tensor, restored.decode_ns
