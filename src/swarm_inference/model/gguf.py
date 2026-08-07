"""Streaming GGUF metadata and tensor inventory for canonical model discovery."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO


class GGUFParseError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GGUFTensorInfo:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    offset: int
    byte_size: int
    dtype: str
    quantization: str


@dataclass(frozen=True, slots=True)
class GGUFInventory:
    path: Path
    version: int
    metadata: dict[str, Any]
    tensors: tuple[GGUFTensorInfo, ...]
    data_offset: int
    file_size: int

    @property
    def tensor_bytes(self) -> int:
        return sum(item.byte_size for item in self.tensors)


_GGML_TYPES: dict[int, tuple[int, int, str, str]] = {
    0: (4, 1, "float32", "none"),
    1: (2, 1, "float16", "none"),
    2: (18, 32, "quantized", "Q4_0"),
    3: (20, 32, "quantized", "Q4_1"),
    6: (22, 32, "quantized", "Q5_0"),
    7: (24, 32, "quantized", "Q5_1"),
    8: (34, 32, "quantized", "Q8_0"),
    9: (36, 32, "quantized", "Q8_1"),
    10: (84, 256, "quantized", "Q2_K"),
    11: (110, 256, "quantized", "Q3_K"),
    12: (144, 256, "quantized", "Q4_K"),
    13: (176, 256, "quantized", "Q5_K"),
    14: (210, 256, "quantized", "Q6_K"),
    15: (292, 256, "quantized", "Q8_K"),
    16: (66, 256, "quantized", "IQ2_XXS"),
    17: (74, 256, "quantized", "IQ2_XS"),
    18: (98, 256, "quantized", "IQ3_XXS"),
    19: (50, 256, "quantized", "IQ1_S"),
    20: (18, 32, "quantized", "IQ4_NL"),
    21: (110, 256, "quantized", "IQ3_S"),
    22: (82, 256, "quantized", "IQ2_S"),
    23: (136, 256, "quantized", "IQ4_XS"),
    24: (1, 1, "int8", "none"),
    25: (2, 1, "int16", "none"),
    26: (4, 1, "int32", "none"),
    27: (8, 1, "int64", "none"),
    28: (8, 1, "float64", "none"),
    29: (56, 256, "quantized", "IQ1_M"),
    30: (2, 1, "bfloat16", "none"),
}
_SCALAR_FORMATS: dict[int, str] = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


def _read_exact(handle: BinaryIO, length: int) -> bytes:
    value = handle.read(length)
    if len(value) != length:
        raise GGUFParseError("unexpected end of GGUF file")
    return value


def _unpack(handle: BinaryIO, fmt: str) -> Any:
    return struct.unpack(fmt, _read_exact(handle, struct.calcsize(fmt)))[0]


def _read_string(handle: BinaryIO, *, keep: bool = True) -> str | None:
    length = int(_unpack(handle, "<Q"))
    if length > 1 << 30:
        raise GGUFParseError("GGUF string length is implausibly large")
    raw = _read_exact(handle, length)
    if not keep:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GGUFParseError("GGUF string is not valid UTF-8") from exc


def _read_value(handle: BinaryIO, value_type: int, *, keep: bool) -> Any:
    if value_type == 8:
        return _read_string(handle, keep=keep)
    if value_type == 9:
        element_type = int(_unpack(handle, "<I"))
        count = int(_unpack(handle, "<Q"))
        if count > 100_000_000:
            raise GGUFParseError("GGUF metadata array is implausibly large")
        values = [_read_value(handle, element_type, keep=keep) for _ in range(count)]
        return values if keep else None
    fmt = _SCALAR_FORMATS.get(value_type)
    if fmt is None:
        raise GGUFParseError(f"unsupported GGUF metadata value type {value_type}")
    value = _unpack(handle, fmt)
    return value if keep else None


def _keep_metadata(key: str) -> bool:
    if key.startswith("tokenizer."):
        return False
    return key.startswith("general.") or key.endswith(
        (
            "block_count",
            "embedding_length",
            "expert_count",
            "expert_used_count",
            "expert_shared_count",
            "context_length",
            "alignment",
        )
    )


def _tensor_storage(shape: tuple[int, ...], ggml_type: int) -> tuple[int, str, str]:
    description = _GGML_TYPES.get(ggml_type)
    if description is None:
        raise GGUFParseError(f"unsupported GGML tensor type {ggml_type}")
    storage_bytes, block_elements, dtype, quantization = description
    blocks = (math.prod(shape) + block_elements - 1) // block_elements
    return blocks * storage_bytes, dtype, quantization


def inspect_gguf(path: Path) -> GGUFInventory:
    """Read GGUF headers and tensor descriptors without loading tensor contents."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    file_size = resolved.stat().st_size
    with resolved.open("rb") as handle:
        if _read_exact(handle, 4) != b"GGUF":
            raise GGUFParseError("file does not begin with GGUF magic")
        version = int(_unpack(handle, "<I"))
        if version not in {2, 3}:
            raise GGUFParseError(f"unsupported GGUF version {version}")
        tensor_count = int(_unpack(handle, "<Q"))
        metadata_count = int(_unpack(handle, "<Q"))
        if tensor_count > 100_000_000 or metadata_count > 10_000_000:
            raise GGUFParseError("GGUF header counts are implausibly large")
        metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = _read_string(handle)
            assert key is not None
            value_type = int(_unpack(handle, "<I"))
            keep = _keep_metadata(key)
            value = _read_value(handle, value_type, keep=keep)
            if keep:
                metadata[key] = value
        raw_tensors: list[tuple[str, tuple[int, ...], int, int]] = []
        for _ in range(tensor_count):
            name = _read_string(handle)
            assert name is not None
            dimensions = int(_unpack(handle, "<I"))
            if not 0 < dimensions <= 8:
                raise GGUFParseError(f"invalid tensor rank {dimensions} for {name}")
            shape = tuple(int(_unpack(handle, "<Q")) for _ in range(dimensions))
            ggml_type = int(_unpack(handle, "<I"))
            offset = int(_unpack(handle, "<Q"))
            raw_tensors.append((name, shape, ggml_type, offset))
        alignment = int(metadata.get("general.alignment", 32))
        if alignment <= 0 or alignment & (alignment - 1):
            raise GGUFParseError("GGUF alignment must be a positive power of two")
        data_offset = (handle.tell() + alignment - 1) // alignment * alignment
    tensors: list[GGUFTensorInfo] = []
    for name, shape, ggml_type, offset in raw_tensors:
        byte_size, dtype, quantization = _tensor_storage(shape, ggml_type)
        if data_offset + offset + byte_size > file_size:
            raise GGUFParseError(f"tensor {name} extends beyond the GGUF file")
        tensors.append(
            GGUFTensorInfo(name, shape, ggml_type, offset, byte_size, dtype, quantization)
        )
    return GGUFInventory(
        path=resolved,
        version=version,
        metadata=metadata,
        tensors=tuple(tensors),
        data_offset=data_offset,
        file_size=file_size,
    )


__all__ = ["GGUFInventory", "GGUFParseError", "GGUFTensorInfo", "inspect_gguf"]
