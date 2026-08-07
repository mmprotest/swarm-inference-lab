"""Header-only Safetensors inspection for canonical model discovery."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO


class SafetensorsHeaderError(ValueError):
    pass


SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def normalize_safetensors_dtype(value: Any) -> str:
    text = str(value).upper().replace("SAFETENSORS.", "")
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text


@dataclass(frozen=True, slots=True)
class SafetensorsTensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    data_offset: int
    byte_size: int


@dataclass(frozen=True, slots=True)
class SafetensorsIndexInventory:
    """Validated, compact identity for one sharded Safetensors checkpoint."""

    source: str
    tensor_count: int
    shard_names: tuple[str, ...]
    tensors_per_shard: tuple[tuple[str, int], ...]
    mapping_sha256: str
    declared_total_size: int | None = None


def inspect_safetensors_stream(
    handle: BinaryIO,
    *,
    file_size: int,
    source: str,
) -> tuple[SafetensorsTensorInfo, ...]:
    """Read one local or range-backed Safetensors header without tensor data."""

    if file_size < 8:
        raise SafetensorsHeaderError("Safetensors file is shorter than its header prefix")
    prefix = handle.read(8)
    if len(prefix) != 8:
        raise SafetensorsHeaderError(f"Safetensors header prefix is truncated in {source}")
    header_length = struct.unpack("<Q", prefix)[0]
    if header_length <= 0 or header_length > min(file_size - 8, 1 << 30):
        raise SafetensorsHeaderError("Safetensors header length is invalid")
    encoded = handle.read(header_length)
    if len(encoded) != header_length:
        raise SafetensorsHeaderError(f"Safetensors header is truncated in {source}")
    try:
        payload: Any = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetensorsHeaderError("Safetensors header is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise SafetensorsHeaderError("Safetensors header must be an object")
    data_start = 8 + header_length
    tensors: list[SafetensorsTensorInfo] = []
    for name, raw in payload.items():
        if name == "__metadata__":
            continue
        if not isinstance(raw, dict):
            raise SafetensorsHeaderError(f"tensor {name!r} metadata is not an object")
        dtype = raw.get("dtype")
        shape = raw.get("shape")
        offsets = raw.get("data_offsets")
        if (
            not isinstance(dtype, str)
            or not isinstance(shape, list)
            or not all(isinstance(value, int) and value > 0 for value in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
        ):
            raise SafetensorsHeaderError(f"tensor {name!r} has malformed metadata")
        start, end = offsets
        if start < 0 or end < start or data_start + end > file_size:
            raise SafetensorsHeaderError(f"tensor {name!r} data range is invalid")
        tensors.append(
            SafetensorsTensorInfo(
                name=str(name),
                shape=tuple(shape),
                dtype=dtype,
                data_offset=data_start + start,
                byte_size=end - start,
            )
        )
    return tuple(tensors)


def inspect_safetensors(path: Path) -> tuple[SafetensorsTensorInfo, ...]:
    """Read and validate one Safetensors header without mapping tensor data."""

    resolved = path.expanduser().resolve()
    file_size = resolved.stat().st_size
    with resolved.open("rb") as handle:
        return inspect_safetensors_stream(
            handle,
            file_size=file_size,
            source=str(resolved),
        )


def inspect_safetensors_index_payload(
    payload: Any,
    *,
    source: str,
    available_files: Collection[str],
) -> SafetensorsIndexInventory:
    """Validate an index and retain a bounded, content-addressed shard map summary."""

    if not isinstance(payload, dict):
        raise SafetensorsHeaderError("Safetensors index must be an object")
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise SafetensorsHeaderError("Safetensors index has no non-empty weight_map")
    normalized_available = {name.replace("\\", "/") for name in available_files}
    normalized: dict[str, str] = {}
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise SafetensorsHeaderError("Safetensors index contains an invalid tensor name")
        if not isinstance(shard_name, str) or not shard_name:
            raise SafetensorsHeaderError(f"Safetensors index target for {tensor_name!r} is invalid")
        shard = shard_name.replace("\\", "/")
        parts = shard.split("/")
        if (
            shard.startswith("/")
            or ":" in parts[0]
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise SafetensorsHeaderError(
                f"Safetensors index target for {tensor_name!r} is not repository-relative"
            )
        if not shard.casefold().endswith(".safetensors"):
            raise SafetensorsHeaderError(
                f"Safetensors index target for {tensor_name!r} is not a Safetensors shard"
            )
        if shard not in normalized_available:
            raise SafetensorsHeaderError(f"Safetensors index references missing shard {shard!r}")
        normalized[tensor_name] = shard
    canonical = json.dumps(
        sorted(normalized.items()),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    counts = Counter(normalized.values())
    metadata = payload.get("metadata")
    declared_total = metadata.get("total_size") if isinstance(metadata, dict) else None
    if declared_total is not None:
        if isinstance(declared_total, bool) or not isinstance(declared_total, int | float):
            raise SafetensorsHeaderError("Safetensors index metadata.total_size is invalid")
        if (
            isinstance(declared_total, float)
            and (not math.isfinite(declared_total) or not declared_total.is_integer())
        ) or declared_total < 0:
            raise SafetensorsHeaderError("Safetensors index metadata.total_size is invalid")
        declared_total = int(declared_total)
    return SafetensorsIndexInventory(
        source=source,
        tensor_count=len(normalized),
        shard_names=tuple(sorted(counts)),
        tensors_per_shard=tuple(sorted(counts.items())),
        mapping_sha256=hashlib.sha256(canonical).hexdigest(),
        declared_total_size=declared_total,
    )


def inspect_safetensors_index(
    path: Path,
    *,
    available_files: Collection[str],
) -> SafetensorsIndexInventory:
    resolved = path.expanduser().resolve()
    try:
        payload: Any = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetensorsHeaderError("Safetensors index is not valid UTF-8 JSON") from exc
    return inspect_safetensors_index_payload(
        payload,
        source=str(resolved),
        available_files=available_files,
    )


def local_safetensors_weight_map(model_path: Path) -> dict[str, str]:
    """Return the exact tensor-to-shard map without assuming a model family."""

    root = model_path.expanduser().resolve()
    indexes = sorted(root.glob("*.safetensors.index.json"))
    if indexes:
        if len(indexes) > 1:
            raise SafetensorsHeaderError(
                f"multiple Safetensors index files found: {[path.name for path in indexes]}"
            )
        try:
            payload: Any = json.loads(indexes[0].read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SafetensorsHeaderError("Safetensors index is not valid UTF-8 JSON") from exc
        available = tuple(path.name for path in root.iterdir() if path.is_file())
        inspect_safetensors_index_payload(
            payload,
            source=indexes[0].name,
            available_files=available,
        )
        weight_map = payload["weight_map"]
        return {str(name): str(file).replace("\\", "/") for name, file in weight_map.items()}
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise SafetensorsHeaderError(f"no Safetensors weights found under {root}")
    mapping: dict[str, str] = {}
    for file in files:
        for tensor in inspect_safetensors(file):
            if tensor.name in mapping:
                raise SafetensorsHeaderError(
                    f"tensor {tensor.name!r} occurs in multiple source files"
                )
            mapping[tensor.name] = file.name
    return mapping


__all__ = [
    "SAFETENSORS_DTYPE_BYTES",
    "SafetensorsHeaderError",
    "SafetensorsIndexInventory",
    "SafetensorsTensorInfo",
    "inspect_safetensors",
    "inspect_safetensors_index",
    "inspect_safetensors_index_payload",
    "inspect_safetensors_stream",
    "local_safetensors_weight_map",
    "normalize_safetensors_dtype",
]
