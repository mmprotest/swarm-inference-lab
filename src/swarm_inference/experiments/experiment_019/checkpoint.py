"""Byte-exact Kimi K3 checkpoint census and direct sub-tensor loading.

The loader in this module never asks Safetensors to materialize a source tensor
before slicing it.  It maps the source file read-only and copies only the
assigned row-major ranges.  Column slices are represented in the audit as a
strided set of concrete byte ranges.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.model.mxfp4 import MXFP4_GROUP_SIZE, MXFP4Tensor
from swarm_inference.model.safetensors import SAFETENSORS_DTYPE_BYTES

SMALL_PARAMETER_REVIEW_BYTES = 32 * 1024 * 1024
_LAYER = re.compile(r"^language_model\.model\.layers\.(\d+)\.")
_EXPERT = re.compile(r"\.block_sparse_moe\.experts\.(\d+)\.")
_DTYPES: dict[str, np.dtype[Any]] = {
    "BF16": np.dtype("<u2"),
    "F32": np.dtype("<f4"),
    "U8": np.dtype("u1"),
    "I8": np.dtype("i1"),
    "F16": np.dtype("<f2"),
    "I16": np.dtype("<i2"),
    "U16": np.dtype("<u2"),
    "I32": np.dtype("<i4"),
    "U32": np.dtype("<u4"),
    "I64": np.dtype("<i8"),
    "U64": np.dtype("<u8"),
    "F64": np.dtype("<f8"),
    "BOOL": np.dtype("?"),
}


@dataclass(frozen=True, slots=True)
class TensorRecord:
    name: str
    file: str
    dtype: str
    shape: tuple[int, ...]
    data_offset: int
    byte_size: int
    layer_id: int | None
    expert_id: int | None
    role: str

    @property
    def element_size(self) -> int:
        return SAFETENSORS_DTYPE_BYTES[self.dtype]


@dataclass(frozen=True, slots=True)
class ShardRange:
    axis: int
    start: int
    stop: int
    total: int

    @property
    def fraction(self) -> float:
        return (self.stop - self.start) / self.total


def balanced_range(total: int, degree: int, index: int, *, quantum: int = 1) -> ShardRange:
    """Partition ``total`` into balanced, quantum-aligned contiguous ranges."""

    if total <= 0 or degree <= 0 or not 0 <= index < degree or total % quantum:
        raise ValueError("invalid balanced range geometry")
    groups = total // quantum
    start_group = (groups * index) // degree
    stop_group = (groups * (index + 1)) // degree
    return ShardRange(0, start_group * quantum, stop_group * quantum, total)


def tensor_role(name: str) -> str:
    if ".block_sparse_moe.experts." in name:
        return "routed_expert"
    if ".block_sparse_moe.shared_experts." in name:
        return "shared_expert"
    if ".block_sparse_moe.gate." in name:
        return "router"
    if ".block_sparse_moe.routed_expert_" in name:
        return "latent_moe_projection"
    if ".self_attn." in name:
        return "attention"
    if ".mlp." in name:
        return "dense_mlp"
    if "embed_tokens" in name:
        return "embedding"
    if name == "language_model.lm_head.weight":
        return "lm_head"
    if name.startswith("vision_tower."):
        return "vision"
    if name.startswith("mm_projector."):
        return "multimodal_projector"
    if "attn_res" in name or "_res_" in name:
        return "attnres"
    if name == "language_model.model.norm.weight":
        return "final_norm"
    return "other"


class CheckpointCatalog:
    """Read all Safetensors headers without reading tensor payloads."""

    def __init__(self, checkpoint: Path) -> None:
        self.root = checkpoint.expanduser().resolve()
        self.index_path = self.root / "model.safetensors.index.json"
        if not self.index_path.is_file():
            raise FileNotFoundError(self.index_path)
        raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        weight_map = raw.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("checkpoint index has no weight_map")
        self.weight_map = {str(name): str(file) for name, file in weight_map.items()}
        metadata = raw.get("metadata")
        self.declared_total_size = (
            int(metadata["total_size"])
            if isinstance(metadata, dict) and "total_size" in metadata
            else None
        )
        self.index_sha256 = hashlib.sha256(self.index_path.read_bytes()).hexdigest()
        self._headers: dict[str, tuple[int, dict[str, Any]]] = {}
        self._records: dict[str, TensorRecord] | None = None

    def _header(self, file: str) -> tuple[int, dict[str, Any]]:
        cached = self._headers.get(file)
        if cached is not None:
            return cached
        path = self.root / file
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError(f"truncated Safetensors prefix: {path}")
            header_size = struct.unpack("<Q", prefix)[0]
            encoded = handle.read(header_size)
        header = json.loads(encoded.decode("utf-8"))
        if not isinstance(header, dict):
            raise ValueError(f"invalid Safetensors header: {path}")
        value = (8 + int(header_size), header)
        self._headers[file] = value
        return value

    def records(self) -> dict[str, TensorRecord]:
        if self._records is not None:
            return self._records
        records: dict[str, TensorRecord] = {}
        for file in sorted(set(self.weight_map.values())):
            payload_offset, header = self._header(file)
            for name, raw in header.items():
                if name == "__metadata__":
                    continue
                if self.weight_map.get(name) != file or not isinstance(raw, dict):
                    raise ValueError(f"header/index mismatch for {name!r}")
                dtype = str(raw["dtype"])
                shape = tuple(int(value) for value in raw["shape"])
                start, stop = (int(value) for value in raw["data_offsets"])
                expected = math.prod(shape) * SAFETENSORS_DTYPE_BYTES[dtype]
                if stop - start != expected:
                    raise ValueError(f"tensor byte mismatch for {name}")
                layer_match = _LAYER.match(name)
                expert_match = _EXPERT.search(name)
                records[name] = TensorRecord(
                    name=name,
                    file=file,
                    dtype=dtype,
                    shape=shape,
                    data_offset=payload_offset + start,
                    byte_size=stop - start,
                    layer_id=int(layer_match.group(1)) if layer_match else None,
                    expert_id=int(expert_match.group(1)) if expert_match else None,
                    role=tensor_role(name),
                )
        missing = set(self.weight_map).difference(records)
        extra = set(records).difference(self.weight_map)
        if missing or extra:
            raise ValueError(f"checkpoint census mismatch: missing={len(missing)} extra={len(extra)}")
        total = sum(record.byte_size for record in records.values())
        if self.declared_total_size is not None and total != self.declared_total_size:
            raise ValueError(
                f"checkpoint payload total {total} != declared {self.declared_total_size}"
            )
        self._records = records
        return records

    def record(self, name: str) -> TensorRecord:
        try:
            return self.records()[name]
        except KeyError as exc:
            raise KeyError(f"checkpoint has no tensor {name}") from exc

    def lfs_hashes(self) -> dict[str, str]:
        """Return Hugging Face LFS SHA-256 identities recorded at download time."""

        directory = self.root / ".cache" / "huggingface" / "download"
        hashes: dict[str, str] = {}
        for file in sorted(set(self.weight_map.values())):
            metadata = directory / f"{file}.metadata"
            if not metadata.is_file():
                continue
            fields = metadata.read_text(encoding="utf-8").strip().split("\n", 1)[0].split("|")
            if len(fields) >= 2 and re.fullmatch(r"[0-9a-f]{64}", fields[1]):
                hashes[file] = fields[1]
        return hashes

    def census_rows(self) -> list[dict[str, Any]]:
        return [
            {
                **asdict(record),
                "shape": "x".join(str(value) for value in record.shape),
                "checkpoint_index_sha256": self.index_sha256,
            }
            for record in sorted(self.records().values(), key=lambda item: item.name)
        ]


class DirectShardLoader:
    """Load only an assigned tensor slice and retain a byte-range audit."""

    def __init__(self, catalog: CheckpointCatalog) -> None:
        self.catalog = catalog
        self.audit: list[dict[str, Any]] = []

    def _mapping(self, record: TensorRecord) -> np.memmap[Any, Any]:
        dtype = _DTYPES.get(record.dtype)
        if dtype is None:
            raise ValueError(f"unsupported direct-load dtype {record.dtype}")
        return np.memmap(
            self.catalog.root / record.file,
            mode="r",
            dtype=dtype,
            offset=record.data_offset,
            shape=record.shape,
        )

    def load(
        self,
        name: str,
        *,
        worker_id: str,
        purpose: str,
        axis: int | None = None,
        start: int | None = None,
        stop: int | None = None,
        allow_reviewed_full_tensor: bool = False,
    ) -> np.ndarray:
        record = self.catalog.record(name)
        mapped = self._mapping(record)
        if axis is None:
            if not allow_reviewed_full_tensor or record.byte_size > SMALL_PARAMETER_REVIEW_BYTES:
                raise ValueError(
                    f"full tensor load forbidden for {name} ({record.byte_size} bytes)"
                )
            result = np.ascontiguousarray(mapped)
            range_encoding: dict[str, Any] = {
                "kind": "contiguous",
                "offset": record.data_offset,
                "length": record.byte_size,
            }
            shard = {"axis": None, "start": 0, "stop": record.shape[0]}
        else:
            if axis not in (0, 1) or start is None or stop is None:
                raise ValueError("direct slices require axis 0/1 and explicit bounds")
            if axis >= len(record.shape) or not 0 <= start < stop <= record.shape[axis]:
                raise ValueError(f"invalid slice for {name}: axis={axis} {start}:{stop}")
            slices = [slice(None)] * len(record.shape)
            slices[axis] = slice(start, stop)
            result = np.ascontiguousarray(mapped[tuple(slices)])
            if axis == 0:
                row_bytes = math.prod(record.shape[1:]) * record.element_size
                range_encoding = {
                    "kind": "contiguous",
                    "offset": record.data_offset + start * row_bytes,
                    "length": (stop - start) * row_bytes,
                }
            else:
                if len(record.shape) != 2:
                    raise ValueError("axis-1 range encoding currently requires a matrix")
                row_stride = record.shape[1] * record.element_size
                range_encoding = {
                    "kind": "strided",
                    "base_offset": record.data_offset + start * record.element_size,
                    "range_count": record.shape[0],
                    "range_length": (stop - start) * record.element_size,
                    "stride": row_stride,
                    "last_range_stop": record.data_offset
                    + (record.shape[0] - 1) * row_stride
                    + stop * record.element_size,
                }
            shard = {"axis": axis, "start": start, "stop": stop}
        if result.nbytes <= 0 or result.nbytes > record.byte_size:
            raise RuntimeError(f"invalid direct shard byte count for {name}")
        self.audit.append(
            {
                "worker_id": worker_id,
                "purpose": purpose,
                "file": record.file,
                "tensor": name,
                "dtype": record.dtype,
                "source_shape": list(record.shape),
                "requested_shard": shard,
                "byte_ranges": range_encoding,
                "bytes_read": result.nbytes,
                "source_tensor_bytes": record.byte_size,
                "source_fraction_read": result.nbytes / record.byte_size,
                "full_source_tensor_materialized": result.nbytes == record.byte_size,
                "read_mechanism": "read_only_memmap_then_assigned_slice_copy",
            }
        )
        return result

    def reviewed_small(self, name: str, *, worker_id: str, purpose: str) -> np.ndarray:
        return self.load(
            name,
            worker_id=worker_id,
            purpose=purpose,
            allow_reviewed_full_tensor=True,
        )

    def expert_stripe(
        self,
        *,
        layer: int,
        expert: int,
        degree: int,
        stripe: int,
        worker_id: str,
    ) -> tuple[MXFP4Tensor, MXFP4Tensor, MXFP4Tensor]:
        if degree < 2:
            raise ValueError("a direct expert stripe requires degree >= 2")
        prefix = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}"
        intermediate = 3072
        latent = 3584
        partition = balanced_range(
            intermediate, degree, stripe, quantum=MXFP4_GROUP_SIZE
        )

        def row_matrix(stem: str) -> MXFP4Tensor:
            packed_name = f"{prefix}.{stem}.weight_packed"
            scale_name = f"{prefix}.{stem}.weight_scale"
            packed = self.load(
                packed_name,
                worker_id=worker_id,
                purpose="routed_expert_row_stripe",
                axis=0,
                start=partition.start,
                stop=partition.stop,
            )
            scales = self.load(
                scale_name,
                worker_id=worker_id,
                purpose="routed_expert_row_scale_stripe",
                axis=0,
                start=partition.start,
                stop=partition.stop,
            )
            return MXFP4Tensor(
                packed=packed,
                scales=scales,
                input_dimension=latent,
                output_dimension=partition.stop - partition.start,
            )

        down_packed = self.load(
            f"{prefix}.w2.weight_packed",
            worker_id=worker_id,
            purpose="routed_expert_down_column_stripe",
            axis=1,
            start=partition.start // 2,
            stop=partition.stop // 2,
        )
        down_scales = self.load(
            f"{prefix}.w2.weight_scale",
            worker_id=worker_id,
            purpose="routed_expert_down_scale_stripe",
            axis=1,
            start=partition.start // MXFP4_GROUP_SIZE,
            stop=partition.stop // MXFP4_GROUP_SIZE,
        )
        down = MXFP4Tensor(
            packed=down_packed,
            scales=down_scales,
            input_dimension=partition.stop - partition.start,
            output_dimension=latent,
        )
        gate = row_matrix("w1")
        up = row_matrix("w3")
        expected_fraction = 1.0 / degree
        actual = (gate.byte_size + up.byte_size + down.byte_size) / (
            self.catalog.record(f"{prefix}.w1.weight_packed").byte_size
            + self.catalog.record(f"{prefix}.w1.weight_scale").byte_size
            + self.catalog.record(f"{prefix}.w2.weight_packed").byte_size
            + self.catalog.record(f"{prefix}.w2.weight_scale").byte_size
            + self.catalog.record(f"{prefix}.w3.weight_packed").byte_size
            + self.catalog.record(f"{prefix}.w3.weight_scale").byte_size
        )
        if not math.isclose(actual, expected_fraction, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("expert stripe bytes do not match native partition")
        return gate, up, down


__all__ = [
    "CheckpointCatalog",
    "DirectShardLoader",
    "SMALL_PARAMETER_REVIEW_BYTES",
    "ShardRange",
    "TensorRecord",
    "balanced_range",
    "tensor_role",
]
