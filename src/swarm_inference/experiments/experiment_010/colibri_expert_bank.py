"""Exact-byte worker banks for native Colibri OLMoE experts.

The converter deliberately operates on safetensors byte ranges.  Whole expert
tensors are copied verbatim.  Native microshards copy gate/up rows and down
columns without dequantizing or recomputing the original F32 row scales.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import tempfile
import time
from collections.abc import Iterable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

CONVERTER_VERSION = "experiment-010-colibri-expert-bank-v1"
SAFETENSORS_HEADER_LIMIT = 100 * 1024 * 1024
COPY_CHUNK_BYTES = 8 * 1024 * 1024
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
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


@dataclass(frozen=True)
class TensorLocation:
    """One validated tensor payload in a safetensors file."""

    name: str
    path: Path
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int
    payload_offset: int

    @property
    def nbytes(self) -> int:
        return self.data_end - self.data_start

    @property
    def absolute_offset(self) -> int:
        return self.payload_offset + self.data_start


@dataclass(frozen=True)
class ByteRange:
    """A source byte range used to assemble a destination tensor."""

    path: Path
    offset: int
    length: int
    label: str


@dataclass(frozen=True)
class OutputTensor:
    """A safetensors tensor assembled from ordered exact source ranges."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    ranges: tuple[ByteRange, ...]

    @property
    def nbytes(self) -> int:
        return sum(item.length for item in self.ranges)


def _product(values: Sequence[int]) -> int:
    product = 1
    for value in values:
        if value < 0:
            raise ValueError("safetensors shapes cannot contain negative dimensions")
        product *= value
    return product


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        encoded_length = handle.read(8)
        if len(encoded_length) != 8:
            raise ValueError(f"{path} has a truncated safetensors header length")
        header_length = struct.unpack("<Q", encoded_length)[0]
        if header_length < 2 or header_length > SAFETENSORS_HEADER_LIMIT:
            raise ValueError(f"{path} has an invalid safetensors header length")
        encoded_header = handle.read(header_length)
        if len(encoded_header) != header_length:
            raise ValueError(f"{path} has a truncated safetensors header")
    try:
        header = json.loads(encoded_header.rstrip(b" \t\r\n"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} has invalid safetensors JSON") from error
    if not isinstance(header, dict):
        raise ValueError(f"{path} safetensors header is not an object")
    return 8 + header_length, header


def scan_safetensors(root: Path) -> dict[str, TensorLocation]:
    """Return all tensor locations after validating shape, dtype, and bounds."""

    model_root = root.expanduser().resolve()
    files = sorted(model_root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors files found in {model_root}")
    tensors: dict[str, TensorLocation] = {}
    for path in files:
        payload_offset, header = _read_safetensors_header(path)
        payload_bytes = path.stat().st_size - payload_offset
        for name, descriptor in header.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ValueError(f"duplicate safetensors tensor {name}")
            if not isinstance(descriptor, dict):
                raise ValueError(f"invalid safetensors descriptor for {name}")
            dtype = descriptor.get("dtype")
            shape = descriptor.get("shape")
            offsets = descriptor.get("data_offsets")
            if (
                dtype not in _DTYPE_BYTES
                or not isinstance(shape, list)
                or not all(isinstance(value, int) for value in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(value, int) for value in offsets)
            ):
                raise ValueError(f"invalid safetensors descriptor for {name}")
            start, end = offsets
            expected = _product(shape) * _DTYPE_BYTES[dtype]
            if start < 0 or end < start or end > payload_bytes or end - start != expected:
                raise ValueError(f"invalid safetensors payload bounds for {name}")
            tensors[name] = TensorLocation(
                name=name,
                path=path,
                dtype=dtype,
                shape=tuple(shape),
                data_start=start,
                data_end=end,
                payload_offset=payload_offset,
            )
    return tensors


def _load_config(root: Path) -> dict[str, Any]:
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing Colibri model config: {config_path}")
    value = json.loads(config_path.read_text(encoding="utf-8"))
    required = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_experts")
    if not isinstance(value, dict) or any(
        not isinstance(value.get(name), int) or value[name] <= 0 for name in required
    ):
        raise ValueError("Colibri config does not contain valid OLMoE dimensions")
    return value


def model_fingerprint(root: Path) -> str:
    """Hash exact config and safetensors container bytes in stable path order."""

    model_root = root.expanduser().resolve()
    paths = [model_root / "config.json", *sorted(model_root.glob("*.safetensors"))]
    if not paths[0].is_file() or len(paths) == 1:
        raise FileNotFoundError(f"incomplete Colibri model container at {model_root}")
    digest = hashlib.sha256()
    digest.update(b"COLIBRI-EXACT-CONTAINER-V1\0")
    for path in paths:
        relative = path.relative_to(model_root).as_posix().encode("utf-8")
        digest.update(struct.pack(">I", len(relative)))
        digest.update(relative)
        digest.update(struct.pack(">Q", path.stat().st_size))
        with path.open("rb") as handle:
            while chunk := handle.read(COPY_CHUNK_BYTES):
                digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _full_range(tensor: TensorLocation, label: str) -> ByteRange:
    return ByteRange(tensor.path, tensor.absolute_offset, tensor.nbytes, label)


def _read_range(handle: BinaryIO, item: ByteRange) -> bytes:
    handle.seek(item.offset)
    value = handle.read(item.length)
    if len(value) != item.length:
        raise OSError(f"short read from {item.path} at {item.offset}")
    return value


def _ranges_hash(ranges: Iterable[ByteRange]) -> str:
    handles: dict[Path, BinaryIO] = {}
    digest = hashlib.sha256()
    try:
        for item in ranges:
            handle = handles.get(item.path)
            if handle is None:
                handle = item.path.open("rb")
                handles[item.path] = handle
            handle.seek(item.offset)
            remaining = item.length
            while remaining:
                chunk = handle.read(min(remaining, COPY_CHUNK_BYTES))
                if not chunk:
                    raise OSError(f"short read from {item.path} at {item.offset}")
                digest.update(chunk)
                remaining -= len(chunk)
    finally:
        for handle in handles.values():
            handle.close()
    return "sha256:" + digest.hexdigest()


def _tensor_header(tensors: Sequence[OutputTensor]) -> tuple[bytes, dict[str, tuple[int, int]]]:
    header: dict[str, Any] = {
        "__metadata__": {
            "format": "pt",
            "converter": CONVERTER_VERSION,
            "byte_semantics": "native_colibri_exact",
        }
    }
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for tensor in tensors:
        expected = _product(tensor.shape) * _DTYPE_BYTES[tensor.dtype]
        if tensor.nbytes != expected:
            raise ValueError(
                f"output tensor {tensor.name} has {tensor.nbytes} bytes; expected {expected}"
            )
        offsets[tensor.name] = (cursor, cursor + tensor.nbytes)
        header[tensor.name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [cursor, cursor + tensor.nbytes],
        }
        cursor += tensor.nbytes
    encoded = json.dumps(header, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    if len(encoded) > SAFETENSORS_HEADER_LIMIT:
        raise ValueError("generated safetensors header is too large")
    return encoded, offsets


def _write_safetensors(
    path: Path, tensors: Sequence[OutputTensor]
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    encoded_header, relative_offsets = _tensor_header(tensors)
    hashes: dict[str, str] = {}
    absolute_offsets: dict[str, dict[str, int]] = {}
    with ExitStack() as stack:
        source_handles: dict[Path, BinaryIO] = {}
        output = stack.enter_context(path.open("wb"))
        output.write(struct.pack("<Q", len(encoded_header)))
        output.write(encoded_header)
        payload_offset = output.tell()
        for tensor in tensors:
            digest = hashlib.sha256()
            for item in tensor.ranges:
                handle = source_handles.get(item.path)
                if handle is None:
                    handle = stack.enter_context(item.path.open("rb"))
                    source_handles[item.path] = handle
                handle.seek(item.offset)
                remaining = item.length
                while remaining:
                    chunk = handle.read(min(remaining, COPY_CHUNK_BYTES))
                    if not chunk:
                        raise OSError(f"short read from {item.path} at {item.offset}")
                    output.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
            hashes[tensor.name] = "sha256:" + digest.hexdigest()
            start, end = relative_offsets[tensor.name]
            absolute_offsets[tensor.name] = {
                "payload_start": start,
                "payload_end": end,
                "file_start": payload_offset + start,
                "file_end": payload_offset + end,
            }
    return hashes, absolute_offsets


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_sums(root: Path, filenames: Sequence[str]) -> None:
    lines = [f"{_sha256_file(root / name)}  {name}" for name in filenames]
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="ascii")


def _created_at(value: str | None) -> str:
    if value:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("creation timestamp must include a UTC offset")
        return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch is not None:
        return datetime.fromtimestamp(int(epoch), UTC).isoformat().replace("+00:00", "Z")
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _expert_tensor_names(layer_id: int, expert_id: int) -> tuple[str, str]:
    prefix = f"model.layers.{layer_id}.mlp.experts.{expert_id}"
    return f"{prefix}.merged_weight", f"{prefix}.qs"


def _validate_expert(
    tensors: dict[str, TensorLocation],
    *,
    layer_id: int,
    expert_id: int,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[TensorLocation, TensorLocation]:
    weight_name, scale_name = _expert_tensor_names(layer_id, expert_id)
    try:
        weight = tensors[weight_name]
        scales = tensors[scale_name]
    except KeyError as error:
        raise KeyError(f"source container has no native expert tensor {error.args[0]}") from error
    expected_weight = 3 * hidden_size * intermediate_size
    expected_scales = 2 * intermediate_size + hidden_size
    if weight.dtype != "I8" or weight.nbytes != expected_weight:
        raise ValueError(f"{weight_name} is not the expected native merged int8 tensor")
    if scales.dtype != "F32" or scales.nbytes != expected_scales * 4:
        raise ValueError(f"{scale_name} is not the expected native F32 scale tensor")
    return weight, scales


def _scale_summary(scales: TensorLocation) -> dict[str, Any]:
    with scales.path.open("rb") as handle:
        raw = _read_range(
            handle,
            ByteRange(scales.path, scales.absolute_offset, scales.nbytes, "scales"),
        )
    values = np.frombuffer(raw, dtype="<f4")
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{scales.name} contains invalid scale values")
    return {
        "count": int(values.size),
        "dtype": "F32",
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }


def _source_descriptor(tensor: TensorLocation, content_hash: str) -> dict[str, Any]:
    return {
        "name": tensor.name,
        "source_file": tensor.path.name,
        "source_file_start": tensor.absolute_offset,
        "source_file_end": tensor.absolute_offset + tensor.nbytes,
        "shape": list(tensor.shape),
        "dtype": tensor.dtype,
        "bytes": tensor.nbytes,
        "sha256": content_hash,
    }


def _validate_ids(
    entries: Iterable[tuple[int, int]], *, num_layers: int, num_experts: int
) -> list[tuple[int, int]]:
    normalized = sorted(set(entries))
    if not normalized:
        raise ValueError("a worker bank must own at least one expert")
    for layer_id, expert_id in normalized:
        if not 0 <= layer_id < num_layers or not 0 <= expert_id < num_experts:
            raise ValueError(f"expert {layer_id}:{expert_id} is outside the model dimensions")
    return normalized


def _prepare_target(source: Path, output: Path) -> tuple[Path, Path]:
    source = source.expanduser().resolve()
    target = output.expanduser().resolve()
    if source == target or source in target.parents:
        raise ValueError("worker banks must not be created inside the source model container")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing worker bank {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.building-", dir=target.parent))
    return target, stage


def _finish_target(stage: Path, target: Path) -> Path:
    deadline = time.monotonic() + 5.0
    delay = 0.01
    while True:
        try:
            stage.replace(target)
            return target
        except PermissionError as error:
            # OneDrive, indexing, and antivirus filters can briefly retain a
            # handle after the final checksum file closes on Windows. Retrying
            # the same atomic rename is safe while the destination is absent;
            # never turn this into an overwrite retry.
            if target.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing worker bank {target}"
                ) from error
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, 0.25)


def build_expert_bank(
    source_model: Path,
    output_directory: Path,
    *,
    worker_id: str,
    owned_experts: Iterable[tuple[int, int]],
    source_model_fingerprint: str | None = None,
    creation_timestamp: str | None = None,
) -> Path:
    """Create a bank containing only the requested native whole experts."""

    source = source_model.expanduser().resolve()
    config = _load_config(source)
    ownership = _validate_ids(
        owned_experts,
        num_layers=config["num_hidden_layers"],
        num_experts=config["num_experts"],
    )
    fingerprint = source_model_fingerprint or model_fingerprint(source)
    target, stage = _prepare_target(source, output_directory)
    try:
        source_tensors = scan_safetensors(source)
        outputs: list[OutputTensor] = []
        expert_records: list[dict[str, Any]] = []
        source_hashes: dict[str, str] = {}
        total_bytes = 0
        for layer_id, expert_id in ownership:
            weight, scales = _validate_expert(
                source_tensors,
                layer_id=layer_id,
                expert_id=expert_id,
                hidden_size=config["hidden_size"],
                intermediate_size=config["intermediate_size"],
            )
            weight_range = _full_range(weight, "merged_weight")
            scale_range = _full_range(scales, "row_scales")
            weight_hash = _ranges_hash((weight_range,))
            scale_hash = _ranges_hash((scale_range,))
            source_hashes[weight.name] = weight_hash
            source_hashes[scales.name] = scale_hash
            outputs.extend(
                (
                    OutputTensor(weight.name, weight.dtype, weight.shape, (weight_range,)),
                    OutputTensor(scales.name, scales.dtype, scales.shape, (scale_range,)),
                )
            )
            total_bytes += weight.nbytes + scales.nbytes
            expert_records.append(
                {
                    "layer_id": layer_id,
                    "expert_id": expert_id,
                    "weight": _source_descriptor(weight, weight_hash),
                    "scales": _source_descriptor(scales, scale_hash),
                    "quantization": "native_colibri_int8_per_row_f32_scale",
                    "quantization_scales": _scale_summary(scales),
                    "expert_bytes": weight.nbytes + scales.nbytes,
                }
            )
        outputs.sort(key=lambda value: value.name)
        destination_hashes, offsets = _write_safetensors(stage / "experts.safetensors", outputs)
        if destination_hashes != source_hashes:
            raise RuntimeError("destination whole-expert tensor hashes differ from source bytes")
        ownership_document = {
            "schema_version": "1.0",
            "worker_id": worker_id,
            "model_fingerprint": fingerprint,
            "owned_experts": [
                {"layer_id": layer_id, "expert_id": expert_id} for layer_id, expert_id in ownership
            ],
            "owned_microshards": [],
        }
        manifest = {
            "schema_version": "1.0",
            "bank_kind": "native_colibri_whole_experts",
            "converter_version": CONVERTER_VERSION,
            "creation_timestamp": _created_at(creation_timestamp),
            "worker_id": worker_id,
            "source_model_path": str(source),
            "source_model_fingerprint": fingerprint,
            "fingerprint_algorithm": "sha256 exact config and safetensors container bytes v1",
            "hidden_size": config["hidden_size"],
            "intermediate_size": config["intermediate_size"],
            "num_hidden_layers": config["num_hidden_layers"],
            "num_experts": config["num_experts"],
            "quantization_format": "native_colibri_merged_int8_f32_row_scales",
            "owned_layer_expert_pairs": [list(value) for value in ownership],
            "owned_expert_count": len(ownership),
            "total_expert_bytes": total_bytes,
            "source_tensor_hashes": source_hashes,
            "destination_tensor_hashes": destination_hashes,
            "tensor_byte_offsets": offsets,
            "experts": expert_records,
        }
        _write_json(stage / "ownership.json", ownership_document)
        _write_json(stage / "manifest.json", manifest)
        _write_sums(stage, ("experts.safetensors", "manifest.json", "ownership.json"))
        return _finish_target(stage, target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def build_coordinator_container(
    source_model: Path,
    output_directory: Path,
    *,
    source_model_fingerprint: str | None = None,
    creation_timestamp: str | None = None,
) -> Path:
    """Build a dense-only coordinator container from exact source tensor bytes."""

    source = source_model.expanduser().resolve()
    config = _load_config(source)
    fingerprint = source_model_fingerprint or model_fingerprint(source)
    target, stage = _prepare_target(source, output_directory)
    try:
        source_tensors = scan_safetensors(source)
        dense = {
            name: tensor for name, tensor in source_tensors.items() if ".mlp.experts." not in name
        }
        routed = {
            name: tensor for name, tensor in source_tensors.items() if ".mlp.experts." in name
        }
        if not dense or not routed:
            raise ValueError("coordinator conversion requires dense and routed expert tensors")
        outputs = [
            OutputTensor(name, tensor.dtype, tensor.shape, (_full_range(tensor, name),))
            for name, tensor in sorted(dense.items())
        ]
        source_dense_hashes = {
            name: _ranges_hash((_full_range(tensor, name),))
            for name, tensor in sorted(dense.items())
        }
        destination_hashes, offsets = _write_safetensors(stage / "model.safetensors", outputs)
        if destination_hashes != source_dense_hashes:
            raise RuntimeError("coordinator dense tensor hashes differ from source bytes")
        copied_metadata: dict[str, str] = {}
        for source_path in sorted(source.glob("*.json")):
            shutil.copy2(source_path, stage / source_path.name)
            copied_metadata[source_path.name] = _sha256_file(source_path)
        if "config.json" not in copied_metadata or "tokenizer.json" not in copied_metadata:
            raise FileNotFoundError(
                "coordinator container requires exact config and tokenizer JSON"
            )
        excluded = [
            {
                "name": name,
                "source_file": tensor.path.name,
                "dtype": tensor.dtype,
                "shape": list(tensor.shape),
                "bytes": tensor.nbytes,
                "sha256": _ranges_hash((_full_range(tensor, name),)),
            }
            for name, tensor in sorted(routed.items())
        ]
        manifest = {
            "schema_version": "experiment-010-dense-coordinator-container-v1",
            "container_kind": "native_colibri_dense_coordinator",
            "converter_version": CONVERTER_VERSION,
            "creation_timestamp": _created_at(creation_timestamp),
            "source_model_path": str(source),
            "source_model_fingerprint": fingerprint,
            "hidden_size": config["hidden_size"],
            "intermediate_size": config["intermediate_size"],
            "num_hidden_layers": config["num_hidden_layers"],
            "num_experts": config["num_experts"],
            "coordinator_owned_routed_expert_count": 0,
            "coordinator_owned_routed_expert_bytes": 0,
            "excluded_routed_tensor_count": len(routed),
            "excluded_routed_expert_bytes": sum(tensor.nbytes for tensor in routed.values()),
            "dense_tensor_count": len(dense),
            "dense_tensor_bytes": sum(tensor.nbytes for tensor in dense.values()),
            "source_dense_tensor_hashes": source_dense_hashes,
            "destination_dense_tensor_hashes": destination_hashes,
            "tensor_byte_offsets": offsets,
            "copied_metadata_sha256": copied_metadata,
            "excluded_routed_tensors": excluded,
        }
        _write_json(stage / "coordinator_manifest.json", manifest)
        filenames = sorted(path.name for path in stage.iterdir() if path.is_file())
        _write_sums(stage, filenames)
        return _finish_target(stage, target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def verify_coordinator_container(root: Path) -> dict[str, Any]:
    """Verify that a coordinator container has exact dense bytes and no routed bytes."""

    container = root.expanduser().resolve()
    manifest = json.loads((container / "coordinator_manifest.json").read_text(encoding="utf-8"))
    tensors = scan_safetensors(container)
    if any(".mlp.experts." in name for name in tensors):
        raise ValueError("coordinator container contains routed expert tensors")
    recorded = manifest.get("destination_dense_tensor_hashes")
    if not isinstance(recorded, dict) or set(recorded) != set(tensors):
        raise ValueError("coordinator dense tensor inventory differs from its manifest")
    for name, tensor in tensors.items():
        if recorded[name] != _ranges_hash((_full_range(tensor, name),)):
            raise ValueError(f"coordinator dense tensor checksum mismatch: {name}")
    sums: dict[str, str] = {}
    for line in (container / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines():
        digest, filename = line.split("  ", 1)
        sums[filename] = digest
    runtime_generated = {".coli_usage", "hot_pinned.bin"}
    expected = {
        path.name
        for path in container.iterdir()
        if path.is_file() and path.name not in runtime_generated
    } - {"SHA256SUMS.txt"}
    if set(sums) != expected:
        raise ValueError("coordinator checksum inventory is incomplete")
    for filename, digest in sums.items():
        if _sha256_file(container / filename) != digest:
            raise ValueError(f"coordinator file checksum mismatch: {filename}")
    if manifest.get("coordinator_owned_routed_expert_bytes") != 0:
        raise ValueError("coordinator manifest claims routed expert ownership")
    return {
        "valid": True,
        "source_model_fingerprint": manifest["source_model_fingerprint"],
        "dense_tensor_count": len(tensors),
        "dense_tensor_bytes": sum(tensor.nbytes for tensor in tensors.values()),
        "coordinator_owned_routed_expert_count": 0,
        "coordinator_owned_routed_expert_bytes": 0,
        "excluded_routed_expert_bytes": manifest["excluded_routed_expert_bytes"],
    }


def _microshard_ranges(
    weight: TensorLocation,
    scales: TensorLocation,
    *,
    hidden_size: int,
    intermediate_size: int,
    hidden_start: int,
    hidden_end: int,
) -> tuple[tuple[ByteRange, ...], tuple[ByteRange, ...], dict[str, str]]:
    width = hidden_end - hidden_start
    gate_offset = weight.absolute_offset + hidden_start * hidden_size
    up_offset = (
        weight.absolute_offset + intermediate_size * hidden_size + hidden_start * hidden_size
    )
    down_base = weight.absolute_offset + 2 * intermediate_size * hidden_size
    weight_ranges: list[ByteRange] = [
        ByteRange(weight.path, gate_offset, width * hidden_size, "gate_rows"),
        ByteRange(weight.path, up_offset, width * hidden_size, "up_rows"),
    ]
    weight_ranges.extend(
        ByteRange(
            weight.path,
            down_base + row * intermediate_size + hidden_start,
            width,
            f"down_row_{row}_columns",
        )
        for row in range(hidden_size)
    )
    scale_ranges = (
        ByteRange(scales.path, scales.absolute_offset + hidden_start * 4, width * 4, "gate_scales"),
        ByteRange(
            scales.path,
            scales.absolute_offset + (intermediate_size + hidden_start) * 4,
            width * 4,
            "up_scales",
        ),
        ByteRange(
            scales.path,
            scales.absolute_offset + 2 * intermediate_size * 4,
            hidden_size * 4,
            "down_scales",
        ),
    )
    component_hashes = {
        "gate_weight": _ranges_hash((weight_ranges[0],)),
        "up_weight": _ranges_hash((weight_ranges[1],)),
        "down_weight": _ranges_hash(weight_ranges[2:]),
        "gate_scales": _ranges_hash((scale_ranges[0],)),
        "up_scales": _ranges_hash((scale_ranges[1],)),
        "down_scales": _ranges_hash((scale_ranges[2],)),
    }
    return tuple(weight_ranges), scale_ranges, component_hashes


def build_microshard_bank(
    source_model: Path,
    output_directory: Path,
    *,
    worker_id: str,
    owned_microshards: Iterable[tuple[int, int, int, int]],
    layout: str = "planner",
    source_model_fingerprint: str | None = None,
    creation_timestamp: str | None = None,
) -> Path:
    """Create an exact native gate/up/down microshard worker bank."""

    if layout not in {"equal", "asymmetric", "planner"}:
        raise ValueError("microshard layout must be equal, asymmetric, or planner")
    source = source_model.expanduser().resolve()
    config = _load_config(source)
    normalized = sorted(set(owned_microshards))
    if not normalized:
        raise ValueError("a microshard bank must own at least one native slice")
    seen_ranges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for layer_id, expert_id, start, end in normalized:
        _validate_ids(
            ((layer_id, expert_id),),
            num_layers=config["num_hidden_layers"],
            num_experts=config["num_experts"],
        )
        if start < 0 or end <= start or end > config["intermediate_size"]:
            raise ValueError(f"microshard {layer_id}:{expert_id}:{start}:{end} is out of range")
        prior = seen_ranges.setdefault((layer_id, expert_id), [])
        if any(start < prior_end and prior_start < end for prior_start, prior_end in prior):
            raise ValueError(f"overlapping microshard ownership for {layer_id}:{expert_id}")
        prior.append((start, end))
    fingerprint = source_model_fingerprint or model_fingerprint(source)
    target, stage = _prepare_target(source, output_directory)
    try:
        source_tensors = scan_safetensors(source)
        outputs: list[OutputTensor] = []
        records: list[dict[str, Any]] = []
        total_bytes = 0
        selected_hashes: dict[str, str] = {}
        for layer_id, expert_id, start, end in normalized:
            weight, scales = _validate_expert(
                source_tensors,
                layer_id=layer_id,
                expert_id=expert_id,
                hidden_size=config["hidden_size"],
                intermediate_size=config["intermediate_size"],
            )
            weight_ranges, scale_ranges, component_hashes = _microshard_ranges(
                weight,
                scales,
                hidden_size=config["hidden_size"],
                intermediate_size=config["intermediate_size"],
                hidden_start=start,
                hidden_end=end,
            )
            prefix = f"model.layers.{layer_id}.mlp.experts.{expert_id}.microshards.{start}_{end}"
            weight_name = f"{prefix}.merged_weight"
            scale_name = f"{prefix}.qs"
            width = end - start
            outputs.extend(
                (
                    OutputTensor(
                        weight_name,
                        "I8",
                        (3 * config["hidden_size"] * width,),
                        weight_ranges,
                    ),
                    OutputTensor(
                        scale_name,
                        "F32",
                        (2 * width + config["hidden_size"],),
                        scale_ranges,
                    ),
                )
            )
            selected_hashes[weight_name] = _ranges_hash(weight_ranges)
            selected_hashes[scale_name] = _ranges_hash(scale_ranges)
            shard_bytes = (
                3 * config["hidden_size"] * width + (2 * width + config["hidden_size"]) * 4
            )
            total_bytes += shard_bytes
            records.append(
                {
                    "layer_id": layer_id,
                    "expert_id": expert_id,
                    "hidden_start": start,
                    "hidden_end": end,
                    "hidden_width": width,
                    "source_weight_tensor": weight.name,
                    "source_scale_tensor": scales.name,
                    "source_weight_file": weight.path.name,
                    "source_scale_file": scales.path.name,
                    "source_weight_file_start": weight.absolute_offset,
                    "source_scale_file_start": scales.absolute_offset,
                    "destination_weight_tensor": weight_name,
                    "destination_scale_tensor": scale_name,
                    "component_hashes": component_hashes,
                    "selected_weight_sha256": selected_hashes[weight_name],
                    "selected_scale_sha256": selected_hashes[scale_name],
                    "native_weight_values": 3 * config["hidden_size"] * width,
                    "native_scale_count": 2 * width + config["hidden_size"],
                    "shard_bytes": shard_bytes,
                    "quantization": "native_colibri_int8_original_f32_row_scales",
                    "down_scale_policy": "preserve_original_per_output_row_scale",
                }
            )
        outputs.sort(key=lambda value: value.name)
        destination_hashes, offsets = _write_safetensors(stage / "shards.safetensors", outputs)
        if destination_hashes != selected_hashes:
            raise RuntimeError("destination microshard hashes differ from selected native bytes")
        owned = [
            {
                "layer_id": layer_id,
                "expert_id": expert_id,
                "hidden_start": start,
                "hidden_end": end,
            }
            for layer_id, expert_id, start, end in normalized
        ]
        ownership_document = {
            "schema_version": "1.0",
            "worker_id": worker_id,
            "model_fingerprint": fingerprint,
            "owned_experts": [],
            "owned_microshards": owned,
        }
        manifest = {
            "schema_version": "1.0",
            "bank_kind": "native_colibri_microshards",
            "converter_version": CONVERTER_VERSION,
            "creation_timestamp": _created_at(creation_timestamp),
            "worker_id": worker_id,
            "source_model_path": str(source),
            "source_model_fingerprint": fingerprint,
            "fingerprint_algorithm": "sha256 exact config and safetensors container bytes v1",
            "hidden_size": config["hidden_size"],
            "intermediate_size": config["intermediate_size"],
            "num_hidden_layers": config["num_hidden_layers"],
            "num_experts": config["num_experts"],
            "layout": layout,
            "quantization_format": "native_colibri_merged_int8_f32_row_scales",
            "owned_microshards": owned,
            "total_expert_bytes": total_bytes,
            "selected_source_tensor_hashes": selected_hashes,
            "destination_tensor_hashes": destination_hashes,
            "tensor_byte_offsets": offsets,
            "shards": records,
        }
        _write_json(stage / "ownership.json", ownership_document)
        _write_json(stage / "manifest.json", manifest)
        _write_sums(stage, ("shards.safetensors", "manifest.json", "ownership.json"))
        return _finish_target(stage, target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def verify_bank(root: Path) -> dict[str, Any]:
    """Fail closed unless file sums and recorded tensor hashes all reconcile."""

    bank = root.expanduser().resolve()
    manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
    sums: dict[str, str] = {}
    for line in (bank / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines():
        digest, filename = line.split("  ", 1)
        if filename in sums:
            raise ValueError(f"duplicate checksum entry for {filename}")
        sums[filename] = digest
    expected_files = {"manifest.json", "ownership.json"}
    tensor_filename = (
        "experts.safetensors"
        if manifest.get("bank_kind") == "native_colibri_whole_experts"
        else "shards.safetensors"
    )
    expected_files.add(tensor_filename)
    if set(sums) != expected_files:
        raise ValueError("bank checksum inventory is incomplete")
    for filename, digest in sums.items():
        if _sha256_file(bank / filename) != digest:
            raise ValueError(f"bank file checksum mismatch: {filename}")
    tensors = scan_safetensors(bank)
    recorded = manifest.get("destination_tensor_hashes")
    if not isinstance(recorded, dict) or set(recorded) != set(tensors):
        raise ValueError("bank tensor inventory differs from its manifest")
    for name, tensor in tensors.items():
        actual = _ranges_hash((_full_range(tensor, name),))
        if recorded[name] != actual:
            raise ValueError(f"bank tensor checksum mismatch: {name}")
    ownership = json.loads((bank / "ownership.json").read_text(encoding="utf-8"))
    if ownership.get("worker_id") != manifest.get("worker_id"):
        raise ValueError("bank worker identity differs between manifest and ownership")
    return {
        "valid": True,
        "bank_kind": manifest["bank_kind"],
        "worker_id": manifest["worker_id"],
        "tensor_count": len(tensors),
        "tensor_bytes": sum(tensor.nbytes for tensor in tensors.values()),
        "source_model_fingerprint": manifest["source_model_fingerprint"],
    }


def verify_microshard_reconstruction(
    source_model: Path,
    banks: Sequence[Path],
    *,
    layer_id: int,
    expert_id: int,
) -> dict[str, Any]:
    """Reconstruct one full native expert and compare every source byte."""

    source = source_model.expanduser().resolve()
    config = _load_config(source)
    source_tensors = scan_safetensors(source)
    source_weight, source_scales = _validate_expert(
        source_tensors,
        layer_id=layer_id,
        expert_id=expert_id,
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
    )
    with source_weight.path.open("rb") as handle:
        expected_weight = _read_range(handle, _full_range(source_weight, "merged_weight"))
    with source_scales.path.open("rb") as handle:
        expected_scales = _read_range(handle, _full_range(source_scales, "row_scales"))

    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    matrix_bytes = hidden_size * intermediate_size
    reconstructed_weight = bytearray(source_weight.nbytes)
    reconstructed_scales = bytearray(source_scales.nbytes)
    observed_ranges: list[tuple[int, int]] = []
    source_fingerprints: set[str] = set()
    worker_ids: list[str] = []
    down_scales = expected_scales[2 * intermediate_size * 4 :]

    for bank_path in banks:
        bank = bank_path.expanduser().resolve()
        verification = verify_bank(bank)
        if verification["bank_kind"] != "native_colibri_microshards":
            raise ValueError(f"{bank} is not a native microshard bank")
        source_fingerprints.add(verification["source_model_fingerprint"])
        worker_ids.append(verification["worker_id"])
        manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
        tensors = scan_safetensors(bank)
        matching = [
            item
            for item in manifest.get("shards", [])
            if item.get("layer_id") == layer_id and item.get("expert_id") == expert_id
        ]
        for item in matching:
            start = item["hidden_start"]
            end = item["hidden_end"]
            require_bank_ownership(
                bank,
                layer_id=layer_id,
                expert_id=expert_id,
                hidden_start=start,
                hidden_end=end,
            )
            observed_ranges.append((start, end))
            weight_tensor = tensors[item["destination_weight_tensor"]]
            scale_tensor = tensors[item["destination_scale_tensor"]]
            with weight_tensor.path.open("rb") as handle:
                selected_weight = _read_range(handle, _full_range(weight_tensor, "selected_weight"))
            with scale_tensor.path.open("rb") as handle:
                selected_scales = _read_range(handle, _full_range(scale_tensor, "selected_scales"))
            width = end - start
            gate_bytes = width * hidden_size
            reconstructed_weight[start * hidden_size : end * hidden_size] = selected_weight[
                :gate_bytes
            ]
            reconstructed_weight[
                matrix_bytes + start * hidden_size : matrix_bytes + end * hidden_size
            ] = selected_weight[gate_bytes : 2 * gate_bytes]
            selected_down = selected_weight[2 * gate_bytes :]
            for row in range(hidden_size):
                destination_start = 2 * matrix_bytes + row * intermediate_size + start
                source_start = row * width
                reconstructed_weight[destination_start : destination_start + width] = selected_down[
                    source_start : source_start + width
                ]
            reconstructed_scales[start * 4 : end * 4] = selected_scales[: width * 4]
            reconstructed_scales[
                (intermediate_size + start) * 4 : (intermediate_size + end) * 4
            ] = selected_scales[width * 4 : 2 * width * 4]
            if selected_scales[2 * width * 4 :] != down_scales:
                raise ValueError("a microshard did not preserve the original down row scales")
            reconstructed_scales[2 * intermediate_size * 4 :] = down_scales

    expected_ranges: list[tuple[int, int]] = []
    cursor = 0
    for start, end in sorted(observed_ranges):
        if start != cursor:
            raise ValueError("microshard banks do not provide exact contiguous expert coverage")
        expected_ranges.append((start, end))
        cursor = end
    if cursor != intermediate_size:
        raise ValueError("microshard banks do not cover the full native expert")
    if len(source_fingerprints) != 1:
        raise ValueError("microshard banks do not share one source-model fingerprint")
    if bytes(reconstructed_weight) != expected_weight:
        raise ValueError("reconstructed native microshard weights differ from source bytes")
    if bytes(reconstructed_scales) != expected_scales:
        raise ValueError("reconstructed native microshard scales differ from source bytes")
    return {
        "valid": True,
        "layer_id": layer_id,
        "expert_id": expert_id,
        "worker_ids": worker_ids,
        "ranges": [list(value) for value in expected_ranges],
        "source_model_fingerprint": next(iter(source_fingerprints)),
        "weight_sha256": "sha256:" + hashlib.sha256(expected_weight).hexdigest(),
        "scale_sha256": "sha256:" + hashlib.sha256(expected_scales).hexdigest(),
        "native_weight_bytes": len(expected_weight),
        "native_scale_bytes": len(expected_scales),
    }


def require_bank_ownership(
    root: Path,
    *,
    layer_id: int,
    expert_id: int,
    hidden_start: int | None = None,
    hidden_end: int | None = None,
) -> None:
    """Fail explicitly unless a bank manifest grants the requested native bytes."""

    bank = root.expanduser().resolve()
    ownership = json.loads((bank / "ownership.json").read_text(encoding="utf-8"))
    if hidden_start is None or hidden_end is None:
        if hidden_start is not None or hidden_end is not None:
            raise ValueError("microshard ownership checks require both range boundaries")
        allowed = any(
            item.get("layer_id") == layer_id and item.get("expert_id") == expert_id
            for item in ownership.get("owned_experts", [])
            if isinstance(item, dict)
        )
        description = f"whole expert layer={layer_id} expert={expert_id}"
    else:
        if hidden_start < 0 or hidden_end <= hidden_start:
            raise ValueError("microshard ownership checks require a positive range")
        allowed = any(
            item.get("layer_id") == layer_id
            and item.get("expert_id") == expert_id
            and isinstance(item.get("hidden_start"), int)
            and isinstance(item.get("hidden_end"), int)
            and item["hidden_start"] <= hidden_start
            and item["hidden_end"] >= hidden_end
            for item in ownership.get("owned_microshards", [])
            if isinstance(item, dict)
        )
        description = (
            f"microshard layer={layer_id} expert={expert_id} range={hidden_start}:{hidden_end}"
        )
    if not allowed:
        raise PermissionError(f"worker bank forbids unowned {description}")


def _parse_expert(value: str) -> tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expert ownership must be LAYER:EXPERT")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError("expert ownership values must be integers") from error


def _parse_microshard(value: str) -> tuple[int, int, int, int]:
    parts = value.split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("microshard ownership must be LAYER:EXPERT:START:END")
    try:
        return tuple(int(part) for part in parts)  # type: ignore[return-value]
    except ValueError as error:
        raise argparse.ArgumentTypeError("microshard ownership values must be integers") from error


def _parse_hidden_range(value: str) -> tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("hidden range must be START:END")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError("hidden range values must be integers") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("whole", "microshard"):
        child = subparsers.add_parser(command)
        child.add_argument("--source-model", type=Path, required=True)
        child.add_argument("--output-directory", type=Path, required=True)
        child.add_argument("--worker-id", required=True)
        child.add_argument("--source-model-fingerprint")
        child.add_argument("--creation-timestamp")
    subparsers.choices["whole"].add_argument(
        "--owned-expert", action="append", required=True, type=_parse_expert
    )
    subparsers.choices["microshard"].add_argument(
        "--owned-microshard", action="append", default=[], type=_parse_microshard
    )
    subparsers.choices["microshard"].add_argument(
        "--all-experts-hidden-range", type=_parse_hidden_range
    )
    subparsers.choices["microshard"].add_argument(
        "--layout", choices=("equal", "asymmetric", "planner"), default="planner"
    )
    verify = subparsers.add_parser("verify")
    verify.add_argument("bank", type=Path)
    reconstruct = subparsers.add_parser("reconstruct")
    reconstruct.add_argument("--source-model", type=Path, required=True)
    reconstruct.add_argument("--bank", type=Path, action="append", required=True)
    reconstruct.add_argument("--layer-id", type=int, required=True)
    reconstruct.add_argument("--expert-id", type=int, required=True)
    reconstruct.add_argument("--output-json", type=Path)
    coordinator = subparsers.add_parser("coordinator")
    coordinator.add_argument("--source-model", type=Path, required=True)
    coordinator.add_argument("--output-directory", type=Path, required=True)
    coordinator.add_argument("--source-model-fingerprint")
    coordinator.add_argument("--creation-timestamp")
    verify_coordinator = subparsers.add_parser("verify-coordinator")
    verify_coordinator.add_argument("container", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify":
        result = verify_bank(args.bank)
    elif args.command == "verify-coordinator":
        result = verify_coordinator_container(args.container)
    elif args.command == "coordinator":
        path = build_coordinator_container(
            args.source_model,
            args.output_directory,
            source_model_fingerprint=args.source_model_fingerprint,
            creation_timestamp=args.creation_timestamp,
        )
        result = verify_coordinator_container(path)
    elif args.command == "reconstruct":
        result = verify_microshard_reconstruction(
            args.source_model,
            args.bank,
            layer_id=args.layer_id,
            expert_id=args.expert_id,
        )
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            _write_json(args.output_json, result)
    elif args.command == "whole":
        path = build_expert_bank(
            args.source_model,
            args.output_directory,
            worker_id=args.worker_id,
            owned_experts=args.owned_expert,
            source_model_fingerprint=args.source_model_fingerprint,
            creation_timestamp=args.creation_timestamp,
        )
        result = verify_bank(path)
    else:
        owned_microshards = list(args.owned_microshard)
        if args.all_experts_hidden_range:
            if owned_microshards:
                raise ValueError("use explicit microshards or --all-experts-hidden-range, not both")
            config = _load_config(args.source_model.expanduser().resolve())
            start, end = args.all_experts_hidden_range
            owned_microshards = [
                (layer_id, expert_id, start, end)
                for layer_id in range(config["num_hidden_layers"])
                for expert_id in range(config["num_experts"])
            ]
        if not owned_microshards:
            raise ValueError("microshard conversion requires owned native slices")
        path = build_microshard_bank(
            args.source_model,
            args.output_directory,
            worker_id=args.worker_id,
            owned_microshards=owned_microshards,
            layout=args.layout,
            source_model_fingerprint=args.source_model_fingerprint,
            creation_timestamp=args.creation_timestamp,
        )
        result = verify_bank(path)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
