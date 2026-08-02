"""Streaming GGUF metadata and logical tensor inventory for Experiment 008."""

from __future__ import annotations

import hashlib
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from swarm_inference.experiments.experiment_008.schemas import (
    ExpertMicroshard,
    ModelPreflight,
    TensorTile,
)


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
        return sum(tensor.byte_size for tensor in self.tensors)

    @property
    def expert_bytes(self) -> int:
        return sum(
            tensor.byte_size
            for tensor in self.tensors
            if tensor_role(tensor.name).startswith(("routed_expert", "shared_expert"))
        )


# (storage bytes, logical elements per storage block, logical dtype, quantization)
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

_LAYER = re.compile(r"(?:^|\.)blk\.(\d+)(?:\.|$)")


def _read_exact(handle: BinaryIO, length: int) -> bytes:
    value = handle.read(length)
    if len(value) != length:
        raise GGUFParseError("unexpected end of GGUF file")
    return value


def _unpack(handle: BinaryIO, fmt: str) -> Any:
    return struct.unpack(fmt, _read_exact(handle, struct.calcsize(fmt)))[0]


def _read_string(handle: BinaryIO, *, keep: bool = True) -> str | None:
    length = int(_unpack(handle, "<Q"))
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
    suffixes = (
        "architecture",
        "name",
        "file_type",
        "alignment",
        "block_count",
        "embedding_length",
        "expert_count",
        "expert_used_count",
        "expert_shared_count",
        "expert_feed_forward_length",
        "context_length",
    )
    return key.endswith(suffixes)


def _tensor_storage(shape: tuple[int, ...], ggml_type: int) -> tuple[int, str, str]:
    description = _GGML_TYPES.get(ggml_type)
    if description is None:
        raise GGUFParseError(f"unsupported GGML tensor type {ggml_type}")
    storage_bytes, block_elements, dtype, quantization = description
    elements = math.prod(shape)
    blocks = (elements + block_elements - 1) // block_elements
    return blocks * storage_bytes, dtype, quantization


def inspect_gguf(path: Path) -> GGUFInventory:
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
            if dimensions <= 0 or dimensions > 8:
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


def tensor_role(name: str) -> str:
    lowered = name.lower()
    if "token_embd" in lowered or "embed_tokens" in lowered:
        return "embedding"
    if lowered == "output.weight" or "lm_head" in lowered:
        return "output_head"
    if "shexp" in lowered or "shared_expert" in lowered:
        return "shared_expert"
    if "ffn_gate_inp" in lowered or ".mlp.gate.weight" in lowered:
        return "router"
    if "ffn_up_exps" in lowered or (".experts." in lowered and ".up_proj" in lowered):
        return "routed_expert_up_projection"
    if "ffn_gate_exps" in lowered or (".experts." in lowered and ".gate_proj" in lowered):
        return "routed_expert_gate_projection"
    if "ffn_down_exps" in lowered or (".experts." in lowered and ".down_proj" in lowered):
        return "routed_expert_down_projection"
    if any(part in lowered for part in ("attn_q", "attn_k", "attn_v", "attn_output")):
        return "attention_projection"
    if any(part in lowered for part in ("ssm_", "recur", "deltanet")):
        return "recurrent_or_hybrid_attention_state"
    if "norm" in lowered:
        return "normalisation"
    return "other"


def tensor_layer(name: str) -> int:
    match = _LAYER.search(name)
    return int(match.group(1)) if match else -1


def _content_hash(path: Path, offset: int, byte_size: int, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(offset)
        remaining = byte_size
        while remaining:
            block = handle.read(min(chunk_size, remaining))
            if not block:
                raise GGUFParseError("tensor content ended before its declared size")
            digest.update(block)
            remaining -= len(block)
    return f"sha256:{digest.hexdigest()}"


def build_tensor_tiles(
    inventory: GGUFInventory,
    *,
    model_id: str,
    model_revision: str,
    hash_contents: bool,
) -> list[TensorTile]:
    tiles: list[TensorTile] = []
    for tensor in inventory.tensors:
        absolute = inventory.data_offset + tensor.offset
        content_hash = (
            _content_hash(inventory.path, absolute, tensor.byte_size)
            if hash_contents
            else "UNVERIFIED:content hashing disabled for this run"
        )
        role = tensor_role(tensor.name)
        tiles.append(
            TensorTile(
                model_id=model_id,
                model_revision=model_revision,
                layer_id=tensor_layer(tensor.name),
                tensor_name=tensor.name,
                tensor_role=role,
                expert_id=None,
                logical_shape=list(tensor.shape),
                logical_slice={"kind": "full_tensor"},
                physical_layout="GGUF contiguous tensor; dimension 0 is innermost",
                dtype=tensor.dtype,
                quantization=tensor.quantization,
                quantization_metadata={
                    "ggml_type": tensor.ggml_type,
                    "file_offset": absolute,
                },
                accumulator_dtype="float32",
                byte_size=tensor.byte_size,
                content_hash=content_hash,
                allowed_backends=["llamacpp"],
                # Inventory construction precedes a backend load.  Actual
                # residency is reconciled later from the selected plan and
                # backend buffer logs; the IR must not imply an unmeasured CPU
                # placement merely because a tensor is an expert weight.
                current_residency="UNPLANNED",
                planned_execution_device="UNPLANNED",
            )
        )
    return tiles


def _metadata_int(metadata: dict[str, Any], suffix: str, default: int = 0) -> int:
    values = [int(value) for key, value in metadata.items() if key.endswith(suffix)]
    return values[0] if values else default


def build_expert_microshards(
    inventory: GGUFInventory,
    *,
    model_id: str,
    model_revision: str,
    hash_contents: bool,
) -> list[ExpertMicroshard]:
    """Expose whole-expert logical slices when packed GGUF layout makes them contiguous."""

    expert_count = _metadata_int(inventory.metadata, "expert_count")
    if expert_count <= 0:
        return []
    grouped: dict[int, dict[str, GGUFTensorInfo]] = {}
    for tensor in inventory.tensors:
        role = tensor_role(tensor.name)
        if not role.startswith("routed_expert_") or tensor_layer(tensor.name) < 0:
            continue
        if not tensor.shape or tensor.shape[-1] != expert_count:
            continue
        grouped.setdefault(tensor_layer(tensor.name), {})[role] = tensor
    shards: list[ExpertMicroshard] = []
    required_roles = {
        "up": "routed_expert_up_projection",
        "gate": "routed_expert_gate_projection",
        "down": "routed_expert_down_projection",
    }
    for layer_id, by_role in sorted(grouped.items()):
        if not set(required_roles.values()).issubset(by_role):
            continue
        up_info = by_role[required_roles["up"]]
        gate_info = by_role[required_roles["gate"]]
        down_info = by_role[required_roles["down"]]
        if len(up_info.shape) < 3 or len(down_info.shape) < 3:
            continue
        hidden_end = up_info.shape[1]
        if gate_info.shape[1] != hidden_end or down_info.shape[0] != hidden_end:
            raise GGUFParseError(f"layer {layer_id} expert projection dimensions do not match")
        for expert_id in range(expert_count):
            projected: dict[str, TensorTile] = {}
            for short_role, full_role in required_roles.items():
                info = by_role[full_role]
                if info.byte_size % expert_count:
                    raise GGUFParseError(f"{info.name} cannot be divided into whole experts")
                expert_bytes = info.byte_size // expert_count
                absolute = inventory.data_offset + info.offset + expert_id * expert_bytes
                content_hash = (
                    _content_hash(inventory.path, absolute, expert_bytes)
                    if hash_contents
                    else "UNVERIFIED:content hashing disabled for this run"
                )
                projected[short_role] = TensorTile(
                    model_id=model_id,
                    model_revision=model_revision,
                    layer_id=layer_id,
                    tensor_name=info.name,
                    tensor_role=full_role,
                    expert_id=expert_id,
                    logical_shape=list(info.shape[:-1]),
                    logical_slice={
                        "expert_axis": len(info.shape) - 1,
                        "expert_start": expert_id,
                        "expert_end": expert_id + 1,
                        "projection_range": {"hidden_start": 0, "hidden_end": hidden_end},
                    },
                    physical_layout="GGUF packed expert; complete contiguous expert slice",
                    dtype=info.dtype,
                    quantization=info.quantization,
                    quantization_metadata={
                        "ggml_type": info.ggml_type,
                        "file_offset": absolute,
                    },
                    accumulator_dtype="float32",
                    byte_size=expert_bytes,
                    content_hash=content_hash,
                    allowed_backends=["experiment-008-tile-runtime"],
                    current_residency="CPU",
                    planned_execution_device="UNPLANNED",
                )
            shards.append(
                ExpertMicroshard(
                    layer_id=layer_id,
                    expert_id=expert_id,
                    hidden_start=0,
                    hidden_end=hidden_end,
                    up=projected["up"],
                    gate=projected["gate"],
                    down=projected["down"],
                )
            )
    return shards


def build_preflight(
    inventory: GGUFInventory,
    *,
    model_id: str,
    model_revision: str,
    configured_architecture: str,
    configured_quantization: str,
    system_ram_available_bytes: int,
    physical_vram_bytes: int,
    backend: str,
    backend_limitations: list[str],
) -> ModelPreflight:
    metadata = inventory.metadata
    architecture = str(metadata.get("general.architecture", configured_architecture))
    layers = _metadata_int(metadata, "block_count")
    experts = _metadata_int(metadata, "expert_count")
    selected = _metadata_int(metadata, "expert_used_count")
    shared = _metadata_int(metadata, "expert_shared_count")
    if shared == 0 and any(tensor_role(item.name) == "shared_expert" for item in inventory.tensors):
        # Qwen3-Next GGUF encodes one shared expert per layer in ``*_shexp``
        # tensors but does not currently emit an ``expert_shared_count`` key.
        shared = 1
    total = inventory.tensor_bytes
    reasons: list[str] = []
    if experts <= 1:
        reasons.append("model metadata does not describe a sparse MoE")
    if selected <= 0 or selected > experts:
        reasons.append("model metadata does not expose a valid experts-per-token value")
    if layers <= 0:
        reasons.append("model metadata does not expose a positive layer count")
    if inventory.expert_bytes <= 0:
        reasons.append("no routed-expert tensor storage was found")
    if physical_vram_bytes <= 0:
        reasons.append("physical CUDA VRAM capacity was unavailable")
    if total <= physical_vram_bytes:
        reasons.append("total tensor bytes do not exceed physical VRAM")
    if total <= 32 * 1024**3:
        reasons.append("total tensor bytes do not exceed 32 GiB")
    system_ram_required = int(total * 1.05)
    if system_ram_required > system_ram_available_bytes:
        reasons.append(
            "available system RAM is below the tensor inventory plus the declared 5% loading allowance"
        )
    return ModelPreflight(
        model_id=model_id,
        model_revision=model_revision,
        model_architecture=architecture,
        quantization_format=configured_quantization,
        total_tensor_bytes=total,
        total_expert_bytes=inventory.expert_bytes,
        layer_count=layers,
        routed_expert_count=experts,
        experts_selected_per_token=selected,
        shared_expert_count=shared,
        system_ram_required_bytes=system_ram_required,
        system_ram_available_bytes=system_ram_available_bytes,
        physical_vram_bytes=physical_vram_bytes,
        backend_selected=backend,
        backend_limitations=backend_limitations,
        genuinely_exceeds_32gb=total > 32 * 1024**3,
        genuinely_exceeds_physical_vram=total > physical_vram_bytes,
        eligible=not reasons,
        rejection_reasons=reasons,
    )
