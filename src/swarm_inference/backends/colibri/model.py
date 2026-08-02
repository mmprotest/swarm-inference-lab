"""Colibri model-family resolution and safetensors-backed inventory import."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from swarm_inference.backends.colibri.constants import MODEL_FAMILY_ENGINES
from swarm_inference.backends.colibri.schemas import (
    ExpertInventoryEntry,
    ModelInventory,
    NativeQuantizationMetadata,
    TensorInventoryEntry,
)
from swarm_inference.protocol.checksums import sha256_file

_SAFETENSORS_HEADER = struct.Struct("<Q")
_LAYER_RE = re.compile(r"(?:^|\.)layers?\.(\d+)(?:\.|$)")
_EXPERT_RE = re.compile(r"(?:^|\.)experts?\.(\d+)(?:\.|$)")
_PROJECTION_RE = re.compile(r"(?:^|\.)(gate_proj|up_proj|down_proj|merged_weight|qs)(?:\.|$)")
_KIMI_EXPERT_TENSOR_RE = re.compile(r"(?:^|\.)(w1|w2|w3)\.weight_(packed|scale)(?:\.|$)")


def _canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        name = path.name.encode("utf-8")
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _range_hash(path: Path, offset: int, length: int) -> str:
    digest = hashlib.sha256()
    remaining = length
    with path.open("rb") as handle:
        handle.seek(offset)
        while remaining:
            block = handle.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError(f"truncated tensor payload in {path} at offset {offset}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def resolve_model_family(config: dict[str, Any], explicit: str | None = None) -> str:
    """Resolve only families that Colibri v1.4.0 has a real engine for."""

    model_type = str(config.get("model_type", "")).lower().replace("_", "-")
    architecture_values = config.get("architectures")
    if not isinstance(architecture_values, list):
        architecture_values = []
    architectures = " ".join(str(item).lower() for item in architecture_values)
    raw_text_config = config.get("text_config")
    text: dict[str, Any] = raw_text_config if isinstance(raw_text_config, dict) else {}
    nested_type = str(text.get("model_type", "")).lower().replace("_", "-")
    signals = " ".join((model_type, nested_type, architectures))
    if "kimi" in signals:
        detected = "kimi-k3"
    elif "inkling" in signals:
        detected = "inkling"
    elif "olmoe" in signals or "olmoe" in str(config.get("_name_or_path", "")).lower():
        detected = "olmoe"
    elif "glm-moe-dsa" in signals or "glm" in signals:
        detected = "glm-5.2"
    else:
        raise ValueError(
            "unsupported model family; Colibri adapter will not fall back to another model engine"
        )
    if explicit is not None:
        normalized = explicit.strip().lower().replace("_", "-")
        aliases = {"glm": "glm-5.2", "kimi": "kimi-k3", "olmoe": "olmoe", "inkling": "inkling"}
        requested = aliases.get(normalized, normalized)
        if requested != detected:
            raise ValueError(
                f"requested model family {requested!r} does not match detected {detected!r}"
            )
    return detected


def _engine_path(engine_directory: Path, family: str) -> Path:
    basename = MODEL_FAMILY_ENGINES[family]
    candidates = [engine_directory / basename, engine_directory / f"{basename}.exe"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Colibri {family} engine is missing; tried {', '.join(str(item) for item in candidates)}"
    )


def _tensor_role(name: str, expert_id: int | None) -> str:
    kimi_tensor = _KIMI_EXPERT_TENSOR_RE.search(name)
    if expert_id is not None and kimi_tensor:
        projection, storage_kind = kimi_tensor.groups()
        if storage_kind == "scale":
            return "routed_expert_scale"
        # Colibri's Kimi engine evaluates w1 as the gate projection, w3 as
        # the up projection, and w2 as the down projection.  Preserve that
        # semantic mapping instead of treating the checkpoint names as
        # arbitrary byte blobs.
        return {
            "w1": "routed_expert_gate_projection",
            "w2": "routed_expert_down_projection",
            "w3": "routed_expert_up_projection",
        }[projection]
    projection = _PROJECTION_RE.search(name)
    if expert_id is not None and projection:
        return {
            "gate_proj": "routed_expert_gate_projection",
            "up_proj": "routed_expert_up_projection",
            "down_proj": "routed_expert_down_projection",
            "merged_weight": "routed_expert_merged_weight",
            "qs": "routed_expert_scale",
        }[projection.group(1)]
    lowered = name.lower()
    if "shared_expert" in lowered:
        return "shared_expert"
    if "router" in lowered or lowered.endswith("mlp.gate.weight"):
        return "router"
    if "embed" in lowered:
        return "embedding"
    if "lm_head" in lowered or lowered.endswith("output.weight"):
        return "output_head"
    if "norm" in lowered:
        return "normalization"
    if any(item in lowered for item in ("self_attn", "attention", "q_proj", "k_proj", "v_proj")):
        return "attention"
    return "dense_parameter"


def _effective_config(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else config


def _format_hint(config: dict[str, Any], family: str) -> str | None:
    effective = _effective_config(config)
    quant = effective.get("quantization_config")
    if not isinstance(quant, dict):
        quant = config.get("quantization_config")
    if isinstance(quant, dict):
        rendered = json.dumps(quant, sort_keys=True).lower()
        if "mxfp4" in rendered:
            return "mxfp4"
        if "int4" in rendered or '"num_bits": 4' in rendered:
            return "int4"
        if "fp8" in rendered:
            return "fp8"
    if family == "olmoe":
        return "int8_rowwise"
    return None


def _logical_projection_shape(
    *, role: str, packed_shape: list[int], config: dict[str, Any], native_format: str
) -> list[int]:
    if native_format != "mxfp4":
        return packed_shape
    effective = _effective_config(config)
    # Kimi's routed experts operate in the latent dimension, not the model's
    # full hidden width.  The GLU intermediate dimension is the separate
    # moe_intermediate_size field.
    hidden = int(
        effective.get("routed_expert_hidden_size", 0) or effective.get("hidden_size", 0) or 0
    )
    intermediate = int(
        effective.get("moe_intermediate_size", 0) or effective.get("intermediate_size", 0) or 0
    )
    if hidden and intermediate:
        if role.endswith(("gate_projection", "up_projection")):
            return [intermediate, hidden]
        if role.endswith("down_projection"):
            return [hidden, intermediate]
    return packed_shape


def _quantization(
    *, dtype: str, shape: list[int], role: str, byte_size: int, config: dict[str, Any], family: str
) -> NativeQuantizationMetadata:
    hint = _format_hint(config, family)
    is_routed = role.startswith("routed_expert_") and role != "routed_expert_scale"
    native_kimi_mxfp4 = family == "kimi-k3" and role.startswith("routed_expert_")
    if native_kimi_mxfp4 and role == "routed_expert_scale":
        return NativeQuantizationMetadata(
            format_name="ue8m0",
            packing="one_scale_per_group",
            scale_format="ue8m0",
            scale_group_size=32,
            quantization_aware_trained=True,
            reencoding_allowed=False,
            backend_requirements=["colibri-kimi-k3-native-mxfp4"],
            logical_shape=shape,
            packed_shape=shape,
            byte_size=byte_size,
        )
    if (hint == "mxfp4" or native_kimi_mxfp4) and is_routed:
        logical = _logical_projection_shape(
            role=role, packed_shape=shape, config=config, native_format="mxfp4"
        )
        return NativeQuantizationMetadata(
            format_name="mxfp4",
            packing="e2m1_two_nibbles",
            scale_format="ue8m0",
            scale_group_size=32,
            quantization_aware_trained=True,
            reencoding_allowed=False,
            backend_requirements=["colibri-kimi-k3-native-mxfp4"],
            logical_shape=logical,
            packed_shape=shape,
            byte_size=byte_size,
        )
    if family == "olmoe" and role == "routed_expert_merged_weight":
        return NativeQuantizationMetadata(
            format_name="int8_rowwise",
            packing="three_projections_flattened",
            scale_format="float32_per_row",
            scale_group_size=None,
            quantization_aware_trained=False,
            reencoding_allowed=False,
            backend_requirements=["colibri-olmoe-merged-expert"],
            logical_shape=shape,
            packed_shape=shape,
            byte_size=byte_size,
        )
    normalized = dtype.lower()
    format_name = hint if hint and is_routed else normalized
    return NativeQuantizationMetadata(
        format_name=format_name,
        packing="safetensors_native",
        scale_format="none" if role != "routed_expert_scale" else "float32_per_row",
        scale_group_size=None,
        quantization_aware_trained=hint == "mxfp4",
        reencoding_allowed=normalized in {"f32", "f16", "bf16"},
        backend_requirements=[f"colibri-{family}"],
        logical_shape=shape,
        packed_shape=shape,
        byte_size=byte_size,
    )


def _read_safetensors_header(path: Path) -> list[dict[str, Any]]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        raw_length = handle.read(_SAFETENSORS_HEADER.size)
        if len(raw_length) != _SAFETENSORS_HEADER.size:
            raise ValueError(f"invalid safetensors header in {path}")
        header_length = _SAFETENSORS_HEADER.unpack(raw_length)[0]
        if header_length <= 1 or header_length > min(file_size - 8, 256 * 1024 * 1024):
            raise ValueError(f"unsafe safetensors header length in {path}: {header_length}")
        header = json.loads(handle.read(header_length))
    data_base = 8 + header_length
    entries = []
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid tensor metadata for {name} in {path}")
        offsets = metadata.get("data_offsets")
        shape = metadata.get("shape")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(item, int) for item in offsets)
            or not isinstance(shape, list)
            or not all(isinstance(item, int) and item > 0 for item in shape)
        ):
            raise ValueError(f"invalid tensor offsets or shape for {name} in {path}")
        start, end = offsets
        absolute = data_base + start
        if start < 0 or end < start or data_base + end > file_size:
            raise ValueError(f"tensor {name} storage range is out of bounds in {path}")
        entries.append(
            {
                "name": name,
                "dtype": str(metadata.get("dtype", "unknown")),
                "shape": shape,
                "offset": absolute,
                "length": end - start,
            }
        )
    return sorted(entries, key=lambda item: int(item["offset"]))


class ColibriModelInspector:
    """Import actual file offsets without assuming expert-ID storage order."""

    def __init__(self, engine_directory: str | Path) -> None:
        self.engine_directory = Path(engine_directory).expanduser().resolve()

    def inspect(
        self,
        model_path: str | Path,
        *,
        model_id: str | None = None,
        model_revision: str | None = None,
        model_family: str | None = None,
        content_hash_mode: Literal["full", "metadata"] = "metadata",
        execution_backends: list[str] | None = None,
    ) -> tuple[
        ModelInventory,
        list[TensorInventoryEntry],
        list[ExpertInventoryEntry],
        list[NativeQuantizationMetadata],
    ]:
        root = Path(model_path).expanduser().resolve()
        config_path = root / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"model config is missing: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("model config must contain a JSON object")
        family = resolve_model_family(config, model_family)
        engine = _engine_path(self.engine_directory, family)
        config_hash = sha256_file(config_path)
        resolved_model_id = model_id or root.name
        resolved_revision = model_revision or config_hash
        tokenizer_files = [
            path
            for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
            if (path := root / name).is_file()
        ]
        tokenizer_hash = _hash_files(tokenizer_files) if tokenizer_files else "unavailable"
        safetensors = sorted(root.glob("*.safetensors"))
        if not safetensors:
            raise FileNotFoundError(f"no safetensors model files found in {root}")

        raw_entries: list[dict[str, Any]] = []
        model_files: list[dict[str, Any]] = []
        for file_index, path in enumerate(safetensors):
            header_entries = _read_safetensors_header(path)
            model_files.append(
                {
                    "path": str(path),
                    "relative_path": path.relative_to(root).as_posix(),
                    "byte_size": path.stat().st_size,
                    "sha256": sha256_file(path) if content_hash_mode == "full" else None,
                    "tensor_count": len(header_entries),
                }
            )
            for item in header_entries:
                item["path"] = path
                item["file_index"] = file_index
                raw_entries.append(item)

        raw_entries.sort(key=lambda item: (int(item["file_index"]), int(item["offset"])))
        tensors: list[TensorInventoryEntry] = []
        for physical_order, item in enumerate(raw_entries):
            name = str(item["name"])
            layer_match = _LAYER_RE.search(name)
            expert_match = _EXPERT_RE.search(name)
            layer_id = int(layer_match.group(1)) if layer_match else -1
            expert_id = int(expert_match.group(1)) if expert_match else None
            role = _tensor_role(name, expert_id)
            shape = [int(value) for value in item["shape"]]
            byte_size = int(item["length"])
            quant = _quantization(
                dtype=str(item["dtype"]),
                shape=shape,
                role=role,
                byte_size=byte_size,
                config=config,
                family=family,
            )
            identity = {
                "model_id": resolved_model_id,
                "model_revision": resolved_revision,
                "name": name,
                "file": Path(item["path"]).name,
                "offset": item["offset"],
                "length": byte_size,
            }
            tensor_id = _canonical_json_hash(identity)
            if content_hash_mode == "full":
                content_hash = "sha256:" + _range_hash(
                    Path(item["path"]), int(item["offset"]), byte_size
                )
            else:
                content_hash = "metadata-sha256:" + _canonical_json_hash(identity)
            tensors.append(
                TensorInventoryEntry(
                    tensor_id=tensor_id,
                    model_id=resolved_model_id,
                    model_revision=resolved_revision,
                    engine_family=family,
                    layer_id=layer_id,
                    tensor_name=name,
                    tensor_role=role,
                    expert_id=expert_id,
                    logical_shape=quant.logical_shape,
                    byte_size=byte_size,
                    storage_file=str(Path(item["path"])),
                    storage_offset=int(item["offset"]),
                    storage_length=byte_size,
                    quantization=quant,
                    current_tier="nvme",
                    permitted_tiers=["vram", "ram", "nvme"],
                    content_hash=content_hash,
                    execution_backends=execution_backends or ["cpu"],
                    physical_storage_order=physical_order,
                )
            )

        grouped: dict[tuple[int, int], list[TensorInventoryEntry]] = defaultdict(list)
        for tensor in tensors:
            if tensor.expert_id is not None and tensor.layer_id >= 0:
                grouped[(tensor.layer_id, tensor.expert_id)].append(tensor)
        expert_order = sorted(
            grouped,
            key=lambda key: min(item.physical_storage_order for item in grouped[key]),
        )
        experts: list[ExpertInventoryEntry] = []
        for physical_order, key in enumerate(expert_order):
            items = sorted(grouped[key], key=lambda item: item.physical_storage_order)
            formats = sorted({item.quantization.format_name for item in items})
            native_format = "+".join(formats)
            if "mxfp4" in formats and set(formats).issubset({"mxfp4", "ue8m0"}):
                # The UE8M0 sidecars are part of the native MXFP4 format, not
                # a second executable expert representation.
                native_format = "mxfp4"
            experts.append(
                ExpertInventoryEntry(
                    layer_id=key[0],
                    expert_id=key[1],
                    tensor_ids=[item.tensor_id for item in items],
                    total_bytes=sum(item.byte_size for item in items),
                    native_format=native_format,
                    storage_location={
                        "segments": [
                            {
                                "file": item.storage_file,
                                "offset": item.storage_offset,
                                "length": item.storage_length,
                            }
                            for item in items
                        ]
                    },
                    current_tier="nvme",
                    physical_storage_order=physical_order,
                )
            )

        effective = _effective_config(config)
        geometry = {
            "layers": int(effective.get("num_hidden_layers", 0) or 0),
            "experts_per_layer": int(
                effective.get("num_experts", 0) or effective.get("n_routed_experts", 0) or 0
            ),
            "experts_selected_per_token": int(
                effective.get("num_experts_per_token", 0)
                or effective.get("num_experts_per_tok", 0)
                or 0
            ),
            "shared_experts": int(
                effective.get("num_shared_experts", 0) or effective.get("n_shared_experts", 0) or 0
            ),
            "routed_expert_hidden_size": int(
                effective.get("routed_expert_hidden_size", 0)
                or effective.get("moe_intermediate_size", 0)
                or 0
            ),
            "expert_intermediate_size": int(
                effective.get("moe_intermediate_size", 0)
                or effective.get("intermediate_size", 0)
                or 0
            ),
        }
        quant_formats = sorted({tensor.quantization.format_name for tensor in tensors})
        inventory = ModelInventory(
            model_id=resolved_model_id,
            model_revision=resolved_revision,
            model_family=family,
            selected_engine_binary=str(engine),
            engine_build_fingerprint=sha256_file(engine),
            model_config_hash=config_hash,
            tokenizer_hash=tokenizer_hash,
            model_files=model_files,
            quantization_formats=quant_formats,
            expert_geometry=geometry,
            tensor_count=len(tensors),
            expert_count=len(experts),
        )
        unique_quant = {
            _canonical_json_hash(item.quantization.model_dump(mode="json")): item.quantization
            for item in tensors
        }
        return inventory, tensors, experts, list(unique_quant.values())
