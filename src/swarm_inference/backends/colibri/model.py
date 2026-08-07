"""Colibri model-family resolution and safetensors-backed inventory import."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Literal

from swarm_inference.backends.colibri.adapters import default_colibri_adapter_registry
from swarm_inference.backends.colibri.schemas import (
    ExpertInventoryEntry,
    ModelInventory,
    NativeQuantizationMetadata,
    TensorInventoryEntry,
)
from swarm_inference.model.architecture import architecture_from_config
from swarm_inference.model.descriptor import ModelFileDescriptor, ResolvedModelDescriptor
from swarm_inference.protocol.checksums import sha256_file

_SAFETENSORS_HEADER = struct.Struct("<Q")


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
    """Resolve an adapter from exact root/nested architecture metadata."""

    try:
        return (
            default_colibri_adapter_registry()
            .resolve_config(
                config,
                explicit_adapter_id=explicit,
            )
            .adapter_id
        )
    except LookupError as exc:
        raise ValueError(str(exc)) from exc


def _engine_path(engine_directory: Path, family: str) -> Path:
    basename = default_colibri_adapter_registry().get(family).engine_basename
    candidates = [engine_directory / basename, engine_directory / f"{basename}.exe"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Colibri {family} engine is missing; tried {', '.join(str(item) for item in candidates)}"
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
        adapter = default_colibri_adapter_registry().get(family)
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
        mappings = adapter.map_tensor_names(
            tuple(
                (
                    str(item["name"]),
                    tuple(int(value) for value in item["shape"]),
                    str(item["dtype"]),
                    int(item["length"]),
                )
                for item in raw_entries
            ),
            config=config,
        )
        mapping_by_name = {item.tensor_name: item for item in mappings}
        if len(mapping_by_name) != len(raw_entries):
            raise ValueError("Colibri adapter did not map every checkpoint tensor exactly once")
        tensors: list[TensorInventoryEntry] = []
        for physical_order, item in enumerate(raw_entries):
            name = str(item["name"])
            mapping = mapping_by_name[name]
            layer_id = mapping.layer_index
            expert_id = mapping.expert_index
            role = mapping.tensor_role
            byte_size = int(item["length"])
            quant = NativeQuantizationMetadata(
                format_name=mapping.quantization_format,
                packing=mapping.packing,
                scale_format=mapping.scale_format,
                scale_group_size=mapping.scale_group_size,
                quantization_aware_trained=mapping.quantization_aware_trained,
                reencoding_allowed=mapping.reencoding_allowed,
                backend_requirements=[
                    f"colibri-adapter:{adapter.adapter_id}",
                    adapter.engine_basename,
                ],
                logical_shape=list(mapping.logical_shape),
                packed_shape=list(mapping.packed_shape),
                byte_size=byte_size,
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

        architecture = architecture_from_config(config)
        inspection_model = ResolvedModelDescriptor(
            model_id=resolved_model_id,
            revision=resolved_revision,
            content_fingerprint="sha256:" + config_hash,
            source_type="local",
            format="safetensors",
            architecture=architecture.canonical,
            architecture_raw=architecture.raw,
            architecture_source=architecture.source,
            files=tuple(
                ModelFileDescriptor(
                    relative_path=str(item["relative_path"]),
                    size_bytes=int(item["byte_size"]),
                    sha256=item["sha256"],
                )
                for item in model_files
            ),
            weight_bytes=sum(int(item["byte_size"]) for item in model_files),
            tokenizer_identity=tokenizer_hash,
            local_paths=tuple(str(path) for path in safetensors),
            configuration=config,
        )
        adapter.validate_model_identity(
            inspection_model,
            tensor_names=tuple(item.tensor_name for item in mappings),
        )
        profile = adapter.inspect_model(inspection_model)
        expert_descriptors = adapter.describe_experts(mappings, profile)
        inventory_by_name = {item.tensor_name: item for item in tensors}
        experts: list[ExpertInventoryEntry] = []
        ordered_descriptors = sorted(
            expert_descriptors,
            key=lambda descriptor: (
                min(
                    inventory_by_name[name].physical_storage_order
                    for group in descriptor.tensor_groups
                    for name in group.tensor_names
                ),
                descriptor.layer_index,
                descriptor.expert_type,
                descriptor.expert_index,
            ),
        )
        for physical_order, descriptor in enumerate(ordered_descriptors):
            names = tuple(name for group in descriptor.tensor_groups for name in group.tensor_names)
            items = sorted(
                (inventory_by_name[name] for name in names),
                key=lambda item: item.physical_storage_order,
            )
            formats = sorted({item.quantization.format_name for item in items})
            native_format = "+".join(formats)
            if "mxfp4" in formats and set(formats).issubset({"mxfp4", "ue8m0"}):
                # The UE8M0 sidecars are part of the native MXFP4 format, not
                # a second executable expert representation.
                native_format = "mxfp4"
            experts.append(
                ExpertInventoryEntry(
                    layer_id=descriptor.layer_index,
                    expert_id=descriptor.expert_index,
                    expert_type=descriptor.expert_type,
                    tensor_ids=[item.tensor_id for item in items],
                    total_bytes=descriptor.memory_bytes,
                    native_format=native_format,
                    storage_location={
                        "segments": [
                            {
                                "file": item.storage_file,
                                "offset": item.storage_offset,
                                "length": item.storage_length,
                            }
                            for item in items
                        ],
                        "tensor_slices": descriptor.routing_metadata.get("tensor_slices", {}),
                    },
                    current_tier="nvme",
                    physical_storage_order=physical_order,
                    routing_metadata=descriptor.routing_metadata,
                )
            )

        geometry = {
            "layers": profile.layer_count or 0,
            "hidden_size": profile.hidden_size or 0,
            "experts_per_layer": profile.expert_count or 0,
            "experts_selected_per_token": profile.experts_per_token or 0,
            "shared_experts": profile.shared_expert_count,
            "expert_intermediate_size": profile.expert_intermediate_size or 0,
            "routing_kind": profile.routing_kind,
            "checkpoint_layout": profile.checkpoint_layout,
            **profile.architecture_metadata,
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
