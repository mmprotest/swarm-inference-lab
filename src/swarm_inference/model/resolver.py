"""Immutable local and Hugging Face model resolution with selective acquisition."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import psutil

from swarm_inference.model.architecture import (
    ArchitectureIdentity,
    architecture_from_config,
    architecture_from_gguf,
)
from swarm_inference.model.architecture_adapters import default_architecture_adapter_registry
from swarm_inference.model.descriptor import (
    ModelFileDescriptor,
    ResolvedModelDescriptor,
    ResolvedTensorDescriptor,
)
from swarm_inference.model.gguf import (
    GGUFInventory,
    GGUFParseError,
    inspect_gguf,
    inspect_gguf_stream,
)
from swarm_inference.model.quantization import quantization_from_config
from swarm_inference.model.safetensors import (
    SafetensorsHeaderError,
    SafetensorsIndexInventory,
    inspect_safetensors,
    inspect_safetensors_index,
    inspect_safetensors_index_payload,
)
from swarm_inference.model.source import ModelSourceReference, parse_model_source
from swarm_inference.model.variants import (
    ModelVariant,
    VariantCandidate,
    discover_gguf_variants,
    select_variant,
)

_COMMIT = re.compile(r"^[0-9a-fA-F]{40}([0-9a-fA-F]{24})?$")
_METADATA_NAMES = frozenset(
    {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    }
)
_NATIVE_METADATA_NAMES = _METADATA_NAMES | frozenset(
    {
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    }
)


def _is_native_metadata_file(relative_path: str) -> bool:
    name = relative_path.rsplit("/", 1)[-1].casefold()
    return name in _NATIVE_METADATA_NAMES or name.endswith(".safetensors.index.json")


def _safetensors_index_metadata(
    inventory: SafetensorsIndexInventory,
    *,
    descriptor: ModelFileDescriptor,
    sha256: str,
) -> dict[str, Any]:
    return {
        "relative_path": descriptor.relative_path,
        "sha256": sha256,
        "size_bytes": descriptor.size_bytes,
        "etag": descriptor.etag,
        "mapping_sha256": inventory.mapping_sha256,
        "tensor_count": inventory.tensor_count,
        "shard_count": len(inventory.shard_names),
        "shards": list(inventory.shard_names),
        "tensors_per_shard": dict(inventory.tensors_per_shard),
        "declared_total_size": inventory.declared_total_size,
    }


@dataclass(frozen=True, slots=True)
class ResolutionResources:
    aggregate_usable_memory_bytes: int
    local_fast_memory_bytes: int = 0

    @classmethod
    def local_default(cls) -> ResolutionResources:
        return cls(aggregate_usable_memory_bytes=int(psutil.virtual_memory().available))


@dataclass(frozen=True, slots=True)
class ModelResolution:
    descriptor: ResolvedModelDescriptor
    variants: tuple[ModelVariant, ...] = ()
    variant_candidates: tuple[VariantCandidate, ...] = ()
    repository_files: tuple[ModelFileDescriptor, ...] = ()


def _sha256_file(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(model_id: str, revision: str, files: tuple[ModelFileDescriptor, ...]) -> str:
    payload = {
        "model_id": model_id,
        "revision": revision,
        "files": [
            {
                "path": item.relative_path,
                "size": item.size_bytes,
                "sha256": item.sha256,
                "etag": item.etag,
            }
            for item in sorted(files, key=lambda value: value.relative_path)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _config_facts(
    config: dict[str, Any],
) -> tuple[ArchitectureIdentity, int | None, int | None]:
    architecture = architecture_from_config(config)
    text_config = config.get("text_config")
    candidates = (config, text_config) if isinstance(text_config, dict) else (config,)
    layer_count = next(
        (
            int(candidate[key])
            for candidate in candidates
            for key in ("num_hidden_layers", "n_layer", "num_layers")
            if isinstance(candidate.get(key), int)
        ),
        None,
    )
    parameters = next(
        (
            candidate["num_parameters"]
            for candidate in candidates
            if isinstance(candidate.get("num_parameters"), int)
        ),
        None,
    )
    return architecture, layer_count, int(parameters) if isinstance(parameters, int) else None


def _activation_facts(config: dict[str, Any]) -> tuple[int | None, int | None]:
    text_config = config.get("text_config")
    candidates = (config, text_config) if isinstance(text_config, dict) else (config,)
    hidden = next(
        (
            int(candidate[key])
            for candidate in candidates
            for key in ("hidden_size", "n_embd", "d_model")
            if isinstance(candidate.get(key), int) and int(candidate[key]) > 0
        ),
        None,
    )
    dtype = next(
        (
            str(candidate.get("torch_dtype") or candidate.get("dtype")).casefold()
            for candidate in candidates
            if candidate.get("torch_dtype") or candidate.get("dtype")
        ),
        "",
    )
    dtype_bytes = {
        "bfloat16": 2,
        "bf16": 2,
        "float16": 2,
        "fp16": 2,
        "float32": 4,
        "fp32": 4,
        "float64": 8,
        "fp64": 8,
    }.get(dtype)
    return hidden, dtype_bytes


def _configuration_quantization(config: dict[str, Any]) -> str | None:
    return quantization_from_config(config)


def _configuration_modalities(config: dict[str, Any]) -> tuple[str, ...]:
    modalities = ["text"]
    if isinstance(config.get("vision_config"), dict):
        modalities.append("vision")
    if isinstance(config.get("audio_config"), dict):
        modalities.append("audio")
    return tuple(modalities)


def _configuration_features(config: dict[str, Any]) -> tuple[str, ...]:
    effective = config.get("text_config")
    effective = effective if isinstance(effective, dict) else config
    features: list[str] = []
    if isinstance(effective.get("layer_types"), list):
        features.append("heterogeneous-layer-types")
    if any(effective.get(key) is not None for key in ("kv_lora_rank", "q_lora_rank")):
        features.append("latent-attention")
    if any(
        isinstance(effective.get(key), int) and int(effective[key]) > 0
        for key in ("num_experts", "n_routed_experts", "num_local_experts")
    ):
        features.append("routed-experts")
    if any(
        isinstance(effective.get(key), int) and int(effective[key]) > 0
        for key in ("num_shared_experts", "n_shared_experts")
    ):
        features.append("shared-experts")
    if len(_configuration_modalities(config)) > 1:
        features.append("multimodal")
    return tuple(features)


def _gguf_configuration(
    config: dict[str, Any], metadata: dict[str, Any], architecture: ArchitectureIdentity
) -> dict[str, Any]:
    """Expose standard topology keys while preserving exact GGUF metadata."""

    derived: dict[str, Any] = dict(config)
    if architecture.raw and not derived.get("model_type"):
        derived["model_type"] = architecture.raw
    suffixes = {
        "block_count": "num_hidden_layers",
        "embedding_length": "hidden_size",
        "expert_count": "num_experts",
        "expert_used_count": "num_experts_per_tok",
        "expert_shared_count": "num_shared_experts",
        "context_length": "max_position_embeddings",
    }
    for key, value in metadata.items():
        for suffix, target in suffixes.items():
            if key.endswith(suffix) and target not in derived and isinstance(value, int):
                derived[target] = value
    return derived


def _attach_architecture_profile(model: ResolvedModelDescriptor) -> ResolvedModelDescriptor:
    registry = default_architecture_adapter_registry()
    adapter = registry.resolve_model(model)
    if adapter is None:
        return model
    try:
        profile = adapter.inspect(model)
    except ValueError as exc:
        metadata = {**model.artifact_metadata, "architecture_inspection_error": str(exc)}
        return model.model_copy(update={"artifact_metadata": metadata})
    return model.model_copy(update={"architecture_profile": profile})


def _tokenizer_identity(files: tuple[ModelFileDescriptor, ...]) -> str | None:
    tokenizer_json = next(
        (item for item in files if item.relative_path.rsplit("/", 1)[-1] == "tokenizer.json"),
        None,
    )
    if tokenizer_json is not None and tokenizer_json.sha256 is not None:
        return "sha256:" + tokenizer_json.sha256.removeprefix("sha256:").lower()
    selected = [
        item
        for item in files
        if item.relative_path.rsplit("/", 1)[-1] in _METADATA_NAMES
        and "token" in item.relative_path.lower()
    ]
    if not selected:
        return None
    payload = "\n".join(
        f"{item.relative_path}:{item.sha256 or item.etag}:{item.size_bytes}"
        for item in sorted(selected, key=lambda value: value.relative_path)
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _local_files(path: Path) -> tuple[ModelFileDescriptor, ...]:
    root = path if path.is_dir() else path.parent
    paths = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
    files: list[ModelFileDescriptor] = []
    for item in sorted(paths):
        relative = item.relative_to(root).as_posix() if item != root else item.name
        files.append(
            ModelFileDescriptor(
                relative_path=relative,
                size_bytes=item.stat().st_size,
                sha256=_sha256_file(item),
            )
        )
    return tuple(files)


class ModelSourceResolver:
    """Resolve mutable user references, but never make an execution decision."""

    def __init__(
        self,
        *,
        cache_directory: Path | None = None,
        api: Any | None = None,
        metadata_loader: Callable[..., str | Path] | None = None,
        gguf_metadata_loader: Callable[..., GGUFInventory] | None = None,
    ) -> None:
        self.cache_directory = cache_directory.expanduser().resolve() if cache_directory else None
        self._api = api
        self._metadata_loader = metadata_loader
        self._gguf_metadata_loader = gguf_metadata_loader

    def _api_client(self) -> Any:
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi()
        return self._api

    def _load_hub_config(
        self,
        *,
        model_id: str,
        revision: str,
        files: tuple[ModelFileDescriptor, ...],
        summary: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Read the immutable config file instead of trusting Hub summary fields.

        ``model_info.config`` intentionally contains only a search-oriented
        subset for many repositories.  Expert topology, nested text config,
        attention layout, and even dimensions can be absent.  The small
        content-addressed ``config.json`` is therefore part of inspection, not
        deferred weight acquisition.
        """

        candidates = sorted(
            (
                item
                for item in files
                if item.relative_path.rsplit("/", 1)[-1].casefold() == "config.json"
            ),
            key=lambda item: (item.relative_path.count("/"), item.relative_path),
        )
        if not candidates:
            return summary, None
        descriptor = candidates[0]
        loader = self._metadata_loader
        if loader is None:
            from huggingface_hub import hf_hub_download

            loader = hf_hub_download
        loaded = Path(
            loader(
                repo_id=model_id,
                filename=descriptor.relative_path,
                revision=revision,
                cache_dir=str(self.cache_directory) if self.cache_directory else None,
            )
        ).resolve()
        if not loaded.is_file():
            raise RuntimeError(f"Hugging Face config download is unavailable: {loaded}")
        if loaded.stat().st_size != descriptor.size_bytes:
            raise RuntimeError("downloaded config.json size differs from repository metadata")
        digest = _sha256_file(loaded)
        if descriptor.sha256 is not None and digest != descriptor.sha256.removeprefix("sha256:"):
            raise RuntimeError("downloaded config.json digest differs from repository metadata")
        try:
            payload = json.loads(loaded.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Hugging Face config.json is not valid UTF-8 JSON") from exc
        if not isinstance(payload, dict) or not payload:
            raise RuntimeError("Hugging Face config.json must contain a non-empty object")
        return payload, {
            "relative_path": descriptor.relative_path,
            "sha256": digest,
            "size_bytes": descriptor.size_bytes,
            "etag": descriptor.etag,
        }

    def _load_hub_safetensors_index(
        self,
        *,
        model_id: str,
        revision: str,
        files: tuple[ModelFileDescriptor, ...],
    ) -> tuple[SafetensorsIndexInventory, dict[str, Any]] | None:
        """Download and validate the small immutable shard index, never the weights."""

        candidates = sorted(
            (
                item
                for item in files
                if item.relative_path.casefold().endswith(".safetensors.index.json")
            ),
            key=lambda item: (item.relative_path.count("/"), item.relative_path),
        )
        if not candidates:
            return None
        minimum_depth = candidates[0].relative_path.count("/")
        shallow = [item for item in candidates if item.relative_path.count("/") == minimum_depth]
        if len(shallow) != 1:
            names = ", ".join(item.relative_path for item in shallow)
            raise RuntimeError(
                f"ambiguous Safetensors indexes at the same repository depth: {names}"
            )
        descriptor = shallow[0]
        loader = self._metadata_loader
        if loader is None:
            from huggingface_hub import hf_hub_download

            loader = hf_hub_download
        loaded = Path(
            loader(
                repo_id=model_id,
                filename=descriptor.relative_path,
                revision=revision,
                cache_dir=str(self.cache_directory) if self.cache_directory else None,
            )
        ).resolve()
        if not loaded.is_file():
            raise RuntimeError(f"Hugging Face Safetensors index is unavailable: {loaded}")
        if loaded.stat().st_size != descriptor.size_bytes:
            raise RuntimeError("downloaded Safetensors index size differs from repository metadata")
        digest = _sha256_file(loaded)
        if descriptor.sha256 is not None and digest != descriptor.sha256.removeprefix("sha256:"):
            raise RuntimeError(
                "downloaded Safetensors index digest differs from repository metadata"
            )
        try:
            payload = json.loads(loaded.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Hugging Face Safetensors index is not valid UTF-8 JSON") from exc
        inventory = inspect_safetensors_index_payload(
            payload,
            source=descriptor.relative_path,
            available_files=tuple(item.relative_path for item in files),
        )
        return inventory, _safetensors_index_metadata(
            inventory,
            descriptor=descriptor,
            sha256=digest,
        )

    def _load_hub_gguf_metadata(
        self,
        *,
        model_id: str,
        revision: str,
        descriptor: ModelFileDescriptor,
    ) -> GGUFInventory:
        """Range-read one immutable GGUF header without acquiring its weights."""

        if self._gguf_metadata_loader is not None:
            return self._gguf_metadata_loader(
                repo_id=model_id,
                filename=descriptor.relative_path,
                revision=revision,
                file_size=descriptor.size_bytes,
            )
        from huggingface_hub import HfFileSystem

        remote_path = f"{model_id}@{revision}/{descriptor.relative_path}"
        filesystem = HfFileSystem()
        with filesystem.open(remote_path, "rb", block_size=1 << 20) as handle:
            return inspect_gguf_stream(
                handle,
                file_size=descriptor.size_bytes,
                source=descriptor.relative_path,
            )

    @staticmethod
    def _hub_file(item: Any) -> ModelFileDescriptor:
        name = str(getattr(item, "rfilename", getattr(item, "path", "")))
        if not name:
            raise RuntimeError("Hugging Face repository returned a file without a name")
        size = getattr(item, "size", None)
        lfs = getattr(item, "lfs", None)
        if size is None and lfs is not None:
            size = lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)
        if size is None:
            raise RuntimeError(f"Hugging Face repository omitted the size of {name}")
        sha256: str | None = None
        if lfs is not None:
            sha256 = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        etag = getattr(item, "blob_id", None) or sha256
        return ModelFileDescriptor(
            relative_path=name,
            size_bytes=int(size),
            sha256=str(sha256) if sha256 else None,
            etag=str(etag) if etag else None,
        )

    def inspect(
        self,
        source: str | Path | ModelSourceReference,
        *,
        revision: str | None = None,
        variant: str | None = None,
        quantization: str | None = None,
        objective: Literal["speed", "throughput", "capacity", "balanced"] = "balanced",
        resources: ResolutionResources | None = None,
    ) -> ModelResolution:
        reference = (
            source
            if isinstance(source, ModelSourceReference)
            else parse_model_source(source, revision=revision, variant=variant)
        )
        if reference.source_type == "local":
            return self._inspect_local(
                reference,
                quantization=quantization,
                objective=objective,
                resources=resources or ResolutionResources.local_default(),
            )
        return self._inspect_hub(
            reference,
            quantization=quantization,
            objective=objective,
            resources=resources or ResolutionResources.local_default(),
        )

    def resolve(self, *args: Any, **kwargs: Any) -> ResolvedModelDescriptor:
        return self.inspect(*args, **kwargs).descriptor

    async def resolve_async(self, *args: Any, **kwargs: Any) -> ResolvedModelDescriptor:
        return await asyncio.to_thread(self.resolve, *args, **kwargs)

    def _select_files(
        self,
        files: tuple[ModelFileDescriptor, ...],
        *,
        reference: ModelSourceReference,
        quantization: str | None,
        objective: Literal["speed", "throughput", "capacity", "balanced"],
        resources: ResolutionResources,
    ) -> tuple[
        tuple[ModelFileDescriptor, ...],
        tuple[ModelVariant, ...],
        tuple[VariantCandidate, ...],
        str | None,
        str | None,
    ]:
        variants = discover_gguf_variants(files)
        if not variants:
            metadata = tuple(item for item in files if _is_native_metadata_file(item.relative_path))
            # Prefer one complete native weight family. Repositories often keep
            # legacy PyTorch weights beside safetensors; acquiring both doubles
            # transfer/storage without changing execution identity.
            weight_suffixes = (
                (".safetensors",),
                (".bin",),
                (".pt", ".pth"),
            )
            weights = next(
                (
                    tuple(item for item in files if item.relative_path.lower().endswith(suffixes))
                    for suffixes in weight_suffixes
                    if any(item.relative_path.lower().endswith(suffixes) for item in files)
                ),
                (),
            )
            selected = tuple(
                sorted(
                    {item.relative_path: item for item in (*metadata, *weights)}.values(),
                    key=lambda item: item.relative_path,
                )
            )
            return selected or files, (), (), reference.variant, quantization
        selection = select_variant(
            variants,
            objective=objective,
            aggregate_usable_memory_bytes=resources.aggregate_usable_memory_bytes,
            local_fast_memory_bytes=resources.local_fast_memory_bytes,
            requested_variant=reference.variant,
            requested_quantization=quantization,
        )
        metadata = tuple(
            item for item in files if item.relative_path.rsplit("/", 1)[-1] in _METADATA_NAMES
        )
        selected_files = tuple(
            sorted(
                {
                    item.relative_path: item for item in (*selection.selected.files, *metadata)
                }.values(),
                key=lambda item: (
                    not item.relative_path.lower().endswith(".gguf"),
                    item.relative_path,
                ),
            )
        )
        return (
            selected_files,
            variants,
            selection.candidates,
            selection.selected.variant_id,
            selection.selected.quantization,
        )

    def _inspect_local(
        self,
        reference: ModelSourceReference,
        *,
        quantization: str | None,
        objective: Literal["speed", "throughput", "capacity", "balanced"],
        resources: ResolutionResources,
    ) -> ModelResolution:
        assert reference.local_path is not None
        path = reference.local_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        files = _local_files(path)
        selected, variants, candidates, selected_variant, selected_quant = self._select_files(
            files,
            reference=reference,
            quantization=quantization,
            objective=objective,
            resources=resources,
        )
        names = {item.relative_path.lower() for item in selected}
        model_format: Literal["safetensors", "gguf", "pytorch", "unknown"]
        if any(name.endswith(".gguf") for name in names):
            model_format = "gguf"
        elif any(name.endswith(".safetensors") for name in names):
            model_format = "safetensors"
        elif any(name.endswith((".bin", ".pt", ".pth")) for name in names):
            model_format = "pytorch"
        else:
            model_format = "unknown"
        config: dict[str, Any] = {}
        config_path = path / "config.json" if path.is_dir() else path.parent / "config.json"
        if config_path.is_file():
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                config = raw
        architecture, layers, parameters = _config_facts(config)
        hidden_size, activation_dtype_bytes = _activation_facts(config)
        artifact_metadata: dict[str, Any] = {}
        tensor_descriptors: list[ResolvedTensorDescriptor] = []
        local_root = path if path.is_dir() else path.parent
        if model_format == "safetensors":
            index_descriptors = sorted(
                (
                    item
                    for item in selected
                    if item.relative_path.casefold().endswith(".safetensors.index.json")
                ),
                key=lambda item: (item.relative_path.count("/"), item.relative_path),
            )
            if index_descriptors:
                minimum_depth = index_descriptors[0].relative_path.count("/")
                shallow = [
                    item
                    for item in index_descriptors
                    if item.relative_path.count("/") == minimum_depth
                ]
                if len(shallow) != 1:
                    index_names = ", ".join(item.relative_path for item in shallow)
                    raise RuntimeError(
                        f"ambiguous Safetensors indexes at the same directory depth: {index_names}"
                    )
                index_descriptor = shallow[0]
                index_path = local_root / index_descriptor.relative_path
                safetensors_inventory = inspect_safetensors_index(
                    index_path,
                    available_files=tuple(item.relative_path for item in files),
                )
                referenced_shards = set(safetensors_inventory.shard_names)
                selected = tuple(
                    item
                    for item in selected
                    if not item.relative_path.casefold().endswith(".safetensors")
                    or item.relative_path in referenced_shards
                )
                artifact_metadata["safetensors_index"] = _safetensors_index_metadata(
                    safetensors_inventory,
                    descriptor=index_descriptor,
                    sha256=_sha256_file(index_path),
                )
        if model_format == "gguf":
            first_descriptor = next(
                item for item in selected if item.relative_path.lower().endswith(".gguf")
            )
            first = path if path.is_file() else path / first_descriptor.relative_path
            try:
                gguf_inventory = inspect_gguf(first)
            except GGUFParseError:
                raise
            architecture = architecture_from_gguf(
                gguf_inventory.metadata.get("general.architecture"), fallback=architecture
            )
            config = _gguf_configuration(config, gguf_inventory.metadata, architecture)
            layers = next(
                (
                    int(value)
                    for key, value in gguf_inventory.metadata.items()
                    if key.endswith("block_count")
                ),
                layers,
            )
            hidden_size = next(
                (
                    int(value)
                    for key, value in gguf_inventory.metadata.items()
                    if key.endswith("embedding_length") and isinstance(value, int) and value > 0
                ),
                hidden_size,
            )
            artifact_metadata["gguf"] = {
                "version": gguf_inventory.version,
                "data_offset": gguf_inventory.data_offset,
                "metadata": gguf_inventory.metadata,
            }
            tensor_descriptors.extend(
                ResolvedTensorDescriptor(
                    name=item.name,
                    shape=item.shape,
                    dtype=(item.quantization if item.dtype == "quantized" else item.dtype),
                    size_bytes=item.byte_size,
                    source_file=first_descriptor.relative_path,
                    data_offset=gguf_inventory.data_offset + item.offset,
                )
                for item in gguf_inventory.tensors
            )
            if selected_quant is None:
                quantizations = sorted(
                    {
                        item.quantization
                        for item in gguf_inventory.tensors
                        if item.quantization != "none"
                    }
                )
                if len(quantizations) == 1:
                    selected_quant = quantizations[0]
                elif quantizations:
                    selected_quant = "mixed:" + "+".join(quantizations)
        elif model_format == "safetensors":
            inspection_errors: list[str] = []
            for item in selected:
                if not item.relative_path.lower().endswith(".safetensors"):
                    continue
                tensor_path = local_root / item.relative_path
                try:
                    header = inspect_safetensors(tensor_path)
                except SafetensorsHeaderError as exc:
                    inspection_errors.append(f"{item.relative_path}: {exc}")
                    continue
                tensor_descriptors.extend(
                    ResolvedTensorDescriptor(
                        name=tensor.name,
                        shape=tensor.shape,
                        dtype=tensor.dtype,
                        size_bytes=tensor.byte_size,
                        source_file=item.relative_path,
                        data_offset=tensor.data_offset,
                    )
                    for tensor in header
                )
            if inspection_errors:
                artifact_metadata["tensor_inspection_errors"] = tuple(inspection_errors)
        provisional_revision = reference.requested_revision or "local"
        fingerprint = _fingerprint(reference.model_id, provisional_revision, selected)
        resolved_revision = reference.requested_revision or fingerprint
        local_paths = tuple(str((local_root / item.relative_path).resolve()) for item in selected)
        descriptor = ResolvedModelDescriptor(
            model_id=reference.model_id,
            revision=resolved_revision,
            content_fingerprint=_fingerprint(reference.model_id, resolved_revision, selected),
            source_type="local",
            format=model_format,
            architecture=architecture.canonical,
            architecture_raw=architecture.raw,
            architecture_source=architecture.source,
            files=selected,
            variant=selected_variant,
            quantization=selected_quant or _configuration_quantization(config),
            weight_bytes=sum(
                item.size_bytes
                for item in selected
                if item.relative_path.lower().endswith(
                    (".gguf", ".safetensors", ".bin", ".pt", ".pth")
                )
            ),
            layer_count=layers,
            hidden_size=hidden_size,
            activation_dtype_bytes=activation_dtype_bytes,
            parameter_count=parameters,
            tokenizer_identity=_tokenizer_identity(files),
            modalities=_configuration_modalities(config),
            features=_configuration_features(config),
            local_paths=local_paths,
            configuration=config,
            artifact_metadata=artifact_metadata,
            tensors=tuple(tensor_descriptors),
        )
        return ModelResolution(
            _attach_architecture_profile(descriptor), variants, candidates, files
        )

    def _inspect_hub(
        self,
        reference: ModelSourceReference,
        *,
        quantization: str | None,
        objective: Literal["speed", "throughput", "capacity", "balanced"],
        resources: ResolutionResources,
    ) -> ModelResolution:
        info = self._api_client().model_info(
            reference.model_id,
            revision=reference.requested_revision,
            files_metadata=True,
        )
        revision = str(getattr(info, "sha", ""))
        if not _COMMIT.fullmatch(revision):
            raise RuntimeError("Hugging Face did not resolve the reference to an immutable commit")
        siblings = getattr(info, "siblings", None)
        if not siblings:
            raise RuntimeError("Hugging Face repository did not expose a file inventory")
        files = tuple(self._hub_file(item) for item in siblings)
        selected, variants, candidates, selected_variant, selected_quant = self._select_files(
            files,
            reference=reference,
            quantization=quantization,
            objective=objective,
            resources=resources,
        )
        names = {item.relative_path.lower() for item in selected}
        if any(name.endswith(".gguf") for name in names):
            model_format: Literal["safetensors", "gguf", "pytorch", "unknown"] = "gguf"
        elif any(name.endswith(".safetensors") for name in names):
            model_format = "safetensors"
        elif any(name.endswith((".bin", ".pt", ".pth")) for name in names):
            model_format = "pytorch"
        else:
            model_format = "unknown"
        summary_config = getattr(info, "config", None)
        summary_config = summary_config if isinstance(summary_config, dict) else {}
        config, config_identity = self._load_hub_config(
            model_id=reference.model_id,
            revision=revision.lower(),
            files=files,
            summary=summary_config,
        )
        architecture, layers, parameters = _config_facts(config)
        hidden_size, activation_dtype_bytes = _activation_facts(config)
        artifact_metadata: dict[str, Any] = {}
        if config_identity is not None:
            artifact_metadata["config"] = config_identity
        if model_format == "safetensors":
            index_result = self._load_hub_safetensors_index(
                model_id=reference.model_id,
                revision=revision.lower(),
                files=files,
            )
            if index_result is not None:
                index_inventory, index_identity = index_result
                referenced_shards = set(index_inventory.shard_names)
                selected = tuple(
                    item
                    for item in selected
                    if not item.relative_path.casefold().endswith(".safetensors")
                    or item.relative_path in referenced_shards
                )
                artifact_metadata["safetensors_index"] = index_identity
        safetensors_info = getattr(info, "safetensors", None)
        safetensors_total = (
            safetensors_info.get("total")
            if isinstance(safetensors_info, dict)
            else getattr(safetensors_info, "total", None)
        )
        if parameters is None and isinstance(safetensors_total, int):
            parameters = safetensors_total
        if isinstance(safetensors_total, int):
            artifact_metadata["safetensors"] = {"total_parameters": safetensors_total}
        tensor_descriptors: list[ResolvedTensorDescriptor] = []
        if model_format == "gguf":
            first_descriptor = next(
                item for item in selected if item.relative_path.lower().endswith(".gguf")
            )
            inventory = self._load_hub_gguf_metadata(
                model_id=reference.model_id,
                revision=revision.lower(),
                descriptor=first_descriptor,
            )
            gguf_summary = getattr(info, "gguf", None)
            gguf_architecture = inventory.metadata.get("general.architecture")
            architecture = architecture_from_gguf(
                gguf_architecture,
                fallback=architecture,
            )
            config = _gguf_configuration(config, inventory.metadata, architecture)
            layers = next(
                (
                    int(value)
                    for key, value in inventory.metadata.items()
                    if key.endswith("block_count") and isinstance(value, int)
                ),
                layers,
            )
            hidden_size = next(
                (
                    int(value)
                    for key, value in inventory.metadata.items()
                    if key.endswith("embedding_length") and isinstance(value, int) and value > 0
                ),
                hidden_size,
            )
            artifact_metadata["gguf"] = {
                "hub_summary": dict(gguf_summary) if isinstance(gguf_summary, dict) else {},
                "version": inventory.version,
                "data_offset": inventory.data_offset,
                "metadata": inventory.metadata,
                "inspected_file": first_descriptor.relative_path,
            }
            tensor_descriptors.extend(
                ResolvedTensorDescriptor(
                    name=item.name,
                    shape=item.shape,
                    dtype=(item.quantization if item.dtype == "quantized" else item.dtype),
                    size_bytes=item.byte_size,
                    source_file=first_descriptor.relative_path,
                    data_offset=inventory.data_offset + item.offset,
                )
                for item in inventory.tensors
            )
            if parameters is None and inventory.tensors:
                parameters = sum(math.prod(item.shape) for item in inventory.tensors)
            if selected_quant is None:
                quantizations = sorted(
                    {item.quantization for item in inventory.tensors if item.quantization != "none"}
                )
                if len(quantizations) == 1:
                    selected_quant = quantizations[0]
                elif quantizations:
                    selected_quant = "mixed:" + "+".join(quantizations)
        descriptor = ResolvedModelDescriptor(
            model_id=reference.model_id,
            revision=revision.lower(),
            content_fingerprint=_fingerprint(reference.model_id, revision.lower(), selected),
            source_type="huggingface",
            format=model_format,
            architecture=architecture.canonical,
            architecture_raw=architecture.raw,
            architecture_source=architecture.source,
            files=selected,
            variant=selected_variant,
            quantization=selected_quant or _configuration_quantization(config),
            weight_bytes=sum(
                item.size_bytes
                for item in selected
                if item.relative_path.lower().endswith(
                    (".gguf", ".safetensors", ".bin", ".pt", ".pth")
                )
            ),
            layer_count=layers,
            hidden_size=hidden_size,
            activation_dtype_bytes=activation_dtype_bytes,
            parameter_count=parameters,
            tokenizer_identity=_tokenizer_identity(files),
            modalities=_configuration_modalities(config),
            features=_configuration_features(config),
            configuration=config,
            artifact_metadata=artifact_metadata,
            tensors=tuple(tensor_descriptors),
        )
        return ModelResolution(
            _attach_architecture_profile(descriptor), variants, candidates, files
        )

    def acquire(self, descriptor: ResolvedModelDescriptor) -> tuple[Path, ...]:
        """Acquire exactly the resolved files and verify every available digest."""

        if descriptor.source_type == "local":
            local_paths = tuple(Path(item) for item in descriptor.local_paths)
            if len(local_paths) != len(descriptor.files) or not all(
                item.is_file() for item in local_paths
            ):
                raise FileNotFoundError("one or more resolved local model files disappeared")
            return local_paths
        from huggingface_hub import hf_hub_download

        paths: list[Path] = []
        for item in descriptor.files:
            resolved = Path(
                hf_hub_download(
                    repo_id=descriptor.model_id,
                    filename=item.relative_path,
                    revision=descriptor.revision,
                    cache_dir=str(self.cache_directory) if self.cache_directory else None,
                )
            ).resolve()
            if resolved.stat().st_size != item.size_bytes:
                raise RuntimeError(f"downloaded file size differs for {item.relative_path}")
            if item.sha256 is not None and _sha256_file(resolved) != item.sha256:
                raise RuntimeError(f"downloaded file digest differs for {item.relative_path}")
            paths.append(resolved)
        return tuple(paths)

    async def acquire_async(self, descriptor: ResolvedModelDescriptor) -> tuple[Path, ...]:
        return await asyncio.to_thread(self.acquire, descriptor)


__all__ = [
    "ModelResolution",
    "ModelSourceResolver",
    "ResolutionResources",
]
