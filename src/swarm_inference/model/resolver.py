"""Immutable local and Hugging Face model resolution with selective acquisition."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import psutil

from swarm_inference.model.architecture import (
    ArchitectureIdentity,
    architecture_from_config,
    architecture_from_gguf,
)
from swarm_inference.model.descriptor import ModelFileDescriptor, ResolvedModelDescriptor
from swarm_inference.model.gguf import GGUFParseError, inspect_gguf
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

    def __init__(self, *, cache_directory: Path | None = None, api: Any | None = None) -> None:
        self.cache_directory = cache_directory.expanduser().resolve() if cache_directory else None
        self._api = api

    def _api_client(self) -> Any:
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi()
        return self._api

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
            metadata = tuple(
                item
                for item in files
                if item.relative_path.rsplit("/", 1)[-1] in _NATIVE_METADATA_NAMES
            )
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
        if model_format == "gguf":
            first_descriptor = next(
                item for item in selected if item.relative_path.lower().endswith(".gguf")
            )
            first = path if path.is_file() else path / first_descriptor.relative_path
            try:
                inventory = inspect_gguf(first)
            except GGUFParseError:
                raise
            architecture = architecture_from_gguf(
                inventory.metadata.get("general.architecture"), fallback=architecture
            )
            layers = next(
                (
                    int(value)
                    for key, value in inventory.metadata.items()
                    if key.endswith("block_count")
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
        provisional_revision = reference.requested_revision or "local"
        fingerprint = _fingerprint(reference.model_id, provisional_revision, selected)
        resolved_revision = reference.requested_revision or fingerprint
        local_root = path if path.is_dir() else path.parent
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
            quantization=selected_quant,
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
            local_paths=local_paths,
        )
        return ModelResolution(descriptor, variants, candidates, files)

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
        config = getattr(info, "config", None)
        config = config if isinstance(config, dict) else {}
        architecture, layers, parameters = _config_facts(config)
        hidden_size, activation_dtype_bytes = _activation_facts(config)
        if model_format == "gguf":
            gguf_metadata = getattr(info, "gguf", None)
            gguf_architecture = (
                gguf_metadata.get("architecture") if isinstance(gguf_metadata, dict) else None
            )
            architecture = architecture_from_gguf(
                gguf_architecture,
                fallback=architecture,
            )
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
            quantization=selected_quant,
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
        )
        return ModelResolution(descriptor, variants, candidates, files)

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
