"""Content-addressed native-stage artifacts and bounded transfers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, NonNegativeInt, PositiveInt, model_validator
from safetensors import safe_open
from safetensors.torch import save_file

from swarm_inference.cluster.models import (
    ARTIFACT_FORMAT_VERSION,
    ArtifactCacheDocument,
    ArtifactCacheEntry,
    ArtifactChunk,
    ArtifactFile,
    ArtifactLease,
    ArtifactManifest,
    ArtifactTransferStatus,
    ClusterAuditEvent,
)
from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.config.models import StrictModel
from swarm_inference.exceptions import IntegrityError
from swarm_inference.filesystem import replace_atomically
from swarm_inference.model.adapter import (
    ComponentKind,
    NativeModelAdapter,
    NativeModelAdapterRegistry,
    default_native_adapter_registry,
)
from swarm_inference.model.descriptor import ResolvedModelDescriptor
from swarm_inference.model.partition import StageAssignment
from swarm_inference.protocol.cluster import (
    ArtifactOperationRequest,
    ArtifactOperationResponse,
)

_MANIFEST_NAME = "artifact-manifest.json"
_IDENTITY_NAME = "swarm-model-identity.json"
_INDEX_NAME = "model.safetensors.index.json"
_DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024
_MAXIMUM_CHUNK_BYTES = 16 * 1024 * 1024
_DEFAULT_MAXIMUM_TRANSFER_BYTES = 512 * 1024**3
_DEFAULT_MAXIMUM_CHUNKS = 131_072
_COPY_BUFFER_BYTES = 1024 * 1024

_TOKENIZER_ASSET_NAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_COPY_BUFFER_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _normalise_artifact_id(artifact_id: str) -> str:
    value = artifact_id.removeprefix("sha256:").lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("artifact ID must be a SHA-256 content hash")
    return value


def _safe_child(root: Path, relative_path: str) -> Path:
    normalized = relative_path.replace("\\", "/")
    if not normalized or normalized.startswith("/") or ".." in normalized.split("/"):
        raise IntegrityError("artifact contains an unsafe relative path")
    candidate = (root / normalized).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise IntegrityError("artifact path escapes its content directory") from exc
    return candidate


def _atomic_json(path: Path, value: StrictModel | dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, StrictModel):
        payload = value.model_dump_json(indent=2) + "\n"
    else:
        payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        replace_atomically(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _manifest_content_payload(manifest: ArtifactManifest) -> bytes:
    """Hash the complete immutable payload without its self-describing manifest."""

    payload = {
        "artifact_format_version": manifest.artifact_format_version,
        "model_id": manifest.model_id,
        "model_revision": manifest.model_revision,
        "tokenizer_revision": manifest.tokenizer_revision,
        "adapter_id": manifest.adapter_id,
        "source_hashes": dict(sorted(manifest.source_hashes.items())),
        "stage_assignment_id": manifest.stage_assignment_id,
        "dtype": manifest.dtype,
        "quantization": manifest.quantization,
        "layer_start": manifest.layer_start,
        "layer_end": manifest.layer_end,
        "owns_embeddings": manifest.owns_embeddings,
        "owns_final_norm": manifest.owns_final_norm,
        "owns_output_projection": manifest.owns_output_projection,
        "tied_tensor_groups": sorted(sorted(group) for group in manifest.tied_tensor_groups),
        "files": [
            item.model_dump(mode="json")
            for item in sorted(manifest.files, key=lambda value: value.relative_path)
        ],
    }
    if manifest.artifact_format_version >= 2:
        payload.update(
            {
                "model_fingerprint": manifest.model_fingerprint,
                "engine_id": manifest.engine_id,
                "model_format": manifest.model_format,
                "artifact_kind": manifest.artifact_kind,
                "total_bytes": manifest.total_bytes,
            }
        )
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def calculate_manifest_content_hash(manifest: ArtifactManifest) -> str:
    return hashlib.sha256(_manifest_content_payload(manifest)).hexdigest()


def _manifest_total_size(manifest: ArtifactManifest) -> int:
    if manifest.total_size_bytes is None:
        raise IntegrityError("artifact manifest does not declare its total byte size")
    return manifest.total_size_bytes


def load_artifact_manifest(directory: Path) -> ArtifactManifest:
    path = directory.resolve() / _MANIFEST_NAME
    if not path.is_file():
        raise IntegrityError(f"artifact manifest is missing: {path}")
    try:
        return ArtifactManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise IntegrityError(f"artifact manifest is invalid: {path}") from exc


def verify_artifact_directory(
    directory: Path,
    *,
    expected_artifact_id: str | None = None,
) -> ArtifactManifest:
    """Fully verify a published artifact before it can be loaded."""

    root = directory.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise IntegrityError(f"artifact directory is unavailable: {root}")
    manifest = load_artifact_manifest(root)
    artifact_hash = _normalise_artifact_id(manifest.artifact_id)
    if expected_artifact_id is not None and artifact_hash != _normalise_artifact_id(
        expected_artifact_id
    ):
        raise IntegrityError("artifact manifest identity does not match the requested artifact")
    actual_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise IntegrityError("artifact payload cannot contain symbolic links")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative != _MANIFEST_NAME:
                actual_files.add(relative)
    expected_files = {item.relative_path for item in manifest.files}
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise IntegrityError(
            f"artifact payload file set differs from its manifest; missing={missing}, "
            f"unexpected={unexpected}"
        )
    for item in manifest.files:
        path = _safe_child(root, item.relative_path)
        if not path.is_file() or path.stat().st_size != item.size_bytes:
            raise IntegrityError(f"artifact file size mismatch: {item.relative_path}")
        if _sha256_path(path) != item.sha256:
            raise IntegrityError(f"artifact file hash mismatch: {item.relative_path}")
    content_hash = calculate_manifest_content_hash(manifest)
    if content_hash != manifest.content_hash or artifact_hash != content_hash:
        raise IntegrityError("artifact complete content hash is invalid")
    return manifest


def resolve_verified_artifact(cache_root: Path, artifact_id: str) -> Path:
    """Resolve an artifact ID without trusting a host-provided model path."""

    artifact_hash = _normalise_artifact_id(artifact_id)
    directory = cache_root.expanduser().resolve() / artifact_hash
    verify_artifact_directory(directory, expected_artifact_id=artifact_hash)
    return directory


def _assignment_identity(assignment: StageAssignment, *, stage_count: int) -> str:
    payload = {
        "stage_count": stage_count,
        "assignment": assignment.to_dict(),
    }
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )


def _source_revision_evidence(source: Path, model_revision: str) -> None:
    config_path = source / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError("native source config is unavailable or invalid") from exc
    if not isinstance(config, dict):
        raise IntegrityError("native source config must be a JSON object")
    reported = config.get("_commit_hash")
    snapshot_revision = source.name if source.parent.name == "snapshots" else None
    identity_path = source / _IDENTITY_NAME
    identity_revision: str | None = None
    if identity_path.is_file():
        raw = json.loads(identity_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("model_revision"), str):
            identity_revision = raw["model_revision"]
    evidence = [
        value
        for value in (reported, snapshot_revision, identity_revision)
        if isinstance(value, str) and value
    ]
    if not evidence:
        raise IntegrityError("source checkpoint has no immutable revision evidence")
    if any(value != model_revision for value in evidence):
        raise IntegrityError("source checkpoint revision evidence conflicts with the request")


def _tokenizer_revision_evidence(source: Path, tokenizer_revision: str) -> None:
    tokenizer_json = source / "tokenizer.json"
    if tokenizer_revision.startswith("sha256:"):
        if not tokenizer_json.is_file():
            raise IntegrityError("tokenizer SHA-256 identity requires tokenizer.json")
        actual = "sha256:" + _sha256_path(tokenizer_json)
        if actual != tokenizer_revision:
            raise IntegrityError("source tokenizer hash differs from its immutable identity")
        return
    metadata_root = source / ".cache" / "huggingface" / "download"
    revisions: set[str] = set()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        metadata = metadata_root / f"{name}.metadata"
        if metadata.is_file():
            lines = metadata.read_text(encoding="utf-8").splitlines()
            if lines and lines[0].strip():
                revisions.add(lines[0].strip())
    if revisions and revisions != {tokenizer_revision}:
        raise IntegrityError("source tokenizer revision metadata conflicts with the request")


class StageArtifactBuilder:
    """Build adapter-validated stage directories without constructing the model."""

    def __init__(
        self,
        *,
        artifact_root: Path,
        temporary_root: Path,
        adapter_registry: NativeModelAdapterRegistry | None = None,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.artifact_root = artifact_root.expanduser().resolve()
        self.temporary_root = temporary_root.expanduser().resolve()
        self.adapter_registry = adapter_registry or default_native_adapter_registry()
        self.clock_ns = clock_ns
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.temporary_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _selected_tensor_names(
        weight_map: dict[str, str],
        assignment: StageAssignment,
        adapter: NativeModelAdapter,
    ) -> tuple[list[str], list[list[str]], tuple[str, str] | None]:
        selected: list[str] = []
        embedding_names: list[str] = []
        output_names: list[str] = []
        for name in sorted(weight_map):
            component = adapter.map_tensor_to_component(name)
            if component.kind == ComponentKind.EMBEDDING:
                embedding_names.append(name)
                if assignment.owns_embeddings:
                    selected.append(name)
            elif component.kind == ComponentKind.DECODER_LAYER:
                if component.layer_index is None:
                    raise IntegrityError(f"decoder tensor has no layer identity: {name}")
                if assignment.layer_start <= component.layer_index < assignment.layer_end:
                    selected.append(name)
            elif component.kind == ComponentKind.FINAL_NORM and assignment.owns_final_norm:
                selected.append(name)
            elif component.kind == ComponentKind.OUTPUT_HEAD:
                output_names.append(name)
                if assignment.owns_output_projection:
                    selected.append(name)
        tied_groups: list[list[str]] = []
        tied_alias: tuple[str, str] | None = None
        if assignment.owns_output_projection and not output_names:
            source_name = next(
                (name for name in embedding_names if name.endswith(".weight")),
                None,
            )
            if source_name is None:
                raise IntegrityError("output stage has no LM-head tensor or tied embedding tensor")
            alias_name = "lm_head.weight"
            tied_alias = (source_name, alias_name)
            tied_groups.append([source_name, alias_name])
        if not selected and tied_alias is None:
            raise IntegrityError("stage assignment selected no checkpoint tensors")
        return selected, tied_groups, tied_alias

    @staticmethod
    def _copy_metadata(source: Path, temporary: Path, *, include_tokenizer: bool) -> None:
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        _atomic_json(temporary / "config.json", config)
        if include_tokenizer:
            copied = 0
            for name in _TOKENIZER_ASSET_NAMES:
                candidate = source / name
                if candidate.is_file():
                    shutil.copy2(candidate, temporary / name)
                    copied += 1
            if copied == 0:
                raise IntegrityError("stage requiring tokenization assets has no tokenizer files")

    def build(
        self,
        source_model_path: Path,
        *,
        model_id: str,
        model_revision: str,
        tokenizer_revision: str,
        assignment: StageAssignment,
        stage_count: int,
        dtype: str,
        quantization: str = "none",
        model_fingerprint: str | None = None,
        adapter_id: str | None = None,
        before_publish: Callable[[ArtifactManifest], object] | None = None,
    ) -> ArtifactManifest:
        source = source_model_path.expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"source model directory is unavailable: {source}")
        if stage_count <= 0 or not 0 <= assignment.stage_id < stage_count:
            raise ValueError("stage assignment lies outside the requested topology")
        _source_revision_evidence(source, model_revision)
        _tokenizer_revision_evidence(source, tokenizer_revision)
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        adapter = (
            self.adapter_registry.get(adapter_id)
            if adapter_id is not None
            else self.adapter_registry.resolve_config(config)
        )
        if not adapter.supports(config):
            raise IntegrityError(
                f"native adapter {adapter.adapter_id!r} rejected the checkpoint config"
            )
        validate_assignment = getattr(adapter, "validate_stage_assignment", None)
        if not callable(validate_assignment):
            raise IntegrityError(
                f"native adapter {adapter.adapter_id!r} cannot validate stage assignments"
            )
        validate_assignment(
            source,
            assignment=assignment,
            stage_count=stage_count,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
        )
        index_path = source / _INDEX_NAME
        try:
            raw_index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IntegrityError("source native safetensors index is invalid") from exc
        raw_weight_map = raw_index.get("weight_map") if isinstance(raw_index, dict) else None
        if not isinstance(raw_weight_map, dict) or not raw_weight_map:
            raise IntegrityError("source native safetensors index has no weight map")
        weight_map = {str(name): str(shard) for name, shard in raw_weight_map.items()}
        selected, tied_groups, tied_alias = self._selected_tensor_names(
            weight_map, assignment, adapter
        )
        selected_set = set(selected)
        temporary = self.temporary_root / f"artifact-{uuid4().hex}.partial"
        temporary.mkdir(parents=False, exist_ok=False)
        try:
            include_tokenizer = assignment.stage_id in {0, stage_count - 1}
            self._copy_metadata(source, temporary, include_tokenizer=include_tokenizer)
            source_hashes: dict[str, str] = {
                "config.json": _sha256_path(source / "config.json"),
                _INDEX_NAME: _sha256_path(index_path),
            }
            output_weight_map: dict[str, str] = {}
            tensor_names_by_file: dict[str, list[str]] = {}
            by_source_shard: dict[str, list[str]] = {}
            for name in selected:
                by_source_shard.setdefault(weight_map[name], []).append(name)
            for shard_index, (source_shard, names) in enumerate(
                sorted(by_source_shard.items()), start=1
            ):
                source_path = _safe_child(source, source_shard)
                if not source_path.is_file():
                    raise IntegrityError(f"source tensor shard is missing: {source_shard}")
                source_hashes[source_shard] = _sha256_path(source_path)
                output_name = f"stage-{assignment.stage_id:02d}-{shard_index:05d}.safetensors"
                tensors = {}
                with safe_open(source_path, framework="pt", device="cpu") as handle:
                    available = set(handle.keys())
                    for name in sorted(names):
                        if name not in available:
                            raise IntegrityError(f"tensor index references missing tensor {name}")
                        tensors[name] = handle.get_tensor(name)
                save_file(tensors, temporary / output_name, metadata={"format": "pt"})
                for name in sorted(tensors):
                    output_weight_map[name] = output_name
                tensor_names_by_file[output_name] = sorted(tensors)
                del tensors
            if tied_alias is not None:
                source_name, alias_name = tied_alias
                source_shard = weight_map[source_name]
                source_path = _safe_child(source, source_shard)
                source_hashes.setdefault(source_shard, _sha256_path(source_path))
                output_name = f"stage-{assignment.stage_id:02d}-tied.safetensors"
                with safe_open(source_path, framework="pt", device="cpu") as handle:
                    tensor = handle.get_tensor(source_name)
                save_file({alias_name: tensor}, temporary / output_name, metadata={"format": "pt"})
                output_weight_map[alias_name] = output_name
                tensor_names_by_file[output_name] = [alias_name]
                del tensor
            for name in output_weight_map:
                component = adapter.map_tensor_to_component(name)
                if component.kind == ComponentKind.DECODER_LAYER and (
                    component.layer_index is None
                    or not (assignment.layer_start <= component.layer_index < assignment.layer_end)
                ):
                    raise IntegrityError("unassigned layer tensor entered the stage artifact")
            if not selected_set.issubset(output_weight_map):
                raise IntegrityError("stage artifact omitted an assigned tensor")
            _atomic_json(
                temporary / _INDEX_NAME,
                {
                    "metadata": {
                        "total_size": sum(
                            path.stat().st_size for path in temporary.glob("*.safetensors")
                        ),
                        "swarm_artifact_format_version": ARTIFACT_FORMAT_VERSION,
                    },
                    "weight_map": dict(sorted(output_weight_map.items())),
                },
            )
            identity = {
                "schema_version": 1,
                "model_id": model_id,
                "model_revision": model_revision,
                "tokenizer_revision": tokenizer_revision,
                "adapter_id": adapter.adapter_id,
                "stage_assignment_id": _assignment_identity(assignment, stage_count=stage_count),
                "source_hashes": dict(sorted(source_hashes.items())),
            }
            _atomic_json(temporary / _IDENTITY_NAME, identity)
            files: list[ArtifactFile] = []
            for path in sorted(temporary.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(temporary).as_posix()
                    files.append(
                        ArtifactFile(
                            relative_path=relative,
                            size_bytes=path.stat().st_size,
                            sha256=_sha256_path(path),
                            media_type=(
                                "application/x-safetensors"
                                if path.suffix == ".safetensors"
                                else "application/json"
                                if path.suffix == ".json"
                                else "application/octet-stream"
                            ),
                            tensor_names=tensor_names_by_file.get(relative, []),
                        )
                    )
            fingerprint_payload = json.dumps(
                {
                    "model_id": model_id,
                    "model_revision": model_revision,
                    "source_hashes": dict(sorted(source_hashes.items())),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            resolved_model_fingerprint = model_fingerprint or (
                "sha256:" + hashlib.sha256(fingerprint_payload).hexdigest()
            )
            total_bytes = sum(item.size_bytes for item in files)
            provisional = ArtifactManifest(
                artifact_id="0" * 64,
                model_id=model_id,
                model_revision=model_revision,
                model_fingerprint=resolved_model_fingerprint,
                tokenizer_revision=tokenizer_revision,
                engine_id="native-stage",
                model_format="safetensors",
                artifact_kind="native-stage",
                adapter_id=adapter.adapter_id,
                source_hashes=dict(sorted(source_hashes.items())),
                stage_assignment_id=_assignment_identity(assignment, stage_count=stage_count),
                dtype=dtype,
                quantization=quantization,
                content_hash="0" * 64,
                layer_start=assignment.layer_start,
                layer_end=assignment.layer_end,
                owns_embeddings=assignment.owns_embeddings,
                owns_final_norm=assignment.owns_final_norm,
                owns_output_projection=assignment.owns_output_projection,
                tied_tensor_groups=tied_groups,
                files=files,
                total_size_bytes=total_bytes,
                total_bytes=total_bytes,
                created_at_unix_ns=self.clock_ns(),
            )
            content_hash = calculate_manifest_content_hash(provisional)
            manifest = provisional.model_copy(
                update={"artifact_id": content_hash, "content_hash": content_hash}
            )
            _atomic_json(temporary / _MANIFEST_NAME, manifest)
            verify_artifact_directory(temporary, expected_artifact_id=content_hash)
            destination = self.artifact_root / content_hash
            if destination.exists():
                verify_artifact_directory(destination, expected_artifact_id=content_hash)
                shutil.rmtree(temporary)
            else:
                if before_publish is not None:
                    before_publish(manifest)
                os.replace(temporary, destination)
                verify_artifact_directory(destination, expected_artifact_id=content_hash)
            return manifest
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


class ModelArtifactBuilder:
    """Publish selected immutable model files as one engine-neutral artifact."""

    def __init__(self, *, artifact_root: Path, temporary_root: Path) -> None:
        self.artifact_root = artifact_root.expanduser().resolve()
        self.temporary_root = temporary_root.expanduser().resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.temporary_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _media_type(path: Path) -> str:
        if path.suffix.lower() == ".gguf":
            return "application/x-gguf"
        if path.suffix.lower() == ".safetensors":
            return "application/x-safetensors"
        if path.suffix.lower() == ".json":
            return "application/json"
        return "application/octet-stream"

    def build(
        self,
        descriptor: ResolvedModelDescriptor,
        *,
        engine_id: str,
        before_publish: Callable[[ArtifactManifest], object] | None = None,
    ) -> ArtifactManifest:
        if not descriptor.local_paths:
            raise FileNotFoundError("model files have not been acquired")
        if len(descriptor.files) != len(descriptor.local_paths):
            raise IntegrityError("model descriptor file/path counts differ")
        artifact_kind: Literal["gguf", "converted-model"] = (
            "gguf" if descriptor.format == "gguf" else "converted-model"
        )
        temporary = self.temporary_root / f"artifact-{uuid4().hex}.partial"
        temporary.mkdir(parents=False, exist_ok=False)
        try:
            files: list[ArtifactFile] = []
            source_hashes: dict[str, str] = {}
            for selected, raw_path in zip(
                descriptor.files,
                descriptor.local_paths,
                strict=True,
            ):
                source = Path(raw_path).expanduser().resolve()
                if not source.is_file() or source.stat().st_size != selected.size_bytes:
                    raise IntegrityError(
                        f"acquired model file size differs from metadata: {selected.relative_path}"
                    )
                destination = _safe_child(temporary, selected.relative_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
                actual_hash = _sha256_path(destination)
                if (
                    selected.sha256 is not None
                    and actual_hash != selected.sha256.removeprefix("sha256:").lower()
                ):
                    raise IntegrityError(
                        f"acquired model file hash differs from metadata: {selected.relative_path}"
                    )
                source_hashes[selected.relative_path] = actual_hash
                files.append(
                    ArtifactFile(
                        relative_path=selected.relative_path,
                        size_bytes=selected.size_bytes,
                        sha256=actual_hash,
                        media_type=self._media_type(destination),
                    )
                )
            files.sort(
                key=lambda item: (
                    0 if item.relative_path.lower().endswith(".gguf") else 1,
                    item.relative_path,
                )
            )
            total_bytes = sum(item.size_bytes for item in files)
            provisional = ArtifactManifest(
                artifact_id="0" * 64,
                model_id=descriptor.model_id,
                model_revision=descriptor.revision,
                model_fingerprint=descriptor.content_fingerprint,
                tokenizer_revision=descriptor.tokenizer_identity,
                engine_id=engine_id,
                model_format=descriptor.format,
                artifact_kind=artifact_kind,
                source_hashes=dict(sorted(source_hashes.items())),
                quantization=descriptor.quantization,
                content_hash="0" * 64,
                files=files,
                total_size_bytes=total_bytes,
                total_bytes=total_bytes,
            )
            content_hash = calculate_manifest_content_hash(provisional)
            manifest = provisional.model_copy(
                update={"artifact_id": content_hash, "content_hash": content_hash}
            )
            _atomic_json(temporary / _MANIFEST_NAME, manifest)
            verify_artifact_directory(temporary, expected_artifact_id=content_hash)
            destination = self.artifact_root / content_hash
            if destination.exists():
                verify_artifact_directory(destination, expected_artifact_id=content_hash)
                shutil.rmtree(temporary)
            else:
                if before_publish is not None:
                    before_publish(manifest)
                os.replace(temporary, destination)
                verify_artifact_directory(destination, expected_artifact_id=content_hash)
            return manifest
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def build_native_stage_artifact(
    adapter: NativeModelAdapter,
    *args: Any,
    builder: StageArtifactBuilder | None = None,
    **kwargs: Any,
) -> ArtifactManifest:
    """Adapter entry point for the one canonical stage-artifact builder."""

    remaining = list(args)
    selected_builder = builder
    if selected_builder is None and remaining and isinstance(remaining[0], StageArtifactBuilder):
        selected_builder = remaining.pop(0)
    if selected_builder is None:
        try:
            artifact_root = Path(kwargs.pop("artifact_root"))
            temporary_root = Path(kwargs.pop("temporary_root"))
        except KeyError as exc:
            raise TypeError(
                "native artifact construction requires a builder or artifact/temporary roots"
            ) from exc
        selected_builder = StageArtifactBuilder(
            artifact_root=artifact_root,
            temporary_root=temporary_root,
            adapter_registry=NativeModelAdapterRegistry((adapter,)),
        )
    return selected_builder.build(
        *remaining,
        adapter_id=adapter.adapter_id,
        **kwargs,
    )


class ArtifactTransferResume(StrictModel):
    """Versioned recovery document for a bounded, restart-safe transfer."""

    schema_version: Literal[1] = 1
    document_version: Literal[1] = 1
    transfer_id: str
    artifact_id: str
    source: str
    destination_node_id: str
    completed_chunk_keys: list[str] = Field(default_factory=list)
    bytes_completed: NonNegativeInt = 0
    updated_at_unix_ns: PositiveInt

    @model_validator(mode="after")
    def validate_unique_chunks(self) -> ArtifactTransferResume:
        if len(set(self.completed_chunk_keys)) != len(self.completed_chunk_keys):
            raise ValueError("artifact transfer resume chunks must be unique")
        return self


class ArtifactManager:
    """Own cache verification, leases, resumable transfers, and bounded LRU."""

    def __init__(
        self,
        *,
        state: ClusterStateStore,
        node_id: str,
        storage_limit_bytes: int,
        chunk_size_bytes: int = _DEFAULT_CHUNK_BYTES,
        maximum_transfer_bytes: int = _DEFAULT_MAXIMUM_TRANSFER_BYTES,
        maximum_chunks: int = _DEFAULT_MAXIMUM_CHUNKS,
        clock_ns: Callable[[], int] = time.time_ns,
        event_sink: Callable[[str, dict[str, object]], None] | None = None,
    ) -> None:
        if storage_limit_bytes <= 0:
            raise ValueError("artifact storage limit must be positive")
        if not 0 < chunk_size_bytes <= _MAXIMUM_CHUNK_BYTES:
            raise ValueError("artifact chunks must use a positive bounded size")
        if maximum_transfer_bytes <= 0 or maximum_chunks <= 0:
            raise ValueError("artifact transfer limits must be positive")
        self.state = state
        self.node_id = node_id
        self.storage_limit_bytes = storage_limit_bytes
        self.chunk_size_bytes = chunk_size_bytes
        self.maximum_transfer_bytes = maximum_transfer_bytes
        self.maximum_chunks = maximum_chunks
        self.clock_ns = clock_ns
        self.event_sink = event_sink
        self._lock = threading.RLock()

    def _emit(self, event_type: str, **fields: object) -> None:
        if self.event_sink is not None:
            self.event_sink(event_type, fields)
        self.state.append_audit(
            ClusterAuditEvent(
                event_id=f"event-{uuid4().hex}",
                event_type=event_type,
                timestamp_unix_ns=self.clock_ns(),
                node_id=self.node_id,
                category=str(fields.get("category")) if fields.get("category") else None,
                detail=str(fields.get("detail")) if fields.get("detail") else None,
            )
        )

    def _document(self) -> ArtifactCacheDocument:
        document = self.state.load_artifact_cache()
        now = self.clock_ns()
        active_leases = [
            lease
            for lease in document.leases
            if lease.expires_at_unix_ns is None or lease.expires_at_unix_ns > now
        ]
        active_ids = {lease.lease_id for lease in active_leases}
        entries = [
            entry.model_copy(
                update={
                    "active_lease_ids": sorted(
                        lease_id for lease_id in entry.active_lease_ids if lease_id in active_ids
                    )
                }
            )
            for entry in document.entries
        ]
        if active_leases != document.leases or entries != document.entries:
            document = document.model_copy(update={"leases": active_leases, "entries": entries})
            self.state.save_artifact_cache(document)
        return document

    def _save(self, document: ArtifactCacheDocument) -> None:
        self.state.save_artifact_cache(document)

    def entries(self) -> list[ArtifactCacheEntry]:
        with self._lock:
            return [item.model_copy(deep=True) for item in self._document().entries]

    def transfers(self) -> list[ArtifactTransferStatus]:
        with self._lock:
            return [item.model_copy(deep=True) for item in self._document().transfers]

    def _entry_directory(self, artifact_id: str) -> Path:
        return self.state.paths.artifacts / _normalise_artifact_id(artifact_id)

    def register(self, artifact_id: str) -> ArtifactCacheEntry:
        with self._lock:
            artifact_hash = _normalise_artifact_id(artifact_id)
            directory = self._entry_directory(artifact_hash)
            manifest = verify_artifact_directory(directory, expected_artifact_id=artifact_hash)
            now = self.clock_ns()
            document = self._document()
            existing = next(
                (item for item in document.entries if item.artifact_id == manifest.artifact_id),
                None,
            )
            entry = ArtifactCacheEntry(
                artifact_id=manifest.artifact_id,
                manifest=manifest,
                relative_directory=artifact_hash,
                size_bytes=manifest.total_size_bytes,
                created_at_unix_ns=(existing.created_at_unix_ns if existing is not None else now),
                last_accessed_at_unix_ns=now,
                verified_at_unix_ns=now,
                pinned=existing.pinned if existing is not None else False,
                active_lease_ids=(existing.active_lease_ids if existing is not None else []),
            )
            values = [item for item in document.entries if item.artifact_id != manifest.artifact_id]
            values.append(entry)
            self._save(
                document.model_copy(
                    update={"entries": sorted(values, key=lambda item: item.artifact_id)}
                )
            )
            self._emit("artifact_verified", detail=f"artifact {artifact_hash[:12]} verified")
            return entry.model_copy(deep=True)

    def resolve(self, artifact_id: str) -> Path:
        with self._lock:
            artifact_hash = _normalise_artifact_id(artifact_id)
            document = self._document()
            entry = next(
                (
                    item
                    for item in document.entries
                    if _normalise_artifact_id(item.artifact_id) == artifact_hash
                ),
                None,
            )
            if entry is None:
                if not self._entry_directory(artifact_hash).is_dir():
                    raise FileNotFoundError(f"artifact {artifact_hash} is not cached")
                entry = self.register(artifact_hash)
                document = self._document()
            directory = self.state.paths.artifacts / entry.relative_directory
            verify_artifact_directory(directory, expected_artifact_id=artifact_hash)
            updated = entry.model_copy(
                update={
                    "last_accessed_at_unix_ns": self.clock_ns(),
                    "verified_at_unix_ns": self.clock_ns(),
                }
            )
            entries = [
                updated if item.artifact_id == entry.artifact_id else item
                for item in document.entries
            ]
            self._save(document.model_copy(update={"entries": entries}))
            return directory

    def lease(
        self,
        artifact_id: str,
        *,
        owner: str,
        purpose: Literal["loaded-stage", "deployment", "transfer", "pinned"],
        expires_at_unix_ns: int | None = None,
    ) -> ArtifactLease:
        with self._lock:
            entry = self.register(artifact_id)
            now = self.clock_ns()
            if expires_at_unix_ns is not None and expires_at_unix_ns <= now:
                raise ValueError("artifact lease expiry must be in the future")
            lease = ArtifactLease(
                lease_id=f"lease-{uuid4().hex}",
                artifact_id=entry.artifact_id,
                owner=owner,
                purpose=purpose,
                created_at_unix_ns=now,
                expires_at_unix_ns=expires_at_unix_ns,
            )
            document = self._document()
            entries = [
                item.model_copy(
                    update={"active_lease_ids": sorted([*item.active_lease_ids, lease.lease_id])}
                )
                if item.artifact_id == entry.artifact_id
                else item
                for item in document.entries
            ]
            self._save(
                document.model_copy(
                    update={"entries": entries, "leases": [*document.leases, lease]}
                )
            )
            return lease.model_copy(deep=True)

    def release(self, lease_id: str) -> bool:
        with self._lock:
            document = self._document()
            if not any(item.lease_id == lease_id for item in document.leases):
                return False
            leases = [item for item in document.leases if item.lease_id != lease_id]
            entries = [
                item.model_copy(
                    update={
                        "active_lease_ids": [
                            value for value in item.active_lease_ids if value != lease_id
                        ]
                    }
                )
                for item in document.entries
            ]
            self._save(document.model_copy(update={"entries": entries, "leases": leases}))
            return True

    def pin(self, artifact_id: str, *, pinned: bool = True) -> None:
        with self._lock:
            entry = self.register(artifact_id)
            document = self._document()
            entries = [
                item.model_copy(update={"pinned": pinned})
                if item.artifact_id == entry.artifact_id
                else item
                for item in document.entries
            ]
            self._save(document.model_copy(update={"entries": entries}))

    def evict_to_fit(self, incoming_bytes: int) -> list[str]:
        if incoming_bytes < 0 or incoming_bytes > self.storage_limit_bytes:
            raise ValueError("incoming artifact exceeds the configured storage budget")
        with self._lock:
            document = self._document()
            used = sum(item.size_bytes for item in document.entries)
            required = used + incoming_bytes - self.storage_limit_bytes
            if required <= 0:
                return []
            active_transfers = {
                item.artifact_id
                for item in document.transfers
                if item.state in {"queued", "transferring", "verifying"}
            }
            candidates = sorted(
                (
                    item
                    for item in document.entries
                    if not item.pinned
                    and not item.active_lease_ids
                    and item.artifact_id not in active_transfers
                ),
                key=lambda item: (item.last_accessed_at_unix_ns, item.artifact_id),
            )
            selected: list[ArtifactCacheEntry] = []
            recovered = 0
            for item in candidates:
                selected.append(item)
                recovered += item.size_bytes
                if recovered >= required:
                    break
            if recovered < required:
                raise OSError("artifact storage budget is exhausted by active or pinned artifacts")
            artifact_root = self.state.paths.artifacts.resolve()
            for item in selected:
                directory = (artifact_root / item.relative_directory).resolve()
                try:
                    directory.relative_to(artifact_root)
                except ValueError as exc:
                    raise IntegrityError("cache entry points outside the artifact root") from exc
                if directory.is_dir():
                    shutil.rmtree(directory)
                self._emit("artifact_evicted", detail=f"artifact {item.artifact_id[:12]} evicted")
            removed = {item.artifact_id for item in selected}
            self._save(
                document.model_copy(
                    update={
                        "entries": [
                            item for item in document.entries if item.artifact_id not in removed
                        ]
                    }
                )
            )
            return sorted(removed)

    def _resume_path(self, artifact_id: str) -> Path:
        return self.state.paths.downloads / f"{_normalise_artifact_id(artifact_id)}.resume.json"

    def _partial_directory(self, artifact_id: str) -> Path:
        return self.state.paths.downloads / f"{_normalise_artifact_id(artifact_id)}.partial"

    def _load_resume(self, artifact_id: str) -> ArtifactTransferResume | None:
        path = self._resume_path(artifact_id)
        if not path.is_file():
            return None
        try:
            return ArtifactTransferResume.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise IntegrityError("artifact transfer recovery document is invalid") from exc

    def _save_transfer(
        self, status: ArtifactTransferStatus, resume: ArtifactTransferResume
    ) -> None:
        document = self._document()
        transfers = [item for item in document.transfers if item.transfer_id != status.transfer_id]
        transfers.append(status)
        self._save(
            document.model_copy(
                update={"transfers": sorted(transfers, key=lambda item: item.transfer_id)}
            )
        )
        _atomic_json(self._resume_path(status.artifact_id), resume)

    def _chunk_plan(self, source: Path, manifest: ArtifactManifest) -> list[ArtifactChunk]:
        chunks: list[ArtifactChunk] = []
        chunk_index = 0
        for item in sorted(manifest.files, key=lambda value: value.relative_path):
            path = _safe_child(source, item.relative_path)
            with path.open("rb") as handle:
                offset = 0
                while offset < item.size_bytes:
                    data = handle.read(min(self.chunk_size_bytes, item.size_bytes - offset))
                    if not data:
                        raise IntegrityError("source artifact ended before its declared size")
                    chunks.append(
                        ArtifactChunk(
                            artifact_id=manifest.artifact_id,
                            relative_path=item.relative_path,
                            chunk_index=chunk_index,
                            offset_bytes=offset,
                            size_bytes=len(data),
                            sha256=hashlib.sha256(data).hexdigest(),
                            final_chunk=offset + len(data) == item.size_bytes,
                        )
                    )
                    offset += len(data)
                    chunk_index += 1
                    if len(chunks) > self.maximum_chunks:
                        raise ValueError("artifact transfer exceeds the bounded chunk count")
        return chunks

    @staticmethod
    def _chunk_key(chunk: ArtifactChunk) -> str:
        return f"{chunk.chunk_index}:{chunk.relative_path}:{chunk.offset_bytes}:{chunk.sha256}"

    def write_chunk(
        self,
        *,
        transfer_id: str,
        chunk: ArtifactChunk,
        payload: bytes,
    ) -> None:
        """Validate and persist one bounded peer chunk for a prepared transfer."""

        if len(payload) > self.chunk_size_bytes or len(payload) != chunk.size_bytes:
            raise IntegrityError("artifact chunk length is invalid")
        if hashlib.sha256(payload).hexdigest() != chunk.sha256:
            raise IntegrityError("artifact chunk hash mismatch")
        with self._lock:
            resume = self._load_resume(chunk.artifact_id)
            if resume is None or resume.transfer_id != transfer_id:
                raise IntegrityError("artifact transfer has no matching recovery state")
            key = self._chunk_key(chunk)
            if key in resume.completed_chunk_keys:
                return
            partial = self._partial_directory(chunk.artifact_id)
            partial.mkdir(parents=True, exist_ok=True)
            destination = _safe_child(partial, chunk.relative_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            mode = "r+b" if destination.exists() else "w+b"
            with destination.open(mode) as handle:
                handle.seek(chunk.offset_bytes)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            updated = resume.model_copy(
                update={
                    "completed_chunk_keys": sorted([*resume.completed_chunk_keys, key]),
                    "bytes_completed": resume.bytes_completed + len(payload),
                    "updated_at_unix_ns": self.clock_ns(),
                }
            )
            document = self._document()
            status = next(
                (item for item in document.transfers if item.transfer_id == transfer_id), None
            )
            if status is None:
                raise IntegrityError("artifact transfer status is missing")
            status = status.model_copy(
                update={
                    "state": "transferring",
                    "bytes_completed": updated.bytes_completed,
                    "chunks_completed": len(updated.completed_chunk_keys),
                    "updated_at_unix_ns": self.clock_ns(),
                }
            )
            self._save_transfer(status, updated)

    def _begin_transfer(
        self,
        *,
        manifest: ArtifactManifest,
        source: str,
        chunks_total: int,
    ) -> tuple[ArtifactTransferStatus, ArtifactTransferResume]:
        resume = self._load_resume(manifest.artifact_id)
        document = self._document()
        if resume is not None:
            status = next(
                (
                    item
                    for item in document.transfers
                    if item.transfer_id == resume.transfer_id
                    and item.artifact_id == manifest.artifact_id
                ),
                None,
            )
            if status is None or resume.source != source:
                raise IntegrityError("artifact transfer recovery identity changed")
            if (
                status.bytes_total != manifest.total_size_bytes
                or status.chunks_total != chunks_total
            ):
                raise IntegrityError("artifact transfer recovery bounds changed")
            return status, resume
        now = self.clock_ns()
        transfer_id = f"transfer-{uuid4().hex}"
        status = ArtifactTransferStatus(
            transfer_id=transfer_id,
            artifact_id=manifest.artifact_id,
            source=source,
            destination_node_id=self.node_id,
            state="queued",
            bytes_total=manifest.total_size_bytes,
            bytes_completed=0,
            chunks_total=chunks_total,
            chunks_completed=0,
            started_at_unix_ns=now,
            updated_at_unix_ns=now,
        )
        resume = ArtifactTransferResume(
            transfer_id=transfer_id,
            artifact_id=manifest.artifact_id,
            source=source,
            destination_node_id=self.node_id,
            updated_at_unix_ns=now,
        )
        self._save_transfer(status, resume)
        return status, resume

    def prepare_incoming(
        self,
        *,
        manifest: ArtifactManifest,
        source: str,
        chunks_total: int,
    ) -> ArtifactTransferStatus:
        """Reserve bounded storage and create or resume an authenticated transfer."""

        if not source.strip():
            raise ValueError("artifact transfer source cannot be empty")
        if chunks_total < 0 or chunks_total > self.maximum_chunks:
            raise ValueError("artifact transfer exceeds the bounded chunk count")
        artifact_hash = _normalise_artifact_id(manifest.artifact_id)
        if artifact_hash != calculate_manifest_content_hash(manifest):
            raise IntegrityError("artifact manifest complete content hash is invalid")
        total_size_bytes = _manifest_total_size(manifest)
        if total_size_bytes > self.maximum_transfer_bytes:
            raise ValueError("artifact exceeds the configured transfer byte bound")
        with self._lock:
            destination = self._entry_directory(artifact_hash)
            if destination.is_dir():
                entry = self.register(artifact_hash)
                now = self.clock_ns()
                return ArtifactTransferStatus(
                    transfer_id=f"existing-{artifact_hash[:16]}",
                    artifact_id=artifact_hash,
                    source=source,
                    destination_node_id=self.node_id,
                    state="complete",
                    bytes_total=entry.size_bytes,
                    bytes_completed=entry.size_bytes,
                    chunks_total=chunks_total,
                    chunks_completed=chunks_total,
                    started_at_unix_ns=now,
                    updated_at_unix_ns=now,
                )
            self.evict_to_fit(total_size_bytes)
            status, _ = self._begin_transfer(
                manifest=manifest,
                source=source,
                chunks_total=chunks_total,
            )
            return status.model_copy(deep=True)

    def complete_incoming(
        self,
        *,
        transfer_id: str,
        manifest: ArtifactManifest,
    ) -> ArtifactTransferStatus:
        """Verify every declared byte and atomically publish a prepared transfer."""

        with self._lock:
            document = self._document()
            status = next(
                (item for item in document.transfers if item.transfer_id == transfer_id),
                None,
            )
            if status is None or status.artifact_id != manifest.artifact_id:
                raise IntegrityError("artifact completion has no matching transfer")
            if status.state == "complete":
                self.resolve(manifest.artifact_id)
                return status.model_copy(deep=True)
            resume = self._load_resume(manifest.artifact_id)
            if resume is None or resume.transfer_id != transfer_id:
                raise IntegrityError("artifact completion recovery identity mismatch")
            if (
                status.bytes_completed != manifest.total_size_bytes
                or status.chunks_completed != status.chunks_total
                or resume.bytes_completed != manifest.total_size_bytes
                or len(resume.completed_chunk_keys) != status.chunks_total
            ):
                raise IntegrityError("artifact transfer is incomplete")
            return self._publish_transfer(status, manifest).model_copy(deep=True)

    def _publish_transfer(
        self,
        status: ArtifactTransferStatus,
        manifest: ArtifactManifest,
    ) -> ArtifactTransferStatus:
        partial = self._partial_directory(manifest.artifact_id)
        _atomic_json(partial / _MANIFEST_NAME, manifest)
        status = status.model_copy(
            update={"state": "verifying", "updated_at_unix_ns": self.clock_ns()}
        )
        resume = self._load_resume(manifest.artifact_id)
        if resume is None:
            raise IntegrityError("artifact transfer recovery state disappeared")
        self._save_transfer(status, resume)
        verify_artifact_directory(partial, expected_artifact_id=manifest.artifact_id)
        destination = self._entry_directory(manifest.artifact_id)
        if destination.exists():
            verify_artifact_directory(destination, expected_artifact_id=manifest.artifact_id)
            shutil.rmtree(partial)
        else:
            os.replace(partial, destination)
        self._resume_path(manifest.artifact_id).unlink(missing_ok=True)
        completed = status.model_copy(
            update={
                "state": "complete",
                "bytes_completed": status.bytes_total,
                "chunks_completed": status.chunks_total,
                "updated_at_unix_ns": self.clock_ns(),
                "last_error": None,
            }
        )
        document = self._document()
        transfers = [
            completed if item.transfer_id == completed.transfer_id else item
            for item in document.transfers
        ]
        self._save(document.model_copy(update={"transfers": transfers}))
        self.register(manifest.artifact_id)
        self._emit(
            "artifact_transferred",
            detail=f"artifact {manifest.artifact_id[:12]} transfer complete",
        )
        return completed

    def transfer_from_directory(
        self,
        source_directory: Path,
        *,
        peer_authenticated: bool,
        maximum_chunks_this_call: int | None = None,
    ) -> ArtifactTransferStatus:
        """Copy from an authenticated peer with resumable, hash-bound chunks."""

        if not peer_authenticated:
            raise PermissionError("artifact peer source is not authenticated cluster membership")
        if maximum_chunks_this_call is not None and maximum_chunks_this_call <= 0:
            raise ValueError("per-call artifact chunk limit must be positive")
        source = source_directory.expanduser().resolve()
        manifest = verify_artifact_directory(source)
        total_size_bytes = _manifest_total_size(manifest)
        if total_size_bytes > self.maximum_transfer_bytes:
            raise ValueError("artifact exceeds the configured transfer byte bound")
        chunks = self._chunk_plan(source, manifest)
        with self._lock:
            if self._entry_directory(manifest.artifact_id).is_dir():
                entry = self.register(manifest.artifact_id)
                now = self.clock_ns()
                return ArtifactTransferStatus(
                    transfer_id=f"existing-{manifest.artifact_id[:16]}",
                    artifact_id=manifest.artifact_id,
                    source=str(source),
                    destination_node_id=self.node_id,
                    state="complete",
                    bytes_total=entry.size_bytes,
                    bytes_completed=entry.size_bytes,
                    chunks_total=len(chunks),
                    chunks_completed=len(chunks),
                    started_at_unix_ns=now,
                    updated_at_unix_ns=now,
                )
            self.evict_to_fit(total_size_bytes)
            status, resume = self._begin_transfer(
                manifest=manifest, source=str(source), chunks_total=len(chunks)
            )
            completed = set(resume.completed_chunk_keys)
            written = 0
            for chunk in chunks:
                if self._chunk_key(chunk) in completed:
                    continue
                if maximum_chunks_this_call is not None and written >= maximum_chunks_this_call:
                    break
                source_file = _safe_child(source, chunk.relative_path)
                with source_file.open("rb") as handle:
                    handle.seek(chunk.offset_bytes)
                    payload = handle.read(chunk.size_bytes)
                self.write_chunk(transfer_id=status.transfer_id, chunk=chunk, payload=payload)
                written += 1
            current_resume = self._load_resume(manifest.artifact_id)
            if current_resume is None:
                raise IntegrityError("artifact transfer recovery state disappeared")
            if len(current_resume.completed_chunk_keys) < len(chunks):
                return next(
                    item
                    for item in self._document().transfers
                    if item.transfer_id == status.transfer_id
                ).model_copy(deep=True)
            latest = next(
                item
                for item in self._document().transfers
                if item.transfer_id == status.transfer_id
            )
            return self._publish_transfer(latest, manifest).model_copy(deep=True)


class ArtifactOperationCoordinator:
    """Bounded coordinator directory backed by the node artifact manager."""

    def __init__(self, manager: ArtifactManager) -> None:
        self.manager = manager

    async def handle(self, request: ArtifactOperationRequest) -> ArtifactOperationResponse:
        if request.operation == "locate":
            try:
                self.manager.resolve(request.artifact_id)
            except FileNotFoundError:
                return ArtifactOperationResponse(
                    accepted=True,
                    detail="exact artifact is not present on the coordinator node",
                )
            return ArtifactOperationResponse(
                accepted=True,
                locations=[self.manager.node_id],
                detail="exact verified artifact is present",
            )
        if request.operation == "status":
            matching = [
                item for item in self.manager.transfers() if item.artifact_id == request.artifact_id
            ]
            transfer = max(matching, key=lambda item: item.updated_at_unix_ns, default=None)
            return ArtifactOperationResponse(
                accepted=True,
                transfer=transfer,
                detail="transfer status found" if transfer else "no transfer exists",
            )
        if request.operation == "prepare":
            if request.manifest is None or request.source_node_id is None:
                raise ValueError("artifact prepare requires a manifest and source node")
            if request.chunks_total is None:
                raise ValueError("artifact prepare requires a bounded chunk count")
            if request.maximum_bytes is not None and (
                _manifest_total_size(request.manifest) > request.maximum_bytes
            ):
                raise ValueError("artifact exceeds the caller's transfer byte bound")
            transfer = self.manager.prepare_incoming(
                manifest=request.manifest,
                source=request.source_node_id,
                chunks_total=request.chunks_total,
            )
            return ArtifactOperationResponse(
                accepted=True,
                transfer=transfer,
                detail="artifact transfer prepared",
            )
        if request.operation == "lease":
            if request.source_node_id is None:
                raise ValueError("artifact lease requires an owner node")
            lease = self.manager.lease(
                request.artifact_id,
                owner=request.source_node_id,
                purpose=request.lease_purpose,
                expires_at_unix_ns=request.lease_expires_at_unix_ns,
            )
            return ArtifactOperationResponse(
                accepted=True,
                lease=lease,
                detail="artifact lease acquired",
            )
        if request.operation == "release":
            if request.lease_id is None:
                raise ValueError("artifact release requires a lease ID")
            released = self.manager.release(request.lease_id)
            return ArtifactOperationResponse(
                accepted=released,
                detail="artifact lease released" if released else "artifact lease was not active",
            )
        return ArtifactOperationResponse(
            accepted=False,
            detail=(
                "artifact bytes are transferred by the canonical transactional deployment manager"
            ),
        )


def artifact_chunks(
    directory: Path,
    *,
    chunk_size_bytes: int = _DEFAULT_CHUNK_BYTES,
) -> Iterable[tuple[ArtifactChunk, bytes]]:
    """Yield bounded hash-described chunks for an authenticated transport."""

    if not 0 < chunk_size_bytes <= _MAXIMUM_CHUNK_BYTES:
        raise ValueError("artifact chunks must use a positive bounded size")
    root = directory.expanduser().resolve()
    manifest = verify_artifact_directory(root)
    index = 0
    for item in sorted(manifest.files, key=lambda value: value.relative_path):
        path = _safe_child(root, item.relative_path)
        with path.open("rb") as handle:
            offset = 0
            while data := handle.read(chunk_size_bytes):
                chunk = ArtifactChunk(
                    artifact_id=manifest.artifact_id,
                    relative_path=item.relative_path,
                    chunk_index=index,
                    offset_bytes=offset,
                    size_bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest(),
                    final_chunk=offset + len(data) == item.size_bytes,
                )
                yield chunk, data
                offset += len(data)
                index += 1


__all__ = [
    "ArtifactManager",
    "ArtifactOperationCoordinator",
    "ArtifactTransferResume",
    "ModelArtifactBuilder",
    "StageArtifactBuilder",
    "artifact_chunks",
    "build_native_stage_artifact",
    "calculate_manifest_content_hash",
    "load_artifact_manifest",
    "resolve_verified_artifact",
    "verify_artifact_directory",
]
