"""Fail-closed real-model acceptance for the major open-weight matrix.

The runner resolves publisher repositories to immutable revisions before any
model execution.  A PASS is possible only after the canonical orchestrator
returns generated token IDs for every configured workload; inspection,
planning, acquisition, and model loading are retained as evidence but never
count as inference.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

import psutil
import yaml
from huggingface_hub import HfApi
from pydantic import Field, PositiveInt, model_validator

from swarm_inference.backends.colibri.constants import COLIBRI_COMMIT
from swarm_inference.config.models import StrictModel

if TYPE_CHECKING:
    from swarm_inference.cluster.orchestrator import ClusterRunSummary, RunProgress

_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")
_GIB = 1024**3
_STATUS_VALUES = ("PASS", "FAIL", "BLOCKED_RESOURCE", "BLOCKED_AUTH", "NOT_RUN")


class ValidationStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED_RESOURCE = "BLOCKED_RESOURCE"
    BLOCKED_AUTH = "BLOCKED_AUTH"
    NOT_RUN = "NOT_RUN"


class WorkloadSpec(StrictModel):
    max_new_tokens: PositiveInt
    prompt: str | None = None
    approximate_input_tokens: PositiveInt | None = None
    seed_text: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> WorkloadSpec:
        if bool(self.prompt) == bool(self.seed_text):
            raise ValueError("a workload requires exactly one prompt or seed_text")
        if self.seed_text and self.approximate_input_tokens is None:
            raise ValueError("seed_text workloads require approximate_input_tokens")
        return self


class LocalGgufArtifact(StrictModel):
    """Pinned local GGUF derived from one immutable official checkpoint."""

    path: Path
    sha256: str
    source_revision: str
    quantization: str
    conversion_manifest: Path
    runtime_manifest: Path

    @model_validator(mode="after")
    def validate_pins(self) -> LocalGgufArtifact:
        if not _IMMUTABLE_REVISION.fullmatch(self.source_revision.casefold()):
            raise ValueError("local GGUF source_revision must be an immutable commit")
        digest = self.sha256.removeprefix("sha256:").casefold()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("local GGUF sha256 must contain an exact SHA-256 digest")
        return self


class MajorModelTarget(StrictModel):
    id: str
    family: str
    publisher: str
    model_id: str
    architecture_id: str
    dense_or_moe: Literal["dense", "moe"]
    expected_format: Literal["safetensors", "gguf"]
    mandatory: bool = True
    repetitions: PositiveInt = 3
    streaming_model: bool = False
    require_colibri: bool = False
    comparison_engines: tuple[str, ...] = ()
    quantization: str | None = None
    variant: str | None = None
    local_gguf: LocalGgufArtifact | None = None

    @model_validator(mode="after")
    def validate_identity(self) -> MajorModelTarget:
        namespace, separator, _name = self.model_id.partition("/")
        if not separator or namespace.casefold() != self.publisher.casefold():
            raise ValueError("model_id must be owned by the declared official publisher")
        if self.require_colibri and self.dense_or_moe != "moe":
            raise ValueError("Colibri is required only for sparse-MoE targets")
        if self.require_colibri and "colibri" not in self.comparison_engines:
            raise ValueError("a Colibri-required target must include a forced comparison")
        if self.local_gguf is not None:
            if self.expected_format != "gguf":
                raise ValueError("local_gguf requires expected_format=gguf")
            if self.quantization != self.local_gguf.quantization:
                raise ValueError("target and local GGUF quantization must agree exactly")
        return self


class MajorModelSuite(StrictModel):
    schema_version: Literal[1] = 1
    revision_policy: Literal["resolve-current-official-and-pin"]
    disk_reserve_bytes: int = Field(ge=0)
    working_space_fraction: float = Field(gt=0, le=0.5)
    workloads: dict[str, WorkloadSpec]
    targets: tuple[MajorModelTarget, ...]

    @model_validator(mode="after")
    def validate_matrix(self) -> MajorModelSuite:
        if set(self.workloads) != {"decode", "prefill", "practical"}:
            raise ValueError("the canonical workload set is decode, prefill, and practical")
        target_ids = [item.id for item in self.targets]
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("major-model target IDs must be unique")
        return self


class ArtifactPreflight(StrictModel):
    model_id: str
    revision: str
    required_files: tuple[str, ...]
    required_artifact_bytes: int = Field(ge=0)
    cached_artifact_bytes: int = Field(ge=0)
    remaining_download_bytes: int = Field(ge=0)
    required_working_space_bytes: int = Field(ge=0)
    required_disk_bytes: int = Field(ge=0)
    available_disk_bytes: int = Field(ge=0)
    available_ram_bytes: int = Field(ge=0)
    total_ram_bytes: int = Field(ge=0)
    required_ram_estimate_bytes: int | None = Field(default=None, ge=0)
    required_vram_estimate_bytes: int | None = Field(default=None, ge=0)
    artifact_size_exact: bool
    artifact_size_basis: str
    cache_root: str
    cache_complete: bool
    resource_sufficient: bool
    blocking_reason: str | None = None
    estimate_notes: tuple[str, ...] = ()


def load_suite(path: Path) -> MajorModelSuite:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("major-model matrix must contain a YAML object")
    return MajorModelSuite.model_validate(raw)


class EvidenceDirectory:
    """Atomic evidence writer used even when a run is interrupted."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"evidence directory is not empty: {self.root}")
        for relative in ("logs", "per_model", "charts"):
            (self.root / relative).mkdir(parents=True, exist_ok=True)

    def _replace(self, relative: str, payload: bytes) -> None:
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.partial")
        temporary.write_bytes(payload)
        os.replace(temporary, target)

    def json(self, relative: str, value: object) -> None:
        self._replace(
            relative,
            (json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n").encode(),
        )

    def text(self, relative: str, value: str) -> None:
        self._replace(relative, value.encode("utf-8"))

    def csv(self, relative: str, rows: Iterable[Mapping[str, object]]) -> None:
        materialized = [dict(item) for item in rows]
        columns = sorted({key for row in materialized for key in row}) or ["status", "reason"]
        target = self.root / relative
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.partial")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            if materialized:
                for row in materialized:
                    writer.writerow(
                        {
                            key: (
                                json.dumps(value, sort_keys=True, separators=(",", ":"))
                                if isinstance(value, (dict, list, tuple))
                                else value
                            )
                            for key, value in row.items()
                        }
                    )
            else:
                writer.writerow({"status": "NOT_RUN", "reason": "no measured rows"})
        os.replace(temporary, target)

    def hashes(self) -> None:
        rows: list[str] = []
        for path in sorted(item for item in self.root.rglob("*") if item.is_file()):
            if path.name == "SHA256SUMS.txt" or ".partial" in path.name:
                continue
            digest = _sha256_file(path)
            rows.append(f"{digest}  {path.relative_to(self.root).as_posix()}")
        self.text("SHA256SUMS.txt", "\n".join(rows) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _run_process(arguments: Sequence[str], *, timeout: float = 10) -> str | None:
    try:
        completed = subprocess.run(
            list(arguments),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _dependency_versions() -> dict[str, str | None]:
    names = (
        "swarm-inference-lab",
        "torch",
        "transformers",
        "safetensors",
        "huggingface-hub",
        "numpy",
        "grpcio",
        "psutil",
    )
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _storage_benchmark(root: Path, *, size_bytes: int = 64 * 1024**2) -> dict[str, object]:
    """Measure cache-warm local reads without touching unrelated files."""

    path = root / ".major-model-storage-probe.bin"
    block = bytes((index * 31 + 17) & 0xFF for index in range(1024 * 1024))
    try:
        with path.open("wb", buffering=0) as handle:
            remaining = size_bytes
            while remaining:
                payload = block[: min(len(block), remaining)]
                handle.write(payload)
                remaining -= len(payload)
            os.fsync(handle.fileno())
        started = time.perf_counter()
        read_bytes = 0
        with path.open("rb", buffering=0) as handle:
            while payload := handle.read(4 * 1024 * 1024):
                read_bytes += len(payload)
        sequential_s = time.perf_counter() - started
        rng = random.Random(12)
        operations = min(4096, size_bytes // 4096)
        offsets = [rng.randrange(0, size_bytes // 4096) * 4096 for _ in range(operations)]
        started = time.perf_counter()
        with path.open("rb", buffering=0) as handle:
            for offset in offsets:
                handle.seek(offset)
                if len(handle.read(4096)) != 4096:
                    raise OSError("short random-read probe")
        random_s = time.perf_counter() - started
        return {
            "status": "MEASURED",
            "method": "single-host cache-warm 64 MiB file; deterministic 4 KiB random reads",
            "sequential_read_bytes_per_second": read_bytes / sequential_s,
            "random_read_iops": operations / random_s,
            "probe_bytes": size_bytes,
        }
    except OSError as exc:
        return {
            "status": "UNAVAILABLE",
            "reason": f"{type(exc).__name__}: {exc}",
            "sequential_read_bytes_per_second": None,
            "random_read_iops": None,
        }
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def capture_hardware(root: Path) -> dict[str, object]:
    virtual = psutil.virtual_memory()
    gpu_line = _run_process(
        (
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,pci.bus_id",
            "--format=csv,noheader,nounits",
        )
    )
    cuda_version = _run_process(("nvidia-smi", "--query", "--display=COMPUTE"))
    partitions: list[dict[str, object]] = []
    for partition in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(partition.mountpoint)
        except OSError:
            continue
        partitions.append(
            {
                "device": partition.device,
                "mountpoint": partition.mountpoint,
                "filesystem": partition.fstype,
                "total_bytes": usage.total,
                "free_bytes": usage.free,
            }
        )
    commit = _run_process(("git", "rev-parse", "HEAD"))
    dirty = _run_process(("git", "status", "--porcelain"))
    record: dict[str, object] = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "physical_machine_count": 1,
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_total_bytes": virtual.total,
        "ram_available_bytes": virtual.available,
        "gpu_query": gpu_line,
        "cuda_query": cuda_version,
        "python_version": platform.python_version(),
        "swarm_commit": commit,
        "swarm_worktree_dirty": bool(dirty),
        "colibri_commit": COLIBRI_COMMIT,
        "dependencies": _dependency_versions(),
        "storage": partitions,
        "storage_read_benchmark": _storage_benchmark(root),
        "remote_inference_used": False,
    }
    return record


def _cache_root() -> Path:
    configured = os.environ.get("HF_HUB_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return (Path(hf_home).expanduser() / "hub").resolve()
    return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def _nearest_existing(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate.parent != candidate:
        candidate = candidate.parent
    return candidate


def _required_repository_files(
    siblings: Sequence[object], expected_format: str
) -> tuple[tuple[str, ...], int]:
    metadata_names = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "merges.txt",
        "vocab.json",
        "tokenizer.model",
        "chat_template.jinja",
    }
    files: list[str] = []
    total = 0
    for sibling in siblings:
        name = str(getattr(sibling, "rfilename", ""))
        size = getattr(sibling, "size", None)
        is_weight = (
            name.endswith(".safetensors")
            if expected_format == "safetensors"
            else name.casefold().endswith(".gguf")
        )
        if (
            not is_weight
            and Path(name).name not in metadata_names
            and not name.endswith(".safetensors.index.json")
            and not name.endswith(".py")
        ):
            continue
        if not isinstance(size, int) or size < 0:
            raise ValueError(f"publisher did not report a byte size for required file {name}")
        files.append(name)
        total += size
    if not any(
        name.endswith(".safetensors") or name.casefold().endswith(".gguf") for name in files
    ):
        raise ValueError(f"official repository exposes no {expected_format} weights")
    return tuple(sorted(files)), total


def _cached_bytes(
    cache_root: Path, model_id: str, revision: str, required_files: Sequence[str]
) -> tuple[int, bool]:
    snapshot = (
        cache_root
        / ("models--" + model_id.replace("/", "--"))
        / "snapshots"
        / revision
    )
    total = 0
    complete = True
    for relative in required_files:
        path = snapshot / relative
        try:
            if path.is_file():
                total += path.stat().st_size
            else:
                complete = False
        except OSError:
            complete = False
    return total, complete


def _is_connectivity_failure(exc: BaseException) -> bool:
    """Distinguish an offline host from auth and publisher-contract failures."""

    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        name = type(current).__name__.casefold()
        message = str(current).casefold()
        if name in {
            "connectionerror",
            "connecttimeout",
            "connecttimeouterror",
            "newconnectionerror",
            "maxretryerror",
            "nameresolutionerror",
        }:
            return True
        if any(
            marker in message
            for marker in (
                "failed to establish a new connection",
                "name resolution",
                "network is unreachable",
                "winerror 10013",
                "winerror 10060",
                "winerror 11001",
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def resolve_artifact_preflight(
    target: MajorModelTarget,
    suite: MajorModelSuite,
    *,
    api: HfApi,
    output_root: Path,
) -> tuple[ArtifactPreflight, dict[str, object]]:
    if target.local_gguf is not None:
        return _resolve_local_gguf_preflight(target, suite, api=api, output_root=output_root)
    try:
        info = api.model_info(target.model_id, revision="main", files_metadata=True)
    except Exception as exc:
        if not _is_connectivity_failure(exc):
            raise
        return _resolve_cached_safetensors_preflight(
            target,
            suite,
            output_root=output_root,
            online_resolution_error=f"{type(exc).__name__}: {exc}",
        )
    revision = str(getattr(info, "sha", "")).casefold()
    if not _IMMUTABLE_REVISION.fullmatch(revision):
        raise ValueError(f"publisher did not resolve an immutable revision for {target.model_id}")
    required_files, artifact_bytes = _required_repository_files(
        tuple(getattr(info, "siblings", ()) or ()), target.expected_format
    )
    cache_root = _cache_root()
    cached, complete = _cached_bytes(cache_root, target.model_id, revision, required_files)
    remaining = max(0, artifact_bytes - cached)
    working = max(2 * _GIB, int(artifact_bytes * suite.working_space_fraction))
    cache_volume = shutil.disk_usage(_nearest_existing(cache_root))
    output_volume = shutil.disk_usage(_nearest_existing(output_root))
    available = min(cache_volume.free, output_volume.free)
    required = remaining + working + suite.disk_reserve_bytes
    sufficient = cache_volume.free >= remaining + suite.disk_reserve_bytes and (
        output_volume.free >= working + suite.disk_reserve_bytes
    )
    virtual = psutil.virtual_memory()
    notes = [
        "artifact bytes are publisher-reported required checkpoint/tokenizer files",
        "working space is the larger of 2 GiB and the configured artifact fraction",
    ]
    if target.streaming_model:
        notes.append(
            "RAM/VRAM admission is plan-dependent because the target permits VRAM/RAM/NVMe tiering"
        )
    preflight = ArtifactPreflight(
        model_id=target.model_id,
        revision=revision,
        required_files=required_files,
        required_artifact_bytes=artifact_bytes,
        cached_artifact_bytes=cached,
        remaining_download_bytes=remaining,
        required_working_space_bytes=working,
        required_disk_bytes=required,
        available_disk_bytes=available,
        available_ram_bytes=int(virtual.available),
        total_ram_bytes=int(virtual.total),
        required_ram_estimate_bytes=None,
        required_vram_estimate_bytes=None,
        artifact_size_exact=True,
        artifact_size_basis="publisher-reported required file byte sizes",
        cache_root=str(cache_root),
        cache_complete=complete,
        resource_sufficient=sufficient,
        blocking_reason=(
            None
            if sufficient
            else "required download, working space, and retained disk reserve do not fit"
        ),
        estimate_notes=tuple(notes),
    )
    identity = {
        "target_id": target.id,
        "model_id": target.model_id,
        "revision": revision,
        "resolved_at_utc": datetime.now(UTC).isoformat(),
        "publisher": target.publisher,
        "official_namespace_verified": True,
        "last_modified": str(getattr(info, "last_modified", "")) or None,
    }
    return preflight, identity


def _resolve_cached_safetensors_preflight(
    target: MajorModelTarget,
    suite: MajorModelSuite,
    *,
    output_root: Path,
    online_resolution_error: str,
) -> tuple[ArtifactPreflight, dict[str, object]]:
    """Retain an honest offline resource result from an immutable cached index.

    A Safetensors index records exact tensor payload bytes but not every shard's
    container-header bytes.  The resource calculation therefore adds a one
    percent allowance and marks the size as an estimate.  This path never
    claims that the cached revision is still publisher ``main``.
    """

    if target.expected_format != "safetensors":
        raise ConnectionError(online_resolution_error)
    cache_root = _cache_root()
    repository = cache_root / ("models--" + target.model_id.replace("/", "--"))
    snapshots_root = repository / "snapshots"
    candidates: list[Path] = []
    cached_ref = repository / "refs" / "main"
    if cached_ref.is_file():
        revision = cached_ref.read_text(encoding="utf-8").strip().casefold()
        candidate = snapshots_root / revision
        if _IMMUTABLE_REVISION.fullmatch(revision) and candidate.is_dir():
            candidates.append(candidate)
    if snapshots_root.is_dir():
        for candidate in snapshots_root.iterdir():
            if (
                candidate.is_dir()
                and _IMMUTABLE_REVISION.fullmatch(candidate.name.casefold())
                and candidate not in candidates
            ):
                candidates.append(candidate)
    usable = [
        candidate
        for candidate in candidates
        if (candidate / "config.json").is_file()
        and (candidate / "model.safetensors.index.json").is_file()
    ]
    if len(usable) != 1:
        raise ConnectionError(
            online_resolution_error
            + "; offline fallback requires exactly one immutable cached snapshot with "
            f"config and Safetensors index, found {len(usable)}"
        )
    snapshot = usable[0]
    revision = snapshot.name.casefold()
    index_path = snapshot / "model.safetensors.index.json"
    raw_index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(raw_index, dict):
        raise ValueError("cached Safetensors index must contain an object")
    metadata = raw_index.get("metadata")
    weight_map = raw_index.get("weight_map")
    if not isinstance(metadata, dict) or not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("cached Safetensors index lacks metadata or tensor mappings")
    raw_tensor_bytes = metadata.get("total_size")
    if (
        isinstance(raw_tensor_bytes, bool)
        or not isinstance(raw_tensor_bytes, (int, float))
        or raw_tensor_bytes <= 0
        or int(raw_tensor_bytes) != raw_tensor_bytes
    ):
        raise ValueError("cached Safetensors index lacks a positive total_size")
    tensor_bytes = int(raw_tensor_bytes)
    shards = tuple(sorted({str(value) for value in weight_map.values()}))
    if not shards or any(not name.endswith(".safetensors") for name in shards):
        raise ValueError("cached Safetensors index contains invalid shard names")

    metadata_names = tuple(
        sorted(
            path.name
            for path in snapshot.iterdir()
            if path.is_file()
            and (
                path.suffix in {".json", ".py", ".txt", ".model", ".jinja"}
                or path.name == "chat_template.jinja"
            )
            and path.name not in shards
        )
    )
    metadata_bytes = sum((snapshot / name).stat().st_size for name in metadata_names)
    # Safetensors total_size is the exact tensor payload, but omits container
    # headers.  One percent is intentionally conservative for admission only.
    artifact_bytes = int(tensor_bytes * 1.01) + metadata_bytes
    required_files = tuple(sorted({*shards, *metadata_names}))
    cached, complete = _cached_bytes(cache_root, target.model_id, revision, required_files)
    remaining = max(0, artifact_bytes - cached)
    working = max(2 * _GIB, int(artifact_bytes * suite.working_space_fraction))
    cache_volume = shutil.disk_usage(_nearest_existing(cache_root))
    output_volume = shutil.disk_usage(_nearest_existing(output_root))
    available = min(cache_volume.free, output_volume.free)
    required = remaining + working + suite.disk_reserve_bytes
    sufficient = cache_volume.free >= remaining + suite.disk_reserve_bytes and (
        output_volume.free >= working + suite.disk_reserve_bytes
    )
    virtual = psutil.virtual_memory()
    preflight = ArtifactPreflight(
        model_id=target.model_id,
        revision=revision,
        required_files=required_files,
        required_artifact_bytes=artifact_bytes,
        cached_artifact_bytes=cached,
        remaining_download_bytes=remaining,
        required_working_space_bytes=working,
        required_disk_bytes=required,
        available_disk_bytes=available,
        available_ram_bytes=int(virtual.available),
        total_ram_bytes=int(virtual.total),
        required_ram_estimate_bytes=None,
        required_vram_estimate_bytes=None,
        artifact_size_exact=False,
        artifact_size_basis=(
            "cached model.safetensors.index.json tensor payload plus a 1 percent "
            "Safetensors-container allowance and cached metadata bytes"
        ),
        cache_root=str(cache_root),
        cache_complete=complete,
        resource_sufficient=sufficient,
        blocking_reason=(
            None
            if sufficient
            else "cached immutable checkpoint estimate, working space, and retained disk "
            "reserve do not fit on this machine"
        ),
        estimate_notes=(
            "publisher main could not be resolved from this process",
            "the cached revision is immutable but is not asserted to be current",
            "RAM/VRAM admission remains plan-dependent for tiered execution",
        ),
    )
    identity = {
        "target_id": target.id,
        "model_id": target.model_id,
        "revision": revision,
        "resolved_at_utc": datetime.now(UTC).isoformat(),
        "publisher": target.publisher,
        "official_namespace_verified": False,
        "declared_namespace_matches_publisher": True,
        "current_revision_verified_online": False,
        "online_resolution_error": online_resolution_error,
        "revision_resolution": "cached-immutable-safetensors-index",
        "safetensors_index": str(index_path),
        "safetensors_index_sha256": _sha256_file(index_path),
    }
    return preflight, identity


def _resolve_local_gguf_preflight(
    target: MajorModelTarget,
    suite: MajorModelSuite,
    *,
    api: HfApi,
    output_root: Path,
) -> tuple[ArtifactPreflight, dict[str, object]]:
    artifact = target.local_gguf
    assert artifact is not None
    path = artifact.path.expanduser().resolve()
    conversion_path = artifact.conversion_manifest.expanduser().resolve()
    runtime_path = artifact.runtime_manifest.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"pinned local GGUF is unavailable: {path}")
    actual_digest = _sha256_file(path)
    expected_digest = artifact.sha256.removeprefix("sha256:").casefold()
    if actual_digest != expected_digest:
        raise ValueError("pinned local GGUF digest differs from the validation matrix")
    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    if not isinstance(conversion, dict):
        raise ValueError("local GGUF conversion manifest must be an object")
    expected_conversion = {
        "source_model_id": target.model_id,
        "source_revision": artifact.source_revision,
        "gguf_sha256": expected_digest,
        "quantisation": artifact.quantization,
    }
    mismatches = [
        key for key, expected in expected_conversion.items() if conversion.get(key) != expected
    ]
    if mismatches:
        raise ValueError(
            "local GGUF conversion manifest differs from the target: "
            + ", ".join(sorted(mismatches))
        )
    from swarm_inference.engines.llamacpp_rpc import load_llamacpp_runtime_manifest

    runtime = load_llamacpp_runtime_manifest(runtime_path)
    online_resolution_error: str | None = None
    online_verified = False
    try:
        info = api.model_info(target.model_id, revision="main", files_metadata=False)
        current_revision = str(getattr(info, "sha", "")).casefold()
        if not _IMMUTABLE_REVISION.fullmatch(current_revision):
            raise ValueError("publisher did not return an immutable current revision")
        if current_revision != artifact.source_revision.casefold():
            raise ValueError(
                "pinned local GGUF is not derived from the current official repository revision"
            )
        online_verified = True
    except Exception as exc:
        cache_root = _cache_root()
        snapshot_root = (
            cache_root
            / ("models--" + target.model_id.replace("/", "--"))
            / "snapshots"
            / artifact.source_revision.casefold()
        )
        cached_ref = (
            cache_root
            / ("models--" + target.model_id.replace("/", "--"))
            / "refs"
            / "main"
        )
        cached_revision = (
            cached_ref.read_text(encoding="utf-8").strip().casefold()
            if cached_ref.is_file()
            else ""
        )
        tokenizer_path = snapshot_root / "tokenizer.json"
        tokenizer_digest = (
            _sha256_file(tokenizer_path) if tokenizer_path.is_file() else ""
        )
        cached_snapshot_verified = (
            snapshot_root.is_dir()
            and (snapshot_root / "config.json").is_file()
            and tokenizer_digest == conversion.get("tokenizer_hash")
        )
        if (
            cached_revision != artifact.source_revision.casefold()
            and not cached_snapshot_verified
        ):
            raise
        online_resolution_error = f"{type(exc).__name__}: {exc}"

    artifact_bytes = path.stat().st_size
    working = max(2 * _GIB, int(artifact_bytes * suite.working_space_fraction))
    output_volume = shutil.disk_usage(_nearest_existing(output_root))
    virtual = psutil.virtual_memory()
    required_ram = int(artifact_bytes * 1.05)
    sufficient = (
        output_volume.free >= working + suite.disk_reserve_bytes
        and virtual.available >= required_ram
    )
    preflight = ArtifactPreflight(
        model_id=target.model_id,
        revision=artifact.source_revision.casefold(),
        required_files=(str(path),),
        required_artifact_bytes=artifact_bytes,
        cached_artifact_bytes=artifact_bytes,
        remaining_download_bytes=0,
        required_working_space_bytes=working,
        required_disk_bytes=working + suite.disk_reserve_bytes,
        available_disk_bytes=output_volume.free,
        available_ram_bytes=int(virtual.available),
        total_ram_bytes=int(virtual.total),
        required_ram_estimate_bytes=required_ram,
        required_vram_estimate_bytes=0,
        artifact_size_exact=True,
        artifact_size_basis="verified local GGUF file byte size",
        cache_root=str(path.parent),
        cache_complete=True,
        resource_sufficient=sufficient,
        blocking_reason=(
            None
            if sufficient
            else "pinned GGUF runtime headroom or retained disk reserve does not fit"
        ),
        estimate_notes=(
            "GGUF digest and conversion provenance were verified before execution",
            "source revision is an exact cached official snapshot when online resolution is unavailable",
            "RAM estimate is 105 percent of the GGUF byte size",
        ),
    )
    identity = {
        "target_id": target.id,
        "model_id": target.model_id,
        "revision": artifact.source_revision.casefold(),
        "resolved_at_utc": datetime.now(UTC).isoformat(),
        "publisher": target.publisher,
        "official_namespace_verified": True,
        "current_revision_verified_online": online_verified,
        "online_resolution_error": online_resolution_error,
        "revision_resolution": (
            "official-hub-main" if online_verified else "cached-immutable-snapshot"
        ),
        "artifact_path": str(path),
        "artifact_sha256": actual_digest,
        "conversion_manifest": str(conversion_path),
        "conversion_manifest_sha256": _sha256_file(conversion_path),
        "runtime_manifest": str(runtime_path),
        "runtime_commit": runtime.commit,
        "runtime_binary_hashes": {
            "server": runtime.server_sha256,
            "rpc_server": runtime.rpc_server_sha256,
            **{item.name: digest for item, digest in runtime.architecture_probe_binaries.items()},
        },
    }
    return preflight, identity


def _gpu_sample() -> tuple[int | None, float | None, float | None]:
    line = _run_process(
        (
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ),
        timeout=3,
    )
    if not line:
        return None, None, None
    try:
        first = line.splitlines()[0]
        memory_mib, utilization, power = (float(item.strip()) for item in first.split(",")[:3])
    except (ValueError, IndexError):
        return None, None, None
    return int(memory_mib * 1024**2), utilization, power


class ResourceSampler:
    """Host-wide sampler for a single-machine acceptance run."""

    def __init__(self, interval_seconds: float = 0.25) -> None:
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ram: list[int] = []
        self._cpu: list[float] = []
        self._vram: list[int] = []
        self._gpu: list[float] = []
        self._power: list[float] = []
        self._disk_start: int | None = None
        self._context_start: int | None = None

    def __enter__(self) -> ResourceSampler:
        disk = psutil.disk_io_counters()
        cpu = psutil.cpu_stats()
        self._disk_start = int(disk.read_bytes) if disk is not None else None
        self._context_start = int(cpu.ctx_switches)
        psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(3.0, self.interval_seconds * 4))

    def _sample(self) -> None:
        while not self._stop.is_set():
            virtual = psutil.virtual_memory()
            self._ram.append(int(virtual.total - virtual.available))
            self._cpu.append(float(psutil.cpu_percent(interval=None)))
            vram, gpu, power = _gpu_sample()
            if vram is not None:
                self._vram.append(vram)
            if gpu is not None:
                self._gpu.append(gpu)
            if power is not None:
                self._power.append(power)
            self._stop.wait(self.interval_seconds)

    def result(self) -> dict[str, object]:
        disk = psutil.disk_io_counters()
        cpu = psutil.cpu_stats()
        read_bytes = (
            max(0, int(disk.read_bytes) - self._disk_start)
            if disk is not None and self._disk_start is not None
            else None
        )
        context_switches = (
            max(0, int(cpu.ctx_switches) - self._context_start)
            if self._context_start is not None
            else None
        )
        return {
            "peak_system_ram_bytes": max(self._ram) if self._ram else None,
            "peak_vram_bytes": max(self._vram) if self._vram else None,
            "cpu_utilization_percent": statistics.fmean(self._cpu) if self._cpu else None,
            "gpu_utilization_percent": statistics.fmean(self._gpu) if self._gpu else None,
            "gpu_power_watts": statistics.fmean(self._power) if self._power else None,
            "storage_read_bytes": read_bytes,
            "context_switches": context_switches,
            "sampling_interval_seconds": self.interval_seconds,
            "measurement_scope": "single physical host",
        }


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _first_number(mapping: Mapping[str, Any], names: Sequence[str]) -> float | None:
    wanted = {item.casefold() for item in names}
    pending: list[Mapping[str, Any]] = [mapping]
    while pending:
        current = pending.pop()
        for key, value in current.items():
            if key.casefold() in wanted and isinstance(value, (int, float)) and not isinstance(
                value, bool
            ):
                return float(value)
            if isinstance(value, Mapping):
                pending.append(value)
    return None


def _event_duration_ms(
    events: Sequence[RunProgress], start_event: str, end_event: str
) -> float | None:
    start = next((item.timestamp_unix_ns for item in events if item.event == start_event), None)
    end = next(
        (
            item.timestamp_unix_ns
            for item in events
            if item.event == end_event and (start is None or item.timestamp_unix_ns >= start)
        ),
        None,
    )
    if start is None or end is None:
        return None
    return max(0.0, (end - start) / 1_000_000)


def _component_plan(summary: ClusterRunSummary) -> tuple[list[dict[str, object]], list[str]]:
    selected: object = (
        summary.canonical_decision.selected
        if summary.canonical_decision is not None
        else summary.plan
    )
    components: list[dict[str, object]] = []
    for component in getattr(selected, "components", ()):
        component_type = getattr(component, "component_type", "unknown")
        placement = getattr(component, "placement", None)
        components.append(
            {
                "component_type": getattr(component_type, "value", str(component_type)),
                "engine_id": str(getattr(component, "engine_id", "unknown")),
                "device": str(getattr(placement, "device", "unknown")),
                "worker_ids": list(getattr(placement, "worker_ids", ())),
            }
        )
    workers = sorted(
        worker_id
        for worker_id, role in getattr(selected, "worker_roles", {}).items()
        if role not in {"idle", "background_replica", "storage_cache", "verification"}
    )
    if not components:
        components.append(
            {
                "component_type": "whole-model",
                "engine_id": summary.engine_id,
                "device": "engine-managed",
                "worker_ids": workers,
            }
        )
    return components, workers


def _tokenizer_for(target: MajorModelTarget, revision: str) -> Any:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            target.model_id,
            revision=revision,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception:
        from tokenizers import Tokenizer

        tokenizer_path = (
            _cache_root()
            / ("models--" + target.model_id.replace("/", "--"))
            / "snapshots"
            / revision
            / "tokenizer.json"
        )
        if not tokenizer_path.is_file():
            raise
        return Tokenizer.from_file(str(tokenizer_path.resolve()))


def _encode_ids(tokenizer: Any, prompt: str) -> list[int]:
    encoded = tokenizer.encode(prompt, add_special_tokens=True)
    values = encoded.ids if hasattr(encoded, "ids") else encoded
    return [int(item) for item in values]


def _prompt_and_ids(tokenizer: Any, spec: WorkloadSpec) -> tuple[str, list[int]]:
    if spec.prompt is not None:
        prompt = spec.prompt
    else:
        assert spec.seed_text is not None and spec.approximate_input_tokens is not None
        target = spec.approximate_input_tokens
        low, high = 1, 2
        while len(_encode_ids(tokenizer, spec.seed_text * high)) < target:
            low, high = high, high * 2
        best = spec.seed_text * low
        best_delta = abs(len(_encode_ids(tokenizer, best)) - target)
        while low <= high:
            middle = (low + high) // 2
            candidate = spec.seed_text * middle
            delta = abs(len(_encode_ids(tokenizer, candidate)) - target)
            if delta < best_delta:
                best, best_delta = candidate, delta
            count = len(_encode_ids(tokenizer, candidate))
            if count < target:
                low = middle + 1
            elif count > target:
                high = middle - 1
            else:
                best = candidate
                break
        prompt = best
    ids = _encode_ids(tokenizer, prompt)
    if not ids:
        raise ValueError("official tokenizer produced no prompt token IDs")
    return prompt, ids


def _classify_error(exc: BaseException) -> ValidationStatus:
    name = type(exc).__name__.casefold()
    message = str(exc).casefold()
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code in {401, 403} or any(marker in name for marker in ("gated", "unauthorized")):
        return ValidationStatus.BLOCKED_AUTH
    if isinstance(exc, MemoryError) or any(
        marker in message
        for marker in (
            "out of memory",
            "no space left",
            "disk quota",
            "insufficient memory",
            "cuda error: out of memory",
        )
    ):
        return ValidationStatus.BLOCKED_RESOURCE
    if "node is not paired with a cluster" in message or "no healthy product workers" in message:
        return ValidationStatus.NOT_RUN
    if any(
        marker in message
        for marker in (
            "failed to establish a new connection",
            "network is unreachable",
            "name resolution",
            "winerror 10013",
            "connection timed out",
        )
    ):
        return ValidationStatus.NOT_RUN
    return ValidationStatus.FAIL


def _run_record(
    *,
    target: MajorModelTarget,
    revision: str,
    workload: str,
    repetition: int,
    requested_engine: str | None,
    prompt_token_ids: Sequence[int],
    summary: ClusterRunSummary,
    events: Sequence[RunProgress],
    resources: Mapping[str, object],
) -> dict[str, object]:
    telemetry = summary.telemetry
    telemetry_payload = telemetry.model_dump(mode="json") if telemetry is not None else {}
    engine_metrics = telemetry.engine_metrics if telemetry is not None else {}
    components, workers = _component_plan(summary)
    inter_token = list(telemetry.inter_token_latency_ms) if telemetry is not None else []
    cache_hits = telemetry.cache_hits if telemetry is not None else None
    cache_misses = telemetry.cache_misses if telemetry is not None else None
    cache_total = (cache_hits or 0) + (cache_misses or 0)
    storage_bytes = resources.get("storage_read_bytes")
    effective_bandwidth = (
        float(storage_bytes) / summary.elapsed_seconds
        if isinstance(storage_bytes, int) and summary.elapsed_seconds > 0
        else None
    )
    generated = [int(item) for item in summary.output_token_ids]
    passed = bool(summary.status == "completed" and generated and telemetry is not None)
    component_engines = [str(item["engine_id"]) for item in components]
    record: dict[str, object] = {
        "family": target.family,
        "target_id": target.id,
        "model_id": summary.model_id,
        "revision": summary.model_revision,
        "architecture_id": summary.model_architecture,
        "dense_or_moe": target.dense_or_moe,
        "total_parameters": (
            summary.model_profile.total_parameters if summary.model_profile is not None else None
        ),
        "active_parameters": (
            summary.model_profile.active_parameters if summary.model_profile is not None else None
        ),
        "format": summary.model_format,
        "quantization": summary.quantization,
        "workload": workload,
        "repetition": repetition,
        "requested_engine": requested_engine or "auto",
        "selected_engine": summary.engine_id,
        "composite_backend_components": components,
        "component_engines": component_engines,
        "worker_count": len(workers),
        "logical_microworker_count": max(
            len(telemetry.microshard_assignments) if telemetry is not None else 0,
            len(workers),
        ),
        "load_time_ms": _event_duration_ms(events, "deployment-started", "deployment-ready"),
        "artifact_preparation_time_ms": _event_duration_ms(
            events, "artifact-preparing", "deployment-started"
        ),
        "ttft_ms": telemetry.ttft_ms if telemetry is not None else None,
        "prefill_tokens_per_second": (
            telemetry.prefill_tokens_s if telemetry is not None else None
        ),
        "decode_tokens_per_second": (
            telemetry.decode_tokens_s if telemetry is not None else None
        ),
        "p50_token_latency_ms": _percentile(inter_token, 0.50),
        "p95_token_latency_ms": _percentile(inter_token, 0.95),
        "peak_vram_bytes": resources.get("peak_vram_bytes"),
        "peak_system_ram_bytes": resources.get("peak_system_ram_bytes"),
        "bytes_read_from_storage": storage_bytes,
        "effective_disk_bandwidth_bytes_per_second": effective_bandwidth,
        "expert_cache_hit_rate": (cache_hits / cache_total if cache_hits is not None and cache_total else None),
        "expert_movement_bytes": _first_number(
            engine_metrics,
            ("colibri_expert_movement_bytes", "expert_movement_bytes", "bytes_loaded"),
        ),
        "worker_to_worker_bytes": _first_number(
            engine_metrics,
            ("worker_to_worker_bytes", "worker_to_worker_activation_bytes"),
        ),
        "coordinator_activation_bytes": _first_number(
            engine_metrics,
            ("coordinator_activation_bytes", "coordinator_bound_bytes"),
        ),
        "serial_coordinator_waits": _first_number(
            engine_metrics,
            ("coordinator_serial_waits", "serial_coordinator_waits"),
        ),
        "cpu_utilization_percent": resources.get("cpu_utilization_percent"),
        "gpu_utilization_percent": resources.get("gpu_utilization_percent"),
        "gpu_power_watts": resources.get("gpu_power_watts"),
        "context_switches": resources.get("context_switches"),
        "prompt_token_count": len(prompt_token_ids),
        "prompt_token_ids": list(prompt_token_ids),
        "generated_token_count": len(generated),
        "generated_token_ids": generated,
        "decoded_text": summary.decoded_text,
        "correctness_status": "PASS" if passed else "FAIL",
        "execution_status": "PASS" if passed else "FAIL",
        "elapsed_seconds": summary.elapsed_seconds,
        "telemetry": telemetry_payload,
    }
    measured_fields = (
        "load_time_ms",
        "artifact_preparation_time_ms",
        "ttft_ms",
        "prefill_tokens_per_second",
        "decode_tokens_per_second",
        "p50_token_latency_ms",
        "p95_token_latency_ms",
        "peak_vram_bytes",
        "peak_system_ram_bytes",
        "bytes_read_from_storage",
        "effective_disk_bandwidth_bytes_per_second",
        "expert_cache_hit_rate",
        "expert_movement_bytes",
        "worker_to_worker_bytes",
        "coordinator_activation_bytes",
        "serial_coordinator_waits",
        "cpu_utilization_percent",
        "gpu_utilization_percent",
        "gpu_power_watts",
        "context_switches",
    )
    record["metric_unavailable_reasons"] = {
        field: "canonical engine or single-host sampler did not expose this measurement"
        for field in measured_fields
        if record[field] is None
    }
    return record


async def _local_gguf_run(
    *,
    target: MajorModelTarget,
    revision: str,
    prompt: str,
    max_new_tokens: int,
    requested_engine: str | None,
    runtime_log_root: Path,
) -> tuple[ClusterRunSummary, tuple[RunProgress, ...]]:
    """Run one real local GGUF through the canonical engine lifecycle."""

    from swarm_inference.cluster.orchestrator import ClusterRunSummary, RunProgress
    from swarm_inference.engines.interfaces import ExecutionRequest, InferenceRequest
    from swarm_inference.engines.llamacpp_rpc import (
        LlamaCppRpcEngine,
        LocalLlamaCppLifecycle,
        load_llamacpp_runtime_manifest,
    )
    from swarm_inference.engines.local_capabilities import discover_local_cluster_capabilities
    from swarm_inference.engines.registry import ExecutionEngineRegistry
    from swarm_inference.model.resolver import ModelSourceResolver
    from swarm_inference.runtime.engine_processes import EngineProcessManager
    from swarm_inference.runtime.telemetry import build_inference_telemetry_record

    artifact = target.local_gguf
    assert artifact is not None
    if requested_engine not in {None, "llamacpp-rpc"}:
        raise RuntimeError(
            f"forced engine {requested_engine!r} cannot consume the pinned local GGUF"
        )
    artifact_path = artifact.path.expanduser().resolve()
    runtime_manifest_path = artifact.runtime_manifest.expanduser().resolve()
    resolution = ModelSourceResolver().inspect(
        artifact_path,
        revision=revision,
        quantization=artifact.quantization,
        objective="speed",
    )
    descriptor = resolution.descriptor.model_copy(
        update={"model_id": target.model_id, "revision": revision}
    )
    if descriptor.architecture != target.architecture_id:
        raise RuntimeError(
            f"local GGUF architecture {descriptor.architecture!r} differs from "
            f"target {target.architecture_id!r}"
        )
    manifest = load_llamacpp_runtime_manifest(runtime_manifest_path)
    cluster = discover_local_cluster_capabilities(llamacpp_manifest=runtime_manifest_path)
    worker_id = cluster.workers[0].worker_id
    lifecycle = LocalLlamaCppLifecycle(
        manifest=manifest,
        processes=EngineProcessManager(runtime_log_root),
        worker_id=worker_id,
    )
    engine = LlamaCppRpcEngine(lifecycle=lifecycle)
    engine.bind_acquired_model(
        descriptor,
        tuple(Path(item) for item in descriptor.local_paths),
    )
    registry = ExecutionEngineRegistry((engine,))
    planning_request = ExecutionRequest(
        objective="speed",
        concurrency=1,
        max_context_tokens=8192,
        max_new_tokens=max_new_tokens,
        requested_engine=requested_engine,
    )
    decision = await registry.compete(descriptor, cluster, planning_request)
    plan = decision.selected
    progress: list[RunProgress] = []

    def event(name: str, stage: str, detail: str) -> None:
        progress.append(
            RunProgress(
                event=name,
                stage=stage,
                timestamp_unix_ns=time.time_ns(),
                detail=detail,
            )
        )

    event("deployment-started", "deployment", plan.plan_id)
    deployment = await engine.prepare(plan)
    event("deployment-ready", "deployment", deployment.deployment_id)
    request_id = f"acceptance-{target.id}-{uuid4().hex}"
    request = InferenceRequest(
        request_id=request_id,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        seed=1,
        temperature=0.0,
    )
    output_ids: list[int] = []
    decoded: list[str] = []
    token_times: list[float] = []
    terminal_metrics: dict[str, object] = {}
    inference_events = 0
    started_unix_ns = time.time_ns()
    submitted = time.monotonic()
    try:
        async for inference_event in engine.submit(deployment, request):
            inference_events += 1
            if inference_event.event_type == "token":
                if inference_event.token_id is None:
                    raise RuntimeError("canonical llama.cpp token event omitted its token ID")
                output_ids.append(int(inference_event.token_id))
                decoded.append(inference_event.text)
                token_times.append(time.monotonic())
            elif inference_event.event_type == "completed":
                terminal_metrics = dict(inference_event.telemetry)
            elif inference_event.event_type == "failed":
                raise RuntimeError(inference_event.detail or "llama.cpp inference failed")
        completed = time.monotonic()
        completed_unix_ns = time.time_ns()
        if not output_ids:
            raise RuntimeError("canonical llama.cpp inference returned no generated token IDs")
        telemetry = build_inference_telemetry_record(
            request_id=request_id,
            model=descriptor,
            execution_plan=plan,
            deployed_plan=plan,
            cluster=cluster,
            status="completed",
            submitted_monotonic_s=submitted,
            completed_monotonic_s=completed,
            token_monotonic_s=token_times,
            terminal_metrics=terminal_metrics,
        )
        return (
            ClusterRunSummary(
                run_id=request_id,
                status="completed",
                model_id=target.model_id,
                model_revision=revision,
                tokenizer_revision=revision,
                mode="speed",
                plan=plan,
                model_fingerprint=descriptor.content_fingerprint,
                model_architecture=descriptor.architecture,
                model_profile=descriptor.architecture_profile,
                model_architecture_source=descriptor.architecture_source,
                model_format=descriptor.format,
                total_model_size_bytes=descriptor.weight_bytes,
                variant=descriptor.variant,
                quantization=descriptor.quantization,
                engine_id=engine.engine_id,
                requested_engine=requested_engine,
                engine_revision=manifest.commit,
                engine_runtime_revisions=telemetry.engine_runtime_revisions,
                execution_identity=plan.execution_identity,
                engine_support=decision.support,
                deployment_id=deployment.deployment_id,
                output_token_ids=output_ids,
                decoded_text="".join(decoded),
                event_count=inference_events,
                started_at_unix_ns=started_unix_ns,
                completed_at_unix_ns=completed_unix_ns,
                elapsed_seconds=max(0.0, completed - submitted),
                detail="real local GGUF generation through managed llama.cpp lifecycle",
                telemetry=telemetry,
            ),
            tuple(progress),
        )
    finally:
        await engine.unload(deployment)


async def _canonical_run(
    *,
    target: MajorModelTarget,
    revision: str,
    prompt: str,
    max_new_tokens: int,
    requested_engine: str | None,
    state_root: Path | None,
    maximum_source_bytes: int,
    runtime_log_root: Path,
) -> tuple[ClusterRunSummary, tuple[RunProgress, ...], dict[str, object]]:
    if target.local_gguf is not None:
        with ResourceSampler() as sampler:
            summary, events = await _local_gguf_run(
                target=target,
                revision=revision,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                requested_engine=requested_engine,
                runtime_log_root=runtime_log_root,
            )
        return summary, events, sampler.result()
    from swarm_inference.cluster.orchestrator import ClusterOrchestrator
    from swarm_inference.cluster.state import ClusterStateStore

    events: list[RunProgress] = []
    orchestrator = ClusterOrchestrator(
        state=ClusterStateStore(state_root),
        progress_sink=events.append,
        maximum_source_bytes=maximum_source_bytes,
        source_timeout_seconds=7200,
    )
    with ResourceSampler() as sampler:
        summary = await orchestrator.run(
            model_id=target.model_id,
            model_revision=revision,
            tokenizer_revision=revision,
            variant=target.variant,
            quantization=target.quantization,
            requested_engine=requested_engine,
            prompt=prompt,
            mode="speed",
            dry_run=False,
            max_context_tokens=8192,
            max_new_tokens=max_new_tokens,
            seed=1,
        )
    return summary, tuple(events), sampler.result()


def _average(rows: Sequence[Mapping[str, object]], key: str) -> float | None:
    values = [float(value) for row in rows if isinstance((value := row.get(key)), (int, float))]
    return statistics.fmean(values) if values else None


def _maximum(rows: Sequence[Mapping[str, object]], key: str) -> float | int | None:
    values = [value for row in rows if isinstance((value := row.get(key)), (int, float))]
    return max(values) if values else None


def _model_summary_row(
    target: MajorModelTarget,
    status: ValidationStatus,
    revision: str | None,
    runs: Sequence[Mapping[str, object]],
    preflight: ArtifactPreflight | None,
    reason: str | None,
) -> dict[str, object]:
    successful = [row for row in runs if row.get("execution_status") == "PASS"]
    decode = [row for row in successful if row.get("workload") == "decode"]
    prefill = [row for row in successful if row.get("workload") == "prefill"]
    representative = successful[0] if successful else {}
    component_sets = [
        tuple(str(item) for item in row.get("component_engines", [])) for row in successful
    ]
    colibri = any("colibri" in items for items in component_sets)
    hybrid = any("colibri" in items and len(set(items)) > 1 for items in component_sets)
    storage_bytes = sum(
        int(value)
        for row in successful
        if isinstance((value := row.get("bytes_read_from_storage")), int)
    )
    generated_tokens = sum(
        int(value)
        for row in successful
        if isinstance((value := row.get("generated_token_count")), int)
    )
    selected_engines = sorted(
        {
            str(row.get("selected_engine"))
            for row in successful
            if row.get("selected_engine") is not None
        }
    )
    return {
        "family": target.family,
        "model": target.model_id,
        "revision": revision,
        "architecture": representative.get("architecture_id", target.architecture_id),
        "dense_or_moe": target.dense_or_moe,
        "parameters": representative.get("total_parameters"),
        "active_parameters": representative.get("active_parameters"),
        "format": representative.get("format", target.expected_format),
        "quantization": representative.get("quantization", target.quantization),
        "real_run": status == ValidationStatus.PASS,
        "colibri": colibri,
        "hybrid": hybrid,
        "alternative_engine": [
            item for item in target.comparison_engines if item != "colibri"
        ],
        "selected_engine": selected_engines,
        "decode_tokens_per_second": _average(decode, "decode_tokens_per_second"),
        "prefill_tokens_per_second": _average(prefill, "prefill_tokens_per_second"),
        "ttft_ms": _average(decode, "ttft_ms"),
        "peak_vram_bytes": _maximum(successful, "peak_vram_bytes"),
        "peak_ram_bytes": _maximum(successful, "peak_system_ram_bytes"),
        "disk_bytes_per_token": storage_bytes / generated_tokens if generated_tokens else None,
        "correctness": (
            "PASS"
            if successful and all(row.get("correctness_status") == "PASS" for row in successful)
            else "FAIL"
            if runs
            else "NOT_RUN"
        ),
        "status": status.value,
        "reason": reason,
        "resource_preflight": preflight.model_dump(mode="json") if preflight else None,
    }


def _comparison_row(
    target: MajorModelTarget,
    automatic: Mapping[str, object],
    forced: Mapping[str, object],
    engine: str,
) -> dict[str, object]:
    auto_rate = automatic.get("decode_tokens_per_second")
    forced_rate = forced.get("decode_tokens_per_second")
    ratio = (
        float(forced_rate) / float(auto_rate)
        if isinstance(auto_rate, (int, float))
        and isinstance(forced_rate, (int, float))
        and float(auto_rate) > 0
        else None
    )
    same_precision = (
        automatic.get("format") == forced.get("format")
        and automatic.get("quantization") == forced.get("quantization")
    )
    tokens_match = automatic.get("generated_token_ids") == forced.get("generated_token_ids")
    return {
        "family": target.family,
        "model_id": target.model_id,
        "revision": automatic.get("revision"),
        "automatic_engine": automatic.get("selected_engine"),
        "automatic_components": automatic.get("component_engines"),
        "forced_engine": engine,
        "forced_selected_engine": forced.get("selected_engine"),
        "forced_components": forced.get("component_engines"),
        "format": automatic.get("format"),
        "quantization": automatic.get("quantization"),
        "equivalent_precision": same_precision,
        "automatic_decode_tokens_per_second": auto_rate,
        "forced_decode_tokens_per_second": forced_rate,
        "forced_to_automatic_throughput_ratio": ratio if same_precision else None,
        "token_ids_match": tokens_match,
        "status": "PASS" if same_precision and tokens_match else "FAIL",
    }


def _write_chart(
    output: Path,
    *,
    title: str,
    labels: Sequence[str],
    values: Sequence[float | int | None],
    x_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    present = [(label, float(value)) for label, value in zip(labels, values, strict=True) if value is not None]
    height = max(4.5, len(present) * 0.38 + 1.8)
    figure, axis = plt.subplots(figsize=(11, height), constrained_layout=True)
    if present:
        plot_labels, plot_values = zip(*present, strict=True)
        positions = list(range(len(plot_labels)))
        axis.barh(positions, plot_values, color="#2b6cb0")
        axis.set_yticks(positions, labels=plot_labels)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.25)
    else:
        axis.text(0.5, 0.5, "No real PASS measurements", ha="center", va="center")
        axis.set_xticks([])
        axis.set_yticks([])
    axis.set_title(title)
    axis.set_xlabel(x_label)
    figure.savefig(output, format="svg", bbox_inches="tight")
    plt.close(figure)


def generate_charts(
    root: Path,
    compatibility: Sequence[Mapping[str, object]],
    comparisons: Sequence[Mapping[str, object]],
) -> None:
    labels = [str(row["family"]) for row in compatibility]
    definitions = (
        ("decode_throughput.svg", "Decode throughput by model", "decode_tokens_per_second", "tokens/s"),
        ("prefill_throughput.svg", "Prefill throughput by model", "prefill_tokens_per_second", "tokens/s"),
        ("ttft.svg", "Time to first token by model", "ttft_ms", "milliseconds"),
        ("peak_vram.svg", "Peak VRAM by model", "peak_vram_bytes", "bytes"),
        ("peak_ram.svg", "Peak system RAM by model", "peak_ram_bytes", "bytes"),
        ("storage_read.svg", "Storage bytes per generated token", "disk_bytes_per_token", "bytes/token"),
    )
    for filename, title, key, x_label in definitions:
        _write_chart(
            root / "charts" / filename,
            title=title,
            labels=labels,
            values=[row.get(key) if isinstance(row.get(key), (int, float)) else None for row in compatibility],
            x_label=x_label,
        )
    comparison_labels = [f"{row['family']} ({row['forced_engine']})" for row in comparisons]
    _write_chart(
        root / "charts" / "colibri_vs_alternative.svg",
        title="Forced backend throughput relative to automatic plan",
        labels=comparison_labels,
        values=[
            row.get("forced_to_automatic_throughput_ratio")
            if isinstance(row.get("forced_to_automatic_throughput_ratio"), (int, float))
            else None
            for row in comparisons
        ],
        x_label="forced / automatic decode throughput",
    )


class MajorModelAcceptanceRunner:
    def __init__(
        self,
        *,
        suite: MajorModelSuite,
        evidence: EvidenceDirectory,
        state_root: Path | None,
        selected_ids: frozenset[str],
        execute: bool,
        acquire: bool,
    ) -> None:
        self.suite = suite
        self.evidence = evidence
        self.state_root = state_root
        self.selected_ids = selected_ids
        self.execute = execute
        self.acquire = acquire
        self.api = HfApi()
        self.preflights: dict[str, ArtifactPreflight] = {}
        self.identities: dict[str, dict[str, object]] = {}
        self.resolution_errors: dict[str, tuple[ValidationStatus, str]] = {}
        self.performance: list[dict[str, object]] = []
        self.comparisons: list[dict[str, object]] = []
        self.correctness: dict[str, object] = {}
        self.compatibility: list[dict[str, object]] = []

    def resolve_all(self) -> None:
        for target in self.suite.targets:
            if self.selected_ids and target.id not in self.selected_ids:
                continue
            try:
                preflight, identity = resolve_artifact_preflight(
                    target,
                    self.suite,
                    api=self.api,
                    output_root=self.evidence.root,
                )
                self.preflights[target.id] = preflight
                self.identities[target.id] = identity
            except Exception as exc:
                status = _classify_error(exc)
                self.resolution_errors[target.id] = (status, f"{type(exc).__name__}: {exc}")

    async def _execute_target(
        self,
        target: MajorModelTarget,
        preflight: ArtifactPreflight,
    ) -> tuple[ValidationStatus, str | None, list[dict[str, object]]]:
        revision = preflight.revision
        measured: list[dict[str, object]] = []
        decode_spec = self.suite.workloads["decode"]
        assert decode_spec.prompt is not None
        maximum_source_bytes = max(100 * _GIB, preflight.required_artifact_bytes + 2 * _GIB)
        try:
            warmup_summary, _warmup_events, _warmup_resources = await _canonical_run(
                target=target,
                revision=revision,
                prompt=decode_spec.prompt,
                max_new_tokens=min(8, decode_spec.max_new_tokens),
                requested_engine=None,
                state_root=self.state_root,
                maximum_source_bytes=maximum_source_bytes,
                runtime_log_root=self.evidence.root / "logs" / target.id,
            )
            if warmup_summary.status != "completed" or not warmup_summary.output_token_ids:
                raise RuntimeError("warm-up completed without actual generated token IDs")
            if warmup_summary.model_revision != revision:
                raise RuntimeError("canonical runtime used a different model revision")
            tokenizer = _tokenizer_for(target, revision)
            workloads = {
                name: (*_prompt_and_ids(tokenizer, spec), spec.max_new_tokens)
                for name, spec in self.suite.workloads.items()
            }
            workload_references: dict[str, list[int]] = {}
            for workload_name, (prompt, prompt_ids, max_new_tokens) in workloads.items():
                for repetition in range(1, target.repetitions + 1):
                    summary, events, resources = await _canonical_run(
                        target=target,
                        revision=revision,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        requested_engine=None,
                        state_root=self.state_root,
                        maximum_source_bytes=maximum_source_bytes,
                        runtime_log_root=self.evidence.root / "logs" / target.id,
                    )
                    row = _run_record(
                        target=target,
                        revision=revision,
                        workload=workload_name,
                        repetition=repetition,
                        requested_engine=None,
                        prompt_token_ids=prompt_ids,
                        summary=summary,
                        events=events,
                        resources=resources,
                    )
                    measured.append(row)
                    if row["execution_status"] != "PASS":
                        raise RuntimeError(f"{workload_name} repetition {repetition} did not generate tokens")
                    if row["architecture_id"] != target.architecture_id:
                        raise RuntimeError(
                            f"resolved architecture {row['architecture_id']!r} does not match "
                            f"required {target.architecture_id!r}"
                        )
                    tokens = [int(item) for item in row["generated_token_ids"]]  # type: ignore[union-attr]
                    reference = workload_references.setdefault(workload_name, tokens)
                    if tokens != reference:
                        row["correctness_status"] = "FAIL"
                        raise RuntimeError(
                            f"deterministic output changed for {workload_name} repetition {repetition}"
                        )

            automatic_decode = next(row for row in measured if row["workload"] == "decode")
            forced_evidence: dict[str, object] = {}
            for engine in target.comparison_engines:
                prompt, prompt_ids, max_new_tokens = workloads["decode"]
                try:
                    summary, events, resources = await _canonical_run(
                        target=target,
                        revision=revision,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        requested_engine=engine,
                        state_root=self.state_root,
                        maximum_source_bytes=maximum_source_bytes,
                        runtime_log_root=self.evidence.root / "logs" / target.id,
                    )
                    forced = _run_record(
                        target=target,
                        revision=revision,
                        workload="decode-backend-comparison",
                        repetition=1,
                        requested_engine=engine,
                        prompt_token_ids=prompt_ids,
                        summary=summary,
                        events=events,
                        resources=resources,
                    )
                    comparison = _comparison_row(target, automatic_decode, forced, engine)
                    self.comparisons.append(comparison)
                    forced_evidence[engine] = {
                        "tokens": forced["generated_token_ids"],
                        "components": forced["component_engines"],
                        "status": comparison["status"],
                    }
                    if comparison["status"] != "PASS":
                        raise RuntimeError(f"forced {engine} output differs from automatic output")
                    if engine == "colibri" and "colibri" not in forced["component_engines"]:  # type: ignore[operator]
                        raise RuntimeError("forced Colibri run did not execute a Colibri component")
                except Exception as exc:
                    forced_evidence[engine] = {
                        "status": "FAIL",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    if engine == "colibri" and target.require_colibri:
                        raise
            self.correctness[target.id] = {
                "status": "PASS",
                "deterministic_decoding": True,
                "automatic_reference_tokens": automatic_decode["generated_token_ids"],
                "workload_repetitions_match": True,
                "forced_backends": forced_evidence,
            }
            return ValidationStatus.PASS, None, measured
        except Exception as exc:
            status = _classify_error(exc)
            reason = f"{type(exc).__name__}: {exc}"
            self.correctness[target.id] = {
                "status": "FAIL" if status == ValidationStatus.FAIL else "NOT_RUN",
                "reason": reason,
            }
            self.evidence.text(f"logs/{target.id}.log", reason + "\n")
            return status, reason, measured

    async def run(self) -> str:
        self.resolve_all()
        self.evidence.json(
            "manifest.json",
            {
                "schema_version": 1,
                "run_id": self.evidence.root.name,
                "started_at_utc": datetime.now(UTC).isoformat(),
                "revision_policy": self.suite.revision_policy,
                "resolved_models": self.identities,
                "resolution_errors": self.resolution_errors,
                "execution_requested": self.execute,
                "acquisition_permitted": self.acquire,
                "single_physical_machine_required": True,
                "remote_inference_permitted": False,
                "overall_status": "RUNNING",
            },
        )
        for target in self.suite.targets:
            preflight = self.preflights.get(target.id)
            revision = preflight.revision if preflight else None
            runs: list[dict[str, object]] = []
            reason: str | None = None
            if self.selected_ids and target.id not in self.selected_ids:
                status = ValidationStatus.NOT_RUN
                reason = "target excluded by --only"
            elif target.id in self.resolution_errors:
                status, reason = self.resolution_errors[target.id]
            elif preflight is None:
                status = ValidationStatus.FAIL
                reason = "artifact preflight did not produce a result"
            elif not preflight.resource_sufficient:
                status = ValidationStatus.BLOCKED_RESOURCE
                reason = preflight.blocking_reason
            elif not self.execute:
                status = ValidationStatus.NOT_RUN
                reason = "preflight-only mode explicitly disabled inference"
            elif not self.acquire and not preflight.cache_complete:
                status = ValidationStatus.NOT_RUN
                reason = "required immutable artifact is incomplete and acquisition was disabled"
            else:
                status, reason, runs = await self._execute_target(target, preflight)
                self.performance.extend(runs)
            summary = _model_summary_row(target, status, revision, runs, preflight, reason)
            self.compatibility.append(summary)
            self.evidence.json(
                f"per_model/{target.id}.json",
                {
                    "target": target.model_dump(mode="json"),
                    "identity": self.identities.get(target.id),
                    "preflight": preflight.model_dump(mode="json") if preflight else None,
                    "status": status.value,
                    "reason": reason,
                    "runs": runs,
                    "correctness": self.correctness.get(target.id),
                },
            )
            self._write_tables()

        mandatory = [
            row for row, target in zip(self.compatibility, self.suite.targets, strict=True) if target.mandatory
        ]
        overall = (
            "MAJOR_MODEL_REAL_RUN_PASS"
            if mandatory and all(row["status"] == "PASS" for row in mandatory)
            else "MAJOR_MODEL_REAL_RUN_INCOMPLETE"
        )
        manifest = json.loads((self.evidence.root / "manifest.json").read_text(encoding="utf-8"))
        manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
        manifest["overall_status"] = overall
        manifest["status_counts"] = {
            status: sum(1 for row in self.compatibility if row["status"] == status)
            for status in _STATUS_VALUES
        }
        self.evidence.json("manifest.json", manifest)
        generate_charts(self.evidence.root, self.compatibility, self.comparisons)
        self.evidence.hashes()
        return overall

    def _write_tables(self) -> None:
        self.evidence.json("compatibility_matrix.json", self.compatibility)
        self.evidence.csv("compatibility_matrix.csv", self.compatibility)
        self.evidence.csv("performance.csv", self.performance)
        self.evidence.csv("backend_comparison.csv", self.comparisons)
        self.evidence.json("correctness.json", self.correctness)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/validation/major_open_weight_models.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/major-model-acceptance"),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--no-acquire", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    suite = load_suite(args.config)
    selected = frozenset(
        item.strip() for value in args.only for item in str(value).split(",") if item.strip()
    )
    known = {target.id for target in suite.targets}
    unknown = selected - known
    if unknown:
        raise ValueError(f"unknown --only target(s): {', '.join(sorted(unknown))}")
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid4().hex[:8]}"
    evidence = EvidenceDirectory(args.output / run_id)
    evidence.json("hardware.json", capture_hardware(evidence.root))
    runner = MajorModelAcceptanceRunner(
        suite=suite,
        evidence=evidence,
        state_root=args.state_root,
        selected_ids=selected,
        execute=not args.preflight_only,
        acquire=not args.no_acquire,
    )
    overall = asyncio.run(runner.run())
    print(overall)
    print(evidence.root)
    return 0 if overall == "MAJOR_MODEL_REAL_RUN_PASS" else 1


__all__ = [
    "ArtifactPreflight",
    "EvidenceDirectory",
    "MajorModelAcceptanceRunner",
    "MajorModelSuite",
    "MajorModelTarget",
    "ValidationStatus",
    "capture_hardware",
    "load_suite",
    "main",
    "resolve_artifact_preflight",
]
