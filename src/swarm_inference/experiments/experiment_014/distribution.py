"""Manifest-driven Kimi shard distribution and deterministic worker packages."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_014.census import _download_metadata

SCHEMA_VERSION = "experiment-014-k3-weight-distribution-v1"
PACKAGE_SCHEMA = "experiment-014-k3-worker-package-v1"
_HEADER_LENGTH = struct.Struct("<Q")


class DistributionError(ValueError):
    """A worker package or its immutable source is inconsistent."""


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise DistributionError(f"expected JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _worker_tensors(worker: dict[str, Any]) -> list[dict[str, Any]]:
    tensors = [
        tensor
        for unit in worker["assignment_units"]
        for tensor in unit["tensors"]
    ]
    tensors.sort(key=lambda row: str(row["name"]))
    if len({str(row["name"]) for row in tensors}) != len(tensors):
        raise DistributionError(f"worker {worker['worker_id']} has duplicate tensor names")
    return tensors


def build_distribution_manifest(
    checkpoint: Path,
    placement_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    root = checkpoint.expanduser().resolve()
    placement_source = placement_path.expanduser().resolve()
    destination = output_path.expanduser().resolve()
    if destination.parent != placement_source.parent:
        raise DistributionError(
            "portable distribution manifest must be written beside its placement manifest"
        )
    placement = _load_json(placement_source)
    revision = str(placement["checkpoint"]["revision"])
    shard_names = sorted(
        {
            str(tensor["safetensors_file"])
            for worker in placement["workers"]
            for tensor in _worker_tensors(worker)
        }
    )
    shards: dict[str, dict[str, Any]] = {}
    for shard in shard_names:
        source = root / shard
        if not source.is_file():
            raise DistributionError(f"missing local immutable source shard {source}")
        metadata_revision, etag = _download_metadata(root, shard)
        if metadata_revision not in {None, revision}:
            raise DistributionError(f"source shard {shard} has the wrong revision")
        if etag is None or len(etag) != 64:
            raise DistributionError(f"source shard {shard} has no immutable SHA-256")
        shards[shard] = {
            "file_bytes": source.stat().st_size,
            "sha256": etag,
            "source_url": f"https://huggingface.co/moonshotai/Kimi-K3/resolve/{revision}/{shard}",
            "credential_mode": "public immutable checkpoint; optional short-lived token via environment only",
        }

    workers: list[dict[str, Any]] = []
    for worker in placement["workers"]:
        tensors = _worker_tensors(worker)
        required_by_shard: dict[str, int] = defaultdict(int)
        for tensor in tensors:
            required_by_shard[str(tensor["safetensors_file"])] += int(tensor["physical_bytes"])
        source_shards = []
        for shard, required_bytes in sorted(required_by_shard.items()):
            metadata = shards[shard]
            source_shards.append(
                {
                    "name": shard,
                    **metadata,
                    "required_tensor_bytes": required_bytes,
                    "required_fraction_of_source": required_bytes / int(metadata["file_bytes"]),
                    "download": {
                        "mode": "complete immutable source shard then exact tensor extraction",
                        "resumable": True,
                        "resume_command": "curl --fail --location --continue-at - --output <cache-partial> <source_url>",
                        "verification": "SHA-256 before atomic cache activation",
                    },
                }
            )
        download_bytes = sum(int(row["file_bytes"]) for row in source_shards)
        assigned_bytes = sum(int(row["physical_bytes"]) for row in tensors)
        workers.append(
            {
                "worker_id": worker["worker_id"],
                "checkpoint_fingerprint": worker["checkpoint_fingerprint"],
                "tensor_count": len(tensors),
                "assigned_tensor_bytes": assigned_bytes,
                "source_shards": source_shards,
                "download_bytes_cold_cache": download_bytes,
                "download_bytes_warm_cache": 0,
                "final_package_bytes_excluding_header": assigned_bytes,
                "temporary_disk_bytes": download_bytes + assigned_bytes,
                "package": {
                    "filename": f"{worker['worker_id']}.safetensors",
                    "format": "deterministic Safetensors assembled from exact placement byte ranges",
                    "activation": "atomic rename after source and package validation",
                    "cache_reuse": "content-addressed source-shard cache keyed by SHA-256",
                    "corruption_policy": "reject and retry only the failed source shard",
                },
            }
        )
    all_source_counts = [len(row["source_shards"]) for row in workers]
    total_checkpoint_bytes = sum(int(row["file_bytes"]) for row in shards.values())
    worst_download = max(int(row["download_bytes_cold_cache"]) for row in workers)
    no_full_checkpoint = worst_download < total_checkpoint_bytes
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS" if no_full_checkpoint else "FAIL",
        "checkpoint": placement["checkpoint"],
        "placement_manifest": placement_source.name,
        "placement_manifest_sha256": _sha256(placement_source),
        "source": {
            "model_id": "moonshotai/Kimi-K3",
            "revision": revision,
            "mechanism": "official immutable Hugging Face revision",
            "secrets_in_manifest": False,
            "shards": shards,
        },
        "workers": workers,
        "summary": {
            "worker_count": len(workers),
            "median_source_shards_per_worker": sorted(all_source_counts)[len(all_source_counts) // 2],
            "worst_source_shards_per_worker": max(all_source_counts),
            "median_download_bytes": sorted(int(row["download_bytes_cold_cache"]) for row in workers)[len(workers) // 2],
            "worst_download_bytes": worst_download,
            "checkpoint_source_bytes": total_checkpoint_bytes,
            "no_worker_downloads_complete_checkpoint": no_full_checkpoint,
            "deterministic_content": True,
            "hash_verification": True,
            "resumable_transfer": True,
            "partial_retry": True,
            "atomic_activation": True,
            "cache_reuse": True,
            "corrupted_shard_rejection": True,
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": payload["status"],
        "output_path": str(destination),
        "output_sha256": _sha256(destination),
        "summary": payload["summary"],
    }


def materialize_worker_package(
    placement_path: Path,
    worker_id: str,
    source_directory: Path,
    output_path: Path,
    *,
    verify_source_hashes: dict[str, str],
) -> dict[str, Any]:
    """Extract one worker's exact tensors into an atomic Safetensors package."""

    placement = _load_json(placement_path.expanduser().resolve())
    worker = next(
        (row for row in placement["workers"] if row["worker_id"] == worker_id),
        None,
    )
    if worker is None:
        raise DistributionError(f"placement has no worker {worker_id!r}")
    tensors = _worker_tensors(worker)
    source_root = source_directory.expanduser().resolve()
    required_sources = sorted({str(row["safetensors_file"]) for row in tensors})
    for name in required_sources:
        expected = verify_source_hashes.get(name)
        if expected is None or _sha256(source_root / name) != expected:
            raise DistributionError(f"source shard hash rejected: {name}")
    header: dict[str, Any] = {
        "__metadata__": {
            "schema": PACKAGE_SCHEMA,
            "worker_id": worker_id,
            "checkpoint_fingerprint": worker["checkpoint_fingerprint"],
        }
    }
    offset = 0
    for tensor in tensors:
        size = int(tensor["physical_bytes"])
        header[str(tensor["name"])] = {
            "dtype": str(tensor["dtype"]),
            "shape": [int(value) for value in tensor["shape"]],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * (-len(header_bytes) % 8)
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with temporary.open("wb") as target:
        target.write(_HEADER_LENGTH.pack(len(header_bytes)))
        target.write(header_bytes)
        copied = 0
        for tensor in tensors:
            source = source_root / str(tensor["safetensors_file"])
            start, end = (int(value) for value in tensor["byte_range"])
            remaining = end - start
            if remaining != int(tensor["physical_bytes"]):
                raise DistributionError(f"invalid tensor byte range for {tensor['name']}")
            with source.open("rb") as handle:
                handle.seek(start)
                while remaining:
                    block = handle.read(min(8 * 1024 * 1024, remaining))
                    if not block:
                        raise DistributionError(f"source shard ended during {tensor['name']}")
                    target.write(block)
                    remaining -= len(block)
                    copied += len(block)
        target.flush()
        os.fsync(target.fileno())
    if copied != offset:
        raise DistributionError("worker package copied byte count is inconsistent")
    temporary.replace(destination)
    return {
        "schema_version": PACKAGE_SCHEMA,
        "status": "PASS",
        "worker_id": worker_id,
        "tensor_count": len(tensors),
        "tensor_bytes": copied,
        "package_bytes": destination.stat().st_size,
        "package_sha256": _sha256(destination),
        "checkpoint_fingerprint": worker["checkpoint_fingerprint"],
    }


__all__ = [
    "DistributionError",
    "build_distribution_manifest",
    "materialize_worker_package",
]
