"""Pod-scoped K3 acquisition manifests and resumable content-addressed cache."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from huggingface_hub import get_token

from swarm_inference.experiments.experiment_019.checkpoint import CheckpointCatalog
from swarm_inference.experiments.experiment_019.placement import PlacementResult, assignments_for

RangeFetcher = Callable[[str, int, int | None], Iterable[bytes]]


def _local_huggingface_hashes(catalog: CheckpointCatalog) -> dict[str, str]:
    """Read immutable LFS SHA-256 values recorded by `hf download`."""

    hashes = dict(catalog.lfs_hashes())
    metadata_root = catalog.root / ".cache" / "huggingface" / "download"
    for file in set(catalog.weight_map.values()):
        if file in hashes:
            continue
        metadata = metadata_root / f"{file}.metadata"
        if not metadata.is_file():
            continue
        lines = metadata.read_text(encoding="utf-8").splitlines()
        if len(lines) < 2:
            continue
        candidate = lines[1].strip().lower()
        if len(candidate) == 64 and all(ch in "0123456789abcdef" for ch in candidate):
            hashes[file] = candidate
    return hashes


def http_range_fetcher(url: str, start: int, stop: int | None) -> Iterable[bytes]:
    headers = {"Range": f"bytes={start}-{'' if stop is None else stop - 1}"}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        status = getattr(response, "status", 200)
        if start and status != 206:
            raise RuntimeError("remote source ignored resume Range request")
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            yield block


def probe_remote_range_acquisition(
    url: str,
    local_file: Path,
    *,
    sample_bytes: int = 64 * 1024,
) -> dict[str, Any]:
    """Prove public/gated range access and resume without persisting a token."""

    token = get_token()
    halves: list[bytes] = []
    responses: list[dict[str, Any]] = []
    midpoint = sample_bytes // 2
    for start, stop in ((0, midpoint), (midpoint, sample_bytes)):
        headers = {"Range": f"bytes={start}-{stop - 1}"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=90) as response:
            body = response.read(stop - start)
            status = int(getattr(response, "status", 200))
            responses.append(
                {
                    "requested_start": start,
                    "requested_stop": stop,
                    "status": status,
                    "content_range_present": bool(response.headers.get("Content-Range")),
                    "received_bytes": len(body),
                }
            )
            halves.append(body)
    remote = b"".join(halves)
    with local_file.open("rb") as handle:
        local = handle.read(sample_bytes)
    return {
        "source_url": url,
        "source_file": local_file.name,
        "authentication_required": token is not None,
        "secret_present": token is not None,
        "secret_value_persisted": False,
        "requested_bytes": sample_bytes,
        "received_bytes": len(remote),
        "range_responses": responses,
        "range_supported": all(row["status"] == 206 for row in responses),
        "resume_supported": responses[1]["status"] == 206,
        "local_sha256": hashlib.sha256(local).hexdigest(),
        "remote_sha256": hashlib.sha256(remote).hexdigest(),
        "local_match": remote == local,
    }


class ShardCache:
    """One hash-verified copy per pod with atomic completion semantics."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.objects = self.root / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def _paths(self, sha256: str) -> tuple[Path, Path, Path]:
        if len(sha256) != 64 or any(ch not in "0123456789abcdef" for ch in sha256):
            raise ValueError("lowercase SHA-256 identity required")
        directory = self.objects / sha256[:2]
        directory.mkdir(parents=True, exist_ok=True)
        final = directory / sha256
        return final, directory / f"{sha256}.partial", directory / f"{sha256}.complete"

    def _lock(self, sha256: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(sha256, threading.Lock())

    def acquire(
        self,
        url: str,
        sha256: str,
        expected_bytes: int,
        *,
        fetcher: RangeFetcher = http_range_fetcher,
    ) -> Path:
        final, partial, marker = self._paths(sha256)
        with self._lock(sha256):
            if final.is_file() and marker.is_file():
                metadata = json.loads(marker.read_text(encoding="utf-8"))
                if metadata.get("sha256") == sha256 and final.stat().st_size == expected_bytes:
                    return final
            start = partial.stat().st_size if partial.exists() else 0
            if start > expected_bytes:
                partial.unlink()
                start = 0
            mode = "ab" if start else "wb"
            with partial.open(mode) as handle:
                for block in fetcher(url, start, expected_bytes):
                    remaining = expected_bytes - handle.tell()
                    if remaining <= 0:
                        break
                    handle.write(block[:remaining])
                    handle.flush()
                    os.fsync(handle.fileno())
            if partial.stat().st_size != expected_bytes:
                raise RuntimeError("partial acquisition did not reach expected byte count")
            digest = hashlib.sha256()
            with partial.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != sha256:
                raise RuntimeError("downloaded object hash mismatch")
            os.replace(partial, final)
            marker_value = {
                "sha256": sha256,
                "bytes": expected_bytes,
                "complete": True,
            }
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=marker.parent, delete=False, suffix=".tmp"
            ) as handle:
                json.dump(marker_value, handle, sort_keys=True)
                handle.write("\n")
                temporary = Path(handle.name)
            os.replace(temporary, marker)
            return final


def build_pod_bundles(
    catalog: CheckpointCatalog,
    placement: PlacementResult,
    *,
    remote_repository: str = "moonshotai/Kimi-K3",
) -> dict[str, Any]:
    """Build deterministic file-level pod acquisition plans.

    K3's 93 transformer files each contain exactly one layer.  Pods therefore
    fetch source Safetensors once and all eight local workers map their assigned
    tensor ranges from that shared copy.  Endpoint/vision files may be
    intentionally replicated when their tensors are placed in different pods.
    """

    records = catalog.records()
    file_pods: dict[str, set[str]] = defaultdict(set)
    file_consumers: dict[tuple[str, str], set[str]] = defaultdict(set)
    file_tensor_count: dict[tuple[str, str], int] = defaultdict(int)
    for record in records.values():
        assignments = assignments_for(record, placement.spec)
        pods: set[str] = set()
        for assignment in assignments:
            worker = placement.workers[assignment.worker_index]
            pods.add(worker.pod_id)
            file_consumers[(record.file, worker.pod_id)].add(worker.worker_id)
        for pod in pods:
            file_pods[record.file].add(pod)
            file_tensor_count[(record.file, pod)] += 1
    hashes = _local_huggingface_hashes(catalog)
    bundles: list[dict[str, Any]] = []
    all_files = sorted(set(catalog.weight_map.values()))
    file_sizes = {name: (catalog.root / name).stat().st_size for name in all_files}
    for pod_index in range(placement.spec.pod_count):
        pod = f"pod-{pod_index:03d}"
        files = []
        for file in all_files:
            if pod not in file_pods[file]:
                continue
            files.append(
                {
                    "source_file": file,
                    "source_url": (
                        f"https://huggingface.co/{remote_repository}/resolve/main/{file}"
                    ),
                    "required_byte_ranges": [[0, file_sizes[file]]],
                    "expected_physical_bytes": file_sizes[file],
                    "worker_consumers": sorted(file_consumers[(file, pod)]),
                    "tensor_count": file_tensor_count[(file, pod)],
                    "sha256": hashes.get(file),
                    "hash_source": "local_huggingface_download_lfs_metadata",
                    "shared_once_per_pod": True,
                }
            )
        bundles.append(
            {
                "pod_id": pod,
                "layers": sorted(
                    {
                        layer
                        for worker in placement.workers
                        if worker.pod_id == pod
                        for layer in worker.assigned_layers
                    }
                ),
                "files": files,
                "disk_requirement_bytes": sum(row["expected_physical_bytes"] for row in files),
                "estimated_download_bytes": sum(
                    row["expected_physical_bytes"] for row in files
                ),
                "worker_count": sum(worker.pod_id == pod for worker in placement.workers),
            }
        )
    total_transfer = sum(bundle["estimated_download_bytes"] for bundle in bundles)
    return {
        "schema_version": "experiment-020-pod-bundles-v1",
        "remote_repository": remote_repository,
        "strategy": "one pod host downloads each required source Safetensors file once; eight workers memory-map only assigned ranges",
        "full_checkpoint_download_per_worker": False,
        "full_checkpoint_download_per_pod": False,
        "bundles": bundles,
        "checkpoint_payload_bytes": catalog.declared_total_size,
        "checkpoint_physical_file_bytes": sum(file_sizes.values()),
        "total_fleet_transfer_bytes": total_transfer,
        "intentional_file_replication_bytes": total_transfer - sum(file_sizes.values()),
        "maximum_copies_of_one_source_file": max(
            (len(pods) for pods in file_pods.values()), default=0
        ),
        "missing_lfs_hashes": sorted(file for file in all_files if file not in hashes),
        "all_files_hash_addressable": all(file in hashes for file in all_files),
    }


def hydrate_worker_manifest_hashes(
    pod_bundles: Mapping[str, Any], manifest_root: Path
) -> dict[str, Any]:
    """Replace placeholder worker file hashes from the canonical pod bundles."""

    expected: dict[str, dict[str, str]] = defaultdict(dict)
    for bundle in pod_bundles.get("bundles", []):
        for source in bundle.get("files", []):
            file_name = str(source["source_file"])
            digest = str(source.get("sha256") or "")
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"missing canonical SHA-256 for {file_name}")
            for worker_id in source.get("worker_consumers", []):
                expected[str(worker_id)][file_name] = digest

    updated = 0
    verified_pairs = 0
    failures: list[dict[str, Any]] = []
    manifest_paths = sorted(manifest_root.glob("*.json"))
    for path in manifest_paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        worker_id = str(manifest["worker_id"])
        declared_files = set(manifest.get("checkpoint_hashes", {}))
        canonical = expected.get(worker_id, {})
        if declared_files != set(canonical):
            failures.append(
                {
                    "worker_id": worker_id,
                    "missing_from_bundle": sorted(declared_files - set(canonical)),
                    "missing_from_manifest": sorted(set(canonical) - declared_files),
                }
            )
            continue
        manifest["checkpoint_hashes"] = dict(sorted(canonical.items()))
        manifest["checkpoint_hash_source"] = (
            "pod-bundles.json/local_huggingface_download_lfs_metadata"
        )
        manifest["all_checkpoint_hashes_available"] = True
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        updated += 1
        verified_pairs += len(canonical)

    return {
        "schema_version": "experiment-020-worker-manifest-hash-audit-v1",
        "status": (
            "PASS"
            if not failures and updated == len(manifest_paths) == len(expected) == 96
            else "FAIL"
        ),
        "worker_manifest_count": len(manifest_paths),
        "updated_worker_manifests": updated,
        "worker_file_hash_pairs": verified_pairs,
        "placeholder_hashes_remaining": sum(
            "UNAVAILABLE" in path.read_text(encoding="utf-8")
            for path in manifest_paths
        ),
        "failures": failures,
    }


__all__ = [
    "ShardCache",
    "build_pod_bundles",
    "http_range_fetcher",
    "hydrate_worker_manifest_hashes",
    "probe_remote_range_acquisition",
]
