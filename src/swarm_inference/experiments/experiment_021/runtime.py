"""Pre-physical runtime, lifecycle, cache, and correctness receipts for E021."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_020.model_distribution import ShardCache

from .control_plane import run_control_plane_scale
from .io import atomic_write_json, sha256_file, write_csv


def run_shard_cache_startup_self_test() -> dict[str, Any]:
    payload = bytes((index * 37 + 11) & 0xFF for index in range(256 * 1024 + 113))
    digest = hashlib.sha256(payload).hexdigest()
    requests: list[dict[str, Any]] = []

    def fetcher(_url: str, start: int, stop: int | None) -> Iterable[bytes]:
        requests.append({"start": start, "stop": stop})
        upper = len(payload) if stop is None else min(stop, len(payload))
        for offset in range(start, upper, 32768):
            yield payload[offset : min(offset + 32768, upper)]

    with tempfile.TemporaryDirectory(prefix="e021-shard-cache-") as directory:
        cache = ShardCache(Path(directory))
        first = cache.acquire(
            "memory://e021-cache-object",
            digest,
            len(payload),
            fetcher=fetcher,
        )
        first_hash = sha256_file(first)
        fetch_count_after_first = len(requests)
        second = cache.acquire(
            "memory://e021-cache-object",
            digest,
            len(payload),
            fetcher=fetcher,
        )
        cache_hit_without_fetch = len(requests) == fetch_count_after_first
        marker = first.with_name(first.name + ".complete")
        marker_value = json.loads(marker.read_text(encoding="utf-8"))

        corrupted_rejected = False
        wrong_digest = hashlib.sha256(b"different expected content").hexdigest()
        try:
            cache.acquire(
                "memory://e021-corrupt-object",
                wrong_digest,
                len(payload),
                fetcher=fetcher,
            )
        except RuntimeError as exc:
            corrupted_rejected = "hash mismatch" in str(exc)

    status = (
        first == second
        and first_hash == digest
        and cache_hit_without_fetch
        and marker_value.get("complete") is True
        and marker_value.get("sha256") == digest
        and corrupted_rejected
    )
    return {
        "schema_version": "experiment-021-shard-cache-startup-v1",
        "status": "PASS" if status else "FAIL",
        "content_addressed": True,
        "expected_sha256": digest,
        "verified_sha256": first_hash,
        "expected_bytes": len(payload),
        "first_acquisition_fetch_calls": fetch_count_after_first,
        "second_startup_cache_hit_without_fetch": cache_hit_without_fetch,
        "atomic_complete_marker_verified": marker_value.get("complete") is True,
        "corrupted_object_rejected": corrupted_rejected,
        "test_object_persisted": False,
        "real_checkpoint_bundle_startup_tested": False,
        "scope": (
            "production ShardCache acquisition/hash/atomic-marker primitive; "
            "not a full remote K3 bundle acquisition"
        ),
    }


def image_preflight(repo: Path) -> dict[str, Any]:
    source = repo / "artifacts" / "experiment-020" / "deployment" / "docker-image-inspect.json"
    dockerfile = repo / "deployment" / "Dockerfile.e021"
    prior = json.loads(source.read_text(encoding="utf-8")) if source.is_file() else {}
    digest = str(prior.get("digest", ""))
    immutable = digest.startswith("sha256:") and len(digest) == 71
    rejected_topology = (
        int(prior.get("eight_worker_lifecycle", {}).get("explicit_worker_processes", 0))
        == 8
    )
    return {
        "schema_version": "experiment-021-image-preflight-v1",
        "status": "FAIL",
        "linux_image_present": bool(prior.get("present")),
        "linux_amd64": prior.get("os") == "linux" and prior.get("architecture") == "amd64",
        "locally_immutable_digest_present": immutable,
        "local_digest": digest or None,
        "published_registry_digest": None,
        "pinned_for_future_independent_machine_deployment": False,
        "source_dockerfile": str(dockerfile),
        "source_dockerfile_sha256": sha256_file(dockerfile) if dockerfile.is_file() else None,
        "rejected_e020_eight_worker_host_lifecycle_embedded": rejected_topology,
        "failure": (
            "The only built image is local/unpublished and embeds the rejected E020 "
            "eight-workers-per-host lifecycle. It is not an immutable E021 "
            "one-worker-per-independent-machine deployment image."
        ),
    }


def run_control_plane_scaling(
    artifact_root: Path,
    *,
    natural_worker_count: int,
) -> list[dict[str, Any]]:
    counts = list(
        dict.fromkeys((100, 250, natural_worker_count, 500, 1000, 2000))
    )
    rows: list[dict[str, Any]] = []
    for count in counts:
        started = time.time_ns()
        try:
            receipt = run_control_plane_scale(count, connection_batch=100)
            row = {
                **receipt,
                "requested_worker_count": count,
                "machine_count": count,
                "compute_workers_per_machine": 1,
                "independent_machine_worker_ids": True,
                "same_host_collective": False,
                "native_shard_compute_invoked": False,
                "workload": "lightweight authenticated lifecycle/dispatch scale test",
            }
        except Exception as exc:
            row = {
                "schema_version": "experiment-021-control-plane-scaling-v1",
                "status": "FAIL",
                "requested_worker_count": count,
                "worker_count": count,
                "machine_count": count,
                "compute_workers_per_machine": 1,
                "same_host_collective": False,
                "native_shard_compute_invoked": False,
                "failure": f"{type(exc).__name__}: {exc}",
            }
        row["started_unix_ns"] = started
        row["finished_unix_ns"] = time.time_ns()
        rows.append(row)
    write_csv(artifact_root / "control-plane" / "scaling.csv", rows)
    return rows


def production_correctness_receipt(repo: Path, artifact_root: Path) -> dict[str, Any]:
    prior_path = (
        repo
        / "artifacts"
        / "experiment-020"
        / "correctness"
        / "full-93-sharded.json"
    )
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    prior_summary = {
        "artifact": str(prior_path),
        "artifact_sha256": sha256_file(prior_path),
        "status": prior.get("status"),
        "evidence_class": prior.get("evidence_class"),
        "complete_93_layer_graph": prior.get("complete_93_layer_graph"),
        "executed_layers": prior.get("executed_layers"),
        "routes_exact": prior.get("routes_exact"),
        "maximum_relative_l2_error": prior.get("maximum_relative_l2_error"),
        "worker_operation_count": prior.get("worker_operation_count"),
        "wall_seconds": prior.get("wall_seconds"),
        "candidate": prior.get("candidate"),
    }
    receipt = {
        "schema_version": "experiment-021-worker-process-93-layer-v1",
        "status": "FAIL",
        "gate": "complete 93-layer correctness through production worker-process dispatch",
        "worker_process_transport": "authenticated EXECUTE_SHARD frame",
        "execute_shard_invokes_real_native_shard_code": False,
        "complete_93_layer_worker_process_traversal": False,
        "in_process_monolithic_fallback_used": False,
        "test_run_after_model_invalid": False,
        "not_run_reason": (
            "The corrected ordered replay invalidated the resident-worker model, "
            "and the current production daemon still returns a mock partial vector "
            "rather than dispatching native shard primitives. Running the prior "
            "44-minute in-process graph again would not satisfy this gate."
        ),
        "prior_physical_shard_math_control": prior_summary,
        "prior_control_admissible_for_this_gate": False,
        "hidden_relative_l2": None,
        "logit_relative_l2": None,
        "exact_greedy_token": None,
        "state_fingerprints": None,
        "routes_exact": None,
        "decisive_blocker": "production EXECUTE_SHARD native dispatch is unimplemented",
        "evidence_class": "NO_E021_WORKER_PROCESS_CORRECTNESS_EVIDENCE",
    }
    atomic_write_json(artifact_root / "correctness" / "worker-process-93-layer.json", receipt)
    return receipt


def materialize_runtime_receipts(repo: Path, artifact_root: Path) -> dict[str, Any]:
    cache = run_shard_cache_startup_self_test()
    image = image_preflight(repo)
    correctness = production_correctness_receipt(repo, artifact_root)
    receipt = {
        "schema_version": "experiment-021-production-runtime-gates-v1",
        "status": "FAIL",
        "controller_lifecycle_implemented": True,
        "authenticated_transport_implemented": True,
        "production_execute_shard_native_dispatch": False,
        "complete_worker_process_correctness": False,
        "shard_cache_startup_primitive": cache,
        "pinned_image_preflight": image,
        "correctness": correctness,
    }
    atomic_write_json(artifact_root / "runtime" / "shard-cache-startup.json", cache)
    atomic_write_json(artifact_root / "runtime" / "image-preflight.json", image)
    atomic_write_json(artifact_root / "runtime" / "production-gates.json", receipt)
    return receipt


__all__ = [
    "image_preflight",
    "materialize_runtime_receipts",
    "production_correctness_receipt",
    "run_control_plane_scaling",
    "run_shard_cache_startup_self_test",
]
