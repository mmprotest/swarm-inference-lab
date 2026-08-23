"""Build immutable per-worker acquisition bundles for the E025 image."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any

from swarm_inference.model.kimi_tokenizer import KIMI_TOKENIZER_ASSETS

from .io import atomic_write_json, canonical_sha256, read_json, sha256_file, utc_now

SCHEMA_VERSION = "experiment-025-image-input-v1"


def _portable_worker_placement(
    placement: dict[str, Any], worker: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": placement["schema_version"],
        "generated_at_utc": placement["generated_at_utc"],
        "status": "PASS",
        "model_id": placement["model_id"],
        "checkpoint": placement["checkpoint"],
        "topology": placement["topology"],
        "coverage": {
            "scope": "one exact worker package; global coverage is in the frozen E025 manifest",
            "worker_id": worker["worker_id"],
            "tensor_count": worker["tensor_count"],
            "source_weight_bytes": worker["source_weight_bytes"],
            "assignment_sha256": worker["assignment_sha256"],
        },
        "workers": [worker],
    }


def _portable_distribution(
    distribution: dict[str, Any],
    distribution_worker: dict[str, Any],
    placement_path: Path,
) -> dict[str, Any]:
    source_names = {
        str(row["name"]) for row in distribution_worker["source_shards"]
    }
    source = copy.deepcopy(distribution["source"])
    source["shards"] = {
        name: row
        for name, row in source["shards"].items()
        if name in source_names
    }
    return {
        "schema_version": distribution["schema_version"],
        "generated_at_utc": distribution["generated_at_utc"],
        "status": "PASS",
        "checkpoint": distribution["checkpoint"],
        "placement_manifest": placement_path.name,
        "placement_manifest_sha256": sha256_file(placement_path),
        "source": source,
        "workers": [distribution_worker],
        "summary": {
            "worker_count": 1,
            "global_worker_count": int(distribution["summary"]["worker_count"]),
            "checkpoint_source_bytes": int(
                distribution["summary"]["checkpoint_source_bytes"]
            ),
            "no_worker_downloads_complete_checkpoint": True,
            "deterministic_content": True,
            "hash_verification": True,
            "resumable_transfer": True,
            "atomic_activation": True,
        },
    }


def _worker_runtime_template(worker: dict[str, Any], placement: dict[str, Any]) -> dict[str, Any]:
    role = str(worker["worker_role"])
    layer = int(worker["owned_layers"][0])
    worker_index: int | None = None
    if role == "SUB_LAYER_WORKER":
        worker_index = int(str(worker["worker_id"]).rsplit("-", maxsplit=1)[1])
    return {
        "schema_version": "experiment-025-worker-template-v1",
        "worker_id": worker["worker_id"],
        "role": role,
        "layer": layer,
        "worker_index": worker_index,
        "worker_count": 4 if role == "SUB_LAYER_WORKER" else None,
        "assignment_sha256": worker["assignment_sha256"],
        "checkpoint_fingerprint": placement["checkpoint"]["checkpoint_fingerprint"],
        "checkpoint_revision": placement["checkpoint"]["revision"],
        "tensor_count": worker["tensor_count"],
        "source_weight_bytes": worker["source_weight_bytes"],
        "owned_layers": worker["owned_layers"],
        "owned_components": worker["owned_components"],
        "owned_expert_count": worker["owned_expert_count"],
        "owned_expert_ids": worker["owned_expert_ids"],
        "safetensor_source_files": worker["safetensor_source_files"],
        "sub_layer_proof": worker.get("sub_layer_proof"),
        "topology_id": canonical_sha256(placement["topology"]),
        "maximum_context": 64,
        "whole_layer_fallback": False,
    }


def build_image_inputs(
    *,
    checkpoint: Path,
    placement_path: Path,
    distribution_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    root = checkpoint.expanduser().resolve()
    placement = read_json(placement_path.expanduser().resolve())
    distribution = read_json(distribution_path.expanduser().resolve())
    if placement.get("status") != "PASS" or distribution.get("status") != "PASS":
        raise ValueError("E025 image inputs require passing placement and distribution")
    workers = {str(row["worker_id"]): row for row in placement["workers"]}
    distribution_workers = {
        str(row["worker_id"]): row for row in distribution["workers"]
    }
    if set(workers) != set(distribution_workers):
        raise ValueError("E025 placement and distribution worker sets differ")
    destination = output_directory.expanduser().resolve()
    if destination.exists():
        raise ValueError(f"refusing to replace E025 image input directory: {destination}")
    destination.mkdir(parents=True)
    shared = destination / "shared"
    shared.mkdir()
    shared_names = ("config.json", *KIMI_TOKENIZER_ASSETS)
    shared_hashes: dict[str, str] = {}
    for name in shared_names:
        source = root / name
        if not source.is_file():
            raise ValueError(f"checkpoint image input is absent: {source}")
        shutil.copyfile(source, shared / name)
        shared_hashes[name] = sha256_file(shared / name)
    index_rows: list[dict[str, Any]] = []
    for worker_id in sorted(workers):
        worker_directory = destination / "workers" / worker_id
        worker_directory.mkdir(parents=True)
        portable_placement = _portable_worker_placement(
            placement,
            workers[worker_id],
        )
        portable_placement_path = worker_directory / "placement.json"
        atomic_write_json(portable_placement_path, portable_placement)
        portable_distribution = _portable_distribution(
            distribution,
            distribution_workers[worker_id],
            portable_placement_path,
        )
        portable_distribution_path = worker_directory / "distribution.json"
        atomic_write_json(portable_distribution_path, portable_distribution)
        template_path = worker_directory / "worker-template.json"
        atomic_write_json(
            template_path,
            _worker_runtime_template(workers[worker_id], placement),
        )
        index_rows.append(
            {
                "worker_id": worker_id,
                "role": workers[worker_id]["worker_role"],
                "placement_sha256": sha256_file(portable_placement_path),
                "distribution_sha256": sha256_file(portable_distribution_path),
                "worker_template_sha256": sha256_file(template_path),
                "download_bytes_cold_cache": distribution_workers[worker_id][
                    "download_bytes_cold_cache"
                ],
                "assigned_tensor_bytes": distribution_workers[worker_id][
                    "assigned_tensor_bytes"
                ],
            }
        )
    index = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "status": "PASS",
        "placement_sha256": sha256_file(placement_path.expanduser().resolve()),
        "distribution_sha256": sha256_file(
            distribution_path.expanduser().resolve()
        ),
        "checkpoint_config_sha256": shared_hashes["config.json"],
        "shared_asset_sha256": shared_hashes,
        "worker_count": len(index_rows),
        "workers": index_rows,
    }
    atomic_write_json(destination / "index.json", index)
    return {
        **index,
        "output_directory": str(destination),
        "index_sha256": sha256_file(destination / "index.json"),
    }


__all__ = ["SCHEMA_VERSION", "build_image_inputs"]
