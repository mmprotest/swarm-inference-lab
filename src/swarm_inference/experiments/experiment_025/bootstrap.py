"""Paid-worker bootstrap: acquire exact tensors, activate, and serve."""

from __future__ import annotations

import argparse
import base64
import json
import os
import stat
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_014.remote_acquisition import (
    SNAPSHOT_SCHEMA_VERSION,
    AcquisitionError,
    acquire_worker_package,
    activate_worker_snapshot,
)
from swarm_inference.model.kimi_tokenizer import KIMI_TOKENIZER_ASSETS

from .io import atomic_write_json, read_json, sha256_file
from .worker import run_worker_server


def _secret_environment(name: str, destination: Path) -> None:
    encoded = os.environ.get(name)
    if not encoded:
        raise ValueError(f"required E025 secret environment variable is absent: {name}")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError(f"required E025 secret environment is invalid: {name}") from exc
    if not payload:
        raise ValueError(f"required E025 secret environment is empty: {name}")
    destination.write_bytes(payload)
    destination.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _reuse_verified_worker_snapshot(
    *,
    worker_id: str,
    distribution_path: Path,
    placement_path: Path,
    config_path: Path,
    runtime_root: Path,
    snapshot: Path,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Reuse an atomic prior activation without reacquiring or duplicating weights."""

    if not snapshot.exists():
        return None

    def reject(detail: str) -> None:
        raise AcquisitionError(
            "existing E025 worker snapshot failed exact reuse validation: "
            f"{detail}: {snapshot}"
        )

    if not snapshot.is_dir():
        reject("snapshot path is not a directory")
    bootstrap_path = runtime_root / "bootstrap.json"
    required = {
        "activation": snapshot / "activation.json",
        "config": snapshot / "config.json",
        "identity": snapshot / "model-identity.json",
        "index": snapshot / "model.safetensors.index.json",
        "weights": snapshot / "model.safetensors",
        "bootstrap": bootstrap_path,
    }
    missing = sorted(name for name, path in required.items() if not path.is_file())
    if missing:
        reject(f"required files are absent ({', '.join(missing)})")

    print(
        f"[e025-bootstrap:reuse] validating existing snapshot for {worker_id}",
        flush=True,
    )
    try:
        distribution = read_json(distribution_path)
        placement = read_json(placement_path)
        activation = read_json(required["activation"])
        identity = read_json(required["identity"])
        index = read_json(required["index"])
        previous = read_json(required["bootstrap"])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise AcquisitionError(
            f"existing E025 worker snapshot metadata is unreadable: {snapshot}"
        ) from exc

    if distribution.get("status") != "PASS" or placement.get("status") != "PASS":
        reject("current distribution or placement is not passing")
    worker = next(
        (
            row
            for row in placement.get("workers", [])
            if row.get("worker_id") == worker_id
        ),
        None,
    )
    if worker is None:
        reject("current placement has no matching worker")
    checkpoint = placement.get("checkpoint", {})
    expected_names = sorted(
        str(tensor["name"])
        for unit in worker.get("assignment_units", [])
        for tensor in unit.get("tensors", [])
    )
    expected_weight_bytes = sum(
        int(tensor["physical_bytes"])
        for unit in worker.get("assignment_units", [])
        for tensor in unit.get("tensors", [])
    )
    current_config_sha = sha256_file(config_path)
    current_placement_sha = sha256_file(placement_path)
    current_distribution_sha = sha256_file(distribution_path)
    snapshot_config_sha = sha256_file(required["config"])
    index_sha = sha256_file(required["index"])
    identity_sha = sha256_file(required["identity"])

    expected_activation = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "status": "PASS",
        "worker_id": worker_id,
        "checkpoint_fingerprint": checkpoint.get("checkpoint_fingerprint"),
        "checkpoint_revision": checkpoint.get("revision"),
        "placement_manifest_sha256": current_placement_sha,
        "config_sha256": current_config_sha,
        "index_sha256": index_sha,
        "model_identity_sha256": identity_sha,
        "tensor_count": len(expected_names),
        "source_weight_bytes": int(worker.get("source_weight_bytes", -1)),
        "owned_layers": worker.get("owned_layers"),
        "worker_role": worker.get("worker_role"),
        "atomic_activation": True,
    }
    for name, expected in expected_activation.items():
        if activation.get(name) != expected:
            reject(f"activation field {name!r} differs")
    if snapshot_config_sha != current_config_sha:
        reject("activated config differs from the current pinned config")

    expected_index = {
        "metadata": {
            "checkpoint_fingerprint": checkpoint.get("checkpoint_fingerprint"),
            "total_size": expected_weight_bytes,
            "worker_id": worker_id,
        },
        "weight_map": {name: "model.safetensors" for name in expected_names},
    }
    if index != expected_index:
        reject("Safetensors index differs from the current assignment")

    owns_embeddings = bool(
        "embedding" in worker.get("owned_components", [])
        or worker.get("worker_role") == "embedding_dense_stage"
    )
    tokenizer_hashes = {
        str(name): str(digest)
        for name, digest in activation.get("tokenizer_assets_sha256", {}).items()
    }
    expected_asset_names = set(KIMI_TOKENIZER_ASSETS) if owns_embeddings else set()
    if set(tokenizer_hashes) != expected_asset_names:
        reject("activated tokenizer asset set differs")
    if int(activation.get("tokenizer_asset_count", -1)) != len(tokenizer_hashes):
        reject("activated tokenizer asset count differs")
    tokenizer_bytes = 0
    for name, expected_sha in tokenizer_hashes.items():
        asset = snapshot / name
        if not asset.is_file() or sha256_file(asset) != expected_sha:
            reject(f"activated tokenizer asset {name!r} differs")
        tokenizer_bytes += asset.stat().st_size
    if int(activation.get("tokenizer_asset_bytes", -1)) != tokenizer_bytes:
        reject("activated tokenizer asset byte count differs")

    expected_identity = {
        "schema_version": "swarm-model-identity-v1",
        "model_id": "moonshotai/Kimi-K3",
        "model_revision": checkpoint.get("revision"),
        "tokenizer_revision": checkpoint.get("revision"),
        "adapter_id": "kimi_k3_cuda",
        "model_content_fingerprint": checkpoint.get("checkpoint_fingerprint"),
        "config_sha256": current_config_sha,
        "safetensors_index_sha256": index_sha,
        "worker_id": worker_id,
        "assignment_sha256": worker.get("assignment_sha256"),
        "owns_embeddings": owns_embeddings,
        "tokenizer_assets_sha256": tokenizer_hashes,
    }
    if identity != expected_identity:
        reject("model identity differs from the current assignment")

    print(
        f"[e025-bootstrap:reuse] hashing activated weights for {worker_id}",
        flush=True,
    )
    package_sha = sha256_file(required["weights"])
    if activation.get("package_sha256") != package_sha:
        reject("activated weight hash differs")
    if any(snapshot.parent.glob(f"{snapshot.name}.partial-*")):
        reject("partial activation directory remains")

    previous_activation = previous.get("activation", {})
    previous_acquisition = previous.get("acquisition", {})
    if (
        previous.get("schema_version") != "experiment-025-worker-bootstrap-v1"
        or previous.get("status") != "PASS"
        or previous.get("worker_id") != worker_id
        or not isinstance(previous_activation, dict)
        or not isinstance(previous_acquisition, dict)
    ):
        reject("prior bootstrap receipt is invalid")
    for name, value in activation.items():
        if previous_activation.get(name) != value:
            reject(f"prior activation receipt field {name!r} differs")
    acquisition_package = previous_acquisition.get("package", {})
    if (
        previous_acquisition.get("status") != "PASS"
        or previous_acquisition.get("worker_id") != worker_id
        or previous_acquisition.get("distribution_manifest_sha256")
        != current_distribution_sha
        or not isinstance(acquisition_package, dict)
        or acquisition_package.get("package_sha256") != package_sha
    ):
        reject("prior acquisition receipt differs from the activated snapshot")

    reused_acquisition = dict(previous_acquisition)
    reused_acquisition["bootstrap_attempt_mode"] = (
        "REUSED_VERIFIED_EXISTING_SNAPSHOT"
    )
    reused_acquisition["bootstrap_attempt_downloaded_bytes"] = 0
    reused_activation = {
        **activation,
        "snapshot_path": str(snapshot),
        "activation_receipt_sha256": sha256_file(required["activation"]),
        "model_identity_path": str(required["identity"]),
        "partial_directory_absent": True,
        "activation_mode": "REUSED_VERIFIED_EXISTING_SNAPSHOT",
        "reused_verified_existing_snapshot": True,
    }
    print(
        f"[e025-bootstrap:reuse] verified existing snapshot for {worker_id}",
        flush=True,
    )
    return reused_acquisition, reused_activation


def prepare_worker(
    *,
    worker_id: str,
    manifest_root: Path,
    state_root: Path,
    cuda_library: Path,
) -> dict[str, Any]:
    worker_bundle = manifest_root.resolve() / "workers" / worker_id
    shared = manifest_root.resolve() / "shared"
    template = read_json(worker_bundle / "worker-template.json")
    if template.get("worker_id") != worker_id:
        raise ValueError("E025 worker template identity differs")
    runtime_root = state_root.resolve() / worker_id
    runtime_root.mkdir(parents=True, exist_ok=True)
    # Grouped GPU workers acquire concurrently.  A worker-scoped cache keeps
    # resumable .partial objects process-exclusive and matches the frozen
    # per-worker transfer/disk accounting.
    cache = runtime_root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    package = runtime_root / "model.safetensors"
    snapshot = runtime_root / "snapshot"
    distribution_path = worker_bundle / "distribution.json"
    placement_path = worker_bundle / "placement.json"
    shared_config_path = shared / "config.json"
    reused = _reuse_verified_worker_snapshot(
        worker_id=worker_id,
        distribution_path=distribution_path,
        placement_path=placement_path,
        config_path=shared_config_path,
        runtime_root=runtime_root,
        snapshot=snapshot,
    )
    if reused is None:
        acquisition = acquire_worker_package(
            distribution_path,
            worker_id,
            cache,
            package,
            retries=3,
            timeout_seconds=60.0,
        )
        activation = activate_worker_snapshot(
            placement_path,
            worker_id,
            package,
            shared_config_path,
            snapshot,
        )
    else:
        acquisition, activation = reused
    credential = runtime_root / "credential.bin"
    certificate = runtime_root / "server.crt"
    private_key = runtime_root / "server.key"
    _secret_environment("E025_RUN_CREDENTIAL_B64", credential)
    _secret_environment("E025_TLS_CERT_B64", certificate)
    _secret_environment("E025_TLS_KEY_B64", private_key)
    expert_endpoints: list[dict[str, Any]] = []
    if template["role"] == "SUB_LAYER_PARENT":
        encoded_endpoints = os.environ.get("E025_EXPERT_ENDPOINTS_B64")
        if not encoded_endpoints:
            raise ValueError("E025 layer-89 parent has no physical expert endpoints")
        value = json.loads(base64.b64decode(encoded_endpoints, validate=True))
        if not isinstance(value, list) or len(value) != 4:
            raise ValueError("E025 layer-89 physical endpoint list is invalid")
        expert_endpoints = value
    config = {
        "schema_version": "experiment-025-worker-config-v1",
        "worker_id": worker_id,
        "role": template["role"],
        "worker_index": template["worker_index"],
        "worker_count": template["worker_count"],
        "worker": template,
        "snapshot_path": str(snapshot),
        "cuda_library": str(cuda_library.resolve()),
        "telemetry_path": str(runtime_root / "telemetry.jsonl"),
        "ready_path": str(runtime_root / "ready.json"),
        "assignment_sha256": template["assignment_sha256"],
        "checkpoint_fingerprint": template["checkpoint_fingerprint"],
        "topology_id": template["topology_id"],
        "maximum_context": int(os.environ.get("E025_MAXIMUM_CONTEXT", "64")),
        "expert_endpoints": expert_endpoints,
        "bootstrap_receipt_path": str(runtime_root / "bootstrap.json"),
    }
    config_path = runtime_root / "worker-config.json"
    atomic_write_json(config_path, config)
    receipt = {
        "schema_version": "experiment-025-worker-bootstrap-v1",
        "status": "PASS",
        "worker_id": worker_id,
        "acquisition": acquisition,
        "activation": activation,
        "worker_config_path": str(config_path),
        "worker_config_sha256": sha256_file(config_path),
        "cuda_library": str(cuda_library.resolve()),
        "cuda_library_sha256": sha256_file(cuda_library.resolve()),
        "secrets_logged": False,
    }
    atomic_write_json(runtime_root / "bootstrap.json", receipt)
    return {
        **receipt,
        "credential_path": credential,
        "certificate_path": certificate,
        "private_key_path": private_key,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", default=os.environ.get("E025_WORKER_ID"))
    parser.add_argument("--manifest-root", type=Path, default=Path("/opt/swarm/e025"))
    parser.add_argument("--state-root", type=Path, default=Path("/var/lib/swarm/e025"))
    parser.add_argument(
        "--cuda-library",
        type=Path,
        default=Path("/opt/swarm/native/libcoli_cuda-consumer.so"),
    )
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=42525)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    if not arguments.worker_id:
        raise ValueError("E025_WORKER_ID is required")
    prepared = prepare_worker(
        worker_id=str(arguments.worker_id),
        manifest_root=arguments.manifest_root,
        state_root=arguments.state_root,
        cuda_library=arguments.cuda_library,
    )
    config = Path(str(prepared["worker_config_path"]))
    run_worker_server(
        config,
        Path(prepared["credential_path"]),
        Path(prepared["certificate_path"]),
        Path(prepared["private_key_path"]),
        bind=arguments.bind,
        port=arguments.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "prepare_worker"]
