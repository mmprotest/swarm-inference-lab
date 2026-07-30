"""Model-manifest persistence and integrity checks."""

from __future__ import annotations

from pathlib import Path

from swarm_inference.config.models import ModelManifest
from swarm_inference.exceptions import IntegrityError
from swarm_inference.protocol.checksums import sha256_file


def save_manifest(manifest: ModelManifest, path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def load_manifest(path: str | Path) -> ModelManifest:
    resolved = Path(path).expanduser().resolve()
    try:
        return ModelManifest.model_validate_json(resolved.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise IntegrityError(f"invalid model manifest {resolved}: {exc}") from exc


def verify_manifest_shards(
    manifest: ModelManifest,
    model_root: str | Path,
) -> None:
    root = Path(model_root).expanduser().resolve()
    for relative, expected in manifest.shard_hashes.items():
        path = root / relative
        if path.is_file():
            actual = sha256_file(path)
        elif path.is_dir():
            actual = hash_shard_directory(path)
        else:
            raise IntegrityError(f"manifest shard is missing: {path}")
        if actual != expected:
            raise IntegrityError(
                f"shard hash mismatch for {relative}: expected={expected} actual={actual}"
            )


def hash_shard_directory(path: str | Path) -> str:
    import hashlib

    root = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    files = sorted(file for file in root.rglob("*") if file.is_file())
    for file in files:
        relative = file.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        file_hash = sha256_file(file).encode("ascii")
        digest.update(file_hash)
    return digest.hexdigest()


def manifest_summary(manifest: ModelManifest) -> dict[str, object]:
    return {
        "model_id": manifest.model_id,
        "model_revision": manifest.model_revision,
        "architecture": manifest.architecture,
        "layer_count": manifest.layer_count,
        "stage_count": len(manifest.stages),
        "total_weight_bytes": manifest.total_weight_bytes,
        "largest_stage_bytes": max(stage.required_memory_bytes for stage in manifest.stages),
        "compatible_worker_backends": [
            backend.value for backend in manifest.compatible_worker_backends
        ],
        "shards": manifest.shard_hashes,
        "shared_tensors": manifest.shared_tensors,
    }
