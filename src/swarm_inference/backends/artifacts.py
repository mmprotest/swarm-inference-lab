"""Canonical model-to-backend artifact identity and validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from swarm_inference.exceptions import IntegrityError
from swarm_inference.protocol.checksums import sha256_file
from swarm_inference.worker.abi import BackendArtifactMapping


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact_hash(path: Path) -> str:
    """Hash one file or a directory tree without following unrelated paths."""

    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return sha256_file(resolved)
    if not resolved.is_dir():
        raise IntegrityError(f"backend artifact does not exist: {resolved}")
    digest = hashlib.sha256()
    files = sorted(item for item in resolved.rglob("*") if item.is_file())
    if not files:
        raise IntegrityError(f"backend artifact directory is empty: {resolved}")
    for item in files:
        relative = item.relative_to(resolved).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def tokenizer_identity(model_path: Path) -> dict[str, str | int | list[int]]:
    """Derive tokenizer, vocabulary, and special-token identities from HF files."""

    root = model_path.expanduser().resolve()
    tokenizer_path = root / "tokenizer.json"
    config_path = root / "tokenizer_config.json"
    if not tokenizer_path.is_file() or not config_path.is_file():
        raise IntegrityError(f"tokenizer identity files are missing under {root}")
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = tokenizer.get("model", {})
    vocabulary = model.get("vocab")
    if vocabulary is None:
        raise IntegrityError("tokenizer.json does not contain model.vocab")
    if isinstance(vocabulary, dict):
        vocabulary_rows = sorted((str(token), int(index)) for token, index in vocabulary.items())
        vocabulary_size = len(vocabulary_rows)
    elif isinstance(vocabulary, list):
        vocabulary_rows = vocabulary
        vocabulary_size = len(vocabulary)
    else:
        raise IntegrityError("unsupported tokenizer vocabulary representation")
    added_tokens = sorted(
        (
            str(item.get("content", "")),
            int(item.get("id", -1)),
            bool(item.get("special", False)),
        )
        for item in tokenizer.get("added_tokens", [])
    )
    special_ids = sorted(
        {
            int(value)
            for key, value in config.items()
            if key.endswith("_token_id") and isinstance(value, int)
        }
    )
    return {
        "tokenizer_hash": artifact_hash(tokenizer_path),
        "vocabulary_hash": canonical_json_hash(vocabulary_rows),
        "vocabulary_size": vocabulary_size,
        "special_tokens_hash": canonical_json_hash(
            {"added_tokens": added_tokens, "special_ids": special_ids}
        ),
        "special_token_ids": special_ids,
    }


def compare_tokenizers(target_path: Path, draft_path: Path) -> dict[str, Any]:
    target = tokenizer_identity(target_path)
    draft = tokenizer_identity(draft_path)
    checks = {
        "tokenizer_identity": target["tokenizer_hash"] == draft["tokenizer_hash"],
        "vocabulary_identity": target["vocabulary_hash"] == draft["vocabulary_hash"],
        "vocabulary_size_identity": target["vocabulary_size"] == draft["vocabulary_size"],
        "special_token_identity": target["special_tokens_hash"] == draft["special_tokens_hash"],
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "target": target,
        "draft": draft,
        "token_id_comparison_allowed": all(checks.values()),
    }


def validate_mapping(mapping: BackendArtifactMapping) -> dict[str, Any]:
    path = Path(mapping.backend_artifact_path).expanduser().resolve()
    actual = artifact_hash(path)
    valid = actual == mapping.backend_artifact_hash
    return {
        "status": "PASS" if valid else "FAIL",
        "backend_id": mapping.backend_id,
        "path": str(path),
        "expected_hash": mapping.backend_artifact_hash,
        "actual_hash": actual,
        "canonical_revision": mapping.canonical_revision,
        "canonical_partition_hash": mapping.canonical_partition_hash,
    }


def gguf_mapping_from_sidecar(
    sidecar_path: Path,
    *,
    canonical_partition_hash: str,
) -> BackendArtifactMapping:
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    required = {
        "source_model_id",
        "source_revision",
        "gguf_path",
        "conversion_command",
        "conversion_version",
        "quantisation",
        "gguf_sha256",
        "tokenizer_hash",
        "vocabulary_hash",
        "special_tokens_hash",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise IntegrityError(f"GGUF conversion sidecar is missing: {missing}")
    gguf_path = Path(str(payload["gguf_path"])).expanduser().resolve()
    if artifact_hash(gguf_path) != payload["gguf_sha256"]:
        raise IntegrityError("GGUF hash does not match its conversion sidecar")
    return BackendArtifactMapping(
        canonical_model_id=str(payload["source_model_id"]),
        canonical_revision=str(payload["source_revision"]),
        canonical_partition_hash=canonical_partition_hash,
        backend_id="llamacpp",
        backend_artifact_path=str(gguf_path),
        backend_artifact_hash=str(payload["gguf_sha256"]),
        conversion_tool="llama.cpp/convert_hf_to_gguf.py",
        conversion_version=str(payload["conversion_version"]),
        conversion_parameters={
            "command": str(payload["conversion_command"]),
            "quantisation": str(payload["quantisation"]),
        },
        canonical_tensor_mapping=dict(payload.get("canonical_tensor_mapping", {})),
        weight_format=f"GGUF/{payload['quantisation']}",
        conversion_loss=("lossless" if payload["quantisation"] == "F16" else "quantised"),
        tokenizer_hash=str(payload["tokenizer_hash"]),
        vocabulary_hash=str(payload["vocabulary_hash"]),
        special_tokens_hash=str(payload["special_tokens_hash"]),
    )
