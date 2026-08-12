from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from swarm_inference.worker.service import _verify_configured_model_identity


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_configured_worker_snapshot_identity_is_verified_before_registration(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    config = snapshot / "config.json"
    index = snapshot / "model.safetensors.index.json"
    config.write_text("{}\n", encoding="utf-8")
    index.write_text('{"weight_map": {}}\n', encoding="utf-8")
    identity = {
        "schema_version": "swarm-model-identity-v1",
        "model_id": "moonshotai/Kimi-K3",
        "model_revision": "revision",
        "tokenizer_revision": "revision",
        "adapter_id": "kimi_k3_cuda",
        "model_content_fingerprint": "a" * 64,
        "config_sha256": _digest(config),
        "safetensors_index_sha256": _digest(index),
        "worker_id": "k3-worker-017",
        "assignment_sha256": "b" * 64,
        "owns_embeddings": False,
        "tokenizer_assets_sha256": {},
    }
    identity_path = snapshot / "model-identity.json"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")

    verified = _verify_configured_model_identity(
        identity_path,
        snapshot,
        worker_id="k3-worker-017",
    )
    assert verified["model_content_fingerprint"] == "a" * 64

    with pytest.raises(ValueError, match="different worker"):
        _verify_configured_model_identity(
            identity_path,
            snapshot,
            worker_id="k3-worker-018",
        )

    config.write_text('{"tampered": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="mismatches the snapshot"):
        _verify_configured_model_identity(
            identity_path,
            snapshot,
            worker_id="k3-worker-017",
        )
