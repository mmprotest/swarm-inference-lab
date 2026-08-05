from __future__ import annotations

import json
from pathlib import Path

import pytest
from artifact_test_support import MODEL_ID, MODEL_REVISION, tiny_olmoe_snapshot

from swarm_inference.cluster.artifacts import (
    StageArtifactBuilder,
    verify_artifact_directory,
)


def test_stage_artifact_builder_excludes_unowned_layers_and_publishes_atomically(
    tmp_path: Path,
) -> None:
    snapshot, tokenizer_revision, plan = tiny_olmoe_snapshot(tmp_path)
    artifacts = tmp_path / "state" / "artifacts"
    builder = StageArtifactBuilder(
        artifact_root=artifacts,
        temporary_root=tmp_path / "state" / "downloads",
        clock_ns=lambda: 100,
    )
    assignment = plan.assignments[1]

    manifest = builder.build(
        snapshot,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_revision=tokenizer_revision,
        assignment=assignment,
        stage_count=plan.stage_count,
        dtype="float32",
    )

    directory = artifacts / manifest.content_hash
    assert verify_artifact_directory(directory) == manifest
    index = json.loads((directory / "model.safetensors.index.json").read_text("utf-8"))
    tensor_names = sorted(index["weight_map"])
    assert tensor_names
    assert all(
        not name.startswith("model.layers.")
        or assignment.layer_start <= int(name.split(".")[2]) < assignment.layer_end
        for name in tensor_names
    )
    assert not any(name.startswith("model.embed_tokens.") for name in tensor_names)
    assert not any(name.startswith("model.norm.") for name in tensor_names)
    assert not any(name.startswith("lm_head.") for name in tensor_names)
    assert not (directory / "tokenizer.json").exists()
    assert not list((tmp_path / "state" / "downloads").glob("*.partial"))


def test_first_and_final_artifacts_receive_their_required_shared_assets(tmp_path: Path) -> None:
    snapshot, tokenizer_revision, plan = tiny_olmoe_snapshot(tmp_path)
    builder = StageArtifactBuilder(
        artifact_root=tmp_path / "artifacts",
        temporary_root=tmp_path / "downloads",
    )
    manifests = [
        builder.build(
            snapshot,
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            tokenizer_revision=tokenizer_revision,
            assignment=assignment,
            stage_count=plan.stage_count,
            dtype="float32",
        )
        for assignment in (plan.assignments[0], plan.assignments[-1])
    ]

    first = tmp_path / "artifacts" / manifests[0].artifact_id
    final = tmp_path / "artifacts" / manifests[1].artifact_id
    assert (first / "tokenizer.json").is_file()
    assert (final / "tokenizer.json").is_file()
    first_index = json.loads((first / "model.safetensors.index.json").read_text("utf-8"))
    final_index = json.loads((final / "model.safetensors.index.json").read_text("utf-8"))
    assert "model.embed_tokens.weight" in first_index["weight_map"]
    assert "model.norm.weight" in final_index["weight_map"]
    assert "lm_head.weight" in final_index["weight_map"]


def test_stage_artifact_is_not_published_when_storage_reservation_fails(
    tmp_path: Path,
) -> None:
    snapshot, tokenizer_revision, plan = tiny_olmoe_snapshot(tmp_path)
    artifacts = tmp_path / "artifacts"
    downloads = tmp_path / "downloads"
    builder = StageArtifactBuilder(
        artifact_root=artifacts,
        temporary_root=downloads,
    )

    def reject_storage(_manifest: object) -> None:
        raise OSError("artifact storage budget is exhausted")

    with pytest.raises(OSError, match="storage budget"):
        builder.build(
            snapshot,
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            tokenizer_revision=tokenizer_revision,
            assignment=plan.assignments[0],
            stage_count=plan.stage_count,
            dtype="float32",
            before_publish=reject_storage,
        )

    assert not list(artifacts.iterdir())
    assert not list(downloads.glob("*.partial"))
