from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from transformers import OlmoeConfig, OlmoeForCausalLM

from swarm_inference.cluster.artifacts import ArtifactManager, StageArtifactBuilder
from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.model.olmoe import inspect_olmoe_partition_metadata
from swarm_inference.model.partition import build_stage_plan
from swarm_inference.protocol.stage_worker import LoadStageRequest
from swarm_inference.worker.stage_runtime import PersistentStageRuntime

_MODEL_ID = "test/tiny-cluster-olmoe"
_MODEL_REVISION = "b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e"


def _snapshot(tmp_path: Path) -> tuple[Path, str]:
    torch.manual_seed(6006)
    model = OlmoeForCausalLM(
        OlmoeConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
            num_experts_per_tok=2,
            num_experts=4,
            pad_token_id=0,
            eos_token_id=31,
        )
    ).eval()
    snapshot = tmp_path / "snapshot"
    model.save_pretrained(snapshot, safe_serialization=True, max_shard_size="2KB")
    config_path = snapshot / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_commit_hash"] = _MODEL_REVISION
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    tokenizer = b'{"version":"1.0","cluster-transfer":true}'
    (snapshot / "tokenizer.json").write_bytes(tokenizer)
    (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return snapshot, "sha256:" + hashlib.sha256(tokenizer).hexdigest()


@pytest.mark.asyncio
async def test_transferred_stage_artifact_loads_through_canonical_runtime(
    tmp_path: Path,
) -> None:
    snapshot, tokenizer_revision = _snapshot(tmp_path)
    metadata = inspect_olmoe_partition_metadata(
        snapshot,
        model_revision=_MODEL_REVISION,
        tokenizer_revision=tokenizer_revision,
    )
    plan = build_stage_plan(
        snapshot,
        metadata=metadata,
        stage_count=2,
        method="equal",
        memory_limit_bytes=1024**3,
        device="cpu",
    )
    source_state = ClusterStateStore(tmp_path / "source-state")
    manifest = StageArtifactBuilder(
        artifact_root=source_state.paths.artifacts,
        temporary_root=source_state.paths.downloads,
    ).build(
        snapshot,
        model_id=_MODEL_ID,
        model_revision=_MODEL_REVISION,
        tokenizer_revision=tokenizer_revision,
        assignment=plan.assignments[0],
        stage_count=2,
        dtype="float32",
    )
    destination_state = ClusterStateStore(tmp_path / "destination-state")
    manager = ArtifactManager(
        state=destination_state,
        node_id="node-destination",
        storage_limit_bytes=1024**3,
        chunk_size_bytes=1024,
    )
    transfer = manager.transfer_from_directory(
        source_state.paths.artifacts / manifest.artifact_id,
        peer_authenticated=True,
    )
    assert transfer.state == "complete"

    runtime = PersistentStageRuntime(
        worker_id="node-destination/cpu-0",
        device="cpu",
        dtype="float32",
        memory_limit_bytes=1024**3,
        maximum_sessions=4,
        model_cache_dir=destination_state.paths.artifacts,
        artifact_resolver=manager.resolve,
        artifact_lease_acquirer=lambda artifact_id, owner: (
            manager.lease(
                artifact_id,
                owner=owner,
                purpose="loaded-stage",
            ).lease_id
        ),
        artifact_lease_releaser=manager.release,
    )
    try:
        response = await runtime.load_stage(
            LoadStageRequest(
                worker_id="node-destination/cpu-0",
                request_id="load-artifact",
                model_id=_MODEL_ID,
                model_revision=_MODEL_REVISION,
                tokenizer_revision=tokenizer_revision,
                topology_id=plan.topology_id,
                stage_count=2,
                assignment=plan.assignments[0],
                device="cpu",
                dtype="float32",
                artifact_id=manifest.artifact_id,
            )
        )
        assert response.accepted
        assert runtime.loaded_executor is not None
        assert manager.entries()[0].active_lease_ids
    finally:
        await runtime.close()

    assert manager.entries()[0].active_lease_ids == []
