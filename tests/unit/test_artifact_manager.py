from __future__ import annotations

from pathlib import Path

import pytest
from artifact_test_support import MODEL_ID, MODEL_REVISION, tiny_olmoe_snapshot

from swarm_inference.cluster.artifacts import (
    ArtifactManager,
    ArtifactOperationCoordinator,
    StageArtifactBuilder,
    artifact_chunks,
    verify_artifact_directory,
)
from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.exceptions import IntegrityError
from swarm_inference.protocol.cluster import (
    ArtifactOperationRequest,
    ClusterRequestAuthentication,
)


def _authentication() -> ClusterRequestAuthentication:
    return ClusterRequestAuthentication(
        node_id="node-source",
        timestamp_unix_ns=1,
        nonce="test",
        signature="verified-by-rpc-layer",
    )


def _built_artifact(tmp_path: Path) -> tuple[Path, int]:
    snapshot, tokenizer_revision, plan = tiny_olmoe_snapshot(tmp_path, stage_count=2)
    builder = StageArtifactBuilder(
        artifact_root=tmp_path / "source-artifacts",
        temporary_root=tmp_path / "source-downloads",
    )
    manifest = builder.build(
        snapshot,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_revision=tokenizer_revision,
        assignment=plan.assignments[0],
        stage_count=plan.stage_count,
        dtype="float32",
    )
    return tmp_path / "source-artifacts" / manifest.artifact_id, manifest.total_size_bytes


def test_artifact_transfer_resumes_and_rejects_bad_chunks(tmp_path: Path) -> None:
    source, size = _built_artifact(tmp_path)
    state = ClusterStateStore(tmp_path / "destination")
    manager = ArtifactManager(
        state=state,
        node_id="node-destination",
        storage_limit_bytes=size * 3,
        chunk_size_bytes=512,
    )

    partial = manager.transfer_from_directory(
        source, peer_authenticated=True, maximum_chunks_this_call=1
    )
    assert partial.state == "transferring"
    next_chunk, _ = next(
        (chunk, payload)
        for chunk, payload in artifact_chunks(source, chunk_size_bytes=512)
        if chunk.chunk_index == 1
    )
    with pytest.raises(IntegrityError, match="chunk hash"):
        manager.write_chunk(
            transfer_id=partial.transfer_id,
            chunk=next_chunk,
            payload=b"x" * next_chunk.size_bytes,
        )

    complete = manager.transfer_from_directory(source, peer_authenticated=True)
    assert complete.state == "complete"
    destination = manager.resolve(complete.artifact_id)
    assert verify_artifact_directory(destination).artifact_id == complete.artifact_id


def test_artifact_transfer_requires_authenticated_membership(tmp_path: Path) -> None:
    source, size = _built_artifact(tmp_path)
    manager = ArtifactManager(
        state=ClusterStateStore(tmp_path / "destination"),
        node_id="node-destination",
        storage_limit_bytes=size * 2,
    )
    with pytest.raises(PermissionError, match="authenticated"):
        manager.transfer_from_directory(source, peer_authenticated=False)


def test_final_hash_corruption_is_rejected(tmp_path: Path) -> None:
    source, _ = _built_artifact(tmp_path)
    manifest = verify_artifact_directory(source)
    victim = source / manifest.files[0].relative_path
    victim.write_bytes(victim.read_bytes() + b"corrupt")
    with pytest.raises(IntegrityError, match=r"size mismatch|hash mismatch"):
        verify_artifact_directory(source)


def test_active_artifact_cannot_be_evicted(tmp_path: Path) -> None:
    source, size = _built_artifact(tmp_path)
    state = ClusterStateStore(tmp_path / "destination")
    manager = ArtifactManager(
        state=state,
        node_id="node-destination",
        storage_limit_bytes=size + 1,
    )
    status = manager.transfer_from_directory(source, peer_authenticated=True)
    lease = manager.lease(status.artifact_id, owner="topology-1", purpose="loaded-stage")

    with pytest.raises(OSError, match="active or pinned"):
        manager.evict_to_fit(2)
    assert manager.resolve(status.artifact_id).is_dir()
    assert manager.release(lease.lease_id)
    assert manager.evict_to_fit(2) == [status.artifact_id]


@pytest.mark.asyncio
async def test_coordinator_artifact_directory_locates_and_leases_exact_content(
    tmp_path: Path,
) -> None:
    source, size = _built_artifact(tmp_path)
    manager = ArtifactManager(
        state=ClusterStateStore(tmp_path / "destination"),
        node_id="node-destination",
        storage_limit_bytes=size * 2,
    )
    transfer = manager.transfer_from_directory(source, peer_authenticated=True)
    coordinator = ArtifactOperationCoordinator(manager)

    located = await coordinator.handle(
        ArtifactOperationRequest(
            authentication=_authentication(),
            operation="locate",
            artifact_id=transfer.artifact_id,
        )
    )
    assert located.accepted
    assert located.locations == ["node-destination"]

    leased = await coordinator.handle(
        ArtifactOperationRequest(
            authentication=_authentication(),
            operation="lease",
            artifact_id=transfer.artifact_id,
            source_node_id="node-source",
            lease_purpose="deployment",
        )
    )
    assert leased.lease is not None
    released = await coordinator.handle(
        ArtifactOperationRequest(
            authentication=_authentication(),
            operation="release",
            artifact_id=transfer.artifact_id,
            lease_id=leased.lease.lease_id,
        )
    )
    assert released.accepted
