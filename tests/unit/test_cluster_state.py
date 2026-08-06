from __future__ import annotations

import json
from pathlib import Path

import pytest

from swarm_inference.cluster.models import (
    ClusterMetadata,
    NodeMetadata,
    VersionCompatibility,
    node_id_from_fingerprint,
)
from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.exceptions import IntegrityError
from swarm_inference.platforms import default_state_directory
from swarm_inference.security.identity import CoordinatorIdentity, WorkerIdentity


def _cluster(identity: CoordinatorIdentity) -> ClusterMetadata:
    return ClusterMetadata(
        cluster_id="cluster-state",
        name="state-test",
        coordinator_id=node_id_from_fingerprint(identity.public_key_fingerprint),
        coordinator_endpoint="10.0.0.1:50051",
        coordinator_public_key=identity.public_key_b64,
        coordinator_fingerprint=identity.public_key_fingerprint,
        created_at_unix_ns=1,
        runtime_compatibility=VersionCompatibility(
            minimum_runtime_version="0.1.0",
            maximum_runtime_version_exclusive="0.2.0",
        ),
    )


def _node(identity: WorkerIdentity) -> NodeMetadata:
    node_id = node_id_from_fingerprint(identity.public_key_fingerprint)
    return NodeMetadata(
        node_id=node_id,
        public_key=identity.public_key_b64,
        fingerprint=identity.public_key_fingerprint,
        hostname="node",
        operating_system="test",
        architecture="test",
        agent_version="0.1.0",
        runtime_version="0.1.0",
        build_id="test",
        package_lock_hash="2" * 64,
        worker_ids=[f"{node_id}/cpu-0"],
        joined_at_unix_ns=1,
        last_seen_at_unix_ns=1,
    )


def test_platform_default_state_paths() -> None:
    home = Path("/home/test")
    assert default_state_directory(
        system="win32",
        environment={"LOCALAPPDATA": "C:/Users/test/AppData/Local"},
        home_directory=home,
    ) == Path("C:/Users/test/AppData/Local/SwarmInference")
    assert default_state_directory(system="linux", environment={}, home_directory=home) == Path(
        "/home/test/.local/state/swarm-inference"
    )
    assert default_state_directory(system="darwin", environment={}, home_directory=home) == Path(
        "/home/test/Library/Application Support/SwarmInference"
    )


def test_cluster_state_is_strict_atomic_and_preserves_identity(tmp_path: Path) -> None:
    state = ClusterStateStore(tmp_path / "state")
    coordinator = CoordinatorIdentity.generate()
    state.save_cluster(_cluster(coordinator))
    assert state.load_cluster() == _cluster(coordinator)

    worker = WorkerIdentity.generate()
    state.save_node(_node(worker))
    assert state.node(node_id_from_fingerprint(worker.public_key_fingerprint)) == _node(worker)

    other = CoordinatorIdentity.generate()
    with pytest.raises(IntegrityError, match="different cluster identity"):
        state.save_cluster(_cluster(other).model_copy(update={"cluster_id": "other"}))


def test_explicit_schema_zero_migration_rewrites_document(tmp_path: Path) -> None:
    state = ClusterStateStore(tmp_path / "state")
    state.paths.nodes.write_text(
        json.dumps({"schema_version": 0, "nodes": []}),
        encoding="utf-8",
    )
    document = state.load_nodes()
    assert document.schema_version == 2
    rewritten = json.loads(state.paths.nodes.read_text(encoding="utf-8"))
    assert rewritten["schema_version"] == 2
    assert rewritten["document_version"] == 2


def test_unknown_or_future_state_fails_closed(tmp_path: Path) -> None:
    state = ClusterStateStore(tmp_path / "state")
    state.paths.nodes.write_text(
        json.dumps({"schema_version": 3, "document_version": 3, "nodes": []}),
        encoding="utf-8",
    )
    with pytest.raises(IntegrityError, match="future state schema"):
        state.load_nodes()


def test_valid_legacy_coordinator_identity_is_adopted_exactly(tmp_path: Path) -> None:
    legacy_state = tmp_path / ".swarm" / "coordinator"
    legacy_identity = CoordinatorIdentity.load_or_create(legacy_state / "coordinator-identity.json")
    state = ClusterStateStore(tmp_path / "new-state")
    adopted = state.adopt_legacy_coordinator_identity(legacy_state)
    assert adopted == state.paths.coordinator_identity
    assert CoordinatorIdentity.load(adopted).public_key_fingerprint == (
        legacy_identity.public_key_fingerprint
    )
