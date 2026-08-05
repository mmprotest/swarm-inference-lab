from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from swarm_inference.cluster.models import (
    ClusterMetadata,
    NodeMetadata,
    VersionCompatibility,
    node_id_from_fingerprint,
)
from swarm_inference.config.models import Backend
from swarm_inference.security.identity import CoordinatorIdentity, WorkerIdentity


def _compatibility() -> VersionCompatibility:
    return VersionCompatibility(
        minimum_runtime_version="0.1.0",
        maximum_runtime_version_exclusive="0.2.0",
    )


def test_cluster_metadata_binds_coordinator_identity() -> None:
    identity = CoordinatorIdentity.generate()
    metadata = ClusterMetadata(
        cluster_id="cluster-test",
        name="test cluster",
        coordinator_id=node_id_from_fingerprint(identity.public_key_fingerprint),
        coordinator_endpoint="192.168.1.10:50051",
        coordinator_public_key=identity.public_key_b64,
        coordinator_fingerprint=identity.public_key_fingerprint,
        created_at_unix_ns=1,
        runtime_compatibility=_compatibility(),
    )
    assert metadata.security_classification == (
        "trusted-lan-private-network-unencrypted-data-plane"
    )

    with pytest.raises(ValidationError, match="public key and fingerprint"):
        metadata.model_copy(update={"coordinator_fingerprint": "0" * 64}).__class__.model_validate(
            {
                **metadata.model_dump(),
                "coordinator_fingerprint": "0" * 64,
            }
        )


def test_node_identity_and_worker_namespace_are_strict(tmp_path: Path) -> None:
    identity = WorkerIdentity.load_or_create(tmp_path / "node.json")
    node_id = node_id_from_fingerprint(identity.public_key_fingerprint)
    node = NodeMetadata(
        node_id=node_id,
        public_key=identity.public_key_b64,
        fingerprint=identity.public_key_fingerprint,
        hostname="node",
        operating_system="Windows 11",
        architecture="AMD64",
        agent_version="0.1.0",
        runtime_version="0.1.0",
        build_id="build-test",
        package_lock_hash="1" * 64,
        selected_backend=Backend.TORCH_CPU,
        selected_device="cpu",
        worker_ids=[f"{node_id}/cpu-0"],
        joined_at_unix_ns=1,
        last_seen_at_unix_ns=1,
    )
    assert node.worker_ids == [f"{node_id}/cpu-0"]

    with pytest.raises(ValidationError, match="namespaced"):
        NodeMetadata.model_validate({**node.model_dump(), "worker_ids": ["some-other-node/cpu-0"]})


def test_version_compatibility_rejects_inverted_range() -> None:
    with pytest.raises(ValidationError, match="range is inverted"):
        VersionCompatibility(
            minimum_runtime_version="0.1.0",
            maximum_runtime_version_exclusive="0.2.0",
            minimum_product_protocol_minor=2,
            maximum_product_protocol_minor=1,
        )
