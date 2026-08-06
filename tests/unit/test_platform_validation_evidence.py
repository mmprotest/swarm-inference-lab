from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from swarm_inference.cli import app
from swarm_inference.cluster.models import (
    BackendValidationRecord,
    ClusterMetadata,
    NodeMembership,
    NodeMetadata,
    VersionCompatibility,
    aggregate_validation_status,
    node_id_from_fingerprint,
)
from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.config.models import Backend
from swarm_inference.platforms.base import PLATFORM_IMPLEMENTATION_CONTRACT
from swarm_inference.platforms.linux import LinuxPlatformAdapter
from swarm_inference.platforms.windows import WindowsPlatformAdapter
from swarm_inference.security.identity import CoordinatorIdentity, WorkerIdentity
from swarm_inference.security.trust_store import WorkerTrustStore


def _state_with_node(tmp_path: Path) -> tuple[ClusterStateStore, ClusterMetadata, NodeMetadata]:
    state = ClusterStateStore(tmp_path / "state")
    worker = WorkerIdentity.load_or_create(state.paths.node_identity)
    coordinator = CoordinatorIdentity.generate()
    node_id = node_id_from_fingerprint(worker.public_key_fingerprint)
    cluster = ClusterMetadata(
        cluster_id="cluster-validation",
        name="validation",
        coordinator_id=node_id,
        coordinator_endpoint="192.168.1.10:50051",
        coordinator_public_key=coordinator.public_key_b64,
        coordinator_fingerprint=coordinator.public_key_fingerprint,
        created_at_unix_ns=1,
        runtime_compatibility=VersionCompatibility(
            minimum_runtime_version="0.1.0",
            maximum_runtime_version_exclusive="1.0.0",
        ),
    )
    node = NodeMetadata(
        node_id=node_id,
        public_key=worker.public_key_b64,
        fingerprint=worker.public_key_fingerprint,
        hostname="node",
        operating_system="windows 11",
        architecture="AMD64",
        agent_version="0.1.0",
        runtime_version="0.1.0",
        build_id="test",
        package_lock_hash="a" * 64,
        selected_backend=Backend.TORCH_CPU,
        selected_device="cpu",
        joined_at_unix_ns=1,
        last_seen_at_unix_ns=1,
        implementation_status="implemented",
        implementation_reason="Windows x86-64 path is implemented",
    )
    membership = NodeMembership(
        cluster_id=cluster.cluster_id,
        node_id=node_id,
        node_public_key=worker.public_key_b64,
        node_fingerprint=worker.public_key_fingerprint,
        coordinator_public_key=coordinator.public_key_b64,
        coordinator_fingerprint=coordinator.public_key_fingerprint,
        joined_at_unix_ns=1,
    )
    state.save_cluster(cluster)
    state.save_node(node)
    state.save_membership(membership)
    return state, cluster, node


@pytest.mark.parametrize(
    ("adapter_type", "module", "architecture", "system"),
    [
        (WindowsPlatformAdapter, "swarm_inference.platforms.windows", "AMD64", "windows"),
        (LinuxPlatformAdapter, "swarm_inference.platforms.linux", "x86_64", "linux"),
    ],
)
def test_platform_detection_reports_implementation_not_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    adapter_type: type[WindowsPlatformAdapter] | type[LinuxPlatformAdapter],
    module: str,
    architecture: str,
    system: str,
) -> None:
    monkeypatch.setattr(f"{module}.platform.machine", lambda: architecture)
    monkeypatch.setattr(f"{module}.platform.release", lambda: "test-release")
    identity = adapter_type(home_directory=tmp_path).identity()

    assert identity.system == system
    assert identity.implementation_status == "implemented"
    assert "validation requires retained evidence" in identity.implementation_reason
    assert "validation_status" not in identity.model_dump()


def test_cpu_software_evidence_does_not_validate_cuda_or_physical_hardware(
    tmp_path: Path,
) -> None:
    state, _, node = _state_with_node(tmp_path)
    evidence = tmp_path / "cpu-software-gate.json"
    evidence.write_text('{"gate":"cpu-software","status":"PASS"}\n', encoding="utf-8")
    evidence_id = "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest()
    updated = state.record_backend_validation(
        node.node_id,
        BackendValidationRecord(
            backend=Backend.TORCH_CPU,
            platform_system="windows",
            platform_release="11",
            platform_architecture="AMD64",
            software_status="validated",
            physical_status="not-run",
            evidence_id=evidence_id,
            evidence_path=evidence,
            validated_at_unix_ns=2,
            detail="CPU software gate passed",
        ),
    )

    assert updated.software_validation_status == "validated"
    assert updated.physical_validation_status == "not-run"
    assert aggregate_validation_status(
        updated.backend_validations,
        backend=Backend.TORCH_CUDA,
        architecture="AMD64",
        operating_system="windows 11",
    ) == ("not-run", "not-run")


def test_physical_cuda_evidence_is_exactly_platform_and_backend_scoped(tmp_path: Path) -> None:
    evidence = tmp_path / "physical-cuda.json"
    evidence.write_text('{"gate":"physical-cuda","status":"PASS"}\n', encoding="utf-8")
    record = BackendValidationRecord(
        backend=Backend.TORCH_CUDA,
        platform_system="windows",
        platform_release="11",
        platform_architecture="AMD64",
        software_status="validated",
        physical_status="validated",
        evidence_id="sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest(),
        evidence_path=evidence,
        validated_at_unix_ns=3,
        detail="physical CUDA gate passed on the retained machine scope",
    )

    assert aggregate_validation_status(
        [record],
        backend=Backend.TORCH_CUDA,
        architecture="AMD64",
        operating_system="windows 11",
    ) == ("validated", "validated")
    assert aggregate_validation_status(
        [record],
        backend=Backend.TORCH_CPU,
        architecture="AMD64",
        operating_system="windows 11",
    ) == ("not-run", "not-run")
    assert aggregate_validation_status(
        [record],
        backend=Backend.TORCH_CUDA,
        architecture="ARM64",
        operating_system="windows 11",
    ) == ("not-run", "not-run")
    assert aggregate_validation_status(
        [record],
        backend=Backend.TORCH_CUDA,
        architecture="AMD64",
        operating_system="windows 12",
    ) == ("not-run", "not-run")


def test_legacy_state_migration_preserves_cluster_membership_and_trust(tmp_path: Path) -> None:
    state, cluster, node = _state_with_node(tmp_path)
    trust = WorkerTrustStore(state.paths.security / "trusted-workers.json")
    trust.trust(node.fingerprint, label="retained worker")
    legacy = node.model_dump(mode="json")
    for field in (
        "implementation_status",
        "implementation_reason",
        "software_validation_status",
        "physical_validation_status",
        "backend_validations",
        "validation_migration_note",
    ):
        legacy.pop(field, None)
    legacy.update(
        {
            "document_version": 1,
            "validation_status": "validated",
            "platform_support_status": "validated",
        }
    )
    state.paths.nodes.write_text(
        json.dumps({"schema_version": 1, "document_version": 1, "nodes": [legacy]}),
        encoding="utf-8",
    )

    migrated = state.load_nodes().nodes[0]
    assert migrated.node_id == node.node_id
    assert migrated.implementation_status == "implemented"
    assert migrated.software_validation_status == "not-run"
    assert migrated.physical_validation_status == "not-run"
    assert "not promoted" in (migrated.validation_migration_note or "")
    assert state.load_cluster() == cluster
    assert state.membership(node.node_id) is not None
    assert trust.contains(node.fingerprint)


def test_node_status_serializes_separate_implementation_and_validation_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state, _, node = _state_with_node(tmp_path)

    class Services:
        async def status(self, _definition: object) -> object:
            return SimpleNamespace(model_dump=lambda **_kwargs: {"installed": False})

    monkeypatch.setattr(
        "swarm_inference.commands.node.build_context",
        lambda _root: (state, SimpleNamespace(), SimpleNamespace(), Services()),
    )
    monkeypatch.setattr(
        "swarm_inference.commands.node.service_definition", lambda *_args, **_kwargs: object()
    )
    result = CliRunner().invoke(
        app,
        ["node", "status", "--state-root", str(state.paths.root), "--json"],
    )
    assert result.exit_code == 0, result.output
    metadata = json.loads(result.stdout)["metadata"]
    assert metadata["implementation_status"] == "implemented"
    assert metadata["software_validation_status"] == "not-run"
    assert metadata["physical_validation_status"] == "not-run"
    assert "platform_support_status" not in metadata
    assert "validation_status" not in metadata
    assert metadata["node_id"] == node.node_id


def test_platform_support_documentation_matches_the_runtime_contract() -> None:
    rows: dict[str, frozenset[str]] = {}
    for line in Path("docs/platform-support.md").read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `") or "implemented" not in line:
            continue
        cells = [cell.strip().strip("`") for cell in line.strip("|").split("|")]
        if len(cells) >= 3:
            rows[cells[0]] = frozenset(item.strip() for item in cells[1].split(","))
    assert rows == PLATFORM_IMPLEMENTATION_CONTRACT
