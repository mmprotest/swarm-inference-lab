from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swarm_inference.cli import app
from swarm_inference.exceptions import IntegrityError
from swarm_inference.security.identity import (
    CoordinatorIdentity,
    WorkerIdentity,
    create_identity_file,
    inspect_identity_file,
)
from swarm_inference.security.trust_store import WorkerTrustStore


def test_versioned_identity_creation_is_stable_and_refuses_overwrite(tmp_path: Path) -> None:
    identity_path = tmp_path / "worker.json"
    identity, metadata = create_identity_file(identity_path, kind="worker")

    assert metadata.identity_kind == "worker"
    assert metadata.format_version == 1
    assert metadata.fingerprint == identity.public_key_fingerprint
    assert WorkerIdentity.load(identity_path).public_key_fingerprint == metadata.fingerprint
    assert inspect_identity_file(identity_path).fingerprint == metadata.fingerprint

    original = identity_path.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        create_identity_file(identity_path, kind="worker")
    assert identity_path.read_bytes() == original

    replacement, replacement_metadata = create_identity_file(
        identity_path,
        kind="worker",
        force=True,
    )
    assert replacement_metadata.fingerprint == replacement.public_key_fingerprint
    assert replacement_metadata.fingerprint != metadata.fingerprint


def test_identity_kind_and_malformed_documents_fail_closed(tmp_path: Path) -> None:
    coordinator_path = tmp_path / "coordinator.json"
    coordinator, metadata = create_identity_file(coordinator_path, kind="coordinator")
    assert isinstance(coordinator, CoordinatorIdentity)
    assert metadata.identity_kind == "coordinator"
    assert CoordinatorIdentity.load(coordinator_path).public_key_fingerprint == metadata.fingerprint
    with pytest.raises(IntegrityError, match="does not match required kind"):
        WorkerIdentity.load(coordinator_path)

    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"format_version": 99}', encoding="utf-8")
    with pytest.raises(IntegrityError, match="unsupported identity document type"):
        inspect_identity_file(malformed)

    damaged = json.loads(coordinator_path.read_text(encoding="utf-8"))
    damaged["fingerprint"] = "0" * 64
    coordinator_path.write_text(json.dumps(damaged), encoding="utf-8")
    with pytest.raises(IntegrityError, match="fingerprint does not match"):
        inspect_identity_file(coordinator_path)


def test_trust_store_is_atomic_deduplicated_reloadable_and_audited(tmp_path: Path) -> None:
    store = WorkerTrustStore(tmp_path / "state" / "trusted-workers.json")
    first = WorkerIdentity.generate().public_key_fingerprint
    second = WorkerIdentity.generate().public_key_fingerprint

    record, added = store.trust(second, label="worker-b")
    assert added and record.fingerprint == second
    record, added = store.trust(first, label="worker-a", notes="rack one")
    assert added and record.fingerprint == first
    record, added = store.trust(first, label="worker-a-renamed")
    assert not added
    assert record.label == "worker-a-renamed"
    assert record.notes == "rack one"

    document = store.load()
    assert [item.fingerprint for item in document.workers] == sorted([first, second])
    assert store.contains(first)
    assert not list(store.path.parent.glob("*.tmp"))
    assert store.audit_path.read_text(encoding="utf-8").count("\n") == 3

    assert store.untrust(first)
    assert not store.untrust(first)
    assert not store.contains(first)
    assert store.contains(second)


def test_identity_cli_json_never_discloses_private_key(tmp_path: Path) -> None:
    runner = CliRunner()
    identity_path = tmp_path / "worker.json"
    coordinator_state = tmp_path / "coordinator"

    created = runner.invoke(
        app,
        ["identity", "create", "--path", str(identity_path), "--kind", "worker", "--json"],
    )
    assert created.exit_code == 0, created.output
    public_metadata = json.loads(created.output)
    assert public_metadata["identity_kind"] == "worker"
    assert "private" not in created.output.lower()
    private_record = json.loads(identity_path.read_text(encoding="utf-8"))["private_key"]
    assert private_record["value"] not in created.output

    shown = runner.invoke(app, ["identity", "show", "--path", str(identity_path), "--json"])
    fingerprint = runner.invoke(
        app,
        ["identity", "fingerprint", "--path", str(identity_path), "--json"],
    )
    trusted = runner.invoke(
        app,
        [
            "identity",
            "trust",
            "--coordinator-state",
            str(coordinator_state),
            "--identity",
            str(identity_path),
            "--label",
            "worker-1",
            "--json",
        ],
    )
    listed = runner.invoke(
        app,
        ["identity", "list-trusted", "--coordinator-state", str(coordinator_state), "--json"],
    )
    untrusted = runner.invoke(
        app,
        [
            "identity",
            "untrust",
            "--coordinator-state",
            str(coordinator_state),
            "--identity",
            str(identity_path),
            "--json",
        ],
    )

    for result in (shown, fingerprint, trusted, listed, untrusted):
        assert result.exit_code == 0, result.output
        assert private_record["value"] not in result.output
    assert json.loads(shown.output)["fingerprint"] == public_metadata["fingerprint"]
    assert json.loads(fingerprint.output)["fingerprint"] == public_metadata["fingerprint"]
    assert json.loads(trusted.output)["status"] == "trusted"
    assert len(json.loads(listed.output)["workers"]) == 1
    assert json.loads(untrusted.output)["status"] == "untrusted"


def test_identity_cli_overwrite_refusal_and_invalid_input(tmp_path: Path) -> None:
    runner = CliRunner()
    path = tmp_path / "worker.json"
    first = runner.invoke(app, ["identity", "create", "--path", str(path)])
    second = runner.invoke(app, ["identity", "create", "--path", str(path)])
    malformed = tmp_path / "bad.json"
    malformed.write_text("not json", encoding="utf-8")
    shown = runner.invoke(app, ["identity", "show", "--path", str(malformed)])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 1
    assert "already exists" in second.output
    assert shown.exit_code == 1
    assert "invalid legacy identity file" in shown.output
