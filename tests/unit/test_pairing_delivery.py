from __future__ import annotations

import base64
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from swarm_inference.cli import app
from swarm_inference.cluster.models import (
    ClusterMetadata,
    NodeMetadata,
    PairingSession,
    VersionCompatibility,
    node_id_from_fingerprint,
)
from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.commands._common import _redact, redact_text
from swarm_inference.protocol.cluster import PairingCreateResponse
from swarm_inference.security.identity import CoordinatorIdentity

_INVITATION = json.dumps(
    {
        "key": base64.urlsafe_b64encode(b"k" * 32).decode().rstrip("="),
        "secret": base64.urlsafe_b64encode(b"s" * 32).decode().rstrip("="),
        "session": "session-1",
        "version": 1,
    },
    separators=(",", ":"),
    sort_keys=True,
).encode()
SECRET = base64.urlsafe_b64encode(_INVITATION).decode().rstrip("=")
URI = f"swarm://192.168.1.20:55000/join/{SECRET}"
REDACTED = "swarm://192.168.1.20:55000/join/REDACTED"


def _pairing() -> PairingCreateResponse:
    return PairingCreateResponse(
        session_id="session-1",
        pairing_uri=URI,
        redacted_uri=REDACTED,
        expires_at_unix_ns=2_000_000_000,
    )


def _single_json(stdout: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    payload, end = decoder.raw_decode(stdout.lstrip())
    assert not stdout.lstrip()[end:].strip()
    assert isinstance(payload, dict)
    return payload


def _cluster_fixture(state: ClusterStateStore) -> tuple[ClusterMetadata, NodeMetadata]:
    coordinator = CoordinatorIdentity.generate()
    identity = state.load_or_create_node_identity()
    node_id = node_id_from_fingerprint(identity.public_key_fingerprint)
    cluster = ClusterMetadata(
        cluster_id="cluster-pairing-output",
        name="pairing-output",
        coordinator_id=node_id,
        coordinator_endpoint="192.168.1.20:55000",
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
        public_key=identity.public_key_b64,
        fingerprint=identity.public_key_fingerprint,
        hostname="coordinator",
        operating_system="windows 11",
        architecture="AMD64",
        agent_version="0.1.0",
        runtime_version="0.1.0",
        build_id="test",
        package_lock_hash="1" * 64,
        joined_at_unix_ns=1,
        last_seen_at_unix_ns=1,
        implementation_status="implemented",
        implementation_reason="test implementation",
    )
    return cluster, node


def test_pairing_commands_emit_one_json_document(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_pair(*_args: object, **_kwargs: object) -> PairingCreateResponse:
        return _pairing()

    monkeypatch.setattr("swarm_inference.commands.cluster._create_pairing", fake_pair)
    pair_root = tmp_path / "pair"
    pair_result = CliRunner().invoke(
        app,
        ["cluster", "pair", "--state-root", str(pair_root), "--json"],
    )
    assert pair_result.exit_code == 0, pair_result.output
    pair_payload = _single_json(pair_result.stdout)
    assert pair_payload["status"] == "ready"
    assert SECRET not in pair_result.stdout

    create_root = tmp_path / "create"
    state = ClusterStateStore(create_root)
    cluster, node = _cluster_fixture(state)

    class Services:
        async def install(self, _definition: object) -> object:
            return SimpleNamespace(installed=True, detail="installed")

    runtime = SimpleNamespace(prepare_configuration=lambda **_kwargs: None)
    platform = SimpleNamespace(service_mode="windows-task")
    monkeypatch.setattr(
        "swarm_inference.commands.cluster.build_context",
        lambda _root: (state, platform, runtime, Services()),
    )
    monkeypatch.setattr(
        "swarm_inference.commands.cluster._bootstrap_cluster",
        lambda *_args, **_kwargs: (cluster, node),
    )
    monkeypatch.setattr(
        "swarm_inference.commands.cluster.service_definition", lambda *_args, **_kwargs: object()
    )

    async def ready(*_args: object, **_kwargs: object) -> object:
        return SimpleNamespace(state="ready", reason=None)

    monkeypatch.setattr("swarm_inference.commands.cluster.wait_for_runtime", ready)
    output = tmp_path / "create-invitation.uri"
    create_result = CliRunner().invoke(
        app,
        [
            "cluster",
            "create",
            "--name",
            cluster.name,
            "--state-root",
            str(create_root),
            "--yes",
            "--json",
            "--pairing-output",
            str(output),
        ],
    )
    assert create_result.exit_code == 0, create_result.output
    create_payload = _single_json(create_result.stdout)
    assert create_payload["cluster"]["cluster_id"] == cluster.cluster_id  # type: ignore[index]
    assert create_payload["pairing"]["invitation_file"] == str(output.resolve())  # type: ignore[index]
    assert SECRET not in create_result.stdout


def test_protected_invitation_file_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = ClusterStateStore(tmp_path / "state")
    destination = tmp_path / "nested" / "invitation.uri"
    import swarm_inference.cluster.state as state_module

    real_link = state_module.os.link
    publishes: list[tuple[Path, Path]] = []

    def observed_link(source: str | bytes | Path, target: str | bytes | Path) -> None:
        source_path, target_path = Path(source), Path(target)
        assert source_path.name.startswith(f".{destination.name}.")
        assert source_path.read_text(encoding="utf-8") == URI
        assert not target_path.exists()
        publishes.append((source_path, target_path))
        real_link(source, target)

    monkeypatch.setattr(state_module.os, "link", observed_link)
    path, protection, _ = state.write_pairing_invitation(
        session_id="session-1",
        pairing_uri=URI,
        output_path=destination,
    )
    assert path == destination.resolve()
    assert destination.read_text(encoding="utf-8") == URI
    assert len(publishes) == 1
    assert not list(destination.parent.glob(f".{destination.name}.*.tmp"))
    if os.name != "nt":
        assert protection == "posix-0600"
        assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    monkeypatch.setattr(state_module.os, "link", real_link)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        state.write_pairing_invitation(
            session_id="session-1",
            pairing_uri=URI.replace(SECRET, "replacement"),
            output_path=destination,
        )
    assert destination.read_text(encoding="utf-8") == URI


def test_pairing_secret_is_absent_from_machine_channels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = 0

    async def fake_pair(*_args: object, **_kwargs: object) -> PairingCreateResponse:
        nonlocal calls
        calls += 1
        return _pairing()

    monkeypatch.setattr("swarm_inference.commands.cluster._create_pairing", fake_pair)
    output = tmp_path / "invitation.uri"
    result = CliRunner().invoke(
        app,
        [
            "cluster",
            "pair",
            "--state-root",
            str(tmp_path / "state"),
            "--json",
            "--pairing-output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = _single_json(result.stdout)
    assert calls == 1
    assert output.read_text(encoding="utf-8") == URI
    assert SECRET not in result.stdout
    assert SECRET not in getattr(result, "stderr", "")
    assert SECRET not in json.dumps(payload)
    for path in (tmp_path / "state").rglob("*"):
        if path.is_file() and path.resolve() != output.resolve():
            assert SECRET.encode() not in path.read_bytes()

    rejected = CliRunner().invoke(
        app,
        [
            "cluster",
            "pair",
            "--state-root",
            str(tmp_path / "other"),
            "--json",
            "--pairing-output",
            "-",
        ],
    )
    assert rejected.exit_code != 0
    assert calls == 1
    assert SECRET not in rejected.output


def test_human_pairing_uri_is_displayed_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_pair(*_args: object, **_kwargs: object) -> PairingCreateResponse:
        return _pairing()

    monkeypatch.setattr("swarm_inference.commands.cluster._create_pairing", fake_pair)
    result = CliRunner().invoke(
        app,
        ["cluster", "pair", "--state-root", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert result.stdout.count(URI) == 1
    assert result.stdout.count(SECRET) == 1
    assert f'swarm node join "{URI}"' in result.stdout


def test_nested_pairing_redaction_and_expired_cleanup(tmp_path: Path) -> None:
    redacted = _redact({"outer": [{"pairing_uri": URI}, {"detail": f"received {URI}"}]})
    assert SECRET not in json.dumps(redacted)
    assert SECRET not in redact_text(RuntimeError(f"failed for {URI}"))

    state = ClusterStateStore(tmp_path / "state", clock_ns=lambda: 200)
    active = PairingSession(
        session_id="active",
        coordinator_endpoint="192.168.1.20:55000",
        coordinator_ephemeral_public_key="cHVibGlj",
        created_at_unix_ns=100,
        expires_at_unix_ns=300,
    )
    expired = PairingSession(
        session_id="expired",
        coordinator_endpoint="192.168.1.20:55000",
        coordinator_ephemeral_public_key="cHVibGlj",
        created_at_unix_ns=1,
        expires_at_unix_ns=100,
        state="expired",
    )
    state.save_pairing_session(active)
    state.save_pairing_session(expired)
    active_path = state.default_pairing_invitation_path("active")
    expired_path = state.default_pairing_invitation_path("expired")
    active_path.write_text(URI, encoding="utf-8")
    expired_path.write_text(URI, encoding="utf-8")
    removed = state.cleanup_expired_pairing_invitations()
    assert removed == [expired_path]
    assert active_path.is_file()
    assert not expired_path.exists()
