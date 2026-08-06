from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from swarm_inference.cli import app
from swarm_inference.protocol.cluster import PairingCreateResponse


def test_cluster_command_tree_is_complete() -> None:
    result = CliRunner().invoke(app, ["cluster", "--help"])

    assert result.exit_code == 0, result.output
    for command in ("create", "pair", "status", "nodes", "revoke", "start", "stop", "delete"):
        assert command in result.output


def test_pairing_json_redacts_single_use_secret(monkeypatch, tmp_path: Path) -> None:
    secret = "this-is-a-sensitive-pairing-secret"

    async def fake_create(*_args: object, **_kwargs: object) -> PairingCreateResponse:
        return PairingCreateResponse(
            session_id="session-1",
            pairing_uri=f"swarm+pair://10.0.0.2:50051/session-1?secret={secret}",
            redacted_uri="swarm+pair://10.0.0.2:50051/session-1?secret=<redacted>",
            expires_at_unix_ns=2_000_000_000,
        )

    monkeypatch.setattr("swarm_inference.commands.cluster._create_pairing", fake_create)
    result = CliRunner().invoke(
        app,
        ["cluster", "pair", "--state-root", str(tmp_path), "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["pairing"]["redacted_uri"].endswith("secret=<redacted>")
    assert Path(payload["pairing"]["invitation_file"]).read_text(encoding="utf-8").endswith(secret)
    assert secret not in result.output


def test_cluster_create_rejects_empty_name_before_service_mutation(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "cluster",
            "create",
            "--name",
            " ",
            "--state-root",
            str(tmp_path),
            "--coordinator-endpoint",
            "192.168.1.20:55000",
            "--json",
            "--yes",
        ],
    )

    assert result.exit_code != 0
    assert "cluster-create" in result.output
    assert not (tmp_path / "cluster.json").exists()
