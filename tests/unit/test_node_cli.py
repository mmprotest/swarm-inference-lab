from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swarm_inference.cli import app
from swarm_inference.commands.node import _parse_bytes


@pytest.mark.parametrize(
    ("value", "expected"),
    [("20GB", 20_000_000_000), ("2 GiB", 2 * 1024**3), ("4096", 4096)],
)
def test_node_memory_and_storage_quantities(value: str, expected: int) -> None:
    assert _parse_bytes(value) == expected


@pytest.mark.parametrize("value", ["", "0GB", "-1GB", "1.5B", "many"])
def test_node_memory_quantity_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        _parse_bytes(value)


def test_node_command_tree_is_complete() -> None:
    result = CliRunner().invoke(app, ["node", "--help"])

    assert result.exit_code == 0, result.output
    for command in (
        "join",
        "status",
        "configure",
        "doctor",
        "leave",
        "install-service",
        "uninstall-service",
        "update",
    ):
        assert command in result.output


def test_unpaired_node_status_is_strict_json_without_identity_secrets(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["node", "status", "--state-root", str(tmp_path), "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["cluster"] is None
    assert payload["runtime"] is None
    assert payload["node_id"].startswith("node-")
    assert "private_key" not in result.output.lower()
    assert "pairing_secret" not in result.output.lower()
