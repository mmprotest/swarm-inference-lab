from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from swarm_inference.cli import app
from swarm_inference.cluster.models import (
    ClusterMetadata,
    VersionCompatibility,
    node_id_from_fingerprint,
)
from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.commands._common import EXIT_PERMISSION, require_confirmation
from swarm_inference.security.identity import CoordinatorIdentity


class _InteractiveInput:
    def isatty(self) -> bool:
        return True


def test_interactive_acceptance_refusal_and_yes_are_enforced_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import swarm_inference.commands._common as common

    monkeypatch.setattr(common.sys, "stdin", _InteractiveInput())
    prompts: list[str] = []
    changed = False

    def accept(prompt: str, *, default: bool) -> bool:
        prompts.append(prompt)
        assert default is False
        return True

    monkeypatch.setattr(common.typer, "confirm", accept)
    require_confirmation("Revoke cluster node trust", yes=False)
    changed = True
    assert changed
    assert prompts == ["Revoke cluster node trust. Continue?"]

    changed = False
    monkeypatch.setattr(common.typer, "confirm", lambda *_args, **_kwargs: False)
    with pytest.raises(PermissionError, match="cancelled; no changes were made"):
        require_confirmation("Delete local cluster state", yes=False)
        changed = True
    assert not changed

    monkeypatch.setattr(
        common.typer,
        "confirm",
        lambda *_args, **_kwargs: pytest.fail("--yes must bypass the prompt"),
    )
    require_confirmation("Install the node service", yes=True)


@pytest.mark.parametrize(
    ("json_output", "ndjson", "mode"),
    [(True, False, "JSON"), (False, True, "NDJSON")],
)
def test_machine_readable_confirmation_never_prompts(
    monkeypatch: pytest.MonkeyPatch,
    json_output: bool,
    ndjson: bool,
    mode: str,
) -> None:
    import swarm_inference.commands._common as common

    monkeypatch.setattr(
        common.typer,
        "confirm",
        lambda *_args, **_kwargs: pytest.fail("machine-readable mode must not prompt"),
    )
    with pytest.raises(PermissionError, match=rf"requires --yes in {mode} mode"):
        require_confirmation(
            "Change node configuration",
            yes=False,
            json_output=json_output,
            ndjson=ndjson,
        )


def test_non_interactive_stdin_without_yes_fails_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import swarm_inference.commands._common as common

    monkeypatch.setattr(common.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    with pytest.raises(PermissionError, match="interactive stdin or --yes"):
        require_confirmation("Leave the cluster", yes=False)


@pytest.mark.parametrize(
    "arguments",
    [
        ["cluster", "create", "--name", "confirm-test", "--json"],
        ["cluster", "revoke", "node-12345678", "--json"],
        ["cluster", "delete", "--json"],
        [
            "node",
            "join",
            "swarm+pair://192.168.1.10:50051/join?session=s&secret=never-print&key=k",
            "--json",
        ],
        ["node", "configure", "--backend", "torch-cpu", "--json"],
        ["node", "leave", "--json"],
        ["node", "update", "--source-wheel", "{wheel}", "--json"],
    ],
)
def test_mutating_json_commands_require_yes_before_state_mutation(
    tmp_path: Path,
    arguments: list[str],
) -> None:
    wheel = tmp_path / "update.whl"
    wheel.write_bytes(b"not reached")
    state_root = tmp_path / "state"
    resolved = [str(wheel) if item == "{wheel}" else item for item in arguments]
    resolved.extend(["--state-root", str(state_root)])
    result = CliRunner().invoke(app, resolved)

    assert result.exit_code == EXIT_PERMISSION, result.output
    assert "requires --yes in JSON mode" in result.output
    assert not state_root.exists()
    assert "never-print" not in result.output


def test_mutating_ndjson_join_requires_yes_before_parsing_or_mutation(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    secret = "ndjson-pairing-secret"
    result = CliRunner().invoke(
        app,
        [
            "node",
            "join",
            f"swarm+pair://192.168.1.10:50051/join?session=s&secret={secret}&key=k",
            "--ndjson",
            "--state-root",
            str(state_root),
        ],
    )
    assert result.exit_code == EXIT_PERMISSION, result.output
    assert "requires --yes in NDJSON mode" in result.output
    assert secret not in result.output
    assert not state_root.exists()


def _paired_state(tmp_path: Path) -> ClusterStateStore:
    state = ClusterStateStore(tmp_path / "paired")
    node_identity = state.load_or_create_node_identity()
    coordinator = CoordinatorIdentity.generate()
    state.save_cluster(
        ClusterMetadata(
            cluster_id="cluster-confirm",
            name="confirm",
            coordinator_id=node_id_from_fingerprint(node_identity.public_key_fingerprint),
            coordinator_endpoint="192.168.1.10:50051",
            coordinator_public_key=coordinator.public_key_b64,
            coordinator_fingerprint=coordinator.public_key_fingerprint,
            created_at_unix_ns=1,
            runtime_compatibility=VersionCompatibility(
                minimum_runtime_version="0.1.0",
                maximum_runtime_version_exclusive="1.0.0",
            ),
        )
    )
    return state


@pytest.mark.parametrize("command", ["install-service", "uninstall-service"])
def test_service_commands_require_yes_before_calling_service_manager(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: str,
) -> None:
    state = _paired_state(tmp_path)

    class Services:
        async def install(self, _definition: object) -> object:
            pytest.fail("service installation must not be reached")

        async def uninstall(self, _definition: object) -> object:
            pytest.fail("service uninstallation must not be reached")

    monkeypatch.setattr(
        "swarm_inference.commands.node.build_context",
        lambda _root: (state, SimpleNamespace(), SimpleNamespace(), Services()),
    )
    result = CliRunner().invoke(
        app,
        ["node", command, "--state-root", str(state.paths.root), "--json"],
    )
    assert result.exit_code == EXIT_PERMISSION, result.output
    assert "requires --yes in JSON mode" in result.output


def test_unpaired_service_install_has_exact_remediation(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "node",
            "install-service",
            "--state-root",
            str(tmp_path / "unpaired"),
            "--yes",
            "--json",
        ],
    )
    assert result.exit_code != 0
    assert "node is not paired; create or join a cluster first" in result.output
