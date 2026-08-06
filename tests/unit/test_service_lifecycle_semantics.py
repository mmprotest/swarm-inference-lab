from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from swarm_inference.cli import app
from swarm_inference.cluster.models import (
    ClusterMetadata,
    NodeMetadata,
    VersionCompatibility,
    node_id_from_fingerprint,
)
from swarm_inference.cluster.pairing import PairingInvitation
from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.config.models import Backend
from swarm_inference.protocol.cluster import PairingCreateResponse
from swarm_inference.security.identity import CoordinatorIdentity


def _cluster_and_node(state: ClusterStateStore) -> tuple[ClusterMetadata, NodeMetadata]:
    identity = state.load_or_create_node_identity()
    coordinator = CoordinatorIdentity.generate()
    node_id = node_id_from_fingerprint(identity.public_key_fingerprint)
    cluster = ClusterMetadata(
        cluster_id="cluster-service-lifecycle",
        name="service-lifecycle",
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
        public_key=identity.public_key_b64,
        fingerprint=identity.public_key_fingerprint,
        hostname="node",
        operating_system="windows 11",
        architecture="AMD64",
        agent_version="0.1.0",
        runtime_version="0.1.0",
        build_id="test",
        package_lock_hash="b" * 64,
        joined_at_unix_ns=1,
        last_seen_at_unix_ns=1,
        implementation_status="implemented",
        implementation_reason="test platform",
    )
    return cluster, node


def _pairing() -> PairingCreateResponse:
    return PairingCreateResponse(
        session_id="session-service",
        pairing_uri=PairingInvitation(
            coordinator_endpoint="192.168.1.10:50051",
            session_id="session-service",
            pairing_secret=b"s" * 32,
            coordinator_ephemeral_public_key=b"k" * 32,
        ).uri(),
        redacted_uri=PairingInvitation(
            coordinator_endpoint="192.168.1.10:50051",
            session_id="session-service",
            pairing_secret=b"s" * 32,
            coordinator_ephemeral_public_key=b"k" * 32,
        ).redacted_uri(),
        expires_at_unix_ns=2_000_000_000,
    )


def _configuration(service_mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        service_mode=service_mode,
        backend_selection=SimpleNamespace(
            selected_backend=Backend.TORCH_CPU,
            selected_device="cpu",
            selected_dtype="bfloat16",
        ),
        memory_budget=SimpleNamespace(limit_bytes=1024**3),
        storage_limit_bytes=2 * 1024**3,
        endpoints=SimpleNamespace(
            control_advertised_endpoint="192.168.1.20:51001",
            data_advertised_endpoint="192.168.1.20:51002",
        ),
    )


def test_cluster_create_installs_service_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = ClusterStateStore(tmp_path / "state")
    cluster, node = _cluster_and_node(state)
    install_calls: list[object] = []

    class Services:
        async def install(self, definition: object) -> object:
            install_calls.append(definition)
            return SimpleNamespace(installed=True, detail="installed and started")

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

    async def pairing(*_args: object, **_kwargs: object) -> PairingCreateResponse:
        return _pairing()

    monkeypatch.setattr("swarm_inference.commands.cluster.wait_for_runtime", ready)
    monkeypatch.setattr("swarm_inference.commands.cluster._create_pairing", pairing)
    result = CliRunner().invoke(
        app,
        [
            "cluster",
            "create",
            "--name",
            cluster.name,
            "--state-root",
            str(state.paths.root),
            "--yes",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(install_calls) == 1
    assert json.loads(result.stdout)["service"]["mode"] == "windows-task"


def test_cluster_create_foreground_never_installs_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = ClusterStateStore(tmp_path / "state")
    cluster, node = _cluster_and_node(state)

    class Services:
        async def install(self, _definition: object) -> object:
            pytest.fail("foreground cluster creation must not install a service")

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

    class Agent:
        async def wait(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    async def foreground(*_args: object, **_kwargs: object) -> tuple[object, object, Agent]:
        return SimpleNamespace(state="ready", reason=None), _pairing(), Agent()

    async def finish(waitable: object, *_args: object, **_kwargs: object) -> None:
        await waitable  # type: ignore[misc]

    monkeypatch.setattr("swarm_inference.commands.cluster._foreground_create", foreground)
    monkeypatch.setattr(
        "swarm_inference.commands.cluster.install_shutdown_signal_handlers",
        lambda _event: lambda: None,
    )
    monkeypatch.setattr("swarm_inference.commands.cluster.wait_for_service_shutdown", finish)
    result = CliRunner().invoke(
        app,
        [
            "cluster",
            "create",
            "--name",
            cluster.name,
            "--state-root",
            str(state.paths.root),
            "--foreground",
            "--yes",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["service"]["mode"] == "foreground"


@pytest.mark.parametrize(
    ("foreground", "expected_mode", "expected_installs"),
    [(False, "windows-task", 1), (True, "foreground", 0)],
)
def test_node_join_owns_service_installation_unless_foreground(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    foreground: bool,
    expected_mode: str,
    expected_installs: int,
) -> None:
    state = ClusterStateStore(tmp_path / ("foreground" if foreground else "service"))
    cluster, node = _cluster_and_node(state)
    install_calls: list[object] = []

    class Services:
        async def install(self, definition: object) -> object:
            install_calls.append(definition)
            return SimpleNamespace(installed=True, detail="installed and started")

    configuration = _configuration(expected_mode)

    class Runtime:
        def prepare_configuration(self, **kwargs: object) -> object:
            assert kwargs["service_mode"] == expected_mode
            return configuration

        def worker_id(self, _configuration: object) -> str:
            return f"{node.node_id}/cpu-0"

    monkeypatch.setattr(
        "swarm_inference.commands.node.build_context",
        lambda _root: (
            state,
            SimpleNamespace(service_mode="windows-task"),
            Runtime(),
            Services(),
        ),
    )
    monkeypatch.setattr("swarm_inference.commands.node._pending_metadata", lambda *_args: node)

    async def pair(*_args: object, **_kwargs: object) -> object:
        return SimpleNamespace(cluster=cluster)

    async def ready(*_args: object, **_kwargs: object) -> object:
        return SimpleNamespace(state="ready", reason=None)

    async def foreground_agent(
        *_args: object,
        on_started: object,
        **_kwargs: object,
    ) -> object:
        status = SimpleNamespace(state="ready", reason=None)
        on_started(status)  # type: ignore[operator]
        return status

    monkeypatch.setattr("swarm_inference.commands.node._pair", pair)
    monkeypatch.setattr("swarm_inference.commands.node.wait_for_runtime", ready)
    monkeypatch.setattr("swarm_inference.commands.node._foreground_agent", foreground_agent)
    monkeypatch.setattr(
        "swarm_inference.commands.node.service_definition", lambda *_args, **_kwargs: object()
    )
    arguments = [
        "node",
        "join",
        _pairing().pairing_uri,
        "--state-root",
        str(state.paths.root),
        "--yes",
        "--json",
    ]
    if foreground:
        arguments.append("--foreground")
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert len(install_calls) == expected_installs
    assert json.loads(result.stdout)["service_mode"] == expected_mode
    assert _pairing().pairing_uri not in result.output


def test_post_pair_install_service_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = ClusterStateStore(tmp_path / "paired")
    cluster, _ = _cluster_and_node(state)
    state.save_cluster(cluster)
    calls = 0

    class Services:
        async def install(self, _definition: object) -> object:
            nonlocal calls
            calls += 1
            return SimpleNamespace(
                installed=True,
                detail="already installed and running",
                model_dump=lambda **_kwargs: {
                    "installed": True,
                    "running": True,
                    "detail": "already installed and running",
                },
            )

    monkeypatch.setattr(
        "swarm_inference.commands.node.build_context",
        lambda _root: (state, SimpleNamespace(), SimpleNamespace(), Services()),
    )
    monkeypatch.setattr(
        "swarm_inference.commands.node.service_definition", lambda *_args, **_kwargs: object()
    )
    for _ in range(2):
        result = CliRunner().invoke(
            app,
            [
                "node",
                "install-service",
                "--state-root",
                str(state.paths.root),
                "--yes",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["installed"] is True
    assert calls == 2
