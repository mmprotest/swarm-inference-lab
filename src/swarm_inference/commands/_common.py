"""Shared, non-shell command composition for the cluster product CLI."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import typer

from swarm_inference import __version__
from swarm_inference.cluster.agent import NodeAgent, NodeAgentOptions
from swarm_inference.cluster.models import (
    ClusterMetadata,
    NodeRuntimeMetadata,
    node_id_from_fingerprint,
)
from swarm_inference.cluster.runtime_manager import RuntimeManager
from swarm_inference.cluster.service_manager import ServiceManager
from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.exceptions import (
    BackendIncompatibleError,
    CompatibilityError,
    ConfigurationError,
    IntegrityError,
    MemoryLimitExceededError,
    TransportError,
)
from swarm_inference.platforms import get_platform_adapter
from swarm_inference.platforms.base import PlatformAdapter, ServiceDefinition

EXIT_PERMISSION = 10
EXIT_CONNECTIVITY = 11
EXIT_COMPATIBILITY = 12
EXIT_CAPACITY = 13
EXIT_ARTIFACT_INTEGRITY = 14
EXIT_EXECUTION = 15

_SENSITIVE_KEYS = (
    "private_key",
    "pairing_secret",
    "session_key",
    "aes_key",
    "raw_proof",
    "prompt",
)
_PAIRING_SECRET_IN_URI = re.compile(r"(?i)(swarm\+pair://[^\s\"']*[?&]secret=)([^&\s\"']+)")
_OPAQUE_PAIRING_URI = re.compile(r"(?i)swarm://[^\s\"']+/join/[A-Za-z0-9_-]+")


def state_store(root: Path | None) -> ClusterStateStore:
    return ClusterStateStore(root)


def build_context(
    root: Path | None,
) -> tuple[ClusterStateStore, PlatformAdapter, RuntimeManager, ServiceManager]:
    state = state_store(root)
    platform = get_platform_adapter()
    runtime = RuntimeManager(state=state, platform=platform)
    services = ServiceManager(platform=platform, state=state)
    return state, platform, runtime, services


def service_definition(
    state: ClusterStateStore,
    cluster: ClusterMetadata,
    *,
    roles: set[str],
) -> ServiceDefinition:
    arguments = [
        "-m",
        "swarm_inference.cli",
        "node",
        "agent",
        "--state-root",
        str(state.paths.root),
    ]
    for role in sorted(roles):
        arguments.extend(("--role", role))
    from swarm_inference.cluster.updates import load_active_runtime_python
    from swarm_inference.native_install import native_install_record

    native_installation = native_install_record()
    active_python = None if native_installation is not None else load_active_runtime_python(state)
    return ServiceDefinition(
        cluster_id=cluster.cluster_id,
        node_id=node_id_from_fingerprint(
            state.load_or_create_node_identity().public_key_fingerprint
        ),
        executable=active_python or Path(sys.executable).absolute(),
        arguments=arguments,
        environment={
            "PYTHONUTF8": "1",
            "SWARM_AGENT_VERSION": __version__,
        },
        working_directory=state.paths.root,
    )


def _redact(value: Any, *, key: str = "") -> Any:
    lowered = key.lower()
    if any(term in lowered for term in _SENSITIVE_KEYS):
        return "<redacted>"
    if lowered == "pairing_uri":
        return "<redacted: use human output to transfer the single-use invitation>"
    if isinstance(value, dict):
        return {str(name): _redact(item, key=str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        redacted = _PAIRING_SECRET_IN_URI.sub(r"\1<redacted>", value)
        return _OPAQUE_PAIRING_URI.sub("swarm://<redacted>/join/<redacted>", redacted)
    return value


def redact_text(value: object) -> str:
    """Remove secret-bearing pairing query values from arbitrary diagnostics."""

    redacted = _PAIRING_SECRET_IN_URI.sub(r"\1<redacted>", str(value))
    return _OPAQUE_PAIRING_URI.sub("swarm://<redacted>/join/<redacted>", redacted)


def require_confirmation(
    action: str,
    *,
    yes: bool,
    json_output: bool = False,
    ndjson: bool = False,
) -> None:
    """Require one human confirmation or an explicit non-interactive ``--yes``.

    The caller must invoke this before its first mutation. ``action`` is fixed
    command-owned text and must never contain a pairing invitation or other secret.
    """

    if yes:
        return
    if json_output or ndjson:
        mode = "JSON" if json_output else "NDJSON"
        raise PermissionError(f"{action} requires --yes in {mode} mode; no changes were made")
    if not sys.stdin.isatty():
        raise PermissionError(f"{action} requires interactive stdin or --yes; no changes were made")
    if not typer.confirm(f"{action}. Continue?", default=False):
        raise PermissionError(f"{action} cancelled; no changes were made")


def emit_document(
    payload: dict[str, Any],
    *,
    json_output: bool,
    ndjson: bool = False,
) -> None:
    safe = _redact(payload)
    assert isinstance(safe, dict)
    if ndjson:
        typer.echo(json.dumps(safe, sort_keys=True, separators=(",", ":")))
    elif json_output:
        typer.echo(json.dumps(safe, indent=2, sort_keys=True))
    else:
        for key, value in safe.items():
            if isinstance(value, (dict, list)):
                typer.echo(f"{key}={json.dumps(value, sort_keys=True)}")
            else:
                typer.echo(f"{key}={value}")


def exit_code_for(exc: BaseException) -> int:
    if isinstance(exc, PermissionError):
        return EXIT_PERMISSION
    if isinstance(exc, (TransportError, TimeoutError, ConnectionError, OSError)):
        return EXIT_CONNECTIVITY
    if isinstance(exc, (CompatibilityError, ConfigurationError, BackendIncompatibleError)):
        return EXIT_COMPATIBILITY
    if isinstance(exc, (MemoryError, MemoryLimitExceededError)):
        return EXIT_CAPACITY
    if isinstance(exc, IntegrityError):
        return EXIT_ARTIFACT_INTEGRITY
    return EXIT_EXECUTION


def fail(stage: str, exc: BaseException, *, node_id: str | None = None) -> None:
    category = {
        EXIT_PERMISSION: "permission",
        EXIT_CONNECTIVITY: "connectivity",
        EXIT_COMPATIBILITY: "compatibility",
        EXIT_CAPACITY: "capacity",
        EXIT_ARTIFACT_INTEGRITY: "artifact-integrity",
        EXIT_EXECUTION: "execution",
    }[exit_code_for(exc)]
    affected = f" node={node_id}" if node_id else ""
    detail = redact_text(exc)
    typer.echo(
        f"failed_stage={stage}{affected} category={category} detail={detail} retry_safe=true",
        err=True,
    )
    raise typer.Exit(exit_code_for(exc))


async def wait_for_runtime(
    state: ClusterStateStore,
    node_id: str,
    *,
    timeout_seconds: float = 90.0,
) -> NodeRuntimeMetadata:
    deadline = time.monotonic() + timeout_seconds
    latest: NodeRuntimeMetadata | None = None
    while time.monotonic() < deadline:
        latest = state.load_runtime(node_id)
        if latest is not None and latest.state in {"ready", "blocked", "failed"}:
            return latest
        await asyncio.sleep(0.25)
    detail = latest.reason if latest is not None else "agent produced no runtime status"
    raise TimeoutError(f"node agent readiness timed out after {timeout_seconds:.0f}s: {detail}")


async def run_agent_foreground(
    state: ClusterStateStore,
    platform: PlatformAdapter,
    runtime: RuntimeManager,
    *,
    roles: set[str],
) -> NodeRuntimeMetadata:
    from typing import cast

    from swarm_inference.cluster.agent import NodeAgentRole
    from swarm_inference.runtime.shutdown import (
        install_shutdown_signal_handlers,
        wait_for_service_shutdown,
    )

    agent = NodeAgent(
        state=state,
        platform=platform,
        runtime_manager=runtime,
        options=NodeAgentOptions(roles={cast(NodeAgentRole, item) for item in roles}),
    )
    status = await agent.start()
    if status.state in {"blocked", "failed"}:
        await agent.stop()
        return status
    stop_event = asyncio.Event()
    restore = install_shutdown_signal_handlers(stop_event)
    try:
        await wait_for_service_shutdown(agent.wait(), stop_event, shutdown=agent.stop)
    finally:
        restore()
        await agent.stop()
    return agent.status


def bootstrap_package_identity() -> tuple[str, str]:
    import hashlib

    build_id = os.environ.get("SWARM_BUILD_ID", f"swarm-inference-lab-{__version__}")
    package_lock_hash = (
        os.environ.get("SWARM_PACKAGE_LOCK_HASH")
        or hashlib.sha256(f"{__version__}:{build_id}".encode()).hexdigest()
    )
    return build_id, package_lock_hash


__all__ = [
    "EXIT_ARTIFACT_INTEGRITY",
    "EXIT_CAPACITY",
    "EXIT_COMPATIBILITY",
    "EXIT_CONNECTIVITY",
    "EXIT_EXECUTION",
    "EXIT_PERMISSION",
    "bootstrap_package_identity",
    "build_context",
    "emit_document",
    "fail",
    "redact_text",
    "require_confirmation",
    "run_agent_foreground",
    "service_definition",
    "state_store",
    "wait_for_runtime",
]
