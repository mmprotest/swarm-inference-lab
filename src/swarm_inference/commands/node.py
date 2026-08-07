"""Persistent node onboarding, configuration, diagnostics, and service commands."""

from __future__ import annotations

import asyncio
import socket
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, cast

import typer

from swarm_inference import __version__
from swarm_inference.cluster.agent import NodeAgent, NodeAgentOptions, NodeAgentRole
from swarm_inference.cluster.models import (
    BackendValidationRecord,
    NodeMetadata,
    node_id_from_fingerprint,
)
from swarm_inference.cluster.pairing import (
    PairingClient,
    PairingInvitation,
    create_cluster_authentication,
)
from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.commands._common import (
    bootstrap_package_identity,
    build_context,
    emit_document,
    fail,
    require_confirmation,
    service_definition,
    wait_for_runtime,
)
from swarm_inference.config.models import Backend
from swarm_inference.coordinator.service import CoordinatorClient
from swarm_inference.host import split_endpoint
from swarm_inference.platforms.base import FirewallRuleSpec
from swarm_inference.protocol.cluster import NodeLeaveRequest
from swarm_inference.runtime.shutdown import (
    install_shutdown_signal_handlers,
    wait_for_service_shutdown,
)
from swarm_inference.security.tls import (
    TlsBootstrapClientConfig,
    tls_public_key_pem,
)

node_app = typer.Typer(
    name="node",
    help="Join, inspect, configure, and service this persistent cluster node.",
    no_args_is_help=True,
)

_BYTE_SUFFIXES = {
    "b": 1,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "tb": 1000**4,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}


def _parse_bytes(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = value.strip().lower().replace(" ", "")
    for suffix in sorted(_BYTE_SUFFIXES, key=len, reverse=True):
        if normalized.endswith(suffix):
            number = normalized[: -len(suffix)]
            try:
                result = float(number) * _BYTE_SUFFIXES[suffix]
            except ValueError as exc:
                raise ValueError(f"invalid byte quantity {value!r}") from exc
            if result <= 0 or not result.is_integer():
                raise ValueError("byte quantity must resolve to a positive whole byte count")
            return int(result)
    try:
        result = int(normalized)
    except ValueError as exc:
        raise ValueError(f"invalid byte quantity {value!r}") from exc
    if result <= 0:
        raise ValueError("byte quantity must be positive")
    return result


def _pending_metadata(
    state: ClusterStateStore,
    invitation: PairingInvitation,
    *,
    rotate_transport_key: bool = False,
) -> NodeMetadata:
    del invitation
    identity = state.load_or_create_node_identity()
    node_id = node_id_from_fingerprint(identity.public_key_fingerprint)
    platform = build_context(state.paths.root)[1]
    platform_identity = platform.identity()
    build_id, lock_hash = bootstrap_package_identity()
    now = time.time_ns()
    return NodeMetadata(
        node_id=node_id,
        public_key=identity.public_key_b64,
        fingerprint=identity.public_key_fingerprint,
        hostname=socket.gethostname(),
        operating_system=f"{platform_identity.system} {platform_identity.release}",
        architecture=platform_identity.architecture,
        agent_version=__version__,
        runtime_version=__version__,
        build_id=build_id,
        package_lock_hash=lock_hash,
        joined_at_unix_ns=now,
        last_seen_at_unix_ns=now,
        service_mode=platform.service_mode,
        implementation_status=platform_identity.implementation_status,
        implementation_reason=platform_identity.implementation_reason,
        tls_public_key_pem=tls_public_key_pem(
            state.prepare_node_tls_private_key(rotate=rotate_transport_key)
        ),
    )


async def _pair(
    state: ClusterStateStore,
    pairing_uri: str,
    metadata: NodeMetadata,
) -> Any:
    invitation = PairingInvitation.parse(pairing_uri)
    pairing_tls = (
        TlsBootstrapClientConfig(
            ca_certificate=state.materialize_pairing_ca(invitation.coordinator_certificate_pem)
        )
        if invitation.coordinator_certificate_pem is not None
        else None
    )
    client = CoordinatorClient(
        invitation.coordinator_endpoint,
        timeout_s=20.0,
        tls=pairing_tls,
    )
    pairing = PairingClient(
        state=state,
        identity=state.load_or_create_node_identity(),
    )
    try:
        return await pairing.join(
            pairing_uri,
            node_metadata=metadata,
            hello_rpc=client.pairing_hello,
            complete_rpc=client.pairing_complete,
        )
    finally:
        await client.close()


async def _foreground_agent(
    state: ClusterStateStore,
    *,
    roles: set[NodeAgentRole],
    on_started: Callable[[Any], None] | None = None,
) -> Any:
    _, platform, runtime, _ = build_context(state.paths.root)
    agent = NodeAgent(
        state=state,
        platform=platform,
        runtime_manager=runtime,
        options=NodeAgentOptions(roles=roles),
    )
    status = await agent.start()
    if on_started is not None:
        on_started(status)
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


@node_app.command("join")
def join_command(
    pairing_uri: Annotated[str, typer.Argument(metavar="PAIRING_URI")],
    state_root: Annotated[Path | None, typer.Option()] = None,
    foreground: Annotated[bool, typer.Option("--foreground")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    ndjson: Annotated[bool, typer.Option("--ndjson")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
    rotate_transport_key: Annotated[
        bool,
        typer.Option(
            "--rotate-transport-key",
            help="Issue a new TLS key/certificate while retaining the durable node identity.",
        ),
    ] = False,
) -> None:
    """Pair once, auto-configure the worker, and start its persistent agent."""

    try:
        require_confirmation(
            "Join the cluster and start its worker service",
            yes=yes,
            json_output=json_output,
            ndjson=ndjson,
        )
        invitation = PairingInvitation.parse(pairing_uri)
        state, platform, runtime, services = build_context(state_root)
        metadata = (
            _pending_metadata(
                state,
                invitation,
                rotate_transport_key=True,
            )
            if rotate_transport_key
            else _pending_metadata(state, invitation)
        )
        result = asyncio.run(_pair(state, pairing_uri, metadata))
        configuration = runtime.prepare_configuration(
            node_id=metadata.node_id,
            cluster=result.cluster,
            service_mode="foreground" if foreground else platform.service_mode,
        )
        if ndjson:
            emit_document(
                {
                    "event": "pairing-consumed",
                    "cluster_id": result.cluster.cluster_id,
                    "node_id": metadata.node_id,
                },
                json_output=False,
                ndjson=True,
            )

        def output_status(status: Any) -> None:
            emit_document(
                {
                    "status": status.state,
                    "reason": status.reason,
                    "cluster_id": result.cluster.cluster_id,
                    "node_id": metadata.node_id,
                    "worker_id": runtime.worker_id(configuration),
                    "backend": configuration.backend_selection.selected_backend.value,
                    "device": configuration.backend_selection.selected_device,
                    "dtype": configuration.backend_selection.selected_dtype,
                    "memory_limit_bytes": configuration.memory_budget.limit_bytes,
                    "storage_limit_bytes": configuration.storage_limit_bytes,
                    "control_endpoint": configuration.endpoints.control_advertised_endpoint,
                    "data_endpoint": configuration.endpoints.data_advertised_endpoint,
                    "service_mode": configuration.service_mode,
                },
                json_output=json_output,
                ndjson=ndjson,
            )

        if foreground:
            status = asyncio.run(
                _foreground_agent(
                    state,
                    roles={"worker"},
                    on_started=output_status,
                )
            )
        else:
            definition = service_definition(state, result.cluster, roles={"worker"})

            async def install() -> Any:
                installed = await services.install(definition)
                if not installed.installed:
                    raise RuntimeError(
                        f"user service installation failed: {installed.detail}; "
                        "retry with --foreground"
                    )
                return await wait_for_runtime(state, metadata.node_id)

            status = asyncio.run(install())
            output_status(status)
        if status.state in {"blocked", "failed"}:
            raise RuntimeError(status.reason or "joined node did not become ready")
    except (OSError, RuntimeError, ValueError, PermissionError) as exc:
        fail("node-join", exc)


@node_app.command("status")
def status_command(
    state_root: Annotated[Path | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    try:
        state, _, _, services = build_context(state_root)
        identity = state.load_or_create_node_identity()
        node_id = node_id_from_fingerprint(identity.public_key_fingerprint)
        cluster = state.load_cluster()
        runtime_status = state.load_runtime(node_id)
        metadata = state.node(node_id)
        configuration = state.load_node_configuration()
        service_status = None
        if cluster is not None:
            definition = service_definition(
                state,
                cluster,
                roles=(
                    {"coordinator", "worker"} if cluster.coordinator_id == node_id else {"worker"}
                ),
            )
            service_status = asyncio.run(services.status(definition))
        emit_document(
            {
                "node_id": node_id,
                "cluster": cluster.model_dump(mode="json") if cluster else None,
                "metadata": metadata.model_dump(mode="json") if metadata else None,
                "runtime": runtime_status.model_dump(mode="json") if runtime_status else None,
                "configuration": (configuration.model_dump(mode="json") if configuration else None),
                "service": service_status.model_dump(mode="json") if service_status else None,
            },
            json_output=json_output,
        )
    except (OSError, RuntimeError, ValueError, PermissionError) as exc:
        fail("node-status", exc)


@node_app.command("configure")
def configure_command(
    state_root: Annotated[Path | None, typer.Option()] = None,
    backend: Annotated[Backend | None, typer.Option()] = None,
    memory_limit: Annotated[str | None, typer.Option("--memory-limit")] = None,
    memory_percent: Annotated[float | None, typer.Option(min=1, max=100)] = None,
    storage_limit: Annotated[str | None, typer.Option("--storage-limit")] = None,
    control_endpoint: Annotated[str | None, typer.Option()] = None,
    data_endpoint: Annotated[str | None, typer.Option()] = None,
    interface: Annotated[str | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    """Persist reviewed backend, memory, storage, interface, or endpoint overrides."""

    try:
        require_confirmation(
            "Change node configuration and restart its service if installed",
            yes=yes,
            json_output=json_output,
        )
        state, platform, runtime, services = build_context(state_root)
        cluster = state.load_cluster()
        if cluster is None:
            raise RuntimeError("node is not paired with a cluster")
        identity = state.load_or_create_node_identity()
        node_id = node_id_from_fingerprint(identity.public_key_fingerprint)
        previous = state.load_node_configuration()
        configuration = runtime.prepare_configuration(
            node_id=node_id,
            cluster=cluster,
            backend_override=backend,
            memory_limit_override_bytes=_parse_bytes(memory_limit),
            memory_percent_override=memory_percent,
            storage_limit_bytes=_parse_bytes(storage_limit),
            control_endpoint_override=control_endpoint,
            data_endpoint_override=data_endpoint,
            interface_override=interface,
            service_mode=previous.service_mode if previous else platform.service_mode,
        )
        definition = service_definition(
            state,
            cluster,
            roles=({"coordinator", "worker"} if cluster.coordinator_id == node_id else {"worker"}),
        )

        async def restart() -> None:
            status = await services.status(definition)
            if status.installed:
                await services.stop(definition)
                restarted = await services.start(definition)
                if not restarted.running:
                    raise RuntimeError(restarted.detail)

        asyncio.run(restart())
        emit_document(configuration.model_dump(mode="json"), json_output=json_output)
    except (OSError, RuntimeError, ValueError, PermissionError) as exc:
        fail("node-configure", exc)


@node_app.command("doctor")
def doctor_command(
    state_root: Annotated[Path | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Probe real device operation, platform support, state, and service tooling."""

    try:
        state, platform, runtime, _ = build_context(state_root)
        report, _ = runtime.select_backend()
        diagnostics = platform.diagnostics()
        platform_identity = platform.identity()
        configuration = state.load_node_configuration()
        metadata = state.node(configuration.node_id) if configuration is not None else None
        retained = list(metadata.backend_validations if metadata is not None else [])
        validation_records = []
        for candidate in report.candidates:
            matching = [
                item
                for item in retained
                if item.backend == candidate.backend
                and item.platform_system == platform_identity.system
                and item.platform_release == platform_identity.release
                and item.platform_architecture.lower() == platform_identity.architecture.lower()
            ]
            validation_records.extend(
                matching
                or [
                    BackendValidationRecord.not_run(
                        backend=candidate.backend,
                        platform=platform_identity,
                    )
                ]
            )
        payload = {
            "status": (
                "pass"
                if all(item.status in {"pass", "warning"} for item in diagnostics)
                else "fail"
            ),
            "platform": platform_identity.model_dump(mode="json"),
            "backend_validation": [item.model_dump(mode="json") for item in validation_records],
            "diagnostics": [item.model_dump(mode="json") for item in diagnostics],
            "backend_selection": report.model_dump(mode="json"),
            "state_root": str(state.paths.root),
            "cluster_configured": state.load_cluster() is not None,
        }
        emit_document(payload, json_output=json_output)
        if payload["status"] == "fail":
            raise typer.Exit(12)
    except typer.Exit:
        raise
    except (OSError, RuntimeError, ValueError, PermissionError) as exc:
        fail("node-doctor", exc)


def _owned_firewall(state: ClusterStateStore) -> FirewallRuleSpec | None:
    cluster = state.load_cluster()
    configuration = state.load_node_configuration()
    if cluster is None or configuration is None:
        return None
    probe = configuration.endpoints.probe_advertised_endpoint
    return FirewallRuleSpec(
        cluster_id=cluster.cluster_id,
        node_id=configuration.node_id,
        control_ports=[split_endpoint(configuration.endpoints.control_advertised_endpoint)[1]],
        data_ports=[
            split_endpoint(configuration.endpoints.data_advertised_endpoint)[1],
            *([split_endpoint(probe)[1]] if probe else []),
        ],
        private_subnets=["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"],
    )


@node_app.command("leave")
def leave_command(
    state_root: Annotated[Path | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    """Authenticate departure, remove owned service/firewall state, preserve identity history."""

    try:
        require_confirmation(
            "Leave the cluster and remove owned service and firewall state",
            yes=yes,
            json_output=json_output,
        )
        state, _, _, services = build_context(state_root)
        cluster = state.load_cluster()
        if cluster is None:
            raise RuntimeError("node is not paired with a cluster")
        identity = state.load_or_create_node_identity()
        node_id = node_id_from_fingerprint(identity.public_key_fingerprint)
        membership = state.membership(node_id)
        if membership is None:
            raise RuntimeError("node has no durable membership")
        body = {"node_id": node_id}
        request = NodeLeaveRequest(
            authentication=create_cluster_authentication(
                identity=identity,
                node_id=node_id,
                action="node-leave",
                body=body,
            ),
            node_id=node_id,
        )
        definition = service_definition(
            state,
            cluster,
            roles=({"coordinator", "worker"} if cluster.coordinator_id == node_id else {"worker"}),
        )
        firewall = _owned_firewall(state)

        async def leave() -> Any:
            client = CoordinatorClient(
                cluster.coordinator_endpoint,
                timeout_s=15.0,
                tls=state.coordinator_tls_client_config(),
            )
            try:
                response = await client.node_leave(request)
            finally:
                await client.close()
            await services.stop(definition)
            await services.uninstall(definition)
            if firewall is not None:
                await services.remove_firewall(firewall)
            return response

        response = asyncio.run(leave())
        state.save_membership(membership.model_copy(update={"status": "left"}))
        emit_document(response.model_dump(mode="json"), json_output=json_output)
    except (OSError, RuntimeError, ValueError, PermissionError) as exc:
        fail("node-leave", exc)


def _service_operation(
    operation: str,
    *,
    state_root: Path | None,
    json_output: bool,
    yes: bool,
) -> None:
    state, _, _, services = build_context(state_root)
    cluster = state.load_cluster()
    if cluster is None:
        raise RuntimeError("node is not paired; create or join a cluster first")
    require_confirmation(
        f"{operation.capitalize()} the cluster node service",
        yes=yes,
        json_output=json_output,
    )
    identity = state.load_or_create_node_identity()
    node_id = node_id_from_fingerprint(identity.public_key_fingerprint)
    definition = service_definition(
        state,
        cluster,
        roles={"coordinator", "worker"} if cluster.coordinator_id == node_id else {"worker"},
    )
    method = services.install if operation == "install" else services.uninstall
    status = asyncio.run(method(definition))
    emit_document(status.model_dump(mode="json"), json_output=json_output)
    if operation == "install" and not status.installed:
        raise RuntimeError(status.detail)


@node_app.command("install-service")
def install_service_command(
    state_root: Annotated[Path | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    try:
        _service_operation(
            "install",
            state_root=state_root,
            json_output=json_output,
            yes=yes,
        )
    except (OSError, RuntimeError, ValueError, PermissionError) as exc:
        fail("service-install", exc)


@node_app.command("uninstall-service")
def uninstall_service_command(
    state_root: Annotated[Path | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    try:
        _service_operation(
            "uninstall",
            state_root=state_root,
            json_output=json_output,
            yes=yes,
        )
    except (OSError, RuntimeError, ValueError, PermissionError) as exc:
        fail("service-uninstall", exc)


@node_app.command("agent", hidden=True)
def agent_command(
    state_root: Annotated[Path | None, typer.Option()] = None,
    role: Annotated[list[str] | None, typer.Option("--role")] = None,
) -> None:
    """Internal service entrypoint; it owns no inference implementation."""

    try:
        values = set(role or ["worker"])
        invalid = values - {"coordinator", "worker"}
        if invalid:
            raise ValueError(f"invalid node-agent roles: {sorted(invalid)}")
        state = ClusterStateStore(state_root)
        status = asyncio.run(
            _foreground_agent(
                state,
                roles={cast(NodeAgentRole, item) for item in values},
            )
        )
        if status.state in {"failed", "blocked"}:
            raise RuntimeError(status.reason or "node agent stopped without readiness")
    except (OSError, RuntimeError, ValueError, PermissionError) as exc:
        fail("node-agent", exc)


@node_app.command("update")
def update_command(
    source_wheel: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    state_root: Annotated[Path | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    """Developer/offline recovery: update a non-native runtime from an explicit wheel."""

    try:
        from swarm_inference.native_install import native_install_record

        if native_install_record() is not None:
            raise RuntimeError(
                "native-windows runtime is installer-owned; use `swarm update` for normal updates"
            )
        require_confirmation(
            "Update the node runtime and restart its service",
            yes=yes,
            json_output=json_output,
        )
        from swarm_inference.cluster.updates import RuntimeUpdater

        state, platform, _, services = build_context(state_root)
        updater = RuntimeUpdater(
            state=state,
            platform=platform,
            services=services,
        )
        cluster = state.load_cluster()
        if cluster is None:
            raise RuntimeError("node is not paired with a cluster")
        identity = state.load_or_create_node_identity()
        node_id = node_id_from_fingerprint(identity.public_key_fingerprint)
        definition = service_definition(
            state,
            cluster,
            roles=({"coordinator", "worker"} if cluster.coordinator_id == node_id else {"worker"}),
        )
        result = asyncio.run(updater.update(source_wheel, definition))
        emit_document(result.model_dump(mode="json"), json_output=json_output)
        if result.status != "committed":
            raise RuntimeError(result.detail)
    except (OSError, RuntimeError, ValueError, PermissionError) as exc:
        fail("node-update", exc)


__all__ = ["node_app"]
