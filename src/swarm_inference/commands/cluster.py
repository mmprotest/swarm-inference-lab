"""High-level cluster lifecycle commands backed by the canonical runtimes."""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import time
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import typer

from swarm_inference import __version__
from swarm_inference.cluster.agent import NodeAgent, NodeAgentOptions
from swarm_inference.cluster.models import (
    SECURE_WAN_SECURITY_CLASSIFICATION,
    ClusterAuditEvent,
    ClusterMetadata,
    NodeMembership,
    NodeMetadata,
    VersionCompatibility,
    node_id_from_fingerprint,
)
from swarm_inference.cluster.pairing import create_cluster_authentication
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
from swarm_inference.coordinator.service import CoordinatorClient
from swarm_inference.platforms.base import FirewallRuleSpec
from swarm_inference.protocol.cluster import (
    ClusterRevokeRequest,
    ClusterStatusRequest,
    PairingCreateRequest,
    PairingCreateResponse,
    PairingDeliveryResult,
)
from swarm_inference.runtime.shutdown import (
    install_shutdown_signal_handlers,
    wait_for_service_shutdown,
)
from swarm_inference.security.identity import CoordinatorIdentity
from swarm_inference.security.tls import (
    certificate_sha256,
    create_cluster_ca_certificate,
    issue_node_certificate,
    tls_public_key_pem,
)
from swarm_inference.security.trust_store import WorkerTrustStore

cluster_app = typer.Typer(
    name="cluster",
    help="Create and operate a persistent authenticated swarm cluster.",
    no_args_is_help=True,
)


def _auth_context(state: ClusterStateStore) -> tuple[ClusterMetadata, str, Any]:
    cluster = state.load_cluster()
    if cluster is None:
        raise RuntimeError("no local cluster metadata; run 'swarm cluster create' or join")
    identity = state.load_or_create_node_identity()
    node_id = node_id_from_fingerprint(identity.public_key_fingerprint)
    membership = state.membership(node_id)
    if membership is None or membership.status != "active":
        raise PermissionError("local node has no active cluster membership")
    return cluster, node_id, identity


async def _create_pairing(
    state: ClusterStateStore,
    *,
    ttl_seconds: int,
) -> PairingCreateResponse:
    cluster, node_id, identity = _auth_context(state)
    body = {"ttl_seconds": ttl_seconds}
    request = PairingCreateRequest(
        authentication=create_cluster_authentication(
            identity=identity,
            node_id=node_id,
            action="pairing-create",
            body=body,
        ),
        ttl_seconds=ttl_seconds,
    )
    client = CoordinatorClient(
        cluster.coordinator_endpoint,
        timeout_s=15.0,
        tls=state.coordinator_tls_client_config(),
    )
    try:
        return await client.pairing_create(request)
    finally:
        await client.close()


def _bootstrap_cluster(
    state: ClusterStateStore,
    *,
    name: str,
    endpoint_override: str | None,
) -> tuple[ClusterMetadata, NodeMetadata]:
    state.adopt_legacy_coordinator_identity(Path(".swarm/coordinator"))
    coordinator_identity = CoordinatorIdentity.load_or_create(state.paths.coordinator_identity)
    node_identity = state.load_or_create_node_identity()
    node_id = node_id_from_fingerprint(node_identity.public_key_fingerprint)
    existing = state.load_cluster()
    if existing is not None:
        if existing.name != name:
            raise ValueError(
                f"state already belongs to cluster {existing.name!r}; refusing replacement"
            )
        if existing.coordinator_id != node_id:
            raise PermissionError("only the pinned coordinator node can adopt cluster state")
        node = state.node(node_id)
        if node is None:
            raise RuntimeError("coordinator cluster state has no local node metadata")
        return existing, node
    _, platform, runtime, _ = build_context(state.paths.root)
    endpoint = runtime.select_coordinator_endpoint(
        node_id=node_id,
        override=endpoint_override,
    )
    now = time.time_ns()
    cluster_id = f"cluster-{uuid4().hex[:12]}"
    coordinator_certificate = create_cluster_ca_certificate(
        coordinator_identity,
        cluster_id=cluster_id,
    )
    cluster = ClusterMetadata(
        cluster_id=cluster_id,
        name=name,
        coordinator_id=node_id,
        coordinator_endpoint=endpoint,
        coordinator_public_key=coordinator_identity.public_key_b64,
        coordinator_fingerprint=coordinator_identity.public_key_fingerprint,
        coordinator_certificate_pem=coordinator_certificate,
        coordinator_certificate_sha256=certificate_sha256(coordinator_certificate),
        security_classification=SECURE_WAN_SECURITY_CLASSIFICATION,
        created_at_unix_ns=now,
        runtime_compatibility=VersionCompatibility(
            minimum_runtime_version=__version__,
            maximum_runtime_version_exclusive="1.0.0",
        ),
    )
    build_id, lock_hash = bootstrap_package_identity()
    platform_identity = platform.identity()
    node_tls_private_key = state.prepare_node_tls_private_key(rotate=True)
    node_tls_public_key = tls_public_key_pem(node_tls_private_key)
    node_certificate = issue_node_certificate(
        coordinator_identity,
        ca_certificate_pem=coordinator_certificate,
        cluster_id=cluster.cluster_id,
        node_public_key_b64=node_identity.public_key_b64,
        node_fingerprint=node_identity.public_key_fingerprint,
        node_tls_public_key_pem=node_tls_public_key,
    )
    node = NodeMetadata(
        node_id=node_id,
        public_key=node_identity.public_key_b64,
        fingerprint=node_identity.public_key_fingerprint,
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
        tls_certificate_pem=node_certificate,
        tls_certificate_sha256=certificate_sha256(node_certificate),
        tls_public_key_pem=node_tls_public_key,
    )
    membership = NodeMembership(
        cluster_id=cluster.cluster_id,
        node_id=node_id,
        node_public_key=node.public_key,
        node_fingerprint=node.fingerprint,
        coordinator_public_key=cluster.coordinator_public_key,
        coordinator_fingerprint=cluster.coordinator_fingerprint,
        joined_at_unix_ns=now,
    )
    state.materialize_coordinator_tls(coordinator_identity, coordinator_certificate)
    state.materialize_node_tls(
        node_identity,
        certificate_pem=node_certificate,
        ca_certificate_pem=coordinator_certificate,
    )
    state.save_cluster(cluster)
    state.save_node(node)
    state.save_membership(membership)
    WorkerTrustStore(state.paths.security / "trusted-workers.json").trust(
        node.fingerprint,
        label=f"{cluster.name} coordinator node",
        notes="automatically trusted during cluster creation",
    )
    state.append_audit(
        ClusterAuditEvent(
            event_id=uuid4().hex,
            event_type="cluster_created",
            timestamp_unix_ns=now,
            cluster_id=cluster.cluster_id,
            node_id=node_id,
            detail="TLS 1.3 cluster trust and durable node identities created",
        )
    )
    return cluster, node


async def _foreground_create(
    state: ClusterStateStore,
    *,
    ttl_seconds: int,
) -> tuple[Any, PairingCreateResponse, NodeAgent]:
    _, platform, runtime, _ = build_context(state.paths.root)
    agent = NodeAgent(
        state=state,
        platform=platform,
        runtime_manager=runtime,
        options=NodeAgentOptions(roles={"coordinator", "worker"}),
    )
    status = await agent.start()
    pairing = await _create_pairing(state, ttl_seconds=ttl_seconds)
    return status, pairing, agent


def _preflight_pairing_output(
    pairing_output: Path | None,
    *,
    machine_readable: bool,
    force: bool,
) -> None:
    if pairing_output is None:
        return
    if str(pairing_output) == "-":
        if machine_readable:
            raise ValueError("--pairing-output - is forbidden in machine-readable mode")
        return
    destination = pairing_output.expanduser().resolve()
    if not force and os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite existing file: {destination}")


def _deliver_pairing(
    state: ClusterStateStore,
    pairing: PairingCreateResponse,
    *,
    json_output: bool,
    pairing_output: Path | None,
    force: bool,
    cluster_name: str | None = None,
) -> PairingDeliveryResult:
    if pairing_output is not None and str(pairing_output) == "-":
        # This is one of only two intentional secret-bearing outputs. It is
        # human-only and is never represented by the returned public model.
        _emit_join_command(pairing.pairing_uri, cluster_name=cluster_name)
        return PairingDeliveryResult(
            session_id=pairing.session_id,
            expires_at_unix_ns=pairing.expires_at_unix_ns,
            redacted_uri=pairing.redacted_uri,
            delivery="interactive",
        )
    if json_output or pairing_output is not None:
        destination, protection, limitation = state.write_pairing_invitation(
            session_id=pairing.session_id,
            pairing_uri=pairing.pairing_uri,
            output_path=pairing_output,
            force=force,
        )
        return PairingDeliveryResult(
            session_id=pairing.session_id,
            expires_at_unix_ns=pairing.expires_at_unix_ns,
            redacted_uri=pairing.redacted_uri,
            delivery="protected-file",
            invitation_file=destination,
            permission_protection=protection,
            permission_limitation=limitation,
        )
    # This is the other intentional secret-bearing output: one complete command
    # for an interactive human with no invitation-file transfer.
    _emit_join_command(pairing.pairing_uri, cluster_name=cluster_name)
    return PairingDeliveryResult(
        session_id=pairing.session_id,
        expires_at_unix_ns=pairing.expires_at_unix_ns,
        redacted_uri=pairing.redacted_uri,
        delivery="interactive",
    )


def _emit_join_command(pairing_uri: str, *, cluster_name: str | None) -> None:
    if any(character in pairing_uri for character in ('"', "\r", "\n")):
        raise ValueError("pairing URI cannot be quoted safely for an interactive command")
    if cluster_name is not None:
        typer.echo(f"Cluster ready: {cluster_name}")
        typer.echo()
    typer.echo("Run this command on the machine joining the cluster:")
    typer.echo()
    typer.echo(f'swarm node join "{pairing_uri}"')


def _create_payload(
    *,
    cluster: ClusterMetadata,
    status: Any,
    service_mode: str,
    delivery: PairingDeliveryResult,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status.state,
        "reason": status.reason,
        "cluster": {
            "cluster_id": cluster.cluster_id,
            "name": cluster.name,
            "coordinator_endpoint": cluster.coordinator_endpoint,
            "security_boundary": cluster.security_classification,
        },
        "service": {"mode": service_mode},
        "pairing": delivery.model_dump(mode="json", exclude_none=True),
    }


@cluster_app.command("create")
def create_command(
    name: Annotated[str, typer.Option("--name", help="Stable human cluster name.")],
    state_root: Annotated[Path | None, typer.Option(help="Override product state root.")] = None,
    coordinator_endpoint: Annotated[
        str | None,
        typer.Option(help="Routed advertised coordinator endpoint override."),
    ] = None,
    ttl_seconds: Annotated[int, typer.Option(min=1, max=3600)] = 600,
    foreground: Annotated[
        bool,
        typer.Option("--foreground", help="Run the agent in this terminal."),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
    pairing_output: Annotated[
        Path | None,
        typer.Option("--pairing-output", help="Protected destination for the one-time URI."),
    ] = None,
    force_pairing_output: Annotated[
        bool,
        typer.Option("--force-pairing-output", help="Replace an existing invitation file."),
    ] = False,
) -> None:
    """Create/adopt coordinator state, start its agent, and issue one invitation."""

    try:
        _preflight_pairing_output(
            pairing_output,
            machine_readable=json_output,
            force=force_pairing_output,
        )
        require_confirmation(
            "Create cluster state and start its node service",
            yes=yes,
            json_output=json_output,
        )
        state, platform, runtime, services = build_context(state_root)
        cluster, node = _bootstrap_cluster(
            state,
            name=name,
            endpoint_override=coordinator_endpoint,
        )
        runtime.prepare_configuration(
            node_id=node.node_id,
            cluster=cluster,
            service_mode="foreground" if foreground else platform.service_mode,
        )
        if foreground:

            async def run() -> None:
                status, pairing, agent = await _foreground_create(
                    state,
                    ttl_seconds=ttl_seconds,
                )
                delivery = _deliver_pairing(
                    state,
                    pairing,
                    json_output=json_output,
                    pairing_output=pairing_output,
                    force=force_pairing_output,
                    cluster_name=cluster.name,
                )
                if json_output or pairing_output is not None:
                    emit_document(
                        _create_payload(
                            cluster=cluster,
                            status=status,
                            service_mode="foreground",
                            delivery=delivery,
                        ),
                        json_output=json_output,
                    )
                if status.state in {"blocked", "failed"}:
                    await agent.stop()
                    raise RuntimeError(status.reason or "node agent did not become ready")
                stop_event = asyncio.Event()
                restore = install_shutdown_signal_handlers(stop_event)
                try:
                    await wait_for_service_shutdown(
                        agent.wait(),
                        stop_event,
                        shutdown=agent.stop,
                    )
                finally:
                    restore()
                    await agent.stop()

            asyncio.run(run())
            return
        definition = service_definition(
            state,
            cluster,
            roles={"coordinator", "worker"},
        )

        async def install_and_pair() -> tuple[Any, PairingCreateResponse]:
            installed = await services.install(definition)
            if not installed.installed:
                raise RuntimeError(
                    f"user service installation failed: {installed.detail}; retry with --foreground"
                )
            status = await wait_for_runtime(state, node.node_id)
            pairing = await _create_pairing(state, ttl_seconds=ttl_seconds)
            return status, pairing

        status, pairing = asyncio.run(install_and_pair())
        delivery = _deliver_pairing(
            state,
            pairing,
            json_output=json_output,
            pairing_output=pairing_output,
            force=force_pairing_output,
            cluster_name=cluster.name,
        )
        if json_output or pairing_output is not None:
            emit_document(
                _create_payload(
                    cluster=cluster,
                    status=status,
                    service_mode=platform.service_mode,
                    delivery=delivery,
                ),
                json_output=json_output,
            )
        if status.state in {"blocked", "failed"}:
            raise RuntimeError(status.reason or "coordinator node did not become ready")
    except (OSError, RuntimeError, ValueError, PermissionError) as exc:
        fail("cluster-create", exc)


@cluster_app.command("pair")
def pair_command(
    state_root: Annotated[Path | None, typer.Option()] = None,
    ttl_seconds: Annotated[int, typer.Option(min=1, max=3600)] = 600,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    pairing_output: Annotated[
        Path | None,
        typer.Option("--pairing-output", help="Protected destination for the one-time URI."),
    ] = None,
    force_pairing_output: Annotated[
        bool,
        typer.Option("--force-pairing-output", help="Replace an existing invitation file."),
    ] = False,
) -> None:
    """Create a bounded, single-use pairing invitation on the coordinator."""

    try:
        _preflight_pairing_output(
            pairing_output,
            machine_readable=json_output,
            force=force_pairing_output,
        )
        state = ClusterStateStore(state_root)
        pairing = asyncio.run(_create_pairing(state, ttl_seconds=ttl_seconds))
        cluster = state.load_cluster()
        delivery = _deliver_pairing(
            state,
            pairing,
            json_output=json_output,
            pairing_output=pairing_output,
            force=force_pairing_output,
            cluster_name=cluster.name if cluster is not None else None,
        )
        if json_output or pairing_output is not None:
            emit_document(
                {
                    "schema_version": 1,
                    "status": "ready",
                    "pairing": delivery.model_dump(mode="json", exclude_none=True),
                },
                json_output=json_output,
            )
    except (OSError, RuntimeError, ValueError, PermissionError) as exc:
        fail("pairing-create", exc)


async def _status(state: ClusterStateStore) -> Any:
    cluster, node_id, identity = _auth_context(state)
    body = {"include_artifacts": True, "include_network": True}
    request = ClusterStatusRequest(
        authentication=create_cluster_authentication(
            identity=identity,
            node_id=node_id,
            action="cluster-status",
            body=body,
        )
    )
    client = CoordinatorClient(
        cluster.coordinator_endpoint,
        timeout_s=15.0,
        tls=state.coordinator_tls_client_config(),
    )
    try:
        return await client.cluster_status(request)
    finally:
        await client.close()


@cluster_app.command("status")
def status_command(
    state_root: Annotated[Path | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Report node, reachability, link, artifact, and runtime evidence."""

    try:
        response = asyncio.run(_status(ClusterStateStore(state_root)))
        emit_document(response.model_dump(mode="json"), json_output=json_output)
    except (OSError, RuntimeError, ValueError, PermissionError) as exc:
        fail("cluster-status", exc)


@cluster_app.command("nodes")
def nodes_command(
    state_root: Annotated[Path | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List every paired node and its current role or exclusion reason."""

    try:
        response = asyncio.run(_status(ClusterStateStore(state_root)))
        payload = {
            "cluster_id": response.cluster.cluster_id,
            "nodes": [item.model_dump(mode="json") for item in response.nodes],
        }
        emit_document(payload, json_output=json_output)
    except (OSError, RuntimeError, ValueError, PermissionError) as exc:
        fail("cluster-nodes", exc)


@cluster_app.command("revoke")
def revoke_command(
    node_id_to_revoke: Annotated[str, typer.Argument(metavar="NODE_ID")],
    reason: Annotated[str, typer.Option()] = "revoked by cluster coordinator",
    state_root: Annotated[Path | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    """Remove node trust for registration, deployment, and recovery."""

    try:
        require_confirmation(
            "Revoke cluster node trust",
            yes=yes,
            json_output=json_output,
        )
        state = ClusterStateStore(state_root)
        cluster, node_id, identity = _auth_context(state)
        body = {"node_id": node_id_to_revoke, "reason": reason}
        request = ClusterRevokeRequest(
            authentication=create_cluster_authentication(
                identity=identity,
                node_id=node_id,
                action="cluster-revoke",
                body=body,
            ),
            node_id=node_id_to_revoke,
            reason=reason,
        )

        async def revoke() -> Any:
            client = CoordinatorClient(
                cluster.coordinator_endpoint,
                timeout_s=15.0,
                tls=state.coordinator_tls_client_config(),
            )
            try:
                return await client.cluster_revoke(request)
            finally:
                await client.close()

        response = asyncio.run(revoke())
        emit_document(response.model_dump(mode="json"), json_output=json_output)
        if not response.revoked:
            raise typer.Exit(3)
    except typer.Exit:
        raise
    except (OSError, RuntimeError, ValueError, PermissionError) as exc:
        fail("cluster-revoke", exc, node_id=node_id_to_revoke)


def _definition_from_state(state: ClusterStateStore) -> tuple[Any, Any, Any]:
    cluster, node_id, _ = _auth_context(state)
    platform = build_context(state.paths.root)[1]
    roles = {"coordinator", "worker"} if node_id == cluster.coordinator_id else {"worker"}
    return platform, cluster, service_definition(state, cluster, roles=roles)


@cluster_app.command("start")
def start_command(
    state_root: Annotated[Path | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    try:
        require_confirmation("Start the cluster node service", yes=yes, json_output=json_output)
        state = ClusterStateStore(state_root)
        _, _, _, services = build_context(state.paths.root)
        _, _, definition = _definition_from_state(state)
        status = asyncio.run(services.start(definition))
        emit_document(status.model_dump(mode="json"), json_output=json_output)
        if not status.running:
            raise RuntimeError(status.detail)
    except (OSError, RuntimeError, ValueError, PermissionError) as exc:
        fail("cluster-start", exc)


@cluster_app.command("stop")
def stop_command(
    state_root: Annotated[Path | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    try:
        require_confirmation("Stop the cluster node service", yes=yes, json_output=json_output)
        state = ClusterStateStore(state_root)
        _, _, _, services = build_context(state.paths.root)
        _, _, definition = _definition_from_state(state)
        status = asyncio.run(services.stop(definition))
        emit_document(status.model_dump(mode="json"), json_output=json_output)
    except (OSError, RuntimeError, ValueError, PermissionError) as exc:
        fail("cluster-stop", exc)


@cluster_app.command("delete")
def delete_command(
    state_root: Annotated[Path | None, typer.Option()] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Confirm local cluster deletion.")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Uninstall this node service and remove only its explicit state root."""

    try:
        require_confirmation(
            "Delete local cluster service, firewall rules, and state",
            yes=yes,
            json_output=json_output,
        )
        state = ClusterStateStore(state_root)
        root = state.paths.root.resolve()
        if root == Path(root.anchor) or root == Path.home().resolve():
            raise ValueError("refusing to delete a broad filesystem root")
        _, cluster, definition = _definition_from_state(state)
        _, _, _, services = build_context(root)
        configuration = state.load_node_configuration()

        async def remove() -> None:
            await services.stop(definition)
            await services.uninstall(definition)
            if configuration is not None:
                from swarm_inference.host import split_endpoint

                await services.remove_firewall(
                    FirewallRuleSpec(
                        cluster_id=cluster.cluster_id,
                        node_id=configuration.node_id,
                        control_ports=[
                            split_endpoint(configuration.endpoints.control_advertised_endpoint)[1]
                        ],
                        data_ports=[
                            split_endpoint(configuration.endpoints.data_advertised_endpoint)[1]
                        ],
                        private_subnets=[
                            "10.0.0.0/8",
                            "172.16.0.0/12",
                            "192.168.0.0/16",
                        ],
                    )
                )

        asyncio.run(remove())
        shutil.rmtree(root)
        emit_document(
            {"status": "deleted", "state_root": str(root), "recoverable": False},
            json_output=json_output,
        )
    except (OSError, RuntimeError, ValueError, PermissionError) as exc:
        fail("cluster-delete", exc)


__all__ = ["cluster_app"]
