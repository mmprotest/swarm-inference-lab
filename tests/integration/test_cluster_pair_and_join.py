from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from swarm_inference.cluster.models import (
    ClusterMetadata,
    NodeMetadata,
    VersionCompatibility,
    node_id_from_fingerprint,
)
from swarm_inference.cluster.pairing import (
    PairingClient,
    PairingManager,
    create_cluster_authentication,
)
from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.config.product import ProductCoordinatorConfig
from swarm_inference.coordinator.service import (
    CoordinatorClient,
    CoordinatorCore,
    CoordinatorRpcServer,
)
from swarm_inference.protocol.cluster import (
    ClusterStatusRequest,
    NodeUpdateRequest,
    ReachabilityCheckRequest,
)
from swarm_inference.security.identity import WorkerIdentity
from swarm_inference.security.trust_store import WorkerTrustStore


def _node(identity: WorkerIdentity) -> NodeMetadata:
    node_id = node_id_from_fingerprint(identity.public_key_fingerprint)
    now = time.time_ns()
    return NodeMetadata(
        node_id=node_id,
        public_key=identity.public_key_b64,
        fingerprint=identity.public_key_fingerprint,
        hostname="rpc-node",
        operating_system="test",
        architecture="test",
        agent_version="0.1.0",
        runtime_version="0.1.0",
        build_id="rpc-test",
        package_lock_hash="4" * 64,
        worker_ids=[],
        joined_at_unix_ns=now,
        last_seen_at_unix_ns=now,
    )


@pytest.mark.asyncio
async def test_pair_and_join_over_canonical_coordinator_rpc(tmp_path: Path) -> None:
    state = ClusterStateStore(tmp_path / "cluster")
    trust_path = state.paths.security / "trusted-workers.json"
    core = CoordinatorCore(
        product_config=ProductCoordinatorConfig(trust_store_path=trust_path),
        state_directory=state.paths.coordinator_runtime_directory,
    )
    server = CoordinatorRpcServer(core)
    port = await server.start("127.0.0.1:0")
    endpoint = f"127.0.0.1:{port}"
    assert core.coordinator_identity is not None
    cluster = ClusterMetadata(
        cluster_id="cluster-rpc",
        name="rpc-test",
        coordinator_id=node_id_from_fingerprint(core.coordinator_identity.public_key_fingerprint),
        coordinator_endpoint=endpoint,
        coordinator_public_key=core.coordinator_identity.public_key_b64,
        coordinator_fingerprint=core.coordinator_identity.public_key_fingerprint,
        created_at_unix_ns=time.time_ns(),
        runtime_compatibility=VersionCompatibility(
            minimum_runtime_version="0.1.0",
            maximum_runtime_version_exclusive="0.2.0",
        ),
    )
    state.save_cluster(cluster)
    manager = PairingManager(
        state=state,
        trust_store=WorkerTrustStore(trust_path),
        coordinator_identity=core.coordinator_identity,
        cluster=cluster,
    )
    core.attach_cluster_control(manager)
    invitation = await manager.create_session(endpoint)

    identity = WorkerIdentity.load_or_create(tmp_path / "node" / "identity.json")
    node_state = ClusterStateStore(tmp_path / "node" / "state")
    pairing_client = PairingClient(state=node_state, identity=identity)
    rpc = CoordinatorClient(endpoint)
    try:
        result = await pairing_client.join(
            invitation.uri(),
            node_metadata=_node(identity),
            hello_rpc=rpc.pairing_hello,
            complete_rpc=rpc.pairing_complete,
        )

        async def accept_and_close(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            del reader
            writer.close()
            await writer.wait_closed()

        control_listener = await asyncio.start_server(accept_and_close, "127.0.0.1", 0)
        data_listener = await asyncio.start_server(accept_and_close, "127.0.0.1", 0)
        control_port = control_listener.sockets[0].getsockname()[1]
        data_port = data_listener.sockets[0].getsockname()[1]
        metadata = node_state.node(result.membership.node_id)
        assert metadata is not None
        metadata = metadata.model_copy(
            update={
                "control_endpoint": f"127.0.0.1:{control_port}",
                "data_endpoint": f"127.0.0.1:{data_port}",
                "worker_ids": [f"{metadata.node_id}/cpu-0"],
            }
        )
        update_body = {"metadata": metadata.model_dump(mode="json")}
        update_auth = create_cluster_authentication(
            identity=identity,
            node_id=metadata.node_id,
            action="node-update",
            body=update_body,
        )
        update = await rpc.node_update(
            NodeUpdateRequest(authentication=update_auth, metadata=metadata)
        )
        assert update.accepted
        reach_body = {"node_id": metadata.node_id, "timeout_ms": 3000}
        reach_auth = create_cluster_authentication(
            identity=identity,
            node_id=metadata.node_id,
            action="reachability-check",
            body=reach_body,
        )
        reachability = await rpc.reachability_check(
            ReachabilityCheckRequest(
                authentication=reach_auth,
                node_id=metadata.node_id,
            )
        )
        assert reachability.control_reachable
        assert reachability.data_reachable
        control_listener.close()
        data_listener.close()
        await control_listener.wait_closed()
        await data_listener.wait_closed()

        body = {"include_artifacts": True, "include_network": True}
        authentication = create_cluster_authentication(
            identity=identity,
            node_id=result.membership.node_id,
            action="cluster-status",
            body=body,
        )
        status = await rpc.cluster_status(ClusterStatusRequest(authentication=authentication))
        assert status.cluster.cluster_id == "cluster-rpc"
        assert [item.metadata.node_id for item in status.nodes] == [result.membership.node_id]
        assert manager.trust_store.contains(identity.public_key_fingerprint)
    finally:
        await rpc.close()
        await server.stop(grace_s=0)
