from __future__ import annotations

import time
from pathlib import Path

import pytest

from swarm_inference.cluster.models import (
    ClusterMetadata,
    NodeMembership,
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
from swarm_inference.config.models import (
    Backend,
    OperationKind,
    StageBenchmark,
    WorkerCapability,
)
from swarm_inference.config.product import ProductCoordinatorConfig
from swarm_inference.coordinator.service import (
    CoordinatorClient,
    CoordinatorCore,
    CoordinatorRpcServer,
)
from swarm_inference.exceptions import TransportError
from swarm_inference.protocol.cluster import ClusterRevokeRequest
from swarm_inference.protocol.messages import RegistrationRequest
from swarm_inference.protocol.stage_ring import STAGE_RING_PROTOCOL_VERSION
from swarm_inference.security.identity import WorkerIdentity
from swarm_inference.security.signatures import canonical_json_bytes
from swarm_inference.security.trust_store import WorkerTrustStore


def _metadata(identity: WorkerIdentity) -> NodeMetadata:
    node_id = node_id_from_fingerprint(identity.public_key_fingerprint)
    now = time.time_ns()
    return NodeMetadata(
        node_id=node_id,
        public_key=identity.public_key_b64,
        fingerprint=identity.public_key_fingerprint,
        hostname=node_id,
        operating_system="injected",
        architecture="x86_64",
        agent_version="0.1.0",
        runtime_version="0.1.0",
        build_id="revocation-integration",
        package_lock_hash="e" * 64,
        worker_ids=[f"{node_id}/cpu-0"],
        control_endpoint="127.0.0.1:51001",
        data_endpoint="127.0.0.1:51002",
        joined_at_unix_ns=now,
        last_seen_at_unix_ns=now,
    )


def _capability(identity: WorkerIdentity) -> WorkerCapability:
    node_id = node_id_from_fingerprint(identity.public_key_fingerprint)
    worker_id = f"{node_id}/cpu-0"
    return WorkerCapability(
        worker_id=worker_id,
        node_id=node_id,
        public_key=identity.public_key_b64,
        hostname=node_id,
        operating_system="injected",
        architecture="x86_64",
        backend=Backend.TORCH_CPU,
        cpu_model="injected",
        logical_cpu_count=2,
        total_ram_bytes=1024**3,
        available_ram_bytes=1024**3,
        supported_dtypes=["float32"],
        stage_benchmarks=[
            StageBenchmark(
                worker_class="integration",
                operation=OperationKind.DECODE,
                sequence_length=1,
                batch_size=1,
                mean_ms=1,
                median_ms=1,
                p95_ms=1,
                samples=3,
                measured=True,
                device="cpu",
                dtype="float32",
                measured_at_unix_ns=time.time_ns(),
                measurement_source="selected-device-torch",
            )
        ],
        upload_bandwidth_bytes_s=1_000_000,
        download_bandwidth_bytes_s=1_000_000,
        coordinator_latency_ms=1,
        memory_limit_bytes=1024**3,
        endpoint="127.0.0.1:51001",
        control_endpoint="127.0.0.1:51001",
        data_plane_endpoint="127.0.0.1:51002",
        device_identifier="cpu",
        stage_ring_protocol_version=STAGE_RING_PROTOCOL_VERSION,
        supported_model_adapters=["olmoe"],
        supported_stage_execution_backends=["canonical-contiguous-olmoe"],
        supported_activation_dtypes=["float32"],
        configured_memory_limit_bytes=1024**3,
        stage_runtime_enabled=True,
    )


def _registration(identity: WorkerIdentity) -> RegistrationRequest:
    capability = _capability(identity)
    nonce = "revocation-registration"
    return RegistrationRequest(
        capability=capability,
        benchmark_nonce=nonce,
        signature=identity.sign(
            canonical_json_bytes(
                {
                    "capability": capability.model_dump(mode="json"),
                    "benchmark_nonce": nonce,
                }
            )
        ),
    )


@pytest.mark.asyncio
async def test_revoked_paired_node_cannot_register_or_enter_deployment_trust(
    tmp_path: Path,
) -> None:
    state = ClusterStateStore(tmp_path / "coordinator")
    coordinator_node_identity = WorkerIdentity.load_or_create(state.paths.node_identity)
    coordinator_node_id = node_id_from_fingerprint(coordinator_node_identity.public_key_fingerprint)
    trust_path = state.paths.security / "trusted-workers.json"
    core = CoordinatorCore(
        product_config=ProductCoordinatorConfig(
            coordinator_id=coordinator_node_id,
            trust_store_path=trust_path,
        ),
        state_directory=state.paths.coordinator_runtime_directory,
        coordinator_identity_path=state.paths.coordinator_identity,
    )
    server = CoordinatorRpcServer(core)
    port = await server.start("127.0.0.1:0")
    endpoint = f"127.0.0.1:{port}"
    assert core.coordinator_identity is not None
    cluster = ClusterMetadata(
        cluster_id="cluster-revocation",
        name="revocation",
        coordinator_id=coordinator_node_id,
        coordinator_endpoint=endpoint,
        coordinator_public_key=core.coordinator_identity.public_key_b64,
        coordinator_fingerprint=core.coordinator_identity.public_key_fingerprint,
        created_at_unix_ns=time.time_ns(),
        runtime_compatibility=VersionCompatibility(
            minimum_runtime_version="0.1.0",
            maximum_runtime_version_exclusive="1.0.0",
        ),
    )
    coordinator_membership = NodeMembership(
        cluster_id=cluster.cluster_id,
        node_id=coordinator_node_id,
        node_public_key=coordinator_node_identity.public_key_b64,
        node_fingerprint=coordinator_node_identity.public_key_fingerprint,
        coordinator_public_key=cluster.coordinator_public_key,
        coordinator_fingerprint=cluster.coordinator_fingerprint,
        joined_at_unix_ns=time.time_ns(),
    )
    state.save_cluster(cluster)
    state.save_membership(coordinator_membership)
    trust = WorkerTrustStore(trust_path)
    manager = PairingManager(
        state=state,
        trust_store=trust,
        coordinator_identity=core.coordinator_identity,
        cluster=cluster,
    )
    core.attach_cluster_control(manager)
    invitation = await manager.create_session(endpoint)
    joined_identity = WorkerIdentity.generate()
    joined_state = ClusterStateStore(tmp_path / "joined")
    rpc = CoordinatorClient(endpoint)
    try:
        joined = await PairingClient(state=joined_state, identity=joined_identity).join(
            invitation.uri(),
            node_metadata=_metadata(joined_identity),
            hello_rpc=rpc.pairing_hello,
            complete_rpc=rpc.pairing_complete,
        )
        assert (await rpc.register(_registration(joined_identity))).accepted
        assert core.is_worker_trusted(f"{joined.membership.node_id}/cpu-0")

        body = {"node_id": joined.membership.node_id, "reason": "integration revocation"}
        revoked = await rpc.cluster_revoke(
            ClusterRevokeRequest(
                authentication=create_cluster_authentication(
                    identity=coordinator_node_identity,
                    node_id=coordinator_node_id,
                    action="cluster-revoke",
                    body=body,
                ),
                node_id=joined.membership.node_id,
                reason="integration revocation",
            )
        )
        assert revoked.revoked
        assert not trust.contains(joined_identity.public_key_fingerprint)
        assert not core.is_worker_trusted(f"{joined.membership.node_id}/cpu-0")
        with pytest.raises(TransportError, match=r"not trusted|PERMISSION_DENIED"):
            await rpc.register(_registration(joined_identity))
    finally:
        await rpc.close()
        await server.stop(grace_s=0)
