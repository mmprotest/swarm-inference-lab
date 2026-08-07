from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from swarm_inference.cluster.models import (
    ClusterMetadata,
    NodeMembership,
    NodeMetadata,
    VersionCompatibility,
    node_id_from_fingerprint,
)
from swarm_inference.cluster.orchestrator import ClusterOrchestrator
from swarm_inference.cluster.pairing import PairingManager
from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.config.models import (
    Backend,
    OperationKind,
    QueueConfig,
    StageBenchmark,
    WorkerCapability,
)
from swarm_inference.config.product import ProductCoordinatorConfig
from swarm_inference.coordinator.service import (
    CoordinatorCore,
    CoordinatorRpcServer,
)
from swarm_inference.protocol.messages import RegistrationRequest
from swarm_inference.protocol.stage_ring import STAGE_RING_PROTOCOL_VERSION
from swarm_inference.security.identity import WorkerIdentity
from swarm_inference.security.signatures import canonical_json_bytes
from swarm_inference.security.trust_store import WorkerTrustStore
from swarm_inference.worker.agent import WorkerAgent
from swarm_inference.worker.stage_runtime import PersistentStageRuntime
from swarm_inference.worker.stage_service import PersistentStageWorkerService

MODEL_REVISION = "c" * 40


def _snapshot(root: Path) -> tuple[Path, str]:
    torch.manual_seed(815)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=32,
            tie_word_embeddings=False,
            attention_dropout=0,
            pad_token_id=0,
            eos_token_id=31,
        )
    ).eval()
    snapshot = root / "snapshot"
    model.save_pretrained(snapshot, safe_serialization=True, max_shard_size="2KB")
    config_path = snapshot / "config.json"
    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    configuration["_commit_hash"] = MODEL_REVISION
    config_path.write_text(json.dumps(configuration, sort_keys=True), encoding="utf-8")
    tokenizer = b'{"version":"1.0","auto-run":true}'
    (snapshot / "tokenizer.json").write_bytes(tokenizer)
    (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return snapshot, "sha256:" + hashlib.sha256(tokenizer).hexdigest()


def _capability(worker_id: str, identity: WorkerIdentity) -> WorkerCapability:
    return WorkerCapability(
        worker_id=worker_id,
        node_id=worker_id.split("/", 1)[0],
        public_key=identity.public_key_b64,
        hostname=worker_id,
        operating_system="injected-windows",
        architecture="AMD64",
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
        upload_bandwidth_bytes_s=100_000_000,
        download_bandwidth_bytes_s=100_000_000,
        coordinator_latency_ms=1,
        memory_limit_bytes=1024**3,
        endpoint="127.0.0.1:1",
        control_endpoint="127.0.0.1:1",
        data_plane_endpoint="127.0.0.1:1",
        device_identifier="cpu",
        stage_ring_protocol_version=STAGE_RING_PROTOCOL_VERSION,
        supported_model_adapters=["qwen3_dense"],
        supported_stage_execution_backends=["canonical-native-stage"],
        supported_activation_dtypes=["float32"],
        configured_memory_limit_bytes=1024**3,
        stage_runtime_enabled=True,
    )


@pytest.mark.asyncio
async def test_high_level_dry_run_uses_real_rpc_catalog_and_bounded_planner(
    tmp_path: Path,
) -> None:
    snapshot, tokenizer_revision = _snapshot(tmp_path)
    state = ClusterStateStore(tmp_path / "cluster-state")
    node_identity = WorkerIdentity.load_or_create(state.paths.node_identity)
    coordinator_node_id = node_id_from_fingerprint(node_identity.public_key_fingerprint)
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
        cluster_id="cluster-auto-run",
        name="auto-run",
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
    membership = NodeMembership(
        cluster_id=cluster.cluster_id,
        node_id=coordinator_node_id,
        node_public_key=node_identity.public_key_b64,
        node_fingerprint=node_identity.public_key_fingerprint,
        coordinator_public_key=cluster.coordinator_public_key,
        coordinator_fingerprint=cluster.coordinator_fingerprint,
        joined_at_unix_ns=time.time_ns(),
    )
    state.save_cluster(cluster)
    state.save_membership(membership)
    state.save_node(
        NodeMetadata(
            node_id=coordinator_node_id,
            public_key=node_identity.public_key_b64,
            fingerprint=node_identity.public_key_fingerprint,
            hostname="coordinator",
            operating_system="injected-windows",
            architecture="AMD64",
            agent_version="0.1.0",
            runtime_version="0.1.0",
            build_id="integration",
            package_lock_hash="d" * 64,
            joined_at_unix_ns=membership.joined_at_unix_ns,
            last_seen_at_unix_ns=membership.joined_at_unix_ns,
        )
    )
    trust = WorkerTrustStore(trust_path)
    control = PairingManager(
        state=state,
        trust_store=trust,
        coordinator_identity=core.coordinator_identity,
        cluster=cluster,
    )
    core.attach_cluster_control(control)

    services: list[PersistentStageWorkerService] = []
    try:
        for index in range(4):
            identity = WorkerIdentity.generate()
            worker_id = f"node-{index:08x}/cpu-0"
            capability = _capability(worker_id, identity)
            runtime = PersistentStageRuntime(
                worker_id=worker_id,
                device="cpu",
                dtype="float32",
                memory_limit_bytes=1024**3,
                maximum_sessions=4,
                configured_model_path=snapshot,
                capability=capability,
            )
            service = PersistentStageWorkerService(
                agent=WorkerAgent(
                    capability=capability,
                    identity=identity,
                    queue_config=QueueConfig(capacity=8),
                ),
                stage_runtime=runtime,
            )
            control_port, data_port = await service.start(
                control_listen_endpoint="127.0.0.1:0",
                data_listen_endpoint="127.0.0.1:0",
            )
            assert data_port is not None
            capability.endpoint = f"127.0.0.1:{control_port}"
            capability.control_endpoint = capability.endpoint
            capability.data_plane_endpoint = f"127.0.0.1:{data_port}"
            trust.trust(identity.public_key_fingerprint, label=worker_id)
            nonce = f"register-{index}"
            registration = RegistrationRequest(
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
            assert (await core.register(registration)).accepted
            services.append(service)

        summary = await ClusterOrchestrator(
            state=state,
            network_refresh_wait_seconds=0,
        ).run(
            model_id=str(snapshot),
            model_revision=MODEL_REVISION,
            tokenizer_revision=tokenizer_revision,
            prompt="not serialized",
            dry_run=True,
            mode="speed",
        )

        assert summary.status == "dry-run"
        assert 1 <= summary.plan.stage_count <= 4
        assert summary.plan.report.search_method == "bounded-deterministic-beam-search"
        assert summary.artifact_ids == []
        assert not state.load_artifact_cache().entries
    finally:
        for service in reversed(services):
            await service.stop()
        await server.stop(grace_s=0)
