from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np
import pytest
import torch

from swarm_inference.engines.topology import TopologyDomain
from swarm_inference.execution.expert import (
    ExpertStore,
    deterministic_expert,
    execute_expert,
    slice_expert_weights,
)
from swarm_inference.execution.microshard import MicroshardRange
from swarm_inference.execution.moe import (
    MicroshardFanoutMode,
    MicroshardRemoteBackend,
    MicroshardTarget,
    select_microshard_fanout,
)
from swarm_inference.protocol.expert import (
    DelegatedMicroshardNode,
    DelegatedMicroshardOperation,
    ExpertExecutionMode,
    ExpertExecutionRequest,
    ExpertResponseMode,
    ExpertRouteParticipant,
    ReductionMode,
    SignedExpertRouteLease,
    TransportCodec,
    expert_route_lease_hash,
    sign_expert_route_lease,
)
from swarm_inference.security.identity import CoordinatorIdentity, WorkerIdentity
from swarm_inference.transport.expert import ExpertTransportClient
from swarm_inference.worker.expert_service import ExpertWorkerRuntime, ExpertWorkerServer


def _ownership(worker_id: str, index: int, content_hash: str) -> dict[str, Any]:
    return {
        "worker_id": worker_id,
        "layer_id": 0,
        "expert_id": 0,
        "hidden_start": index * 2,
        "hidden_end": (index + 1) * 2,
        "logical_intermediate_dimension": 16,
        "content_hash": content_hash,
        "quantization_group_size": None,
    }


@pytest.mark.asyncio
async def test_canonical_worker_owned_delegation_is_observable_and_exact() -> None:
    coordinator = CoordinatorIdentity.generate()
    stage_identity = WorkerIdentity.generate()
    full = deterministic_expert(latent_dimension=4, intermediate_dimension=16, seed=1213)
    activation = np.arange(12, dtype=np.float32).reshape(3, 4) / np.float32(11)
    expected = execute_expert(activation, full)
    workers: list[tuple[WorkerIdentity, ExpertWorkerRuntime, ExpertWorkerServer, str, Any]] = []
    for index in range(8):
        worker_id = f"canonical-{index:02d}"
        identity = WorkerIdentity.generate()
        shard = slice_expert_weights(full, hidden_start=index * 2, hidden_end=(index + 1) * 2)
        owned = _ownership(worker_id, index, shard.content_hash)
        store = ExpertStore(
            owned={(0, 0)},
            loader=lambda _layer, _expert, value=shard: value,
            residency_budget_bytes=shard.byte_size,
            cache_budget_bytes=shard.byte_size,
        )
        runtime = ExpertWorkerRuntime(
            worker_id=worker_id,
            identity=identity,
            model_id="test/canonical-delegation",
            model_revision="revision",
            model_fingerprint="model-fingerprint",
            quantization_fingerprint="quantization-fingerprint",
            store=store,
            roles={"expert-microshard", "reducer"},
            owned_microshards=[owned],
            maximum_concurrent_requests=1,
            require_authenticated_routes=True,
            trusted_coordinators={"coordinator": coordinator.public_key_b64},
        )
        server = ExpertWorkerServer(runtime, host="127.0.0.1", port=0)
        host, port = await server.start()
        workers.append((identity, runtime, server, f"{host}:{port}", owned))

    now = time.time_ns()
    lease = SignedExpertRouteLease(
        topology_id="canonical-delegated-topology",
        route_generation=13,
        model_id="test/canonical-delegation",
        model_revision="revision",
        model_fingerprint="model-fingerprint",
        quantization_fingerprint="quantization-fingerprint",
        participants=[
            ExpertRouteParticipant(
                worker_id="stage-owner",
                worker_public_key=stage_identity.public_key_b64,
                worker_public_key_fingerprint=stage_identity.public_key_fingerprint,
                endpoint="127.0.0.1:1",
                roles=["contiguous-stage"],
                model_fingerprint="model-fingerprint",
                quantization_fingerprint="quantization-fingerprint",
            ),
            *[
                ExpertRouteParticipant(
                    worker_id=f"canonical-{index:02d}",
                    worker_public_key=identity.public_key_b64,
                    worker_public_key_fingerprint=identity.public_key_fingerprint,
                    endpoint=endpoint,
                    roles=["expert-microshard", "reducer"],
                    owned_microshards=[owned],
                    model_fingerprint="model-fingerprint",
                    quantization_fingerprint="quantization-fingerprint",
                )
                for index, (identity, _runtime, _server, endpoint, owned) in enumerate(workers)
            ],
        ],
        lease_issued_unix_ns=now,
        lease_expiry_unix_ns=now + 60_000_000_000,
        nonce="h012-013-route",
        coordinator_identity="coordinator",
        coordinator_public_key=coordinator.public_key_b64,
        coordinator_public_key_fingerprint=coordinator.public_key_fingerprint,
    )
    lease = sign_expert_route_lease(lease, coordinator)
    targets: list[MicroshardTarget] = []
    for index, (_identity, runtime, _server, endpoint, owned) in enumerate(workers):
        runtime.install_route(lease)
        targets.append(
            MicroshardTarget(
                ownership=MicroshardRange(
                    worker_id=f"canonical-{index:02d}",
                    layer_id=0,
                    expert_id=0,
                    hidden_start=int(owned["hidden_start"]),
                    hidden_end=int(owned["hidden_end"]),
                    logical_intermediate_dimension=16,
                    content_hash=str(owned["content_hash"]),
                ),
                client=ExpertTransportClient(endpoint, timeout_s=5),
                endpoint=endpoint,
            )
        )

    def backend(mode: str) -> MicroshardRemoteBackend:
        value = MicroshardRemoteBackend(
            targets={(0, 0): targets},
            model_id="test/canonical-delegation",
            model_revision="revision",
            model_fingerprint="model-fingerprint",
            quantization_fingerprint="quantization-fingerprint",
            topology_id="canonical-delegated-topology",
            route_generation=13,
            maximum_parallel_requests=8,
            fanout_branching_factor=2,
            reduction_branching_factor=2,
            fanout_mode=mode,
            topology_domain=TopologyDomain.LOCAL_FAST,
        )
        value.configure_route(lease, identity=stage_identity, worker_id="stage-owner")
        value.open_session(f"session-{mode}")
        return value

    flat = backend("flat")
    delegated = backend("delegated")
    try:
        flat_output, flat_event = await asyncio.to_thread(
            flat.execute_expert_rows,
            session_id="session-flat",
            request_id="flat",
            token_position=0,
            layer_id=0,
            expert_id=0,
            activation=torch.from_numpy(activation),
            deadline_ns=time.time_ns() + 10_000_000_000,
        )
        delegated_output, event = await asyncio.to_thread(
            delegated.execute_expert_rows,
            session_id="session-delegated",
            request_id="delegated",
            token_position=0,
            layer_id=0,
            expert_id=0,
            activation=torch.from_numpy(activation),
            deadline_ns=time.time_ns() + 10_000_000_000,
        )
        np.testing.assert_allclose(flat_output.numpy(), expected, rtol=2e-6, atol=2e-8)
        np.testing.assert_allclose(delegated_output.numpy(), expected, rtol=2e-6, atol=2e-8)
        assert flat_event.root_dispatches == 8
        assert flat_event.root_leaf_rpcs == 8
        assert event.fanout_mode == "delegated"
        assert event.root_dispatches == 2
        assert event.root_messages == 4
        assert event.root_leaf_rpcs == 0
        assert event.worker_to_worker_messages == 12
        assert event.total_messages == 16
        assert event.fanout_depth == 3
        assert event.intermediate_reductions == 4
        all_events = [
            item
            for _identity, runtime, _server, _endpoint, _owned in workers
            for item in runtime.status()["delegation_events"]
            if item["event"] == "delegated_microshard_reduced"
        ]
        assert len(all_events) == 8
        assert len({item["process_id"] for item in all_events}) == 1
        assert sum(bool(item["intermediate_reduction"]) for item in all_events) == 4
        assert sum(len(item["child_worker_ids"]) for item in all_events) == 6
    finally:
        flat.close()
        delegated.close()
        await asyncio.gather(*(server.close() for _, _, server, _, _ in workers))


def test_canonical_fanout_selector_keeps_wan_coarse_and_flat_selectable() -> None:
    assert (
        select_microshard_fanout(
            requested="auto",
            worker_count=8,
            branch_factor=2,
            topology_domain="local-fast",
        ).selected
        == MicroshardFanoutMode.DELEGATED
    )
    assert (
        select_microshard_fanout(
            requested="auto",
            worker_count=512,
            branch_factor=8,
            topology_domain="wan",
        ).selected
        == MicroshardFanoutMode.FLAT
    )
    assert (
        select_microshard_fanout(
            requested="flat",
            worker_count=1000,
            branch_factor=8,
            topology_domain="local-fast",
        ).selected
        == MicroshardFanoutMode.FLAT
    )
    with pytest.raises(ValueError, match="local-fast"):
        select_microshard_fanout(
            requested="delegated",
            worker_count=32,
            branch_factor=8,
            topology_domain="wan",
        )


def test_delegated_subtree_is_strictly_bound_to_signed_route() -> None:
    coordinator = CoordinatorIdentity.generate()
    stage = WorkerIdentity.generate()
    identities = [WorkerIdentity.generate() for _ in range(4)]
    full = deterministic_expert(latent_dimension=4, intermediate_dimension=8, seed=1313)
    shards = [
        slice_expert_weights(full, hidden_start=index * 2, hidden_end=(index + 1) * 2)
        for index in range(4)
    ]
    descriptors = [
        {
            "worker_id": f"route-{index}",
            "layer_id": 0,
            "expert_id": 0,
            "hidden_start": index * 2,
            "hidden_end": (index + 1) * 2,
            "logical_intermediate_dimension": 8,
            "content_hash": shard.content_hash,
            "quantization_group_size": None,
        }
        for index, shard in enumerate(shards)
    ]
    store = ExpertStore(
        owned={(0, 0)},
        loader=lambda _layer, _expert: shards[0],
        residency_budget_bytes=shards[0].byte_size,
        cache_budget_bytes=shards[0].byte_size,
    )
    runtime = ExpertWorkerRuntime(
        worker_id="route-0",
        identity=identities[0],
        model_id="test/route-binding",
        model_revision="revision",
        model_fingerprint="model-fingerprint",
        quantization_fingerprint="quantization-fingerprint",
        store=store,
        roles={"expert-microshard", "reducer"},
        owned_microshards=[descriptors[0]],
        require_authenticated_routes=True,
        trusted_coordinators={"coordinator": coordinator.public_key_b64},
    )
    now = time.time_ns()
    lease = sign_expert_route_lease(
        SignedExpertRouteLease(
            topology_id="route-binding",
            route_generation=13,
            model_id="test/route-binding",
            model_revision="revision",
            model_fingerprint="model-fingerprint",
            quantization_fingerprint="quantization-fingerprint",
            participants=[
                ExpertRouteParticipant(
                    worker_id="stage-owner",
                    worker_public_key=stage.public_key_b64,
                    worker_public_key_fingerprint=stage.public_key_fingerprint,
                    endpoint="127.0.0.1:1",
                    roles=["contiguous-stage"],
                    model_fingerprint="model-fingerprint",
                    quantization_fingerprint="quantization-fingerprint",
                ),
                *[
                    ExpertRouteParticipant(
                        worker_id=f"route-{index}",
                        worker_public_key=identity.public_key_b64,
                        worker_public_key_fingerprint=identity.public_key_fingerprint,
                        endpoint=f"127.0.0.1:{5000 + index}",
                        roles=["expert-microshard", "reducer"],
                        owned_microshards=[descriptor],
                        model_fingerprint="model-fingerprint",
                        quantization_fingerprint="quantization-fingerprint",
                    )
                    for index, (identity, descriptor) in enumerate(
                        zip(identities, descriptors, strict=True)
                    )
                ],
            ],
            lease_issued_unix_ns=now,
            lease_expiry_unix_ns=now + 60_000_000_000,
            nonce="route-binding",
            coordinator_identity="coordinator",
            coordinator_public_key=coordinator.public_key_b64,
            coordinator_public_key_fingerprint=coordinator.public_key_fingerprint,
        ),
        coordinator,
    )
    runtime.install_route(lease)

    def node(index: int) -> DelegatedMicroshardNode:
        descriptor = descriptors[index]
        return DelegatedMicroshardNode(
            worker_id=f"route-{index}",
            endpoint=f"127.0.0.1:{5000 + index}",
            layer_id=0,
            expert_id=0,
            hidden_start=int(descriptor["hidden_start"]),
            hidden_end=int(descriptor["hidden_end"]),
            logical_intermediate_dimension=8,
            content_hash=str(descriptor["content_hash"]),
            ordering_key=(
                f"{int(descriptor['hidden_start']):020d}:"
                f"{int(descriptor['hidden_end']):020d}:route-{index}"
            ),
        )

    valid_root = node(0).model_copy(update={"children": [node(1)]}, deep=True)

    def request_for(operation: DelegatedMicroshardOperation) -> ExpertExecutionRequest:
        return ExpertExecutionRequest(
            request_id=f"{operation.operation_id}:worker:route-0",
            session_id="route-session",
            token_position=0,
            sequence_id=0,
            route_generation=13,
            topology_id="route-binding",
            model_id="test/route-binding",
            model_revision="revision",
            model_fingerprint="model-fingerprint",
            quantization_fingerprint="quantization-fingerprint",
            layer_id=0,
            batch_rows=1,
            latent_dimension=4,
            expert_ids=[0],
            expert_hashes={0: shards[0].content_hash},
            routing_weights=[1.0],
            top_k=1,
            response_mode=ExpertResponseMode.PER_WORKER_FAST,
            activations={},
            deadline_ns=operation.deadline_ns,
            execution_mode=ExpertExecutionMode.MICROSHARD,
            determinism_mode="exact",
            compression=TransportCodec.RAW_FP32,
            hidden_start=0,
            hidden_end=2,
            reduction_mode=ReductionMode.FIXED_ORDER_FP32,
            metadata={"delegation": operation.model_dump(mode="json")},
        )

    def operation_for(root: DelegatedMicroshardNode) -> DelegatedMicroshardOperation:
        return DelegatedMicroshardOperation(
            operation_id="route-operation",
            execution_generation=13,
            parent_worker_id="stage-owner",
            branch_factor=2,
            deadline_ns=time.time_ns() + 10_000_000_000,
            route_lease_identity=expert_route_lease_hash(lease),
            trace_id="route-trace",
            parent_span_id="stage",
            node=root,
        )

    valid = operation_for(valid_root)
    assert runtime._validate_delegation(request_for(valid), "stage-owner") == valid

    wrong_endpoint = operation_for(
        valid_root.model_copy(
            update={"children": [node(1).model_copy(update={"endpoint": "127.0.0.1:9999"})]},
            deep=True,
        )
    )
    unleased = operation_for(
        valid_root.model_copy(
            update={
                "children": [
                    node(1).model_copy(
                        update={
                            "worker_id": "unleased",
                            "ordering_key": "00000000000000000002:00000000000000000004:unleased",
                        }
                    )
                ]
            },
            deep=True,
        )
    )
    duplicate = operation_for(valid_root.model_copy(update={"children": [node(0)]}, deep=True))
    unbounded = operation_for(
        valid_root.model_copy(update={"children": [node(1), node(2), node(3)]}, deep=True)
    )
    stale = valid.model_copy(update={"execution_generation": 12}, deep=True)
    cases = [
        (wrong_endpoint, "stage-owner", "endpoint"),
        (unleased, "stage-owner", "unleased"),
        (duplicate, "stage-owner", "duplicate"),
        (unbounded, "stage-owner", "branch factor"),
        (stale, "stage-owner", "stale or mismatched"),
        (valid, "wrong-parent", "parent"),
    ]
    for operation, parent, message in cases:
        with pytest.raises(Exception, match=message):
            runtime._validate_delegation(request_for(operation), parent)

    malformed = valid_root.model_dump(mode="json")
    malformed["children"] = [node(2).model_dump(mode="json"), node(1).model_dump(mode="json")]
    raw_operation = valid.model_dump(mode="json")
    raw_operation["node"] = malformed
    malformed_request = request_for(valid).model_copy(
        update={"metadata": {"delegation": raw_operation}}, deep=True
    )
    with pytest.raises(ValueError, match="uniquely ordered"):
        runtime._validate_delegation(malformed_request, "stage-owner")
