from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import pytest

from swarm_inference.experiments.experiment_010.coordinator import StableExpertCoordinator
from swarm_inference.experiments.experiment_010.dispatch import ExpertDispatcher
from swarm_inference.experiments.experiment_010.expert import (
    ExpertStore,
    deterministic_expert,
)
from swarm_inference.experiments.experiment_010.relay import ExpertRelayManager
from swarm_inference.experiments.experiment_010.schemas import (
    DataPlane,
    ExpertExecutionRequest,
    FailureType,
    RecoveryStrategy,
    WorkerBudget,
    WorkerManifest,
)
from swarm_inference.experiments.experiment_010.transport import (
    NETWORK_PROFILES,
    ExpertTransportClient,
)
from swarm_inference.experiments.experiment_010.verification import (
    TrustController,
    reconcile_expert_ownership,
)
from swarm_inference.experiments.experiment_010.worker import (
    ExpertWorkerManager,
    fixture_ownership_entry,
)
from swarm_inference.worker.abi import (
    TensorPayload,
    WorkerJob,
    WorkerJobStatus,
    WorkerJobType,
    tensor_payload_from_array,
)
from swarm_inference.worker.universal import UniversalWorkerClient

MODEL = {
    "model_id": "experiment-010-test-model",
    "model_revision": "test-revision",
    "quantization_fingerprint": "test-quantization",
    "model_fingerprint": "test-model-fingerprint",
}
LATENT = 8
INTERMEDIATE = 16


def _save(path: Path, weights: Any) -> None:
    np.savez(
        path,
        up=weights.up,
        gate=weights.gate,
        down=weights.down,
        hidden_start=np.asarray(weights.hidden_offset, dtype=np.int64),
        logical_intermediate_dimension=np.asarray(weights.logical_width, dtype=np.int64),
    )


def _request(request_id: str, expert_ids: list[int] | None = None) -> ExpertExecutionRequest:
    selected = expert_ids or [0]
    return ExpertExecutionRequest(
        request_id=request_id,
        model_id=MODEL["model_id"],
        model_revision=MODEL["model_revision"],
        quantization_fingerprint=MODEL["quantization_fingerprint"],
        layer_id=0,
        batch_rows=2,
        latent_dimension=LATENT,
        expert_ids=selected,
        routing_weights=[1 / len(selected)] * len(selected),
        activations={},
        deadline_ns=time.time_ns() + 10_000_000_000,
    )


def _budget(worker_id: str, root: Path, expert_bytes: int) -> WorkerBudget:
    affinity = psutil.Process().cpu_affinity()
    return WorkerBudget(
        worker_id=worker_id,
        memory_budget_bytes=max(expert_bytes, 1 << 20),
        expert_residency_budget_bytes=expert_bytes,
        cache_budget_bytes=expert_bytes,
        thread_count=1,
        cpu_affinity=[affinity[0]],
        storage_directory=str(root),
        device="cpu",
        backend="numpy",
        physical_memory_limit=False,
    )


def _trust() -> TrustController:
    return TrustController(**MODEL, sampled_duplicate_fraction=1.0, quarantine_failures=2)


@pytest.fixture(scope="module")
def rpc_system(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("experiment-010-rpc")
    weights = {
        expert: deterministic_expert(
            latent_dimension=LATENT,
            intermediate_dimension=INTERMEDIATE,
            seed=1010 + expert,
        )
        for expert in range(3)
    }
    paths = {}
    for expert, value in weights.items():
        path = root / f"expert-{expert}.npz"
        _save(path, value)
        paths[expert] = path
    activation = np.random.default_rng(1010).normal(0, 0.1, (2, LATENT)).astype(np.float32)
    manager = ExpertWorkerManager(root / "workers")
    clients: dict[str, ExpertTransportClient] = {}
    processes = {}
    manifests = {}
    specifications = {"worker-a": [0, 1], "worker-b": [2], "worker-c": [0]}
    try:
        for worker_id, experts in specifications.items():
            entries = [fixture_ownership_entry(0, expert, paths[expert]) for expert in experts]
            size = sum(int(entry["logical_bytes"]) for entry in entries)
            process = manager.start(
                worker_id=worker_id,
                owned_experts=entries,
                budget=_budget(worker_id, root / worker_id, size),
                loader_type="npz",
                **MODEL,
            )
            client = ExpertTransportClient(process.endpoint)
            manifest = WorkerManifest.model_validate(client.control("manifest")["manifest"])
            clients[worker_id] = client
            processes[worker_id] = process
            manifests[worker_id] = manifest
        local_store = ExpertStore(
            owned={(0, expert) for expert in weights},
            loader=lambda _layer, expert: weights[expert],
            residency_budget_bytes=sum(value.byte_size for value in weights.values()),
            cache_budget_bytes=sum(value.byte_size for value in weights.values()),
        )
        yield {
            "root": root,
            "weights": weights,
            "paths": paths,
            "activation": activation,
            "manager": manager,
            "clients": clients,
            "processes": processes,
            "manifests": manifests,
            "local_store": local_store,
        }
    finally:
        manager.close()


def _registered_trust(system: dict[str, Any], workers: tuple[str, ...]) -> TrustController:
    trust = _trust()
    for worker_id in workers:
        manifest = WorkerManifest.model_validate(
            system["clients"][worker_id].control("manifest")["manifest"]
        )
        trust.register(manifest, signature_secret=system["processes"][worker_id].signature_secret)
    return trust


def _execute_fault(
    system: dict[str, Any], fault: FailureType, request_id: str
) -> tuple[Any, np.ndarray, np.ndarray, TrustController]:
    client = system["clients"]["worker-c"]
    trust = _registered_trust(system, ("worker-c",))
    request = _request(request_id)
    reference, _ = system["local_store"].execute(request, system["activation"])
    client.control("configure_fault", fault_type=fault.value, remaining=1)
    response, result, _ = client.execute(request, system["activation"])
    return response, result, reference, trust


def test_worker_process_isolation(rpc_system: dict[str, Any]) -> None:
    pids = {process.process.pid for process in rpc_system["processes"].values()}
    control_endpoints = {process.control_endpoint for process in rpc_system["processes"].values()}
    assert len(pids) == 3
    assert len(control_endpoints) == 3
    assert psutil.Process().pid not in pids


def test_worker_universal_abi_moe_expert(rpc_system: dict[str, Any]) -> None:
    process = rpc_system["processes"]["worker-a"]
    assert process.negotiated_protocol["major"] == 1
    assert "moe-expert" in process.negotiated_protocol["capabilities"]
    assert process.universal_identity["worker_id"] == "worker-a"
    assert (
        process.universal_capabilities["backend_details"]["expert_data_endpoint"]
        == process.endpoint
    )
    request = _request(f"universal-moe-{time.time_ns()}")
    payload = tensor_payload_from_array(
        rpc_system["activation"],
        tensor_id="universal-expert-input",
        request_id=request.request_id,
        stage_id=0,
        token_position=0,
        sequence_length=rpc_system["activation"].shape[0],
        model_revision=MODEL["model_revision"],
        partition_hash=MODEL["model_fingerprint"],
        route_generation=0,
        logical_dtype="float32",
    )
    job = WorkerJob(
        job_id=f"job-{time.time_ns()}",
        request_id=request.request_id,
        role=WorkerJobType.MOE_EXPERT,
        model_id=MODEL["model_id"],
        model_revision=MODEL["model_revision"],
        input_payload=payload,
        deadline_ms=10_000,
        metadata={"expert_request": request.model_dump(mode="json")},
    )
    host, raw_port = process.control_endpoint.rsplit(":", 1)

    async def submit() -> Any:
        client = UniversalWorkerClient(host, int(raw_port), timeout_seconds=5.0)
        assert (await client.heartbeat())["worker_id"] == "worker-a"
        return await client.submit(job)

    result = asyncio.run(submit())
    assert result.status == WorkerJobStatus.ACCEPTED
    assert isinstance(result.output_payload, TensorPayload)
    reference, _ = rpc_system["local_store"].execute(request, rpc_system["activation"])
    np.testing.assert_array_equal(result.output_payload.to_tensor().array, reference)


def test_worker_affinity(rpc_system: dict[str, Any]) -> None:
    expected = psutil.Process().cpu_affinity()[0]
    for process in rpc_system["processes"].values():
        assert psutil.Process(process.process.pid).cpu_affinity() == [expected]


def test_worker_memory_budget() -> None:
    weights = deterministic_expert(latent_dimension=8, intermediate_dimension=16, seed=1)
    store = ExpertStore(
        owned={(0, 0)},
        loader=lambda _layer, _expert: weights,
        residency_budget_bytes=weights.byte_size - 1,
        cache_budget_bytes=weights.byte_size - 1,
    )
    with pytest.raises(MemoryError, match="exceeds"):
        store.get(0, 0)


def test_worker_cache_isolation(rpc_system: dict[str, Any]) -> None:
    request = _request("cache-isolation")
    rpc_system["clients"]["worker-a"].execute(request, rpc_system["activation"])
    before = rpc_system["clients"]["worker-a"].control("manifest")["manifest"]
    rpc_system["clients"]["worker-b"].control("cache_drop")
    after = rpc_system["clients"]["worker-a"].control("manifest")["manifest"]
    assert before["cache_bytes"] > 0
    assert after["cache_bytes"] == before["cache_bytes"]


def test_worker_model_fingerprint(rpc_system: dict[str, Any]) -> None:
    assert rpc_system["manifests"]["worker-a"].model_fingerprint == MODEL["model_fingerprint"]
    assert rpc_system["manifests"]["worker-a"].control_endpoint
    assert rpc_system["manifests"]["worker-a"].universal_worker_abi["job_role"] == "moe_expert"


def test_expert_ownership_disjoint(rpc_system: dict[str, Any]) -> None:
    result = reconcile_expert_ownership(
        [rpc_system["manifests"]["worker-a"], rpc_system["manifests"]["worker-b"]],
        expected={(0, 0), (0, 1), (0, 2)},
    )
    assert result["disjoint"] is True


def test_expert_ownership_reconciliation(rpc_system: dict[str, Any]) -> None:
    result = reconcile_expert_ownership(
        [rpc_system["manifests"]["worker-a"], rpc_system["manifests"]["worker-b"]],
        expected={(0, 0), (0, 1), (0, 2)},
    )
    assert result["valid"] is True
    assert result["missing"] == []


def _whole_outputs(system: dict[str, Any], *, coalesced: bool) -> tuple[np.ndarray, Any]:
    request = _request(f"local-whole-{coalesced}", [0, 1])
    reference, _ = system["local_store"].execute(request, system["activation"])
    coordinator = StableExpertCoordinator(
        clients={"worker-a": system["clients"]["worker-a"]},
        whole_ownership={(0, 0): "worker-a", (0, 1): "worker-a"},
        latent_dimension=LATENT,
        **{key: MODEL[key] for key in ("model_id", "model_revision", "quantization_fingerprint")},
    )
    result = coordinator.execute_whole_layer(
        system["activation"],
        layer_id=0,
        expert_ids=[0, 1],
        routing_weights=[0.5, 0.5],
        coalesced=coalesced,
        request_id=f"remote-whole-{coalesced}-{time.time_ns()}",
    )
    return reference, result


def test_whole_expert_rpc_equivalence(rpc_system: dict[str, Any]) -> None:
    reference, result = _whole_outputs(rpc_system, coalesced=True)
    np.testing.assert_array_equal(result.output, reference)


def test_whole_expert_rpc_token_identity(rpc_system: dict[str, Any]) -> None:
    reference, result = _whole_outputs(rpc_system, coalesced=True)
    expected_tokens = np.argmax(reference, axis=-1)
    actual_tokens = np.argmax(result.output, axis=-1)
    np.testing.assert_array_equal(actual_tokens, expected_tokens)


def test_direct_tcp_data_plane(rpc_system: dict[str, Any]) -> None:
    response, result, metrics = rpc_system["clients"]["worker-a"].execute(
        _request(f"direct-{time.time_ns()}"), rpc_system["activation"]
    )
    assert response.worker_id == "worker-a"
    assert result.shape == rpc_system["activation"].shape
    assert metrics["messages_sent"] == 1


def test_relayed_tcp_data_plane(rpc_system: dict[str, Any]) -> None:
    with ExpertRelayManager(rpc_system["root"] / f"relay-{time.time_ns()}") as relays:
        relay = relays.start(
            target_endpoint=rpc_system["processes"]["worker-a"].endpoint,
            profile=NETWORK_PROFILES["loopback_unshaped"],
        )
        client = ExpertTransportClient(relay.endpoint, data_plane=DataPlane.RELAYED_TCP)
        response, result, _ = client.execute(
            _request(f"relay-{time.time_ns()}"), rpc_system["activation"]
        )
    assert response.worker_id == "worker-a"
    assert np.isfinite(result).all()


def test_shared_memory_data_plane(rpc_system: dict[str, Any]) -> None:
    client = ExpertTransportClient(
        rpc_system["processes"]["worker-a"].endpoint,
        data_plane=DataPlane.SHARED_MEMORY,
    )
    _, result, metrics = client.execute(
        _request(f"shared-{time.time_ns()}"), rpc_system["activation"]
    )
    assert result.shape == rpc_system["activation"].shape
    assert metrics["shared_memory_bytes"] > rpc_system["activation"].nbytes


def test_data_plane_byte_accounting(rpc_system: dict[str, Any]) -> None:
    _, result, metrics = rpc_system["clients"]["worker-a"].execute(
        _request(f"bytes-{time.time_ns()}"), rpc_system["activation"]
    )
    assert metrics["request_bytes"] > rpc_system["activation"].nbytes
    assert metrics["response_bytes"] > result.nbytes
    assert metrics["payload_bytes"] == rpc_system["activation"].nbytes


def test_naive_expert_rpc(rpc_system: dict[str, Any]) -> None:
    _, result = _whole_outputs(rpc_system, coalesced=False)
    assert result.metrics["protocol"] == "naive_per_expert"
    assert result.metrics["messages_per_layer"] == 2


def test_coalesced_layer_rpc(rpc_system: dict[str, Any]) -> None:
    _, result = _whole_outputs(rpc_system, coalesced=True)
    assert result.metrics["protocol"] == "coalesced_per_layer"
    assert result.metrics["messages_per_layer"] == 1


def test_coalescing_reduces_messages(rpc_system: dict[str, Any]) -> None:
    _, naive = _whole_outputs(rpc_system, coalesced=False)
    _, coalesced = _whole_outputs(rpc_system, coalesced=True)
    assert coalesced.metrics["messages_per_layer"] < naive.metrics["messages_per_layer"]
    assert coalesced.metrics["activation_payload_bytes"] < naive.metrics["activation_payload_bytes"]


def test_capacity_no_worker_holds_full_model(rpc_system: dict[str, Any]) -> None:
    a = set(rpc_system["manifests"]["worker-a"].owned_experts["0"])
    b = set(rpc_system["manifests"]["worker-b"].owned_experts["0"])
    assert len(a) < 3 and len(b) < 3


def test_capacity_global_inventory_complete(rpc_system: dict[str, Any]) -> None:
    owned = set(rpc_system["manifests"]["worker-a"].owned_experts["0"])
    owned |= set(rpc_system["manifests"]["worker-b"].owned_experts["0"])
    assert owned == {0, 1, 2}


def test_hedged_execution(rpc_system: dict[str, Any]) -> None:
    trust = _registered_trust(rpc_system, ("worker-a", "worker-c"))
    request = _request(f"hedge-{time.time_ns()}")
    reference, _ = rpc_system["local_store"].execute(request, rpc_system["activation"])
    dispatcher = ExpertDispatcher(
        {key: rpc_system["clients"][key] for key in ("worker-a", "worker-c")},
        trust=trust,
    )
    result = dispatcher.execute(
        request,
        rpc_system["activation"],
        primary_worker="worker-a",
        alternate_workers=("worker-c",),
        recovery_strategy=RecoveryStrategy.HEDGED_DUPLICATE,
        reference=reference,
    )
    assert result.metrics["correctness"] is True
    assert result.metrics["duplicate_results"] == 1


def test_timeout_local_fallback(rpc_system: dict[str, Any]) -> None:
    trust = _registered_trust(rpc_system, ("worker-c",))
    request = _request(f"local-fallback-{time.time_ns()}")
    reference, _ = rpc_system["local_store"].execute(request, rpc_system["activation"])

    def local_executor(
        selected: ExpertExecutionRequest, activation: np.ndarray
    ) -> tuple[Any, np.ndarray]:
        response, output, _ = rpc_system["clients"]["worker-c"].execute(selected, activation)
        return response, output

    dispatcher = ExpertDispatcher({}, trust=trust, local_executor=local_executor)
    result = dispatcher.execute(
        request,
        rpc_system["activation"],
        primary_worker="missing",
        recovery_strategy=RecoveryStrategy.TIMEOUT_LOCAL_FALLBACK,
        reference=reference,
    )
    assert result.metrics["recovered"] is True
    np.testing.assert_array_equal(result.result, reference)


def test_wrong_expert_detection(rpc_system: dict[str, Any]) -> None:
    response, result, reference, trust = _execute_fault(
        rpc_system, FailureType.WRONG_EXPERT, f"wrong-expert-{time.time_ns()}"
    )
    assert not trust.verify(
        _request(response.request_id), response, result, reference=reference
    ).accepted


def test_wrong_revision_detection(rpc_system: dict[str, Any]) -> None:
    response, result, reference, trust = _execute_fault(
        rpc_system, FailureType.WRONG_MODEL_REVISION, f"wrong-revision-{time.time_ns()}"
    )
    decision = trust.verify(_request(response.request_id), response, result, reference=reference)
    assert "wrong_model_revision" in decision.reasons


def test_bit_flip_detection(rpc_system: dict[str, Any]) -> None:
    response, result, reference, trust = _execute_fault(
        rpc_system, FailureType.BIT_FLIP, f"bit-flip-{time.time_ns()}"
    )
    assert not trust.verify(
        _request(response.request_id), response, result, reference=reference
    ).accepted


def test_zero_result_detection(rpc_system: dict[str, Any]) -> None:
    response, result, reference, trust = _execute_fault(
        rpc_system, FailureType.ZERO_RESULT, f"zero-{time.time_ns()}"
    )
    assert not trust.verify(
        _request(response.request_id), response, result, reference=reference
    ).accepted


def test_challenge_activation_detection(rpc_system: dict[str, Any]) -> None:
    response, result, reference, trust = _execute_fault(
        rpc_system, FailureType.WRONG_EXPERT, f"challenge-{time.time_ns()}"
    )
    request = _request(response.request_id).model_copy(update={"challenge": True})
    decision = trust.verify(request, response, result, reference=reference)
    assert not decision.accepted
    assert trust.reputations["worker-c"].challenge_failures == 1


def test_sampled_duplicate_verification(rpc_system: dict[str, Any]) -> None:
    trust = _registered_trust(rpc_system, ("worker-a", "worker-c"))
    request = _request(f"duplicate-{time.time_ns()}")
    left = rpc_system["clients"]["worker-a"].execute(request, rpc_system["activation"])[1]
    right = rpc_system["clients"]["worker-c"].execute(request, rpc_system["activation"])[1]
    decision = trust.compare_duplicate("worker-a", left, "worker-c", right, exact=True)
    assert trust.should_duplicate(request) is True
    assert decision.accepted is True


def test_worker_quarantine(rpc_system: dict[str, Any]) -> None:
    trust = _registered_trust(rpc_system, ("worker-c",))
    for index in range(2):
        request = _request(f"quarantine-{index}-{time.time_ns()}")
        reference, _ = rpc_system["local_store"].execute(request, rpc_system["activation"])
        rpc_system["clients"]["worker-c"].control(
            "configure_fault", fault_type=FailureType.WRONG_EXPERT.value, remaining=1
        )
        response, result, _ = rpc_system["clients"]["worker-c"].execute(
            request, rpc_system["activation"]
        )
        assert not trust.verify(request, response, result, reference=reference).accepted
    assert trust.reputations["worker-c"].quarantined is True


def test_reputation_update(rpc_system: dict[str, Any]) -> None:
    trust = _registered_trust(rpc_system, ("worker-a",))
    request = _request(f"reputation-{time.time_ns()}")
    reference, _ = rpc_system["local_store"].execute(request, rpc_system["activation"])
    response, result, metrics = rpc_system["clients"]["worker-a"].execute(
        request, rpc_system["activation"]
    )
    decision = trust.verify(
        request,
        response,
        result,
        reference=reference,
        latency_ns=metrics["request_elapsed_ns"],
    )
    assert decision.accepted
    assert trust.reputations["worker-a"].requests_completed == 1
    assert trust.reputations["worker-a"].latency_ns


@pytest.mark.parametrize("fault", [FailureType.WORKER_TERMINATION, FailureType.WORKER_PAUSE])
def test_real_worker_recovery(
    rpc_system: dict[str, Any], tmp_path: Path, fault: FailureType
) -> None:
    manager = ExpertWorkerManager(tmp_path / "victims")
    entries = [fixture_ownership_entry(0, 0, rpc_system["paths"][0])]
    size = int(entries[0]["logical_bytes"])
    victim = manager.start(
        worker_id=f"victim-{fault.value}",
        owned_experts=entries,
        budget=_budget(f"victim-{fault.value}", tmp_path / fault.value, size),
        loader_type="npz",
        **MODEL,
    )
    timeout = 0.02 if fault == FailureType.WORKER_PAUSE else 1.0
    victim_client = ExpertTransportClient(victim.endpoint, timeout_s=timeout)
    trust = _registered_trust(rpc_system, ("worker-c",))
    victim_manifest = WorkerManifest.model_validate(victim_client.control("manifest")["manifest"])
    trust.register(victim_manifest, signature_secret=victim.signature_secret)
    dispatcher = ExpertDispatcher(
        {victim.worker_id: victim_client, "worker-c": rpc_system["clients"]["worker-c"]},
        trust=trust,
    )
    request = _request(f"recover-{fault.value}-{time.time_ns()}")
    reference, _ = rpc_system["local_store"].execute(request, rpc_system["activation"])
    victim_client.control(
        "configure_fault",
        fault_type=fault.value,
        remaining=1,
        fixed_delay_ms=100,
    )
    try:
        result = dispatcher.execute(
            request,
            rpc_system["activation"],
            primary_worker=victim.worker_id,
            alternate_workers=("worker-c",),
            recovery_strategy=RecoveryStrategy.TIMEOUT_ALTERNATE_WORKER,
            reference=reference,
        )
        assert result.metrics["recovered"] is True
        np.testing.assert_array_equal(result.result, reference)
    finally:
        manager.close()


def test_worker_termination_recovery(rpc_system: dict[str, Any], tmp_path: Path) -> None:
    test_real_worker_recovery(rpc_system, tmp_path, FailureType.WORKER_TERMINATION)


def test_worker_pause_recovery(rpc_system: dict[str, Any], tmp_path: Path) -> None:
    test_real_worker_recovery(rpc_system, tmp_path, FailureType.WORKER_PAUSE)


def test_orphan_worker_cleanup(rpc_system: dict[str, Any], tmp_path: Path) -> None:
    manager = ExpertWorkerManager(tmp_path / "orphan")
    entries = [fixture_ownership_entry(0, 0, rpc_system["paths"][0])]
    worker = manager.start(
        worker_id="orphan",
        owned_experts=entries,
        budget=_budget("orphan", tmp_path / "orphan-storage", int(entries[0]["logical_bytes"])),
        loader_type="npz",
        **MODEL,
    )
    pid = worker.process.pid
    manager.close()
    assert not psutil.pid_exists(pid)
    assert manager.lifecycle_records[0]["exit_code"] == 0
    assert manager.lifecycle_records[0]["shutdown_via_universal_worker"] is True
