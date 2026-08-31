from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from swarm_inference.experiments.experiment_025.providers.base import (
    EndpointMode,
    ProviderMode,
    ProviderMutationForbidden,
    ProviderMutationPolicy,
    ProviderOperation,
)
from swarm_inference.experiments.experiment_025.providers.runpod import (
    RunPodProvider,
    graphql_operation,
    normalize_pod,
    normalize_port_mappings,
    redact_provider_payload,
    resolve_endpoint,
)
from swarm_inference.experiments.experiment_025.runpod_cleanup import (
    AppendOnlyRunPodLedger,
    cleanup_from_ledger,
    safe_e025_name,
    select_cleanup_targets,
)
from swarm_inference.experiments.experiment_025.runpod_inventory import (
    ReadOnlyRunPodCli,
    normalize_graphql_matrix,
    sanitize_account,
)
from swarm_inference.experiments.experiment_025.runpod_lifecycle import (
    FragmentEndpointCoordinator,
    FragmentIdentity,
    PodLifecycle,
    PodLifecycleState,
    WorkerLifecycle,
    simulate_full_control_flow,
)
from swarm_inference.experiments.experiment_025.runpod_operator import (
    _p2_network_request,
    lightweight_start_code,
    runtime_create_payload,
    select_stage_manifests,
)
from swarm_inference.experiments.experiment_025.runpod_planning import (
    BACKBONE_GPU_ID,
    IMAGE_DIGEST,
    FrozenRole,
    assert_manifest_safety,
    build_role_manifests,
    decode_worker_specs,
    model_distribution_plan,
    pack_roles,
    paid_canary_budget,
    storage_plan,
    topology_metrics,
)
from swarm_inference.experiments.experiment_025.runpod_security import (
    scan_runpod_artifacts,
)
from swarm_inference.experiments.experiment_025.secrets import (
    create_transport_material,
)
from swarm_inference.experiments.experiment_025.supervisor import (
    BASE_PORT,
    _child_environment,
    _decode_specs,
)

RUN_ID = "20260819T013016Z"


class RecordingTransport:
    def __init__(self, response: Any = None) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def request(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


def _role(layer: int, role: str = "BACKBONE_STAGE", index: int | None = None) -> FrozenRole:
    worker_id = (
        f"e025-layer-089-sub-{index:02d}"
        if role == "SUB_LAYER_WORKER" and index is not None
        else f"e025-stage-{layer:03d}-parent"
        if role == "SUB_LAYER_PARENT"
        else f"e025-stage-{layer:03d}"
    )
    shared = layer // 2
    return FrozenRole(
        worker_id=worker_id,
        role=role,
        layer=layer,
        worker_index=index,
        assignment_sha256=f"assignment-{worker_id}",
        source_weight_bytes=1000 + layer,
        download_bytes_cold_cache=5000,
        assigned_tensor_bytes=900 + layer,
        temporary_disk_bytes=3000 + layer,
        source_shards=(
            {"name": f"model-{shared:05d}.safetensors", "bytes": 5000, "sha256": f"s{shared}"},
        ),
        maximum_context=64,
    )


def frozen_roles() -> list[FrozenRole]:
    roles = [
        _role(layer, "SUB_LAYER_PARENT" if layer == 89 else "BACKBONE_STAGE") for layer in range(93)
    ]
    roles.extend(_role(89, "SUB_LAYER_WORKER", index) for index in range(4))
    return roles


def manifests(count: int = 4) -> list[dict[str, Any]]:
    return build_role_manifests(
        pack_roles(frozen_roles(), gpu_count_per_backbone_pod=count),
        backbone_datacenters=["EU-CZ-1"],
        fragment_datacenters=["EU-RO-1"],
        endpoint_mode="RUNPOD_PUBLIC_TCP",
    )


@pytest.mark.parametrize(
    "method,args",
    [
        ("create_pod", ({"name": "blocked"},)),
        ("start_pod", ("pod",)),
        ("restart_pod", ("pod",)),
        ("reset_pod", ("pod",)),
        ("update_pod", ("pod", {})),
        ("stop_pod", ("pod",)),
        ("delete_pod", ("pod",)),
        ("create_network_volume", ({"name": "blocked"},)),
        ("delete_network_volume", ("volume",)),
    ],
)
def test_mutation_firewall_blocks_every_rest_mutation_before_transport(
    method: str,
    args: tuple[Any, ...],
) -> None:
    transport = RecordingTransport()
    provider = RunPodProvider(transport=transport)
    with pytest.raises(ProviderMutationForbidden):
        getattr(provider, method)(*args)
    assert transport.calls == []


def test_graphql_semantic_guard_allows_queries_and_blocks_mutations() -> None:
    transport = RecordingTransport({"data": {"gpuTypes": []}})
    provider = RunPodProvider(transport=transport)
    assert provider.graphql_query({"query": "# comment\nquery Inventory { gpuTypes { id } }"})
    assert transport.calls[0]["method"] == "POST"
    assert transport.calls[0]["operation"] is ProviderOperation.GRAPHQL_QUERY
    with pytest.raises(ProviderMutationForbidden):
        provider.graphql_query({"query": "mutation Create { podCreate { id } }"})
    with pytest.raises(ProviderMutationForbidden):
        provider.graphql_query({"query": "subscription Events { pod { id } }"})
    assert graphql_operation("{ gpuTypes { id } }") is ProviderOperation.GRAPHQL_QUERY
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    ("flag", "environment", "expected"),
    [
        (False, {}, ProviderMode.READ_ONLY_PREPARATION),
        (True, {}, ProviderMode.READ_ONLY_PREPARATION),
        (False, {"E025_RUNPOD_ALLOW_RENTAL": "YES"}, ProviderMode.READ_ONLY_PREPARATION),
        (True, {"E025_RUNPOD_ALLOW_RENTAL": "YES"}, ProviderMode.PAID_RUN),
    ],
)
def test_paid_mode_requires_two_independent_intents(
    flag: bool,
    environment: dict[str, str],
    expected: ProviderMode,
) -> None:
    assert (
        ProviderMutationPolicy.from_intent(
            allow_paid_run=flag,
            environment=environment,
        ).mode
        is expected
    )


def test_read_only_cli_facade_rejects_create_without_spawning(tmp_path: Path) -> None:
    executable = tmp_path / "runpodctl.exe"
    executable.touch()
    cli = ReadOnlyRunPodCli(executable)
    with pytest.raises(PermissionError):
        cli.run("pod", "create", "--gpuType", "RTX 3090")
    assert cli.invocations == []


def test_graphql_inventory_parser_preserves_count_specific_stock() -> None:
    row: dict[str, Any] = {"id": BACKBONE_GPU_ID, "displayName": "RTX 3090", "memoryInGb": 24}
    for prefix in ("s", "c"):
        for count in range(1, 9):
            row[f"{prefix}{count}"] = (
                {
                    "stockStatus": "Low",
                    "uninterruptablePrice": count * 0.5,
                    "availableGpuCounts": None,
                }
                if prefix == "s" and count <= 3
                else None
            )
    parsed = normalize_graphql_matrix({"data": {"g0": [row]}})
    secure = parsed[0]["tiers"]["SECURE"]
    assert secure["schedulable_gpu_counts_inferred_from_non_null_lowest_price"] == [1, 2, 3]
    assert secure["availableGpuCounts_field"] is None


def test_account_sanitization_never_retains_identity_strings() -> None:
    sanitized = sanitize_account(
        {
            "id": "user-sensitive-id",
            "email": "person@example.test",
            "clientBalance": 0,
            "currentSpendPerHr": 0,
            "spendLimit": 80,
        }
    )
    encoded = json.dumps(sanitized)
    assert "person@example.test" not in encoded
    assert "user-sensitive-id" not in encoded
    assert sanitized["email"]["utf8_bytes"] == len("person@example.test")


def test_runpod_response_parsing_extracts_physical_and_network_metadata() -> None:
    allocation = normalize_pod(
        {
            "id": "pod-1",
            "name": f"e025-rp-{RUN_ID}-bb-000-003",
            "machineId": "machine-physical-7",
            "desiredStatus": "RUNNING",
            "runtimeStatus": "RUNNING",
            "cloudType": "SECURE",
            "gpu": {
                "id": BACKBONE_GPU_ID,
                "count": 4,
                "memoryInGb": 24,
                "securePrice": 0.5,
            },
            "dataCenter": {"id": "EU-CZ-1", "country": "CZ"},
            "publicIp": "192.0.2.10",
            "portMappings": {
                "42525/tcp": [{"HostPort": "50101"}],
                "42526": 50102,
            },
            "imageName": "immutable-tag",
            "createdAt": "2026-08-28T00:00:00Z",
        }
    )
    assert allocation.machine_id == "machine-physical-7"
    assert allocation.datacenter_id == "EU-CZ-1"
    assert allocation.gpu_count == 4
    assert allocation.gpu_vram_bytes == 24 * 1024**3
    assert allocation.cost_per_hour == 2.0
    assert allocation.port_mappings == {42525: 50101, 42526: 50102}
    assert allocation.private_identity == "pod-1.runpod.internal"


def test_port_mapping_parser_ignores_malformed_values() -> None:
    assert normalize_port_mappings(
        {"bad": "x", "42525/tcp": [{"HostPort": "bad"}, {"host_port": 51000}]}
    ) == {42525: 51000}


def test_all_endpoint_modes_are_explicit() -> None:
    allocation = normalize_pod(
        {
            "id": "pod-a",
            "publicIp": "192.0.2.4",
            "portMappings": {"42525": 50001},
        }
    )
    local = resolve_endpoint(
        allocation=allocation,
        worker_id="worker",
        internal_port=42525,
        mode=EndpointMode.POD_LOCAL,
    )
    private = resolve_endpoint(
        allocation=allocation,
        worker_id="worker",
        internal_port=42525,
        mode=EndpointMode.RUNPOD_GLOBAL_PRIVATE,
    )
    public = resolve_endpoint(
        allocation=allocation,
        worker_id="worker",
        internal_port=42525,
        mode=EndpointMode.RUNPOD_PUBLIC_TCP,
    )
    assert local.authority == "127.0.0.1:42525"
    assert private.authority == "pod-a.runpod.internal:42525"
    assert public.authority == "192.0.2.4:50001"


@pytest.mark.parametrize(
    ("count", "expected_pods"),
    [(1, 97), (2, 52), (4, 29), (8, 18)],
)
def test_contiguous_packing_and_parent_isolation(count: int, expected_pods: int) -> None:
    pods = pack_roles(frozen_roles(), gpu_count_per_backbone_pod=count)
    assert len(pods) == expected_pods
    fragments = [pod for pod in pods if pod["pod_class"] == "FRAGMENT"]
    parent = [pod for pod in pods if pod["pod_class"] == "LAYER89_PARENT"]
    assert len(fragments) == 4
    assert all(pod["requested_gpu_count"] == 1 for pod in fragments)
    assert len(parent) == 1 and parent[0]["requested_gpu_count"] == 1
    for pod in pods:
        if pod["pod_class"] == "BACKBONE":
            layers = [int(role["layer"]) for role in pod["roles"]]
            assert layers == list(range(layers[0], layers[-1] + 1))


def test_topology_metrics_do_not_infer_host_independence_from_pods() -> None:
    pods = pack_roles(frozen_roles(), gpu_count_per_backbone_pod=3)
    metrics = topology_metrics(
        pods,
        backbone_price_per_gpu_hour=0.5,
        fragment_price_bounds_per_gpu_hour=(0.13, 0.18),
        supported=True,
        support_reason="test",
        stock_confidence="LOW",
    )
    assert metrics["total_pod_count"] == 36
    assert metrics["required_unique_physical_host_count_minimum"] == 4
    assert metrics["cross_pod_backbone_boundaries"] == 31
    assert metrics["intra_pod_backbone_boundaries"] == 61


@pytest.mark.parametrize("count", [1, 2, 4, 8])
def test_pod_manifest_roundtrip_matches_supervisor_contract(count: int) -> None:
    manifest = next(
        row
        for row in manifests(count)
        if row["pod_class"] == "BACKBONE" and int(row["requested_gpu_count"]) == count
    )
    specs = decode_worker_specs(manifest["E025_WORKER_SPECS_B64"])
    supervisor_specs = _decode_specs(
        manifest["E025_WORKER_SPECS_B64"],
        default_port=BASE_PORT,
    )
    assert specs == supervisor_specs
    assert [row["gpu_slot"] for row in specs] == list(range(count))
    assert [row["port"] for row in specs] == list(range(BASE_PORT, BASE_PORT + count))
    assert len({row["worker_id"] for row in specs}) == count
    for spec in specs:
        environment = _child_environment(spec)
        assert environment["CUDA_VISIBLE_DEVICES"] == str(spec["gpu_slot"])
        assert environment["E025_WORKER_ID"] == spec["worker_id"]
        assert "E025_WORKER_SPECS_B64" not in environment


def test_parent_manifest_injects_four_generation_tagged_endpoints() -> None:
    parent = next(row for row in manifests() if row["pod_class"] == "LAYER89_PARENT")
    specs = decode_worker_specs(parent["E025_WORKER_SPECS_B64"])
    endpoints = specs[0]["expert_endpoints"]
    assert len(endpoints) == 4
    assert {row["worker_index"] for row in endpoints} == set(range(4))
    assert all(row["fragment_endpoint_generation"] == 0 for row in endpoints)
    assert all("RUNPOD_PUBLIC" in row["host"] for row in endpoints)


def test_manifest_safety_storage_and_distribution_charge_worker_scoped_overlap() -> None:
    rows = manifests(4)
    assert assert_manifest_safety(rows)["status"] == "PASS"
    storage = storage_plan(rows)
    distribution = model_distribution_plan(rows)
    assert storage["network_volumes_created"] == 0
    assert storage["total_requested_container_disk_gb"] > 0
    assert (
        distribution["shared_cache_analysis"]["process_safety"] == "SAFE_BY_WORKER_PATH_ISOLATION"
    )
    assert distribution["shared_cache_analysis"]["worker_image_change_required"] is False
    assert any(
        row["overlap_bytes_current_implementation"] > 0
        for row in distribution["pods"]
        if len(row["worker_ids"]) > 1
    )


def test_paid_canary_budget_never_prices_p3_as_a_single_gpu_canary() -> None:
    budget = paid_canary_budget(
        backbone_price=0.5,
        fragment_low_price=0.13,
        fragment_high_price=0.18,
        preferred_backbone_gpu_count_per_pod=1,
        fragment_stock_located=False,
        headline_storage_gb=1000,
    )
    stages = {row["stage"]: row for row in budget["stages"]}
    p3 = stages["P3_MULTI_GPU_SECURE_3090"]
    assert p3["backbone_gpus"] == 2
    assert p3["current_stock_supported"] is False
    assert "P3_MULTI_GPU_SECURE_3090" in budget["unsupported_stage_costs_are_projections"]
    assert stages["P4_LAYER89_PROVIDER_CANARY"]["current_stock_supported"] is False


def test_runtime_payload_has_no_network_volume_and_secrets_are_ephemeral() -> None:
    manifest = next(row for row in manifests() if row["pod_class"] == "BACKBONE")
    payload = runtime_create_payload(
        manifest,
        secrets={
            "E025_RUN_CREDENTIAL_B64": "credential-value",
            "E025_TLS_CERT_B64": "certificate-value",
            "E025_TLS_KEY_B64": "private-key-value",
        },
        fragment_gpu_id=None,
        lightweight=False,
    )
    assert payload.get("networkVolumeId") is None
    assert payload["imageName"].endswith("e025-20260819t013016z")
    assert payload["env"]["E025_IMAGE_DIGEST"] == IMAGE_DIGEST
    redacted = redact_provider_payload(payload)
    assert redacted["env"]["E025_TLS_KEY_B64"]["redacted"] is True
    assert "private-key-value" not in json.dumps(redacted)


def test_p2_uses_two_secure_single_gpu_manifests_and_no_k3_command() -> None:
    selected = select_stage_manifests(manifests(4), "p2")
    assert len(selected) == 2
    assert all(row["pod_class"] == "NETWORK_CANARY" for row in selected)
    assert all(row["requested_gpu_count"] == 1 for row in selected)
    payload = runtime_create_payload(
        selected[0],
        secrets={
            "E025_RUN_CREDENTIAL_B64": "a",
            "E025_TLS_CERT_B64": "b",
            "E025_TLS_KEY_B64": "c",
        },
        fragment_gpu_id=None,
        lightweight=True,
    )
    assert payload["gpuTypeIds"] == [BACKBONE_GPU_ID]
    assert payload["globalNetworking"] is True
    assert payload["env"]["E025_RUNPOD_MODE"] == "LIGHTWEIGHT_ALLOCATION_REHEARSAL_NO_K3"
    assert "bootstrap" not in " ".join(payload["dockerStartCmd"])


def test_fragment_endpoint_generations_and_duplicate_machine_rejection() -> None:
    coordinator = FragmentEndpointCoordinator()
    first = FragmentIdentity("e025-layer-089-sub-00", "pod-0", "machine-a", "h0", 42525)
    duplicate = FragmentIdentity("e025-layer-089-sub-01", "pod-1", "machine-a", "h1", 42525)
    assert coordinator.publish(first) == (True, None)
    assert coordinator.publish(duplicate) == (False, first.worker_id)
    assert coordinator.replacements[0]["expensive_model_download_started"] is False
    for index in range(1, 4):
        accepted, _ = coordinator.publish(
            FragmentIdentity(
                f"e025-layer-089-sub-{index:02d}",
                f"pod-{index + 1}",
                f"machine-{index}",
                f"h{index}",
                42525,
            )
        )
        assert accepted
    assert coordinator.complete
    old_generation = coordinator.generation
    coordinator.retire("e025-layer-089-sub-00")
    coordinator.publish(
        FragmentIdentity("e025-layer-089-sub-00", "pod-new", "machine-new", "hn", 42525)
    )
    assert coordinator.generation > old_generation
    assert all(
        row["fragment_endpoint_generation"] == coordinator.generation
        for row in coordinator.endpoints()
    )


def test_child_failure_invalidates_whole_pod_group() -> None:
    pod = PodLifecycle(
        logical_pod_id="pod",
        worker_ids=("a", "b"),
        state=PodLifecycleState.READY_HEALTHY,
        workers={
            "a": WorkerLifecycle("a", 0, 42525, PodLifecycleState.READY_HEALTHY),
            "b": WorkerLifecycle("b", 1, 42526, PodLifecycleState.READY_HEALTHY),
        },
    )
    pod.mark_worker("a", PodLifecycleState.UNHEALTHY, reason="exit")
    assert pod.state is PodLifecycleState.REPLACEMENT_REQUIRED
    assert all(worker.state is PodLifecycleState.UNHEALTHY for worker in pod.workers.values())


def test_full_fake_provider_orchestration_exercises_replacement_and_cleanup() -> None:
    result = simulate_full_control_flow(manifests(4))
    assert result["status"] == "PASS"
    assert result["evidence_class"] == "SIMULATED_PROVIDER_CONTROL_FLOW"
    assert result["provider_network_calls"] == 0
    assert result["provider_mutations"] == []
    assert result["simultaneously_healthy_logical_roles"] == 97
    assert result["fragment_duplicate_machine_rejected"] is True
    assert result["parent_generation_restart_exercised"] is True
    assert result["pod_group_churn_exercised"] is True
    assert result["simulated_cost_accrual_usd"] > 0
    assert result["zero_live_simulated_pods"] is True


class DelayedDeleteProvider:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.delete_calls: list[str] = []

    def list_pods(self) -> list[dict[str, Any]]:
        return list(self.rows)

    def delete_pod(self, pod_id: str) -> None:
        self.delete_calls.append(pod_id)
        if self.delete_calls.count(pod_id) >= 2:
            self.rows = [row for row in self.rows if row["id"] != pod_id]


def test_cleanup_is_exact_idempotent_and_confirms_async_absence(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = AppendOnlyRunPodLedger(ledger_path, run_id=RUN_ID)
    name = f"e025-rp-{RUN_ID}-p1-single"
    ledger.record_created(pod_id="owned", pod_name=name, stage="p1")
    provider = DelayedDeleteProvider(
        [
            {"id": "owned", "name": name},
            {"id": "unrelated", "name": "another-project"},
        ]
    )
    receipt = cleanup_from_ledger(
        provider=provider,
        ledger_path=ledger_path,
        run_id=RUN_ID,
        reason="test",
        retry_seconds=0,
    )
    assert receipt["status"] == "PASS"
    assert provider.delete_calls == ["owned", "owned"]
    assert provider.rows == [{"id": "unrelated", "name": "another-project"}]
    assert receipt["unrelated_pods_touched"] == 0
    second = cleanup_from_ledger(
        provider=provider,
        ledger_path=ledger_path,
        run_id=RUN_ID,
        reason="test-again",
        retry_seconds=0,
    )
    assert second["status"] == "PASS"
    assert provider.delete_calls == ["owned", "owned"]


def test_cleanup_requires_both_ledger_id_and_exact_name(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    name = f"e025-rp-{RUN_ID}-p1-single"
    AppendOnlyRunPodLedger(ledger_path, run_id=RUN_ID).record_created(
        pod_id="owned",
        pod_name=name,
        stage="p1",
    )
    selected = select_cleanup_targets(
        live_pods=[
            {"id": "owned", "name": "renamed-unrelated"},
            {"id": "not-owned", "name": name},
        ],
        ledger_path=ledger_path,
        run_id=RUN_ID,
    )
    assert selected == []
    assert safe_e025_name(name, RUN_ID)
    assert not safe_e025_name(f"e025-rp-{RUN_ID}-unsafe/name", RUN_ID)


def test_secret_scan_accepts_descriptors_and_rejects_raw_values(tmp_path: Path) -> None:
    (tmp_path / "safe.json").write_text(
        json.dumps(
            {
                "RUNPOD_API_KEY": {
                    "redacted": True,
                    "utf8_bytes": 40,
                    "sha256": "0" * 64,
                },
                "tokenizer_identity": "not-a-secret-key-name",
            }
        ),
        encoding="utf-8",
    )
    assert scan_runpod_artifacts(tmp_path)["status"] == "PASS"
    (tmp_path / "unsafe.json").write_text(
        json.dumps({"RUNPOD_API_KEY": "rpa_abcdefghijklmnopqrstuvwxyz123456"}),
        encoding="utf-8",
    )
    failed = scan_runpod_artifacts(tmp_path)
    assert failed["status"] == "FAIL"
    assert {row["kind"] for row in failed["findings"]} >= {
        "RAW_SECRET_VALUE",
        "RUNPOD_API_KEY_PATTERN",
    }


def test_local_p2_tls_hmac_probe_roundtrip(tmp_path: Path) -> None:
    material = create_transport_material(tmp_path / "transport", RUN_ID)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = int(reservation.getsockname()[1])
    environment = dict(os.environ)
    environment.update(
        {
            "E025_LIGHTWEIGHT_PORTS": str(port),
            "E025_RUN_CREDENTIAL_B64": base64.b64encode(
                Path(material["credential_path"]).read_bytes()
            ).decode(),
            "E025_TLS_CERT_B64": base64.b64encode(
                Path(material["certificate_path"]).read_bytes()
            ).decode(),
            "E025_TLS_KEY_B64": base64.b64encode(
                Path(material["private_key_path"]).read_bytes()
            ).decode(),
            "TEMP": str(tmp_path),
            "TMP": str(tmp_path),
        }
    )
    process = subprocess.Popen(
        [sys.executable, "-c", lightweight_start_code(network_probe=True)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        result = _p2_network_request(
            host="127.0.0.1",
            port=port,
            material=material,
            payload_bytes=7_168 * 4,
            deadline_epoch=time.time() + 15,
        )
        assert result["tls_certificate_pinned"] is True
        assert result["hmac_authenticated"] is True
        assert result["payload_bytes"] == 7_168 * 4
        assert result["end_to_end_mbps"] > 0
    finally:
        process.terminate()
        process.wait(timeout=10)
