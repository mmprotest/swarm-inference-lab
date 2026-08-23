from __future__ import annotations

import base64
import hashlib
import io
import json
import struct
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from swarm_inference.experiments.experiment_014.remote_acquisition import (
    AcquisitionError,
    activate_worker_snapshot,
)
from swarm_inference.experiments.experiment_025 import bootstrap as bootstrap_module
from swarm_inference.experiments.experiment_025 import (
    provisioning as provisioning_module,
)
from swarm_inference.experiments.experiment_025 import stages as stages_module
from swarm_inference.experiments.experiment_025 import watchdog as watchdog_module
from swarm_inference.experiments.experiment_025.constants import (
    GIB,
    ROUTED_EXPERTS,
    SUB_LAYER_TARGET,
    SUB_LAYER_WORKERS,
    TRANSFORMER_LAYERS,
)
from swarm_inference.experiments.experiment_025.controller import load_endpoints
from swarm_inference.experiments.experiment_025.headline import (
    _ready_groups_with_concurrent_backbone,
)
from swarm_inference.experiments.experiment_025.image import (
    anonymous_registry_manifest_probe,
)
from swarm_inference.experiments.experiment_025.io import (
    atomic_write_json,
    sha256_file,
)
from swarm_inference.experiments.experiment_025.placement import split_layer_worker
from swarm_inference.experiments.experiment_025.supervisor import (
    _child_environment,
    _decode_specs,
)
from swarm_inference.experiments.experiment_025.vast_lifecycle import (
    AppendOnlyLifecycleLedger,
    Offer,
    VastClient,
    rank_grouped_offers_for_workers,
    rank_offers_for_workers,
    summarize_lifecycle_costs,
)
from swarm_inference.experiments.experiment_025.wire import (
    Action,
    pack_payload,
    unpack_payload,
)
from swarm_inference.experiments.experiment_025.worker import _execution_receipt


def _source_worker() -> dict[str, object]:
    def tensor(name: str, size: int) -> dict[str, object]:
        return {
            "name": name,
            "physical_bytes": size,
            "safetensors_file": "model-00001-of-00096.safetensors",
            "byte_range": [0, size],
        }

    core_bytes = 1_267_744_256
    units: list[dict[str, object]] = [
        {
            "unit_id": "layer-89-core",
            "kind": "layer_core",
            "routed_expert": None,
            "weight_bytes": core_bytes,
            "tensors": [tensor("layer.89.core", core_bytes)],
        }
    ]
    for expert in range(ROUTED_EXPERTS):
        units.append(
            {
                "unit_id": f"layer-89-expert-{expert}",
                "kind": "routed_expert",
                "routed_expert": expert,
                "weight_bytes": 17_547_264,
                "tensors": [tensor(f"layer.89.expert.{expert}", 17_547_264)],
            }
        )
    return {
        "worker_id": "k3-worker-089",
        "worker_role": "moe_stage",
        "assignment_units": units,
        "assignment_sha256": "a" * 64,
        "owned_layers": [89],
        "owned_components": ["layer_89"],
        "checkpoint_fingerprint": "b" * 64,
        "tensor_count": len(units),
        "source_weight_bytes": core_bytes + ROUTED_EXPERTS * 17_547_264,
        "memory": {
            "planned_total_vram_bytes": 23_133_329_818,
            "safety_headroom_bytes": 2_576_980_378,
            "measured_cuda_baseline_bytes": 1_761_148_928,
        },
    }


def test_promoted_layer_89_split_is_disjoint_necessary_and_fits() -> None:
    parent, fragments, proof = split_layer_worker(_source_worker())
    assert parent["owned_expert_ids"] == []
    assert parent["local_routed_expert_count"] == 0
    assert len(fragments) == SUB_LAYER_WORKERS
    assert proof["all_experts_owned_once"] is True
    assert proof["all_complete_layers_exceed_usable_vram"] is True
    assert proof["all_fragments_fit"] is True
    assert (
        proof["negative_control"]["frozen_placement_without_sub_layer_group_valid"]
        is False
    )
    owned = [set(worker["owned_expert_ids"]) for worker in fragments]
    assert set.union(*owned) == set(range(ROUTED_EXPERTS))
    assert all(not left & right for index, left in enumerate(owned) for right in owned[index + 1 :])
    for index, worker in enumerate(fragments):
        assert all(expert % 4 == index for expert in worker["owned_expert_ids"])
        assert worker["memory"]["planned_runtime_peak_bytes"] < 8 * GIB * 0.9


def test_headline_monitors_backbone_while_preserving_fragment_parent_gate() -> None:
    backbone_started = threading.Event()
    fragments_ready = threading.Event()
    allow_backbone_finish = threading.Event()
    fragment_count = 0
    fragment_lock = threading.Lock()

    def ready_group(group_id: str) -> list[object]:
        nonlocal fragment_count
        if group_id.startswith("backbone"):
            backbone_started.set()
            assert allow_backbone_finish.wait(timeout=2.0)
            return [group_id]
        assert backbone_started.wait(timeout=2.0)
        with fragment_lock:
            fragment_count += 1
            if fragment_count == 2:
                fragments_ready.set()
        return [group_id]

    def ready_parent_after_fragments(fragments: list[object]) -> list[object]:
        assert fragments_ready.is_set()
        assert set(fragments) == {"fragment-0", "fragment-1"}
        allow_backbone_finish.set()
        return ["parent"]

    fragments, parent, backbone = _ready_groups_with_concurrent_backbone(
        fragment_group_ids=["fragment-0", "fragment-1"],
        backbone_group_ids=["backbone-0", "backbone-1"],
        ready_group=ready_group,
        ready_parent_after_fragments=ready_parent_after_fragments,
    )

    assert set(fragments) == {"fragment-0", "fragment-1"}
    assert parent == ["parent"]
    assert set(backbone) == {"backbone-0", "backbone-1"}


def test_wire_round_trip_is_typed_hashed_and_fail_closed() -> None:
    arrays = {
        "hidden": np.arange(32, dtype=np.float32).reshape(4, 8),
        "experts": np.asarray([1, 7, 9], dtype=np.int32),
    }
    payload = pack_payload(Action.EXECUTE_STAGE, {"position": 2}, arrays)
    action, metadata, observed = unpack_payload(payload)
    assert action is Action.EXECUTE_STAGE
    assert metadata == {"position": 2}
    np.testing.assert_array_equal(observed["hidden"], arrays["hidden"])
    np.testing.assert_array_equal(observed["experts"], arrays["experts"])
    tampered = bytearray(payload)
    tampered[-1] ^= 1
    with pytest.raises(ValueError, match="digest"):
        unpack_payload(bytes(tampered))
    with pytest.raises(ValueError, match="dtype"):
        pack_payload(Action.HEALTH, {}, {"bad": np.ones(2, dtype=np.float64)})


def test_execution_receipt_replaces_raw_model_arrays() -> None:
    raw = {
        "layer_output": np.arange(7168, dtype=np.float32),
        "boundary_output": np.zeros((1, 9, 7168), dtype=np.float32),
        "logits": np.zeros(163_840, dtype=np.float32),
        "selected_expert_ids": [3, 4],
    }
    receipt = _execution_receipt(raw)
    assert "layer_output" not in receipt
    assert "boundary_output" not in receipt
    assert "logits" not in receipt
    assert receipt["layer_output_shape"] == [7168]
    assert receipt["boundary_output_shape"] == [1, 9, 7168]
    assert receipt["logits_shape"] == [163_840]
    assert len(json.dumps(receipt)) < 4096


def test_live_endpoints_require_full_graph_and_distinct_fragment_machines(
    tmp_path: Path,
) -> None:
    workers = [
        {
            "worker_id": (
                f"e025-stage-{layer:03d}-parent"
                if layer == SUB_LAYER_TARGET
                else f"e025-stage-{layer:03d}"
            ),
            "role": "SUB_LAYER_PARENT" if layer == SUB_LAYER_TARGET else "BACKBONE_STAGE",
            "host": "127.0.0.1",
            "port": 42000 + layer,
            "layer": layer,
            "worker_index": None,
            "machine_id": f"backbone-{layer}",
        }
        for layer in range(TRANSFORMER_LAYERS)
    ]
    workers.extend(
        {
            "worker_id": f"e025-layer-089-sub-{index:02d}",
            "role": "SUB_LAYER_WORKER",
            "host": "127.0.0.1",
            "port": 43000 + index,
            "layer": 89,
            "worker_index": index,
            "machine_id": f"fragment-{index}",
        }
        for index in range(SUB_LAYER_WORKERS)
    )
    path = tmp_path / "endpoints.json"
    atomic_write_json(
        path,
        {"schema_version": "experiment-025-live-endpoints-v1", "workers": workers},
    )
    assert len(load_endpoints(path)) == TRANSFORMER_LAYERS + SUB_LAYER_WORKERS
    workers[-1]["machine_id"] = workers[-2]["machine_id"]
    atomic_write_json(
        path,
        {"schema_version": "experiment-025-live-endpoints-v1", "workers": workers},
    )
    with pytest.raises(ValueError, match="distinct machines"):
        load_endpoints(path)


def _offer(
    offer_id: int,
    machine_id: int,
    gpu_name: str,
    *,
    gpu_count: int = 1,
    disk_space_gb: float = 200,
) -> Offer:
    vram = 8.0 if gpu_name == "RTX 3070" else (32.0 if gpu_name == "RTX 5090" else 24.0)
    return Offer(
        offer_id=offer_id,
        machine_id=machine_id,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        gpu_ram_gib=vram,
        reliability=0.995,
        verified=True,
        rentable=True,
        dph_total=0.2 + offer_id / 1000,
        storage_cost_per_gb_month=0.1,
        inet_down_mbps=1000,
        inet_up_mbps=1000,
        inet_down_cost_per_gb=0.01,
        inet_up_cost_per_gb=0.01,
        disk_bw_mbps=1000,
        disk_space_gb=disk_space_gb,
        direct_port_count=2,
        static_ip=True,
        driver_version="580.65.06",
        cuda_max_version=13.0,
        raw={},
    )


def test_offer_plan_prioritizes_cost_and_distinct_sub_layer_machines() -> None:
    workers = [
        {
            "worker_id": f"fragment-{index}",
            "role": "SUB_LAYER_WORKER",
            "download_bytes_cold_cache": 5_000_000_000,
        }
        for index in range(4)
    ]
    workers.append(
        {
            "worker_id": "backbone-0",
            "role": "BACKBONE_STAGE",
            "download_bytes_cold_cache": 18_000_000_000,
        }
    )
    offers = [
        *[_offer(index + 1, 100 + index, "RTX 3070") for index in range(8)],
        *[_offer(100 + index, 200 + index, "RTX 3090") for index in range(4)],
    ]
    plan = rank_offers_for_workers(offers, workers, disk_gb=60)
    assert plan["status"] == "PASS"
    assert plan["worker_count"] == 5
    assert len(set(plan["sub_layer_machine_ids"])) == 4
    assert all(
        row["selected_offer"]["gpu_name"] == "RTX 3070"
        for row in plan["workers"]
        if row["role"] == "SUB_LAYER_WORKER"
    )


def test_vast_raw_offer_units_and_verification_are_normalized() -> None:
    offer = Offer.from_raw(
        {
            "id": 101,
            "machine_id": 202,
            "gpu_name": "RTX 3090",
            "num_gpus": 1,
            "gpu_ram": 24576,
            "reliability2": 0.997,
            "verification": "verified",
            "vericode": 1,
            "is_vm_deverified": False,
            "rentable": True,
            "cuda_max_good": 13.0,
            "driver_version": "580.142",
        }
    )
    assert offer.gpu_ram_gib == 24.0
    assert offer.verified is True
    assert offer.cuda_max_version == 13.0


def test_vast_deverified_machine_is_rejected() -> None:
    offer = Offer.from_raw(
        {
            "id": 101,
            "machine_id": 202,
            "gpu_name": "RTX 3090",
            "num_gpus": 1,
            "gpu_ram": 24576,
            "verification": "verified",
            "vericode": 1,
            "is_vm_deverified": True,
        }
    )
    assert offer.verified is False


def test_grouped_plan_uses_every_rented_gpu_and_preserves_machine_diversity() -> None:
    gib = 1024**3
    workers = [
        {
            "worker_id": f"e025-stage-{index:03d}",
            "role": "BACKBONE_STAGE",
            "download_bytes_cold_cache": 17 * gib,
            "assigned_tensor_bytes": 16 * gib,
            "temporary_disk_bytes": 36 * gib,
        }
        for index in [*range(89), 90, 91, 92]
    ]
    workers.append(
        {
            "worker_id": "e025-stage-089-parent",
            "role": "SUB_LAYER_PARENT",
            "download_bytes_cold_cache": 7 * gib,
            "assigned_tensor_bytes": 2 * gib,
            "temporary_disk_bytes": 12 * gib,
        }
    )
    workers.extend(
        {
            "worker_id": f"e025-layer-089-sub-{index:02d}",
            "role": "SUB_LAYER_WORKER",
            "download_bytes_cold_cache": 18 * gib,
            "assigned_tensor_bytes": 4 * gib,
            "temporary_disk_bytes": 23 * gib,
        }
        for index in range(4)
    )
    offers = [
        *[_offer(index + 1, 10 + index, "RTX 3070") for index in range(10)],
        _offer(100, 100, "RTX 3090"),
        *[
            _offer(
                200 + index,
                200 + index,
                "RTX 3090" if index < 30 else "RTX 5090",
                gpu_count=2,
                disk_space_gb=500,
            )
            for index in range(56)
        ],
    ]
    plan = rank_grouped_offers_for_workers(
        offers,
        workers,
        minimum_disk_gb=60,
        excluded_machine_ids={10},
    )
    assert plan["status"] == "PASS"
    assert plan["worker_count"] == 97
    assert len(plan["sub_layer_machine_ids"]) == 4
    assert len(set(plan["sub_layer_machine_ids"])) == 4
    assert plan["instance_group_count"] >= 40
    assert plan["compatibility_canaries_required"] == ["RTX 5090"]
    assert all(
        len(group["workers"]) == group["selected_offer"]["gpu_count"]
        for group in plan["instance_groups"]
    )
    assert all(not group["unused_rented_gpu_slots"] for group in plan["instance_groups"])
    assert plan["excluded_prior_failed_machine_ids"] == [10]
    assert all(
        group["selected_offer"]["machine_id"] != 10
        and all(row["offer"]["machine_id"] != 10 for row in group["alternates"])
        for group in plan["instance_groups"]
    )
    worker_ids = {
        worker["worker_id"]
        for group in plan["instance_groups"]
        for worker in group["workers"]
    }
    assert worker_ids == {worker["worker_id"] for worker in workers}


def test_grouped_supervisor_rejects_duplicate_gpu_slots() -> None:
    encoded = base64.b64encode(
        json.dumps(
            [
                {"worker_id": "worker-a", "gpu_slot": 0, "port": 42525},
                {"worker_id": "worker-b", "gpu_slot": 0, "port": 42526},
            ]
        ).encode()
    ).decode()
    with pytest.raises(ValueError, match="duplicate physical GPU slots"):
        _decode_specs(encoded, default_port=42525)


def test_single_worker_supervisor_preserves_parent_expert_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = base64.b64encode(b'[{"worker_id":"fragment-0"}]').decode()
    monkeypatch.setenv("E025_EXPERT_ENDPOINTS_B64", frozen)
    environment = _child_environment(
        {
            "worker_id": "e025-stage-089-parent",
            "gpu_slot": 0,
            "maximum_context": 3,
            "preserve_expert_endpoints": True,
        }
    )
    assert environment["E025_EXPERT_ENDPOINTS_B64"] == frozen


def _reusable_worker_bootstrap_fixture(
    root: Path,
) -> tuple[str, Path, Path, Path, Path]:
    worker_id = "e025-stage-001"
    manifest_root = root / "manifests"
    worker_bundle = manifest_root / "workers" / worker_id
    shared = manifest_root / "shared"
    state_root = root / "state"
    runtime_root = state_root / worker_id
    worker_bundle.mkdir(parents=True)
    shared.mkdir(parents=True)
    runtime_root.mkdir(parents=True)

    config_path = shared / "config.json"
    config_path.write_text('{"model_type":"kimi_k3"}\n', encoding="utf-8")
    fingerprint = "b" * 64
    assignment_sha = "a" * 64
    tensor_name = "language_model.model.layers.1.test.weight"
    tensor = {
        "name": tensor_name,
        "physical_bytes": 4,
    }
    placement = {
        "status": "PASS",
        "checkpoint": {
            "checkpoint_fingerprint": fingerprint,
            "revision": "fixture-revision",
            "config_sha256": sha256_file(config_path),
        },
        "workers": [
            {
                "worker_id": worker_id,
                "worker_role": "moe_stage",
                "assignment_units": [{"tensors": [tensor]}],
                "assignment_sha256": assignment_sha,
                "owned_layers": [1],
                "owned_components": ["layer_1"],
                "source_weight_bytes": 4,
            }
        ],
    }
    placement_path = worker_bundle / "placement.json"
    atomic_write_json(placement_path, placement)
    distribution_path = worker_bundle / "distribution.json"
    atomic_write_json(
        distribution_path,
        {
            "status": "PASS",
            "workers": [{"worker_id": worker_id}],
        },
    )
    atomic_write_json(
        worker_bundle / "worker-template.json",
        {
            "worker_id": worker_id,
            "role": "BACKBONE_STAGE",
            "worker_index": 1,
            "worker_count": 93,
            "assignment_sha256": assignment_sha,
            "checkpoint_fingerprint": fingerprint,
            "topology_id": "fixture-topology",
        },
    )

    header = {
        "__metadata__": {
            "schema": "experiment-014-k3-worker-package-v1",
            "worker_id": worker_id,
            "checkpoint_fingerprint": fingerprint,
        },
        tensor_name: {
            "dtype": "BF16",
            "shape": [2],
            "data_offsets": [0, 4],
        },
    }
    encoded_header = json.dumps(
        header, sort_keys=True, separators=(",", ":")
    ).encode()
    encoded_header += b" " * (-len(encoded_header) % 8)
    package = runtime_root / "model.safetensors"
    package.write_bytes(struct.pack("<Q", len(encoded_header)) + encoded_header + b"\0" * 4)
    snapshot = runtime_root / "snapshot"
    activation = activate_worker_snapshot(
        placement_path,
        worker_id,
        package,
        config_path,
        snapshot,
    )
    acquisition = {
        "schema_version": "experiment-014-k3-remote-acquisition-v1",
        "status": "PASS",
        "worker_id": worker_id,
        "distribution_manifest": str(distribution_path.resolve()),
        "distribution_manifest_sha256": sha256_file(distribution_path),
        "downloaded_bytes": 123,
        "package": {
            "status": "PASS",
            "worker_id": worker_id,
            "package_sha256": activation["package_sha256"],
        },
    }
    atomic_write_json(
        runtime_root / "bootstrap.json",
        {
            "schema_version": "experiment-025-worker-bootstrap-v1",
            "status": "PASS",
            "worker_id": worker_id,
            "acquisition": acquisition,
            "activation": activation,
        },
    )
    cuda_library = root / "libcoli_cuda-consumer.so"
    cuda_library.write_bytes(b"fixture-cuda-library")
    return worker_id, manifest_root, state_root, cuda_library, snapshot


def test_prepare_worker_reuses_only_an_exact_verified_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_id, manifest_root, state_root, cuda_library, snapshot = (
        _reusable_worker_bootstrap_fixture(tmp_path)
    )
    snapshot_sha = sha256_file(snapshot / "model.safetensors")

    def unexpected_reacquisition(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("verified snapshot must not be reacquired or reactivated")

    monkeypatch.setattr(
        bootstrap_module, "acquire_worker_package", unexpected_reacquisition
    )
    monkeypatch.setattr(
        bootstrap_module, "activate_worker_snapshot", unexpected_reacquisition
    )
    monkeypatch.setenv("E025_RUN_CREDENTIAL_B64", base64.b64encode(b"credential").decode())
    monkeypatch.setenv("E025_TLS_CERT_B64", base64.b64encode(b"certificate").decode())
    monkeypatch.setenv("E025_TLS_KEY_B64", base64.b64encode(b"private-key").decode())

    prepared = bootstrap_module.prepare_worker(
        worker_id=worker_id,
        manifest_root=manifest_root,
        state_root=state_root,
        cuda_library=cuda_library,
    )

    assert prepared["status"] == "PASS"
    assert (
        prepared["acquisition"]["bootstrap_attempt_mode"]
        == "REUSED_VERIFIED_EXISTING_SNAPSHOT"
    )
    assert prepared["acquisition"]["bootstrap_attempt_downloaded_bytes"] == 0
    assert prepared["acquisition"]["downloaded_bytes"] == 123
    assert prepared["activation"]["reused_verified_existing_snapshot"] is True
    assert sha256_file(snapshot / "model.safetensors") == snapshot_sha
    assert not (state_root / worker_id / "model.safetensors.partial").exists()


def test_prepare_worker_rejects_corrupt_existing_snapshot_without_reacquiring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_id, manifest_root, state_root, cuda_library, snapshot = (
        _reusable_worker_bootstrap_fixture(tmp_path)
    )
    weights = snapshot / "model.safetensors"
    with weights.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"x")

    def unexpected_reacquisition(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("invalid existing snapshot must fail closed")

    monkeypatch.setattr(
        bootstrap_module, "acquire_worker_package", unexpected_reacquisition
    )
    monkeypatch.setattr(
        bootstrap_module, "activate_worker_snapshot", unexpected_reacquisition
    )
    with pytest.raises(AcquisitionError, match="activated weight hash differs"):
        bootstrap_module.prepare_worker(
            worker_id=worker_id,
            manifest_root=manifest_root,
            state_root=state_root,
            cuda_library=cuda_library,
        )


def test_lifecycle_ledger_is_hash_chained_and_create_requires_arming(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = AppendOnlyLifecycleLedger(path, "run-1")
    first = ledger.append("TEST", instance_id=None)
    second = ledger.append("TEST_2", instance_id=42)
    assert second["previous_entry_sha256"] == first["entry_sha256"]
    assert len(ledger.entries()) == 2

    client = VastClient(executable="vastai", ledger=ledger)
    offer = _offer(1, 10, "RTX 3090")
    with pytest.raises(RuntimeError, match="watchdog and GO"):
        client.create_instance(
            run_id="run-1",
            role="BACKBONE_STAGE",
            index=0,
            offer=offer,
            image="example.invalid/e025@sha256:" + "0" * 64,
            disk_gb=60,
            env_options="-p 42525:42525",
            watchdog_receipt=tmp_path / "missing-watchdog.json",
            go_receipt=tmp_path / "missing-go.json",
        )

    rows = path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(rows[0])
    tampered["event"] = "ALTERED"
    rows[0] = json.dumps(tampered)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        ledger.entries()


def test_watchdog_startup_timeout_prearms_cleanup_trigger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _DelayedProcess:
        pid = 4242

        @staticmethod
        def poll() -> None:
            return None

    monkeypatch.setattr(
        watchdog_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: _DelayedProcess(),
    )
    trigger = tmp_path / "WATCHDOG_TRIGGER"
    with pytest.raises(TimeoutError, match="RUNNING receipt"):
        watchdog_module.start_watchdog(
            run_id="run-1",
            stage="canary",
            ledger_path=tmp_path / "instance-ledger.jsonl",
            ttl_seconds=15 * 60,
            receipt_path=tmp_path / "watchdog-receipt.json",
            log_path=tmp_path / "watchdog.jsonl",
            trigger_path=trigger,
            stop_path=tmp_path / "WATCHDOG_STOP",
            cleanup_receipt_path=tmp_path / "cleanup-verification.json",
            startup_timeout_seconds=0.01,
        )
    trigger_receipt = json.loads(trigger.read_text(encoding="utf-8"))
    assert trigger_receipt["reason"] == "WATCHDOG_STARTUP_RECEIPT_TIMEOUT"
    assert trigger_receipt["watchdog_pid"] == 4242
    assert watchdog_module.WATCHDOG_STARTUP_TIMEOUT_SECONDS == 60.0


def test_watchdog_accepts_nonce_bound_receipt_when_windows_pid_differs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "watchdog-receipt.json"

    class _WindowsLauncher:
        pid = 4242

        @staticmethod
        def poll() -> None:
            return None

    def _launch(arguments: list[str], **_kwargs: object) -> _WindowsLauncher:
        nonce = arguments[arguments.index("--startup-nonce") + 1]
        atomic_write_json(
            receipt_path,
            {
                "status": "RUNNING",
                "run_id": "run-1",
                "stage": "canary",
                "watchdog_pid": 9001,
                "startup_nonce_sha256": watchdog_module.hashlib.sha256(
                    nonce.encode("utf-8")
                ).hexdigest(),
            },
        )
        return _WindowsLauncher()

    monkeypatch.setattr(watchdog_module.subprocess, "Popen", _launch)
    receipt = watchdog_module.start_watchdog(
        run_id="run-1",
        stage="canary",
        ledger_path=tmp_path / "instance-ledger.jsonl",
        ttl_seconds=15 * 60,
        receipt_path=receipt_path,
        log_path=tmp_path / "watchdog.jsonl",
        trigger_path=tmp_path / "WATCHDOG_TRIGGER",
        stop_path=tmp_path / "WATCHDOG_STOP",
        cleanup_receipt_path=tmp_path / "cleanup-verification.json",
        startup_timeout_seconds=1.0,
    )
    assert receipt["watchdog_pid"] == 9001
    assert receipt["launcher_pid"] == 4242
    assert receipt["watchdog_pid_matches_launcher"] is False
    assert receipt["startup_identity"] == "per-launch SHA-256 nonce"


def test_watchdog_receipt_write_retries_transient_windows_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "watchdog-receipt.json"
    real_atomic_write_json = watchdog_module.atomic_write_json
    attempts = 0
    sleeps: list[float] = []

    def _flaky_write(path: Path, value: dict[str, object]) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("simulated Windows reader lock")
        real_atomic_write_json(path, value)

    monkeypatch.setattr(watchdog_module, "atomic_write_json", _flaky_write)
    monkeypatch.setattr(watchdog_module.time, "sleep", sleeps.append)
    watchdog_module._write_receipt(receipt_path, {"status": "RUNNING"})

    assert attempts == 2
    assert sleeps == [watchdog_module.WATCHDOG_RECEIPT_WRITE_RETRY_SECONDS]
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == {
        "status": "RUNNING"
    }


def test_watchdog_delays_first_heartbeat_until_after_startup_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    trigger_path = tmp_path / "WATCHDOG_TRIGGER"

    monkeypatch.setattr(
        watchdog_module,
        "_write_receipt",
        lambda *_args, **_kwargs: events.append("receipt"),
    )
    monkeypatch.setattr(
        watchdog_module,
        "append_jsonl",
        lambda *_args, **_kwargs: events.append("log"),
    )
    monkeypatch.setattr(watchdog_module.time, "time", lambda: 0.0)

    def _sleep(_seconds: float) -> None:
        events.append("sleep")
        trigger_path.write_text("triggered\n", encoding="utf-8")

    monkeypatch.setattr(watchdog_module.time, "sleep", _sleep)
    monkeypatch.setattr(
        watchdog_module,
        "destroy_all_from_ledger",
        lambda **_kwargs: {"zero_live_e025_instances": True},
    )
    monkeypatch.setattr(watchdog_module, "atomic_write_json", lambda *_args: None)

    assert (
        watchdog_module.run_watchdog(
            run_id="run-1",
            stage="canary",
            ledger_path=tmp_path / "instance-ledger.jsonl",
            deadline_epoch=100.0,
            receipt_path=tmp_path / "watchdog-receipt.json",
            log_path=tmp_path / "watchdog.jsonl",
            trigger_path=trigger_path,
            stop_path=tmp_path / "WATCHDOG_STOP",
            cleanup_receipt_path=tmp_path / "cleanup-verification.json",
            vast_executable="vastai",
            startup_nonce="nonce",
        )
        == 0
    )
    assert events[:3] == ["receipt", "log", "sleep"]


def test_lifecycle_cost_summary_uses_observed_create_to_destroy_interval(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cost-ledger.jsonl"
    ledger = AppendOnlyLifecycleLedger(path, "run-cost")
    ledger.append(
        "CREATE_CONFIRMED",
        instance_id=42,
        offer_id=7,
        machine_id=9,
        gpu_model="RTX 3090",
        gpu_count=1,
        creation_time="2026-08-19T00:00:00+00:00",
        active_rental_rate_usd_per_hour=1.0,
        storage_rate_usd_per_gb_month=0.72,
        requested_disk_gb=100,
    )
    ledger.append(
        "DESTROY_CONFIRMED",
        instance_id=42,
        destroy_confirmed_time="2026-08-19T01:00:00+00:00",
    )
    summary = summarize_lifecycle_costs(path, "run-cost")
    assert summary["status"] == "PASS"
    assert summary["total_elapsed_instance_seconds"] == 3600
    assert summary["estimated_active_cost_usd"] == pytest.approx(1.0)
    assert summary["estimated_storage_cost_usd"] == pytest.approx(0.1)
    assert "not represented as a provider invoice" in summary["basis"]


def test_worker_readiness_fails_fast_after_seen_instance_disappears(
    tmp_path: Path,
) -> None:
    class _VanishingClient:
        calls = 0

        def show_instances(self) -> list[dict[str, object]]:
            self.calls += 1
            if self.calls == 1:
                return [
                    {
                        "id": 42,
                        "machine_id": 10,
                        "actual_status": "loading",
                        "label": "e025-run-1-backbone_stage-001",
                    }
                ]
            return []

    client = _VanishingClient()
    ledger = AppendOnlyLifecycleLedger(tmp_path / "ledger.jsonl", "run-1")
    with pytest.raises(RuntimeError, match="disappeared after appearing"):
        provisioning_module.wait_for_worker(
            client=client,  # type: ignore[arg-type]
            ledger=ledger,
            run_id="run-1",
            worker_id="worker-1",
            role="BACKBONE_STAGE",
            layer=1,
            worker_index=None,
            offer=_offer(1, 10, "RTX 3090"),
            instance_id=42,
            credential=b"test-credential",
            certificate=tmp_path / "unused.crt",
            image_digest="sha256:" + "0" * 64,
            deadline_epoch=provisioning_module.time.time() + 60,
            poll_seconds=0,
            missing_after_seen_polls=3,
        )
    assert client.calls == 4
    assert [row["event"] for row in ledger.entries()] == ["INSTANCE_RUNNING"]


def test_worker_readiness_bounds_initial_instance_absence(tmp_path: Path) -> None:
    class _AbsentClient:
        calls = 0

        def show_instances(self) -> list[dict[str, object]]:
            self.calls += 1
            return []

    client = _AbsentClient()
    ledger = AppendOnlyLifecycleLedger(tmp_path / "ledger.jsonl", "run-1")
    with pytest.raises(TimeoutError, match="never appeared"):
        provisioning_module.wait_for_worker(
            client=client,  # type: ignore[arg-type]
            ledger=ledger,
            run_id="run-1",
            worker_id="worker-1",
            role="BACKBONE_STAGE",
            layer=1,
            worker_index=None,
            offer=_offer(1, 10, "RTX 3090"),
            instance_id=42,
            credential=b"test-credential",
            certificate=tmp_path / "unused.crt",
            image_digest="sha256:" + "0" * 64,
            deadline_epoch=provisioning_module.time.time() + 60,
            poll_seconds=0,
            initial_presence_timeout_seconds=0,
        )
    assert client.calls == 1
    assert ledger.entries() == []


def test_public_endpoint_is_frozen_before_authenticated_readiness(
    tmp_path: Path,
) -> None:
    class _EndpointClient:
        def show_instances(self) -> list[dict[str, object]]:
            return [
                {
                    "id": 42,
                    "machine_id": 10,
                    "actual_status": "loading",
                    "ports": {
                        "42525/tcp": [
                            {"HostIp": "203.0.113.9", "HostPort": "30123"}
                        ]
                    },
                }
            ]

    ledger = AppendOnlyLifecycleLedger(tmp_path / "ledger.jsonl", "run-1")
    endpoint = provisioning_module.wait_for_public_endpoint(
        client=_EndpointClient(),  # type: ignore[arg-type]
        ledger=ledger,
        worker_id="fragment-0",
        offer=_offer(1, 10, "RTX 3070"),
        instance_id=42,
        deadline_epoch=provisioning_module.time.time() + 60,
        poll_seconds=0,
    )
    assert endpoint == ("203.0.113.9", 30123)
    assert ledger.entries()[0]["event"] == "ENDPOINT_PUBLISHED"
    assert ledger.entries()[0]["final_status"] == "ENDPOINT_ONLY_NOT_READY"


def test_paid_stage_excludes_prior_machine_that_never_became_ready(
    tmp_path: Path,
) -> None:
    rental = tmp_path / "rental"
    failed = rental / "stage-1-backbone-canary-attempt-failed"
    passed = rental / "stage-1-backbone-canary-attempt-passed"
    numerical = rental / "stage-1-backbone-canary-attempt-numerical-fail"
    current = rental / "stage-1-backbone-canary"
    failed.mkdir(parents=True)
    passed.mkdir(parents=True)
    numerical.mkdir(parents=True)
    current.mkdir(parents=True)
    failed_ledger = AppendOnlyLifecycleLedger(
        failed / "instance-ledger.jsonl", "run-1"
    )
    failed_ledger.append(
        "CREATE_CONFIRMED",
        instance_id=42,
        offer_id=7,
        machine_id=10,
    )
    passed_ledger = AppendOnlyLifecycleLedger(
        passed / "instance-ledger.jsonl", "run-1"
    )
    passed_ledger.append(
        "CREATE_CONFIRMED",
        instance_id=84,
        offer_id=8,
        machine_id=20,
    )
    passed_ledger.append(
        "WORKER_READY",
        instance_id=84,
        offer_id=8,
        machine_id=20,
    )
    numerical_ledger = AppendOnlyLifecycleLedger(
        numerical / "instance-ledger.jsonl", "run-1"
    )
    numerical_ledger.append(
        "CREATE_CONFIRMED",
        instance_id=126,
        offer_id=9,
        machine_id=30,
    )
    numerical_ledger.append(
        "WORKER_READY",
        instance_id=126,
        offer_id=9,
        machine_id=30,
    )
    atomic_write_json(
        numerical / "physical-canary.json",
        {"status": "FAIL", "frozen_numerical_gate": 1e-4},
    )
    receipt = stages_module._failed_machine_exclusions(
        current,
        stage_prefix="stage-1-backbone-canary",
    )
    assert receipt["machine_ids"] == [10, 30]
    assert receipt["sources"][0]["observations"][0]["instance_id"] == 42
    assert (
        receipt["sources"][1]["observations"][0]["reason"]
        == "prior stage frozen physical canary failed"
    )


def test_measured_machine_selection_requires_retained_physical_pass(
    tmp_path: Path,
) -> None:
    rental = tmp_path / "rental"
    passed = rental / "stage-1-backbone-canary-pass"
    current = rental / "stage-1-backbone-canary"
    passed.mkdir(parents=True)
    current.mkdir(parents=True)
    atomic_write_json(
        passed / "backbone-canary-result.json",
        {
            "status": "PASS",
            "image_digest": "sha256:historical",
            "worker": {"machine_id": "20"},
        },
    )
    offers = [_offer(1, 10, "RTX 3090"), _offer(2, 20, "RTX 3090")]
    selected, receipt = stages_module._prior_passing_machine_selection(
        current,
        offers,
        machine_id=20,
    )
    assert [offer.machine_id for offer in selected] == [20]
    assert receipt["machine_id"] == 20
    assert receipt["physical_pass_evidence"][0]["status"] == "PASS"
    with pytest.raises(RuntimeError, match="no retained passing physical canary"):
        stages_module._prior_passing_machine_selection(
            current,
            offers,
            machine_id=10,
        )


class _RegistryResponse:
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._body = body
        self.status = status
        self.headers = headers or {}

    def __enter__(self) -> _RegistryResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _maximum: int = -1) -> bytes:
        return self._body

    def getcode(self) -> int:
        return self.status


def _registry_unauthorized(
    request: urllib.request.Request, challenge: str | None = None
) -> urllib.error.HTTPError:
    headers = {"WWW-Authenticate": challenge} if challenge is not None else {}
    return urllib.error.HTTPError(
        request.full_url,
        401,
        "Unauthorized",
        headers,
        io.BytesIO(b""),
    )


def test_anonymous_registry_probe_uses_only_public_bearer_flow() -> None:
    manifest = b'{"schemaVersion":2,"kind":"public-test"}'
    digest = "sha256:" + hashlib.sha256(manifest).hexdigest()
    calls: list[urllib.request.Request] = []

    def opener(
        request: urllib.request.Request, *, timeout: float
    ) -> _RegistryResponse:
        assert timeout == 5
        calls.append(request)
        if len(calls) == 1:
            raise _registry_unauthorized(
                request,
                'Bearer realm="https://ghcr.io/token",service="ghcr.io",'
                'scope="repository:mmprotest/swarm-inference-lab:pull"',
            )
        if "/token?" in request.full_url:
            assert request.get_header("Authorization") is None
            return _RegistryResponse(b'{"token":"anonymous-ephemeral-token"}')
        assert (
            request.get_header("Authorization") == "Bearer anonymous-ephemeral-token"
        )
        return _RegistryResponse(
            manifest,
            headers={
                "Docker-Content-Digest": digest,
                "Content-Type": "application/vnd.oci.image.index.v1+json",
            },
        )

    result = anonymous_registry_manifest_probe(
        f"ghcr.io/mmprotest/swarm-inference-lab@{digest}",
        timeout_seconds=5,
        opener=opener,
    )
    assert result["status"] == "PASS"
    assert result["authentication"] == "anonymous_bearer_token"
    assert result["digest_match"] is True
    assert result["credentials_supplied"] is False
    assert result["docker_credential_store_consulted"] is False
    assert "anonymous-ephemeral-token" not in json.dumps(result)
    assert len(calls) == 3


def test_anonymous_registry_probe_rejects_private_manifest() -> None:
    digest = "sha256:" + "1" * 64
    calls: list[urllib.request.Request] = []

    def opener(
        request: urllib.request.Request, *, timeout: float
    ) -> _RegistryResponse:
        assert timeout == 5
        calls.append(request)
        if len(calls) == 1:
            raise _registry_unauthorized(
                request,
                'Bearer realm="https://ghcr.io/token",service="ghcr.io"',
            )
        if "/token?" in request.full_url:
            return _RegistryResponse(b'{"token":"anonymous-private-token"}')
        raise _registry_unauthorized(request)

    result = anonymous_registry_manifest_probe(
        f"ghcr.io/mmprotest/swarm-inference-lab@{digest}",
        timeout_seconds=5,
        opener=opener,
    )
    assert result["status"] == "FAIL"
    assert result["failed_stage"] == "anonymous_manifest_request"
    assert result["http_status"] == 401
    assert "anonymous-private-token" not in json.dumps(result)
