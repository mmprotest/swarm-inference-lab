"""Acquire, run, prove, and destroy the complete physical E025 fleet."""

from __future__ import annotations

import math
import os
import subprocess
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .constants import (
    EVIDENCE_CLASS,
    FULL_FLEET_TTL_SECONDS,
    GIB,
    MIB,
    MODEL_ID,
    MODEL_REVISION,
    SUB_LAYER_WORKERS,
    TRANSFORMER_LAYERS,
)
from .controller import run_physical_generation
from .io import atomic_write_json, read_json, sha256_file, utc_now
from .provisioning import (
    LiveWorker,
    WorkerCompatibilityError,
    create_worker_group_instance,
    wait_for_worker,
    write_live_endpoints,
)
from .secrets import load_transport_material
from .vast_lifecycle import (
    AppendOnlyLifecycleLedger,
    Offer,
    VastClient,
    destroy_all_from_ledger,
    summarize_lifecycle_costs,
)
from .watchdog import start_watchdog


def _local_gpu_process_snapshot() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) == 3 and fields[0].isdigit():
            try:
                used_memory = int(fields[2])
            except ValueError:
                used_memory = -1
            rows.append(
                {
                    "pid": int(fields[0]),
                    "process_name": fields[1],
                    "used_gpu_memory_mib": used_memory,
                }
            )
    controller_pid = os.getpid()
    return {
        "timestamp": utc_now(),
        "controller_pid": controller_pid,
        "nvidia_smi_returncode": result.returncode,
        "compute_processes": rows,
        "controller_is_gpu_compute_process": any(
            int(row["pid"]) == controller_pid for row in rows
        ),
    }


def _spec(worker_id: str, role: str) -> dict[str, Any]:
    if role == "SUB_LAYER_WORKER":
        return {
            "worker_id": worker_id,
            "role": role,
            "layer": 89,
            "worker_index": int(worker_id.rsplit("-", maxsplit=1)[1]),
        }
    if role == "SUB_LAYER_PARENT":
        return {"worker_id": worker_id, "role": role, "layer": 89, "worker_index": None}
    return {
        "worker_id": worker_id,
        "role": role,
        "layer": int(worker_id.rsplit("-", maxsplit=1)[1]),
        "worker_index": None,
    }


def _offer_candidates(row: dict[str, Any]) -> list[Offer]:
    return [
        Offer(**row["selected_offer"]),
        *(Offer(**value["offer"]) for value in row.get("alternates", [])),
    ]


def _ready_groups_with_concurrent_backbone(
    *,
    fragment_group_ids: list[str],
    backbone_group_ids: list[str],
    ready_group: Callable[[str], list[LiveWorker]],
    ready_parent_after_fragments: Callable[[list[LiveWorker]], list[LiveWorker]],
) -> tuple[list[LiveWorker], list[LiveWorker], list[LiveWorker]]:
    """Monitor backbone readiness while preserving the fragment-before-parent gate."""

    with ThreadPoolExecutor(max_workers=max(1, min(32, len(backbone_group_ids)))) as pool:
        backbone_futures = {
            pool.submit(ready_group, group_id): group_id for group_id in backbone_group_ids
        }
        with ThreadPoolExecutor(max_workers=len(fragment_group_ids)) as fragment_pool:
            fragment_groups = list(fragment_pool.map(ready_group, fragment_group_ids))
        fragments = [worker for group in fragment_groups for worker in group]
        parent = ready_parent_after_fragments(fragments)
        backbone: list[LiveWorker] = []
        for future in as_completed(backbone_futures):
            backbone.extend(future.result())
    return fragments, parent, backbone


def _worker_memory_proof(
    fragments: list[LiveWorker],
    physical_placement: dict[str, Any],
    generation: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    complete_peak = int(
        physical_placement["sub_layer_proof"]["complete_layer_runtime_peak_bytes"]
    )
    generated_token_count = len((generation or {}).get("generated_token_ids", []))
    retained_records = (
        list((generation or {}).get("token_records", []))[-generated_token_count:]
        if generated_token_count
        else []
    )
    rows: list[dict[str, Any]] = []
    for worker in fragments:
        actual_vram = int(worker.ready["gpu"]["vram_mib"]) * MIB
        usable = int(actual_vram * 0.9)
        memory = worker.ready["executor"]["memory_after_load"]
        measured_used = int(memory["total_bytes"]) - int(memory["free_bytes"])
        fragment_peak = measured_used + 448 * MIB
        physical_receipts = [
            receipt
            for token in retained_records
            for receipt in token.get("sub_layer_participation", {}).get("workers", [])
            if str(receipt.get("worker_id")) == worker.worker_id
        ]
        rows.append(
            {
                "worker_id": worker.worker_id,
                "instance_id": worker.instance_id,
                "machine_id": worker.machine_id,
                "gpu_name": worker.ready["gpu"]["gpu_name"],
                "gpu_uuid": worker.ready["gpu"]["gpu_uuid"],
                "physical_vram_bytes": actual_vram,
                "physical_vram_gib": actual_vram / GIB,
                "usable_vram_bytes": usable,
                "complete_layer_peak_bytes": complete_peak,
                "complete_layer_peak_gib": complete_peak / GIB,
                "fragment_measured_used_bytes": measured_used,
                "fragment_reserve_bytes": 448 * MIB,
                "fragment_peak_bytes": fragment_peak,
                "fragment_peak_gib": fragment_peak / GIB,
                "complete_layer_fits": complete_peak <= usable,
                "fragment_fits": fragment_peak <= usable,
                "assigned_layer": 89,
                "assigned_expert_count": worker.ready["assignment"][
                    "owned_expert_count"
                ],
                "assigned_expert_ids": worker.ready["assignment"]["owned_expert_ids"],
                "assigned_source_bytes": worker.ready["assignment"][
                    "source_weight_bytes"
                ],
                "native_primitive": worker.ready["executor"]["native_primitive"],
                "retained_headline_token_count": generated_token_count,
                "retained_headline_execution_count": len(physical_receipts),
                "participated_in_every_retained_headline_token": bool(
                    generated_token_count
                    and len(physical_receipts) == generated_token_count
                ),
                "retained_headline_native_expert_calls": sum(
                    int(receipt.get("native_expert_calls", 0))
                    for receipt in physical_receipts
                ),
                "retained_headline_request_wire_bytes": sum(
                    int(receipt.get("request_wire_bytes", 0))
                    for receipt in physical_receipts
                ),
                "retained_headline_response_wire_bytes": sum(
                    int(receipt.get("response_wire_bytes", 0))
                    for receipt in physical_receipts
                ),
            }
        )
    return rows


def _retrieve_logs(
    client: VastClient,
    workers: list[LiveWorker],
    destination: Path,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    def retrieve(worker: LiveWorker) -> None:
        try:
            text = client.logs(worker.instance_id, tail=2500)
        except BaseException as exc:
            text = f"LOG_RETRIEVAL_FAILED: {type(exc).__name__}: {exc}\n"
        (destination / f"{worker.worker_id}.log").write_text(text, encoding="utf-8")

    unique = {worker.instance_id: worker for worker in workers}
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(retrieve, unique.values()))


def _write_worker_receipts(
    *,
    workers: dict[str, LiveWorker],
    worker_rows: dict[str, dict[str, Any]],
    correctness: dict[str, Any] | None,
    generation: dict[str, Any] | None,
    destination: Path,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for worker_id, worker in sorted(workers.items()):
        phases: dict[str, dict[str, int]] = {}
        for phase, result in (("correctness", correctness), ("headline", generation)):
            executions = 0
            native_expert_calls = 0
            request_bytes = 0
            response_bytes = 0
            for token in (result or {}).get("token_records", []):
                for stage in token.get("stages", []):
                    if str(stage.get("worker_id")) == worker_id:
                        executions += 1
                        request_bytes += int(stage["transport"]["request_wire_bytes"])
                        response_bytes += int(stage["transport"]["response_wire_bytes"])
                    participation = stage.get("execution", {}).get(
                        "external_expert_dispatch", {}
                    )
                    for fragment in participation.get("workers", []):
                        if str(fragment.get("worker_id")) == worker_id:
                            executions += 1
                            native_expert_calls += int(
                                fragment.get("native_expert_calls", 0)
                            )
                            request_bytes += int(fragment.get("request_wire_bytes", 0))
                            response_bytes += int(fragment.get("response_wire_bytes", 0))
            phases[phase] = {
                "execution_count": executions,
                "native_expert_calls": native_expert_calls,
                "request_wire_bytes": request_bytes,
                "response_wire_bytes": response_bytes,
            }
        row = worker_rows[worker_id]
        payload = {
            "schema_version": "experiment-025-physical-worker-evidence-v1",
            "generated_at_utc": utc_now(),
            "worker_id": worker_id,
            "role": worker.role,
            "layer": worker.layer,
            "worker_index": worker.worker_index,
            "instance_group_id": row["instance_group_id"],
            "instance_id": worker.instance_id,
            "machine_id": worker.machine_id,
            "host_redacted": True,
            "port": worker.port,
            "gpu_slot": worker.gpu_slot,
            "gpu": worker.ready["gpu"],
            "assignment": worker.ready["assignment"],
            "assignment_sha256": worker.ready["assignment_sha256"],
            "checkpoint_fingerprint": worker.ready["checkpoint_fingerprint"],
            "image_digest": worker.ready["image_digest"],
            "native_binary_sha256": worker.ready["cuda_library_sha256"],
            "snapshot_activation_sha256": worker.ready[
                "snapshot_activation_sha256"
            ],
            "bootstrap": worker.ready.get("bootstrap"),
            "native_execution_prepare": worker.ready.get("prepare"),
            "whole_layer_fallback": worker.ready["whole_layer_fallback"],
            "controller_compute_fallback": worker.ready[
                "controller_compute_fallback"
            ],
            "execution": phases,
            "post_run_health": {
                phase: (result or {}).get("health", {}).get("workers", {}).get(
                    worker_id
                )
                for phase, result in (
                    ("correctness", correctness),
                    ("headline", generation),
                )
            },
        }
        atomic_write_json(destination / f"{worker_id}.json", payload)


def _write_central_telemetry(
    *,
    correctness: dict[str, Any] | None,
    generation: dict[str, Any] | None,
    failures: list[dict[str, Any]],
    local_gpu_samples: list[dict[str, Any]],
    destination: Path,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    def token_timing(result: dict[str, Any] | None) -> list[dict[str, Any]]:
        return [
            {
                "position": row.get("position"),
                "input_token_id": row.get("input_token_id"),
                "sampled_token_id": row.get("sampled_token_id"),
                "started_at_utc": row.get("started_at_utc"),
                "completed_at_utc": row.get("completed_at_utc"),
                "elapsed_seconds": row.get("elapsed_seconds"),
            }
            for row in (result or {}).get("token_records", [])
        ]

    payload = {
        "schema_version": "experiment-025-central-physical-telemetry-v1",
        "generated_at_utc": utc_now(),
        "correctness": {
            "status": (correctness or {}).get("status"),
            "token_timing": token_timing(correctness),
            "network_wire_bytes": (correctness or {}).get("network_wire_bytes"),
            "worker_health": (correctness or {}).get("health", {}).get("workers", {}),
        },
        "headline": {
            "status": (generation or {}).get("status"),
            "token_timing": token_timing(generation),
            "network_wire_bytes": (generation or {}).get("network_wire_bytes"),
            "worker_health": (generation or {}).get("health", {}).get("workers", {}),
        },
        "offer_or_worker_failures": failures,
        "local_controller_gpu_monitor_samples": local_gpu_samples,
        "secrets_retained": False,
    }
    atomic_write_json(destination / "physical-telemetry.json", payload)


def run_headline_stage(
    *,
    run_id: str,
    stage_root: Path,
    full_fleet_go_path: Path,
    image_receipt_path: Path,
    private_root: Path,
    physical_placement_path: Path,
    checkpoint: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    correctness_output_path: Path,
    generation_output_path: Path,
    summary_output_path: Path,
    prompt: str = "Hi",
    max_new_tokens: int = 16,
    disk_gb: int = 60,
    vast_executable: str = "vastai",
) -> dict[str, Any]:
    stage_root.mkdir(parents=True, exist_ok=True)
    go = read_json(full_fleet_go_path)
    image = read_json(image_receipt_path)
    placement = read_json(physical_placement_path)
    if go.get("status") != "GO" or go.get("run_id") != run_id:
        raise RuntimeError("E025 full headline fleet is not in a frozen GO state")
    if image.get("status") != "PASS" or not image.get("immutable_reference"):
        raise RuntimeError("E025 headline image is not published and immutable")
    plan = go["fleet_plan"]
    if int(plan["worker_count"]) != TRANSFORMER_LAYERS + SUB_LAYER_WORKERS:
        raise RuntimeError("E025 full fleet GO does not contain all 97 workers")
    image_reference = str(image["immutable_reference"])
    image_digest = str(image["immutable_digest"])
    material = load_transport_material(private_root, run_id)
    ledger_path = stage_root / "instance-ledger.jsonl"
    if ledger_path.is_file() and ledger_path.stat().st_size:
        raise RuntimeError("E025 full headline stage ledger already exists")
    ledger = AppendOnlyLifecycleLedger(ledger_path, run_id)
    client = VastClient(executable=vast_executable, ledger=ledger)
    live_e025 = [
        row
        for row in client.show_instances()
        if str(row.get("label", "")).lower().startswith("e025-")
    ]
    if live_e025:
        live_ids = [row.get("id", row.get("instance_id")) for row in live_e025]
        raise RuntimeError(
            f"E025 refuses full acquisition while E025 instances are live: {live_ids}"
        )
    watchdog_paths = {
        "receipt": stage_root / "watchdog-receipt.json",
        "log": stage_root / "watchdog.jsonl",
        "trigger": stage_root / "WATCHDOG_TRIGGER",
        "stop": stage_root / "WATCHDOG_STOP",
        "cleanup": stage_root / "cleanup-verification.json",
    }
    watchdog = start_watchdog(
        run_id=run_id,
        stage="full-headline-fleet",
        ledger_path=ledger_path,
        ttl_seconds=FULL_FLEET_TTL_SECONDS,
        receipt_path=watchdog_paths["receipt"],
        log_path=watchdog_paths["log"],
        trigger_path=watchdog_paths["trigger"],
        stop_path=watchdog_paths["stop"],
        cleanup_receipt_path=watchdog_paths["cleanup"],
        vast_executable=vast_executable,
    )
    stage_deadline = float(watchdog["deadline_epoch"])
    acquisition_deadline = min(stage_deadline - 15 * 60, time.time() + 25 * 60)
    rows = {str(row["worker_id"]): row for row in plan["workers"]}
    groups = {
        str(row["instance_group_id"]): row for row in plan.get("instance_groups", [])
    }
    if not groups or len(groups) != int(plan.get("instance_group_count", 0)):
        raise RuntimeError("E025 full fleet GO has no exact instance-group plan")
    specs = {
        worker_id: _spec(worker_id, str(row["role"]))
        for worker_id, row in rows.items()
    }
    parent_id = "e025-stage-089-parent"
    parent_group_id = str(rows[parent_id]["instance_group_id"])
    initial_group_ids = sorted(set(groups) - {parent_group_id})
    active_offer_ids: set[int] = set()
    failed_offer_ids: set[int] = set()
    active_machine_ids: dict[str, int] = {}
    selection_lock = threading.Lock()
    fleet_abort = threading.Event()
    instances: dict[str, tuple[int, Offer]] = {}
    workers: dict[str, LiveWorker] = {}
    failures: list[dict[str, Any]] = []
    correctness: dict[str, Any] | None = None
    generation: dict[str, Any] | None = None
    cleanup: dict[str, Any]
    lifecycle_costs: dict[str, Any] = {}
    local_gpu_process_evidence = {"before_fleet": _local_gpu_process_snapshot()}
    local_gpu_monitor_samples: list[dict[str, Any]] = []
    local_gpu_monitor_stop = threading.Event()

    def monitor_local_gpu() -> None:
        while not local_gpu_monitor_stop.is_set():
            try:
                local_gpu_monitor_samples.append(_local_gpu_process_snapshot())
            except BaseException as exc:
                local_gpu_monitor_samples.append(
                    {
                        "timestamp": utc_now(),
                        "monitor_error_type": type(exc).__name__,
                        "monitor_error": str(exc),
                    }
                )
            local_gpu_monitor_stop.wait(5.0)

    local_gpu_monitor_thread = threading.Thread(
        target=monitor_local_gpu,
        name=f"e025-local-gpu-monitor-{run_id}",
        daemon=True,
    )
    local_gpu_monitor_thread.start()

    def reserve_offer(group_id: str, candidates: list[Offer]) -> Offer:
        with selection_lock:
            for offer in candidates:
                if offer.offer_id in active_offer_ids:
                    continue
                if offer.offer_id in failed_offer_ids:
                    continue
                if offer.machine_id in active_machine_ids.values():
                    continue
                active_offer_ids.add(offer.offer_id)
                active_machine_ids[group_id] = offer.machine_id
                return offer
        raise RuntimeError(f"no unused frozen alternate remains for {group_id}")

    def release_offer(group_id: str, offer: Offer) -> None:
        with selection_lock:
            active_offer_ids.discard(offer.offer_id)
            active_machine_ids.pop(group_id, None)

    def mark_offer_failed(group_id: str, offer: Offer) -> None:
        with selection_lock:
            failed_offer_ids.add(offer.offer_id)
            active_offer_ids.discard(offer.offer_id)
            active_machine_ids.pop(group_id, None)

    def create_group(
        group_id: str,
        expert_endpoints: list[dict[str, Any]] | None = None,
    ) -> tuple[int, Offer]:
        group = groups[group_id]
        candidates = _offer_candidates(group)
        last: BaseException | None = None
        for _ in range(len(candidates)):
            if fleet_abort.is_set():
                raise RuntimeError("E025 fleet acquisition was aborted")
            if time.time() >= acquisition_deadline:
                raise TimeoutError("E025 fleet acquisition deadline expired")
            offer = reserve_offer(group_id, candidates)
            candidates = [value for value in candidates if value.offer_id != offer.offer_id]
            try:
                group_workers = sorted(
                    group["workers"], key=lambda row: int(row["gpu_slot"])
                )
                worker_specs = [
                    {
                        "worker_id": str(worker["worker_id"]),
                        "gpu_slot": int(worker["gpu_slot"]),
                        "port": int(worker["container_port"]),
                        "maximum_context": 64,
                        "expert_endpoints": (
                            expert_endpoints
                            if str(worker["worker_id"]) == parent_id
                            else None
                        ),
                    }
                    for worker in group_workers
                ]
                instance_id = create_worker_group_instance(
                    client=client,
                    run_id=run_id,
                    group_id=group_id,
                    group_index=sorted(groups).index(group_id),
                    offer=offer,
                    image_reference=image_reference,
                    image_digest=image_digest,
                    disk_gb=int(group["disk_gb"]),
                    material=material,
                    watchdog_receipt=watchdog_paths["receipt"],
                    go_receipt=full_fleet_go_path,
                    worker_specs=worker_specs,
                )
                return instance_id, offer
            except BaseException as exc:
                last = exc
                fatal_contract_failure = isinstance(exc, WorkerCompatibilityError)
                mark_offer_failed(group_id, offer)
                failures.append(
                    {
                        "phase": "CREATE",
                        "instance_group_id": group_id,
                        "offer_id": offer.offer_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "classification": (
                            "CODE_OR_RUNTIME_CONTRACT_FAILURE"
                            if fatal_contract_failure
                            else "TRANSIENT_OFFER_CREATE_FAILURE"
                        ),
                    }
                )
                if fatal_contract_failure:
                    fleet_abort.set()
                    raise
        raise RuntimeError(f"all frozen create candidates failed for {group_id}") from last

    def wait_one(worker_id: str) -> LiveWorker:
        row = rows[worker_id]
        group_id = str(row["instance_group_id"])
        instance_id, offer = instances[group_id]
        spec = specs[worker_id]
        try:
            return wait_for_worker(
                client=client,
                ledger=ledger,
                run_id=run_id,
                worker_id=worker_id,
                role=spec["role"],
                layer=spec["layer"],
                worker_index=spec["worker_index"],
                offer=offer,
                instance_id=instance_id,
                credential=Path(material["credential_path"]).read_bytes(),
                certificate=Path(material["certificate_path"]),
                image_digest=image_digest,
                deadline_epoch=acquisition_deadline,
                container_port=int(row["container_port"]),
                gpu_slot=int(row["gpu_slot"]),
                abort_event=fleet_abort,
            )
        except WorkerCompatibilityError:
            fleet_abort.set()
            raise

    def ready_group(
        group_id: str,
        expert_endpoints: list[dict[str, Any]] | None = None,
    ) -> list[LiveWorker]:
        group_worker_ids = [
            str(row["worker_id"]) for row in groups[group_id]["workers"]
        ]
        attempts = 1 + len(groups[group_id].get("alternates", []))
        last: BaseException | None = None
        for attempt in range(1, attempts + 1):
            try:
                with ThreadPoolExecutor(max_workers=len(group_worker_ids)) as pool:
                    ready = list(pool.map(wait_one, group_worker_ids))
                if len(ready) != len(group_worker_ids):
                    raise RuntimeError("E025 instance group returned partial readiness")
                return ready
            except BaseException as exc:
                last = exc
                fatal_contract_failure = isinstance(exc, WorkerCompatibilityError)
                instance_id, failed_offer = instances[group_id]
                failures.append(
                    {
                        "phase": "READY",
                        "instance_group_id": group_id,
                        "offer_id": failed_offer.offer_id,
                        "instance_id": instance_id,
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "classification": (
                            "CODE_OR_RUNTIME_CONTRACT_FAILURE"
                            if fatal_contract_failure
                            else "TRANSIENT_HOST_OR_BOOTSTRAP_FAILURE"
                        ),
                    }
                )
                try:
                    log = client.logs(instance_id, tail=1000)
                except BaseException as log_exc:
                    log = f"LOG_RETRIEVAL_FAILED: {type(log_exc).__name__}: {log_exc}\n"
                failure_log = stage_root / "failed-group-logs"
                failure_log.mkdir(parents=True, exist_ok=True)
                (failure_log / f"{group_id}-attempt-{attempt}.log").write_text(
                    log, encoding="utf-8"
                )
                client.destroy_instance(
                    instance_id,
                    reason=f"readiness-failed-{group_id}-attempt-{attempt}",
                )
                mark_offer_failed(group_id, failed_offer)
                if (
                    fatal_contract_failure
                    or fleet_abort.is_set()
                    or attempt >= attempts
                    or time.time() >= acquisition_deadline
                ):
                    break
                instances[group_id] = create_group(group_id, expert_endpoints)
        fleet_abort.set()
        raise RuntimeError(f"all frozen readiness candidates failed for {group_id}") from last

    failure: dict[str, Any] | None = None
    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            future_map = {
                pool.submit(create_group, group_id): group_id
                for group_id in initial_group_ids
            }
            for future in as_completed(future_map):
                group_id = future_map[future]
                instances[group_id] = future.result()
        fragment_ids = sorted(
            worker_id
            for worker_id in rows
            if specs[worker_id]["role"] == "SUB_LAYER_WORKER"
        )
        fragment_group_ids = [str(rows[worker_id]["instance_group_id"]) for worker_id in fragment_ids]
        backbone_group_ids = sorted(
            set(groups) - set(fragment_group_ids) - {parent_group_id}
        )

        def ready_parent_after_fragments(
            fragments: list[LiveWorker],
        ) -> list[LiveWorker]:
            fragment_workers = {worker.worker_id: worker for worker in fragments}
            expert_endpoints = [
                fragment_workers[worker_id].expert_endpoint()
                for worker_id in fragment_ids
            ]
            instances[parent_group_id] = create_group(parent_group_id, expert_endpoints)
            return ready_group(parent_group_id, expert_endpoints)

        fragment_workers, parent_workers, backbone_workers = (
            _ready_groups_with_concurrent_backbone(
                fragment_group_ids=fragment_group_ids,
                backbone_group_ids=backbone_group_ids,
                ready_group=ready_group,
                ready_parent_after_fragments=ready_parent_after_fragments,
            )
        )
        for worker in [*fragment_workers, *parent_workers, *backbone_workers]:
            workers[worker.worker_id] = worker
        if len(workers) != TRANSFORMER_LAYERS + SUB_LAYER_WORKERS:
            raise RuntimeError("E025 full fleet did not reach 97 READY workers")
        gpu_uuids = [str(worker.ready["gpu"]["gpu_uuid"]) for worker in workers.values()]
        if len(set(gpu_uuids)) != len(gpu_uuids) or "unknown" in gpu_uuids:
            raise RuntimeError("E025 READY fleet does not expose 97 distinct physical GPUs")
        if time.time() >= acquisition_deadline:
            raise TimeoutError("E025 fleet became ready after its acquisition deadline")
        ordered_workers = sorted(
            workers.values(),
            key=lambda row: (0 if row.role != "SUB_LAYER_WORKER" else 1, row.worker_id),
        )
        endpoint_path = stage_root / "live-endpoints.json"
        write_live_endpoints(endpoint_path, ordered_workers)
        correctness = run_physical_generation(
            endpoints_path=endpoint_path,
            credential_path=Path(material["credential_path"]),
            certificate=Path(material["certificate_path"]),
            stage_zero_snapshot=checkpoint,
            output_path=correctness_output_path,
            prompt="Hi",
            max_new_tokens=2,
            fixture_token_ids=[163584, 18699, 11],
            fixture_hidden_trace=oracle_trace,
            fixture_routes=oracle_routes,
        )
        if correctness.get("status") != "PASS":
            raise RuntimeError("E025 one/two-token physical correctness gate failed")
        remaining_seconds = stage_deadline - time.time()
        correctness_forwards = len(correctness.get("token_records", []))
        measured_seconds_per_forward = float(correctness["model_forward_seconds"]) / max(
            1, correctness_forwards
        )
        safe_generation_seconds = max(0.0, remaining_seconds - 3 * 60)
        maximum_output_tokens_by_ttl = (
            math.floor(0.75 * safe_generation_seconds / measured_seconds_per_forward) - 1
        )
        public_max_new_tokens = min(max_new_tokens, maximum_output_tokens_by_ttl)
        if public_max_new_tokens < 3:
            raise TimeoutError(
                "E025 measured physical latency cannot fit several public tokens plus cleanup"
            )
        generation = run_physical_generation(
            endpoints_path=endpoint_path,
            credential_path=Path(material["credential_path"]),
            certificate=Path(material["certificate_path"]),
            stage_zero_snapshot=checkpoint,
            output_path=generation_output_path,
            prompt=prompt,
            max_new_tokens=public_max_new_tokens,
        )
        generation["requested_max_new_tokens"] = max_new_tokens
        generation["ttl_adapted_max_new_tokens"] = public_max_new_tokens
        generation["correctness_measured_seconds_per_forward"] = (
            measured_seconds_per_forward
        )
        generation.pop("output_path", None)
        generation.pop("output_sha256", None)
        atomic_write_json(generation_output_path, generation)
        generation["output_path"] = str(generation_output_path.resolve())
        generation["output_sha256"] = sha256_file(generation_output_path.resolve())
        if generation.get("status") != "PASS":
            raise RuntimeError("E025 public physical generation failed")
        local_gpu_process_evidence["after_generation"] = _local_gpu_process_snapshot()
    except BaseException as exc:
        fleet_abort.set()
        failure = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "classification": (
                "INCOMPLETE_TIMEOUT"
                if isinstance(exc, TimeoutError)
                else "PHYSICAL_OR_RUNTIME_BLOCKER"
            ),
        }
        failures.append({"phase": "HEADLINE", **failure})
    finally:
        local_gpu_monitor_stop.set()
        local_gpu_monitor_thread.join(timeout=10.0)
        if workers:
            _write_worker_receipts(
                workers=workers,
                worker_rows=rows,
                correctness=correctness,
                generation=generation,
                destination=summary_output_path.parent.parent / "workers",
            )
        _write_central_telemetry(
            correctness=correctness,
            generation=generation,
            failures=failures,
            local_gpu_samples=local_gpu_monitor_samples,
            destination=summary_output_path.parent.parent / "telemetry",
        )
        if workers and stage_deadline - time.time() > 60:
            _retrieve_logs(client, list(workers.values()), stage_root / "logs")
        cleanup = destroy_all_from_ledger(
            ledger_path=ledger_path,
            run_id=run_id,
            reason="full-headline-complete-or-aborted",
            executable=vast_executable,
            attempts=6,
        )
        try:
            lifecycle_costs = summarize_lifecycle_costs(ledger_path, run_id)
            live_offers = {
                offer.offer_id: offer for _, offer in instances.values()
            }
            observed_download_bytes = 0
            observed_ingress_cost = 0.0
            for worker in workers.values():
                downloaded = int(
                    worker.ready.get("bootstrap", {})
                    .get("acquisition", {})
                    .get("downloaded_bytes", 0)
                )
                observed_download_bytes += downloaded
                offer = live_offers.get(worker.offer_id)
                if offer is not None:
                    observed_ingress_cost += (
                        downloaded / 1e9 * offer.inet_down_cost_per_gb
                    )
            lifecycle_costs["ready_worker_observed_download_bytes"] = (
                observed_download_bytes
            )
            lifecycle_costs["ready_worker_observed_ingress_cost_usd"] = (
                observed_ingress_cost
            )
            lifecycle_costs["provider_invoice_claimed"] = False
        except BaseException as cost_exc:
            lifecycle_costs = {
                "schema_version": "experiment-025-ledger-cost-summary-v1",
                "generated_at_utc": utc_now(),
                "status": "INCOMPLETE",
                "error_type": type(cost_exc).__name__,
                "error": str(cost_exc),
                "provider_invoice_claimed": False,
            }
        atomic_write_json(stage_root / "rental-cost-summary.json", lifecycle_costs)
        atomic_write_json(watchdog_paths["cleanup"], cleanup)
        if cleanup["zero_live_e025_instances"]:
            atomic_write_json(
                watchdog_paths["stop"],
                {"timestamp": utc_now(), "reason": "zero live fleet proven"},
            )
    fragment_workers = sorted(
        (worker for worker in workers.values() if worker.role == "SUB_LAYER_WORKER"),
        key=lambda row: int(row.worker_index or 0),
    )
    memory_proof = (
        _worker_memory_proof(fragment_workers, placement, generation)
        if fragment_workers
        else []
    )
    decoded = str(generation.get("decoded_text", "")) if generation else ""
    rented_gpu_count = sum(offer.gpu_count for _, offer in instances.values())
    unused_rented_gpu_count = max(0, rented_gpu_count - len(workers))
    expected_stage_worker_ids = {
        worker_id
        for worker_id, spec in specs.items()
        if spec["role"] in {"BACKBONE_STAGE", "SUB_LAYER_PARENT"}
    }
    generation_records = list((generation or {}).get("token_records", []))
    every_forward_has_exact_stage_coverage = bool(generation_records) and all(
        len(token.get("stages", [])) == TRANSFORMER_LAYERS
        and {
            str(stage.get("worker_id")) for stage in token.get("stages", [])
        }
        == expected_stage_worker_ids
        for token in generation_records
    )
    process_identities = {
        (worker.instance_id, int(worker.ready.get("process_id", -1)))
        for worker in workers.values()
    }
    checkpoint_fingerprints = {
        str(worker.ready.get("checkpoint_fingerprint", ""))
        for worker in workers.values()
    }
    trace_facts = {
        "expected_stage_worker_count": TRANSFORMER_LAYERS,
        "every_forward_has_exact_stage_coverage": (
            every_forward_has_exact_stage_coverage
        ),
        "unique_physical_gpu_uuid_count": len(
            {str(worker.ready["gpu"]["gpu_uuid"]) for worker in workers.values()}
        ),
        "unique_instance_process_count": len(process_identities),
        "worker_checkpoint_fingerprints": sorted(checkpoint_fingerprints),
        "all_workers_report_one_visible_cuda_device": bool(workers)
        and all(
            int(worker.ready["gpu"].get("torch_device_count", 0)) == 1
            for worker in workers.values()
        ),
        "controller_owned_model_tensors": False,
        "final_tokens_returned_by_physical_stage_92": bool(generation_records)
        and all(
            int(token["stages"][-1]["layer"]) == TRANSFORMER_LAYERS - 1
            for token in generation_records
        ),
    }
    gates = {
        "G1_authoritative_kimi_k3": correctness is not None
        and correctness.get("model_id") == MODEL_ID
        and correctness.get("model_revision") == MODEL_REVISION,
        "G2_full_physical_multi_machine_path": generation is not None
        and int(generation.get("physical_machine_count", 0)) > 1,
        "G3_consumer_only_headline_compute": generation is not None
        and bool(generation["registration"]["all_consumer_geforce"])
        and all("GEFORCE RTX" in worker.gpu_name.upper() for worker in workers.values())
        and trace_facts["all_workers_report_one_visible_cuda_device"] is True
        and bool(local_gpu_monitor_samples)
        and all(
            sample.get("nvidia_smi_returncode") == 0
            and sample.get("controller_is_gpu_compute_process") is False
            for sample in local_gpu_monitor_samples
        )
        and all(
            snapshot.get("controller_is_gpu_compute_process") is False
            for snapshot in local_gpu_process_evidence.values()
        ),
        "G4_real_native_execution": correctness is not None
        and correctness.get("status") == "PASS",
        "G5_required_model_coverage": placement.get("status") == "PASS"
        and all(placement["acceptance_gates"].values()),
        "G6_physical_sub_layer_workers": len(fragment_workers) == SUB_LAYER_WORKERS
        and generation is not None
        and generation.get("every_sub_layer_worker_executed_real_compute") is True
        and all(
            row["participated_in_every_retained_headline_token"] is True
            for row in memory_proof
        ),
        "G7_complete_layer_cannot_fit": bool(memory_proof)
        and all(row["complete_layer_fits"] is False for row in memory_proof),
        "G8_fragment_fits": bool(memory_proof)
        and all(row["fragment_fits"] is True for row in memory_proof),
        "G9_no_hidden_full_layer_fallback": generation is not None
        and generation["anti_cheating"]["layer_89_complete_fallback"] is False
        and placement["sub_layer_proof"]["parent_local_routed_experts"] == 0
        and all(worker.ready["whole_layer_fallback"] is False for worker in workers.values()),
        "G10_sub_layer_necessity": placement["sub_layer_proof"]["negative_control"][
            "frozen_placement_without_sub_layer_group_valid"
        ]
        is False
        and go["checklist"]["physical_sub_layer_canary_passed"] is True,
        "G11_numerical_correctness": correctness is not None
        and correctness.get("status") == "PASS",
        "G12_stateful_autoregressive_correctness": correctness is not None
        and correctness.get("stateful_autoregressive_advancement") is True,
        "G13_human_readable_generation": generation is not None
        and len(generation.get("generated_token_ids", [])) >= 3
        and any(character.isalnum() for character in decoded),
        "G14_physical_trace_complete": generation is not None
        and len(generation.get("token_records", [])) >= 2
        and len(workers) == 97
        and every_forward_has_exact_stage_coverage
        and trace_facts["unique_physical_gpu_uuid_count"] == 97
        and trace_facts["unique_instance_process_count"] == 97
        and len(checkpoint_fingerprints) == 1
        and "" not in checkpoint_fingerprints,
        "G15_cleanup": cleanup["zero_live_e025_instances"] is True,
    }
    status = "PASS" if all(gates.values()) else ("INCOMPLETE" if failure else "FAIL")
    payload = {
        "schema_version": "experiment-025-headline-summary-v1",
        "generated_at_utc": utc_now(),
        "status": status,
        "evidence_class": EVIDENCE_CLASS,
        "run_id": run_id,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "image_digest": image_digest,
        "physical_worker_count": len(workers),
        "rented_instance_count": len(instances),
        "rented_consumer_gpu_count": rented_gpu_count,
        "unused_rented_consumer_gpu_count": unused_rented_gpu_count,
        "physical_machine_count": len({worker.machine_id for worker in workers.values()}),
        "consumer_gpu_count": len(workers),
        "qualifying_sub_layer_gpu_count": len(fragment_workers),
        "sub_layer_memory_proof": memory_proof,
        "correctness": correctness,
        "generation": generation,
        "cleanup": cleanup,
        "rental_costs": lifecycle_costs,
        "local_controller_gpu_process_evidence": local_gpu_process_evidence,
        "local_controller_gpu_monitor_samples": local_gpu_monitor_samples,
        "physical_trace_facts": trace_facts,
        "failures": failures,
        "decisive_failure": failure,
        "pass_gates": gates,
        "strongest_public_claim": (
            "Kimi K3, a 2.8-trillion-parameter model, generated real tokens across a "
            "physical consumer-GPU swarm, including four GPUs that executed real "
            "fragments of a K3 layer whose complete runtime peak exceeded their usable VRAM."
            if status == "PASS"
            else None
        ),
    }
    atomic_write_json(summary_output_path, payload)
    return payload


__all__ = ["run_headline_stage"]
