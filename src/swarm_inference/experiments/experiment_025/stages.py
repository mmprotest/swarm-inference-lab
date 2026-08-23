"""Fail-fast paid canary stages with mandatory cleanup and watchdogs."""

from __future__ import annotations

import json
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .canary_runtime import run_physical_stage_fixture, run_physical_sub_layer_canary
from .constants import CANARY_TTL_SECONDS, SUB_LAYER_CANARY_TTL_SECONDS
from .io import atomic_write_json, read_json, utc_now
from .preflight import source_tree_sha256
from .provisioning import (
    LiveWorker,
    WorkerCompatibilityError,
    create_worker_instance,
    provision_worker,
    wait_for_public_endpoint,
    wait_for_worker,
)
from .secrets import create_transport_material, load_transport_material
from .vast_lifecycle import (
    AppendOnlyLifecycleLedger,
    Offer,
    VastClient,
    destroy_all_from_ledger,
    rank_offers_for_workers,
    summarize_lifecycle_costs,
)
from .watchdog import start_watchdog


def _material(private_root: Path, run_id: str) -> dict[str, Any]:
    return (
        load_transport_material(private_root, run_id)
        if private_root.exists()
        else create_transport_material(private_root, run_id)
    )


def _failed_machine_exclusions(
    stage_root: Path,
    *,
    stage_prefix: str,
) -> dict[str, Any]:
    """Return machines that failed readiness or the frozen physical canary."""

    rows: dict[int, list[dict[str, Any]]] = {}
    rental_root = stage_root.parent
    current = stage_root.resolve()
    for directory in sorted(rental_root.glob(f"{stage_prefix}*")):
        if not directory.is_dir() or directory.resolve() == current:
            continue
        ledger_path = directory / "instance-ledger.jsonl"
        if not ledger_path.is_file():
            continue
        entries = [
            json.loads(line)
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        ready_instances = {
            int(entry["instance_id"])
            for entry in entries
            if entry.get("event") == "WORKER_READY"
            and entry.get("instance_id") is not None
        }
        physical_receipts = [
            directory / "physical-canary.json",
            directory / "physical-sub-layer-canary.json",
        ]
        failed_physical_receipt = next(
            (
                receipt
                for receipt in physical_receipts
                if receipt.is_file() and read_json(receipt).get("status") != "PASS"
            ),
            None,
        )
        for entry in entries:
            if entry.get("event") != "CREATE_CONFIRMED":
                continue
            instance_id = entry.get("instance_id")
            machine_id = entry.get("machine_id")
            if instance_id is None or machine_id is None:
                continue
            reached_ready = int(instance_id) in ready_instances
            if reached_ready and failed_physical_receipt is None:
                continue
            rows.setdefault(int(machine_id), []).append(
                {
                    "stage_directory": directory.name,
                    "instance_id": int(instance_id),
                    "offer_id": entry.get("offer_id"),
                    "reason": (
                        "prior stage frozen physical canary failed"
                        if reached_ready
                        else "prior instance never reached WORKER_READY"
                    ),
                    "physical_failure_receipt": (
                        failed_physical_receipt.name
                        if failed_physical_receipt is not None
                        else None
                    ),
                }
            )
    return {
        "schema_version": "experiment-025-failed-machine-exclusions-v1",
        "generated_at_utc": utc_now(),
        "stage_prefix": stage_prefix,
        "policy": (
            "exclude a physical machine after any prior instance for this stage "
            "failed authenticated WORKER_READY or participated in a frozen physical "
            "canary receipt whose status was not PASS"
        ),
        "machine_ids": sorted(rows),
        "sources": [
            {"machine_id": machine_id, "observations": rows[machine_id]}
            for machine_id in sorted(rows)
        ],
    }


def _prior_passing_machine_selection(
    stage_root: Path,
    offers: list[Offer],
    *,
    machine_id: int,
) -> tuple[list[Offer], dict[str, Any]]:
    """Select a live offer only when retained physical evidence passed on that machine."""

    rental_root = stage_root.parent
    current = stage_root.resolve()
    evidence: list[dict[str, Any]] = []
    for directory in sorted(rental_root.glob("stage-1-backbone-canary*")):
        if not directory.is_dir() or directory.resolve() == current:
            continue
        for name in ("backbone-canary-result.json", "physical-canary.json"):
            path = directory / name
            if not path.is_file():
                continue
            receipt = read_json(path)
            worker = receipt.get("worker") or receipt.get("result", {}).get("worker", {})
            if (
                receipt.get("status") == "PASS"
                and int(worker.get("machine_id", -1)) == int(machine_id)
            ):
                evidence.append(
                    {
                        "path": path.relative_to(rental_root).as_posix(),
                        "status": "PASS",
                        "machine_id": int(machine_id),
                        "image_digest": receipt.get("image_digest"),
                    }
                )
    if not evidence:
        raise RuntimeError(
            f"preferred machine {machine_id} has no retained passing physical canary"
        )
    selected = [offer for offer in offers if offer.machine_id == int(machine_id)]
    if not selected:
        raise RuntimeError(
            f"preferred previously passing machine {machine_id} has no live offer"
        )
    return selected, {
        "schema_version": "experiment-025-measured-machine-selection-v1",
        "generated_at_utc": utc_now(),
        "policy": (
            "prefer a currently rentable physical machine only after retained E025 "
            "evidence proves authenticated native K3 correctness on that machine"
        ),
        "machine_id": int(machine_id),
        "matching_live_offer_ids": sorted(offer.offer_id for offer in selected),
        "physical_pass_evidence": evidence,
    }


def _image(image_receipt_path: Path) -> tuple[str, str]:
    receipt = read_json(image_receipt_path)
    if (
        receipt.get("status") != "PASS"
        or receipt.get("push_status") != "PASS"
        or receipt.get("anonymous_registry_resolve_status") != "PASS"
    ):
        raise RuntimeError("E025 paid stage requires a published passing image")
    reference = str(receipt.get("immutable_reference", ""))
    digest = str(receipt.get("immutable_digest", ""))
    if "@sha256:" not in reference or not digest.startswith("sha256:"):
        raise RuntimeError("E025 paid stage image is not immutable")
    return reference, digest


def _require_local_prerequisites(
    *,
    image_receipt_path: Path,
    test_receipt_path: Path,
    rehearsal_receipt_path: Path,
    local_image_canary_path: Path,
) -> None:
    image = read_json(image_receipt_path)
    tests = read_json(test_receipt_path)
    rehearsal = read_json(rehearsal_receipt_path)
    local_image = read_json(local_image_canary_path)
    source_id = image.get("source_id")
    current_source_id = source_tree_sha256(Path.cwd())
    digest = image.get("immutable_digest")
    gates = {
        "image": image.get("status") == "PASS",
        "tests": tests.get("status") == "PASS"
        and tests.get("production_dispatch_functional_test") is True,
        "rehearsal": rehearsal.get("status") == "PASS",
        "tested_source": bool(source_id)
        and current_source_id == source_id
        and tests.get("source_tree_sha256") == source_id
        and rehearsal.get("source_tree_sha256") == source_id,
        "local_immutable_image_canary": local_image.get("status") == "PASS"
        and local_image.get("immutable_digest") == digest,
    }
    if not all(gates.values()):
        raise RuntimeError(f"E025 paid canary local prerequisites failed: {gates}")


def _require_zero_live_e025(client: VastClient) -> None:
    live = [
        row
        for row in client.show_instances()
        if str(row.get("label", "")).lower().startswith("e025-")
    ]
    if live:
        ids = [row.get("id", row.get("instance_id")) for row in live]
        raise RuntimeError(
            f"E025 refuses a new paid stage while E025 instances are live: {ids}"
        )


def _image_worker_requirement(image_index_path: Path, worker_id: str, role: str) -> dict[str, Any]:
    index = read_json(image_index_path)
    row = next(
        (value for value in index["workers"] if value["worker_id"] == worker_id),
        None,
    )
    if row is None or row["role"] != role:
        raise ValueError(f"E025 image has no {role} bundle for {worker_id}")
    return {
        "worker_id": worker_id,
        "role": role,
        "download_bytes_cold_cache": int(row["download_bytes_cold_cache"]),
        "assigned_tensor_bytes": int(row["assigned_tensor_bytes"]),
    }


def _write_go(
    path: Path,
    *,
    run_id: str,
    stage: str,
    image_reference: str,
    offers: list[Offer],
    plan: dict[str, Any],
    budget: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": "experiment-025-paid-stage-go-v1",
        "generated_at_utc": utc_now(),
        "status": "GO",
        "run_id": run_id,
        "stage": stage,
        "image_reference": image_reference,
        "on_demand_only": True,
        "consumer_only": True,
        "offer_snapshot": [asdict(offer) for offer in offers],
        "fleet_plan": plan,
        "redacted_budget": budget,
        "watchdog_required_before_first_create": True,
    }
    atomic_write_json(path, payload)
    return payload


def _paid_stage_budget(
    client: VastClient,
    plan: dict[str, Any],
    *,
    disk_gb: int,
    ttl_seconds: int,
) -> dict[str, Any]:
    budget = client.user_budget()
    selected = [Offer(**row["selected_offer"]) for row in plan["workers"]]
    maximum_cost = sum(
        offer.effective_rate_per_hour(disk_gb) * ttl_seconds / 3600.0
        for offer in selected
    ) + sum(
        float(row["selection_score"]["expected_ingress_cost_usd"])
        for row in plan["workers"]
    )
    budget["maximum_stage_ttl_plus_expected_ingress_usd"] = maximum_cost
    budget["required_safety_multiplier"] = 1.10
    budget["sufficient"] = (
        float(budget["conservative_available_usd"]) >= maximum_cost * 1.10
    )
    if budget["sufficient"] is not True:
        raise RuntimeError("E025 paid canary budget gate failed")
    return budget


def _watchdog_paths(stage_root: Path) -> dict[str, Path]:
    return {
        "ledger": stage_root / "instance-ledger.jsonl",
        "receipt": stage_root / "watchdog-receipt.json",
        "log": stage_root / "watchdog.jsonl",
        "trigger": stage_root / "WATCHDOG_TRIGGER",
        "stop": stage_root / "WATCHDOG_STOP",
        "cleanup": stage_root / "cleanup-verification.json",
    }


def _finish_watchdog(paths: dict[str, Path], cleanup: dict[str, Any]) -> None:
    atomic_write_json(paths["cleanup"], cleanup)
    if cleanup.get("zero_live_e025_instances") is True:
        atomic_write_json(paths["stop"], {"timestamp": utc_now(), "reason": "cleanup proven"})


def _cost_summary(path: Path, run_id: str) -> dict[str, Any]:
    try:
        return summarize_lifecycle_costs(path, run_id)
    except BaseException as exc:
        return {
            "schema_version": "experiment-025-ledger-cost-summary-v1",
            "generated_at_utc": utc_now(),
            "status": "INCOMPLETE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "provider_invoice_claimed": False,
        }


def _failure_classification(exc: BaseException) -> str:
    if isinstance(exc, TimeoutError):
        return "INCOMPLETE_TIMEOUT"
    if isinstance(exc, WorkerCompatibilityError):
        return "CODE_OR_RUNTIME_CONTRACT_FAILURE"
    message = str(exc).lower()
    if any(
        marker in message
        for marker in (
            "vast create failed",
            "terminal status",
            "absent from live query",
            "no public mapping",
        )
    ):
        return "TRANSIENT_HOST_OR_OFFER_FAILURE"
    if isinstance(exc, (ValueError, AssertionError)):
        return "CODE_OR_CONFIGURATION_FAILURE"
    return "PHYSICAL_OR_RUNTIME_DEFECT_REQUIRES_OFFLINE_REVIEW"


def _retrieve_logs(client: VastClient, workers: list[LiveWorker], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for worker in workers:
        try:
            log = client.logs(worker.instance_id, tail=3000)
        except BaseException as exc:
            log = f"LOG_RETRIEVAL_FAILED: {type(exc).__name__}: {exc}\n"
        (output / f"{worker.worker_id}.log").write_text(log, encoding="utf-8")


def run_backbone_canary_stage(
    *,
    run_id: str,
    stage_root: Path,
    image_receipt_path: Path,
    image_index_path: Path,
    test_receipt_path: Path,
    rehearsal_receipt_path: Path,
    local_image_canary_path: Path,
    private_root: Path,
    checkpoint: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    output_path: Path,
    disk_gb: int = 60,
    preferred_machine_id: int | None = None,
    vast_executable: str = "vastai",
) -> dict[str, Any]:
    stage_root.mkdir(parents=True, exist_ok=True)
    ledger_path = stage_root / "instance-ledger.jsonl"
    if ledger_path.is_file() and ledger_path.stat().st_size:
        raise RuntimeError("E025 backbone canary stage ledger already exists")
    _require_local_prerequisites(
        image_receipt_path=image_receipt_path,
        test_receipt_path=test_receipt_path,
        rehearsal_receipt_path=rehearsal_receipt_path,
        local_image_canary_path=local_image_canary_path,
    )
    image_reference, image_digest = _image(image_receipt_path)
    requirement = _image_worker_requirement(
        image_index_path,
        "e025-stage-001",
        "BACKBONE_STAGE",
    )
    ledger = AppendOnlyLifecycleLedger(ledger_path, run_id)
    client = VastClient(executable=vast_executable, ledger=ledger)
    _require_zero_live_e025(client)
    offers = client.search_offers(gpu_names=("RTX 3090", "RTX 3090 Ti"), storage_gb=disk_gb)
    exclusions = _failed_machine_exclusions(
        stage_root,
        stage_prefix="stage-1-backbone-canary",
    )
    atomic_write_json(stage_root / "failed-machine-exclusions.json", exclusions)
    offers = [
        offer for offer in offers if offer.machine_id not in exclusions["machine_ids"]
    ]
    if not offers:
        raise RuntimeError("no Stage 1 offers remain after failed-machine exclusions")
    if preferred_machine_id is not None:
        offers, measured_selection = _prior_passing_machine_selection(
            stage_root,
            offers,
            machine_id=preferred_machine_id,
        )
        atomic_write_json(
            stage_root / "measured-machine-selection.json", measured_selection
        )
    plan = rank_offers_for_workers(offers, [requirement], disk_gb=disk_gb)
    budget = _paid_stage_budget(
        client,
        plan,
        disk_gb=disk_gb,
        ttl_seconds=CANARY_TTL_SECONDS,
    )
    go_path = stage_root / "CANARY_GO.json"
    _write_go(
        go_path,
        run_id=run_id,
        stage="single-backbone-canary",
        image_reference=image_reference,
        offers=offers,
        plan=plan,
        budget=budget,
    )
    paths = _watchdog_paths(stage_root)
    watchdog = start_watchdog(
        run_id=run_id,
        stage="single-backbone-canary",
        ledger_path=paths["ledger"],
        ttl_seconds=CANARY_TTL_SECONDS,
        receipt_path=paths["receipt"],
        log_path=paths["log"],
        trigger_path=paths["trigger"],
        stop_path=paths["stop"],
        cleanup_receipt_path=paths["cleanup"],
        vast_executable=vast_executable,
    )
    deadline = float(watchdog["deadline_epoch"])
    selected = Offer(**plan["workers"][0]["selected_offer"])
    material = _material(private_root, run_id)
    worker: LiveWorker | None = None
    result: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    cleanup: dict[str, Any]
    lifecycle_costs: dict[str, Any] = {}
    try:
        worker = provision_worker(
            client=client,
            ledger=ledger,
            run_id=run_id,
            worker_id="e025-stage-001",
            role="BACKBONE_STAGE",
            layer=1,
            worker_index=None,
            offer=selected,
            image_reference=image_reference,
            image_digest=image_digest,
            disk_gb=disk_gb,
            material=material,
            watchdog_receipt=paths["receipt"],
            go_receipt=go_path,
            deadline_epoch=deadline,
            maximum_context=3,
        )
        result = run_physical_stage_fixture(
            worker=worker,
            checkpoint=checkpoint,
            oracle_trace=oracle_trace,
            oracle_routes=oracle_routes,
            credential_path=Path(material["credential_path"]),
            certificate=Path(material["certificate_path"]),
            output_path=stage_root / "physical-canary.json",
            cycle_id="E025-BACKBONE-CANARY",
        )
    except BaseException as exc:
        failure = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "classification": _failure_classification(exc),
        }
    finally:
        if worker is not None:
            _retrieve_logs(client, [worker], stage_root / "logs")
        cleanup = destroy_all_from_ledger(
            ledger_path=paths["ledger"],
            run_id=run_id,
            reason="single-backbone-canary-complete-or-aborted",
            executable=vast_executable,
            attempts=5,
        )
        lifecycle_costs = _cost_summary(paths["ledger"], run_id)
        atomic_write_json(stage_root / "rental-cost-summary.json", lifecycle_costs)
        _finish_watchdog(paths, cleanup)
    payload = {
        "schema_version": "experiment-025-backbone-canary-stage-v1",
        "generated_at_utc": utc_now(),
        "status": (
            "PASS"
            if result is not None
            and result.get("status") == "PASS"
            and cleanup["zero_live_e025_instances"]
            else "FAIL"
        ),
        "run_id": run_id,
        "image_reference": image_reference,
        "image_digest": image_digest,
        "worker": worker.endpoint() if worker else None,
        "result": result,
        "failure": failure,
        "cleanup": cleanup,
        "rental_costs": lifecycle_costs,
    }
    atomic_write_json(output_path, payload)
    return payload


def run_sub_layer_canary_stage(
    *,
    run_id: str,
    stage_root: Path,
    image_receipt_path: Path,
    image_index_path: Path,
    test_receipt_path: Path,
    rehearsal_receipt_path: Path,
    local_image_canary_path: Path,
    backbone_canary_path: Path,
    private_root: Path,
    checkpoint: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    physical_placement: Path,
    output_path: Path,
    disk_gb: int = 60,
    vast_executable: str = "vastai",
) -> dict[str, Any]:
    stage_root.mkdir(parents=True, exist_ok=True)
    ledger_path = stage_root / "instance-ledger.jsonl"
    if ledger_path.is_file() and ledger_path.stat().st_size:
        raise RuntimeError("E025 sub-layer canary stage ledger already exists")
    _require_local_prerequisites(
        image_receipt_path=image_receipt_path,
        test_receipt_path=test_receipt_path,
        rehearsal_receipt_path=rehearsal_receipt_path,
        local_image_canary_path=local_image_canary_path,
    )
    backbone = read_json(backbone_canary_path)
    frozen_digest = read_json(image_receipt_path).get("immutable_digest")
    if (
        backbone.get("status") != "PASS"
        or backbone.get("image_digest") != frozen_digest
        or backbone.get("cleanup", {}).get("zero_live_e025_instances") is not True
    ):
        raise RuntimeError("E025 sub-layer canary requires a passing cleaned backbone canary")
    image_reference, image_digest = _image(image_receipt_path)
    requirements = [
        _image_worker_requirement(
            image_index_path,
            f"e025-layer-089-sub-{index:02d}",
            "SUB_LAYER_WORKER",
        )
        for index in range(4)
    ]
    requirements.append(
        _image_worker_requirement(
            image_index_path,
            "e025-stage-089-parent",
            "SUB_LAYER_PARENT",
        )
    )
    ledger = AppendOnlyLifecycleLedger(ledger_path, run_id)
    client = VastClient(executable=vast_executable, ledger=ledger)
    _require_zero_live_e025(client)
    offers = [
        *client.search_offers(
            gpu_names=("RTX 3060", "RTX 3070", "RTX 3070 Ti", "RTX 3080", "RTX 3080 Ti"),
            storage_gb=disk_gb,
        ),
        *client.search_offers(
            gpu_names=("RTX 3090", "RTX 3090 Ti"),
            storage_gb=disk_gb,
        ),
    ]
    unique_offers = list({offer.offer_id: offer for offer in offers}.values())
    exclusions = _failed_machine_exclusions(
        stage_root,
        stage_prefix="stage-2-sub-layer-canary",
    )
    atomic_write_json(stage_root / "failed-machine-exclusions.json", exclusions)
    unique_offers = [
        offer
        for offer in unique_offers
        if offer.machine_id not in exclusions["machine_ids"]
    ]
    if not unique_offers:
        raise RuntimeError("no Stage 2 offers remain after failed-machine exclusions")
    plan = rank_offers_for_workers(unique_offers, requirements, disk_gb=disk_gb)
    budget = _paid_stage_budget(
        client,
        plan,
        disk_gb=disk_gb,
        ttl_seconds=SUB_LAYER_CANARY_TTL_SECONDS,
    )
    go_path = stage_root / "SUB_LAYER_CANARY_GO.json"
    _write_go(
        go_path,
        run_id=run_id,
        stage="physical-sub-layer-canary",
        image_reference=image_reference,
        offers=unique_offers,
        plan=plan,
        budget=budget,
    )
    paths = _watchdog_paths(stage_root)
    watchdog = start_watchdog(
        run_id=run_id,
        stage="physical-sub-layer-canary",
        ledger_path=paths["ledger"],
        ttl_seconds=SUB_LAYER_CANARY_TTL_SECONDS,
        receipt_path=paths["receipt"],
        log_path=paths["log"],
        trigger_path=paths["trigger"],
        stop_path=paths["stop"],
        cleanup_receipt_path=paths["cleanup"],
        vast_executable=vast_executable,
    )
    deadline = float(watchdog["deadline_epoch"])
    selected = {
        row["worker_id"]: Offer(**row["selected_offer"]) for row in plan["workers"]
    }
    material = _material(private_root, run_id)
    fragments: list[LiveWorker] = []
    parent: LiveWorker | None = None
    result: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    cleanup: dict[str, Any]
    lifecycle_costs: dict[str, Any] = {}
    try:
        def create_fragment(index: int) -> tuple[int, int, str, int]:
            worker_id = f"e025-layer-089-sub-{index:02d}"
            instance_id = create_worker_instance(
                client=client,
                run_id=run_id,
                worker_id=worker_id,
                role="SUB_LAYER_WORKER",
                layer=89,
                worker_index=index,
                offer=selected[worker_id],
                image_reference=image_reference,
                image_digest=image_digest,
                disk_gb=disk_gb,
                material=material,
                watchdog_receipt=paths["receipt"],
                go_receipt=go_path,
                maximum_context=3,
            )
            host, port = wait_for_public_endpoint(
                client=client,
                ledger=ledger,
                worker_id=worker_id,
                offer=selected[worker_id],
                instance_id=instance_id,
                deadline_epoch=deadline,
            )
            return index, instance_id, host, port

        with ThreadPoolExecutor(max_workers=4) as pool:
            pending_fragments = list(pool.map(create_fragment, range(4)))
        pending_fragments.sort(key=lambda row: row[0])
        parent_instance_id = create_worker_instance(
            client=client,
            run_id=run_id,
            worker_id="e025-stage-089-parent",
            role="SUB_LAYER_PARENT",
            layer=89,
            worker_index=None,
            offer=selected["e025-stage-089-parent"],
            image_reference=image_reference,
            image_digest=image_digest,
            disk_gb=disk_gb,
            material=material,
            watchdog_receipt=paths["receipt"],
            go_receipt=go_path,
            maximum_context=3,
            expert_endpoints=[
                {
                    "worker_id": f"e025-layer-089-sub-{index:02d}",
                    "worker_index": index,
                    "host": host,
                    "port": port,
                    "timeout_seconds": 180.0,
                }
                for index, _instance_id, host, port in pending_fragments
            ],
        )
        abort_readiness = threading.Event()
        credential = Path(material["credential_path"]).read_bytes()
        certificate = Path(material["certificate_path"])

        def await_fragment(
            pending: tuple[int, int, str, int],
        ) -> LiveWorker:
            index, instance_id, _host, _port = pending
            worker_id = f"e025-layer-089-sub-{index:02d}"
            return wait_for_worker(
                client=client,
                ledger=ledger,
                run_id=run_id,
                worker_id=worker_id,
                role="SUB_LAYER_WORKER",
                layer=89,
                worker_index=index,
                offer=selected[worker_id],
                instance_id=instance_id,
                credential=credential,
                certificate=certificate,
                image_digest=image_digest,
                deadline_epoch=deadline,
                abort_event=abort_readiness,
            )

        def await_parent() -> LiveWorker:
            worker_id = "e025-stage-089-parent"
            return wait_for_worker(
                client=client,
                ledger=ledger,
                run_id=run_id,
                worker_id=worker_id,
                role="SUB_LAYER_PARENT",
                layer=89,
                worker_index=None,
                offer=selected[worker_id],
                instance_id=parent_instance_id,
                credential=credential,
                certificate=certificate,
                image_digest=image_digest,
                deadline_epoch=deadline,
                abort_event=abort_readiness,
            )

        with ThreadPoolExecutor(max_workers=5) as pool:
            future_roles = {
                pool.submit(await_fragment, pending): "fragment"
                for pending in pending_fragments
            }
            future_roles[pool.submit(await_parent)] = "parent"
            try:
                for future in as_completed(future_roles):
                    worker = future.result()
                    if future_roles[future] == "parent":
                        parent = worker
                    else:
                        fragments.append(worker)
            except BaseException:
                abort_readiness.set()
                raise
        fragments.sort(key=lambda row: int(row.worker_index or 0))
        if parent is None or len(fragments) != 4:
            raise RuntimeError("E025 Stage 2 readiness completed without all five workers")
        result = run_physical_sub_layer_canary(
            parent=parent,
            fragments=fragments,
            checkpoint=checkpoint,
            oracle_trace=oracle_trace,
            oracle_routes=oracle_routes,
            physical_placement=physical_placement,
            credential_path=Path(material["credential_path"]),
            certificate=Path(material["certificate_path"]),
            output_path=stage_root / "physical-sub-layer-canary.json",
        )
    except BaseException as exc:
        failure = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "classification": _failure_classification(exc),
        }
    finally:
        workers = [*fragments, *([parent] if parent is not None else [])]
        if workers:
            _retrieve_logs(client, workers, stage_root / "logs")
        cleanup = destroy_all_from_ledger(
            ledger_path=paths["ledger"],
            run_id=run_id,
            reason="physical-sub-layer-canary-complete-or-aborted",
            executable=vast_executable,
            attempts=5,
        )
        lifecycle_costs = _cost_summary(paths["ledger"], run_id)
        atomic_write_json(stage_root / "rental-cost-summary.json", lifecycle_costs)
        _finish_watchdog(paths, cleanup)
    payload = {
        "schema_version": "experiment-025-sub-layer-canary-stage-v1",
        "generated_at_utc": utc_now(),
        "status": (
            "PASS"
            if result is not None
            and result.get("status") == "PASS"
            and cleanup["zero_live_e025_instances"]
            else "FAIL"
        ),
        "run_id": run_id,
        "image_reference": image_reference,
        "image_digest": image_digest,
        "parent": parent.endpoint() if parent else None,
        "fragments": [worker.endpoint() for worker in fragments],
        "result": result,
        "failure": failure,
        "cleanup": cleanup,
        "rental_costs": lifecycle_costs,
    }
    atomic_write_json(output_path, payload)
    return payload


__all__ = ["run_backbone_canary_stage", "run_sub_layer_canary_stage"]
