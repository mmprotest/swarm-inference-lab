"""Continue Stage 2 in place while replacing only failed parent instance 48442617."""

from __future__ import annotations

import json
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from swarm_inference.experiments.experiment_025.canary_runtime import (
    run_physical_sub_layer_canary,
)
from swarm_inference.experiments.experiment_025.io import (
    atomic_write_json,
    read_json,
    utc_now,
)
from swarm_inference.experiments.experiment_025.provisioning import (
    LiveWorker,
    _public_endpoint,
    create_worker_instance,
    wait_for_worker,
)
from swarm_inference.experiments.experiment_025.secrets import load_transport_material
from swarm_inference.experiments.experiment_025.stages import (
    _cost_summary,
    _failure_classification,
    _image,
    _retrieve_logs,
)
from swarm_inference.experiments.experiment_025.vast_lifecycle import (
    AppendOnlyLifecycleLedger,
    Offer,
    VastClient,
    destroy_all_from_ledger,
)


RUN_ID = "20260819T013016Z"
FAILED_PARENT_INSTANCE_ID = 48442617
FRAGMENT_INSTANCES = {
    0: 48442376,
    1: 48442375,
    2: 48442374,
    3: 48442373,
}
STAGE_ROOT = Path(__file__).resolve().parent
RUN_ROOT = STAGE_ROOT.parent.parent


def main() -> int:
    ledger_path = STAGE_ROOT / "instance-ledger.jsonl"
    ledger = AppendOnlyLifecycleLedger(ledger_path, RUN_ID)
    client = VastClient(executable="vastai", ledger=ledger)
    go = read_json(STAGE_ROOT / "SUB_LAYER_CANARY_GO.json")
    plan_rows = {row["worker_id"]: row for row in go["fleet_plan"]["workers"]}
    selected = {
        worker_id: Offer(**row["selected_offer"])
        for worker_id, row in plan_rows.items()
    }
    image_reference, image_digest = _image(RUN_ROOT / "preflight" / "deployment-image.json")
    material = load_transport_material(Path(".e025-private").resolve() / RUN_ID, RUN_ID)
    watchdog = read_json(STAGE_ROOT / "progress-watchdog-receipt.json")
    readiness_deadline = float(watchdog["deadline_epoch"]) - 180.0
    credential = Path(material["credential_path"]).read_bytes()
    certificate = Path(material["certificate_path"])
    fragments: list[LiveWorker] = []
    parent: LiveWorker | None = None
    result = None
    failure = None
    replacement: dict[str, object] = {
        "failed_instance_id": FAILED_PARENT_INSTANCE_ID,
        "reason": "provider host-port collision before container start",
    }
    try:
        rows = client.show_instances()
        endpoint_rows = {
            int(row["id"]): row
            for row in rows
            if int(row.get("id", -1)) in set(FRAGMENT_INSTANCES.values())
        }
        if set(endpoint_rows) != set(FRAGMENT_INSTANCES.values()):
            raise RuntimeError("one or more retained Stage 2 fragment instances disappeared")
        expert_endpoints = []
        for index, instance_id in sorted(FRAGMENT_INSTANCES.items()):
            host, port = _public_endpoint(endpoint_rows[instance_id])
            expert_endpoints.append(
                {
                    "worker_id": f"e025-layer-089-sub-{index:02d}",
                    "worker_index": index,
                    "host": host,
                    "port": port,
                    "timeout_seconds": 180.0,
                }
            )

        if not client.destroy_instance(
            FAILED_PARENT_INSTANCE_ID,
            reason="replace parent after provider host-port collision",
        ):
            raise RuntimeError("failed Stage 2 parent could not be destroyed")

        used_machines = {
            selected[f"e025-layer-089-sub-{index:02d}"].machine_id
            for index in range(4)
        }
        used_machines.add(selected["e025-stage-089-parent"].machine_id)
        candidates = [
            Offer(**row["offer"])
            for row in plan_rows["e025-stage-089-parent"]["alternates"]
            if int(row["offer"]["machine_id"]) not in used_machines
            and str(row["offer"]["gpu_name"]) in {"RTX 3090", "RTX 3090 Ti"}
        ]
        parent_instance_id = None
        parent_offer = None
        creation_errors = []
        for offer in candidates:
            try:
                parent_instance_id = create_worker_instance(
                    client=client,
                    run_id=RUN_ID,
                    worker_id="e025-stage-089-parent",
                    role="SUB_LAYER_PARENT",
                    layer=89,
                    worker_index=None,
                    offer=offer,
                    image_reference=image_reference,
                    image_digest=image_digest,
                    disk_gb=60,
                    material=material,
                    watchdog_receipt=STAGE_ROOT / "progress-watchdog-receipt.json",
                    go_receipt=STAGE_ROOT / "SUB_LAYER_CANARY_GO.json",
                    maximum_context=3,
                    expert_endpoints=expert_endpoints,
                )
                parent_offer = offer
                break
            except BaseException as exc:
                creation_errors.append(
                    {
                        "offer_id": offer.offer_id,
                        "machine_id": offer.machine_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        if parent_instance_id is None or parent_offer is None:
            raise RuntimeError(f"all ranked parent alternates failed: {creation_errors}")
        replacement.update(
            {
                "replacement_instance_id": parent_instance_id,
                "replacement_offer_id": parent_offer.offer_id,
                "replacement_machine_id": parent_offer.machine_id,
                "creation_errors": creation_errors,
            }
        )
        atomic_write_json(STAGE_ROOT / "parent-replacement.json", replacement)

        abort_readiness = threading.Event()

        def await_fragment(index: int) -> LiveWorker:
            worker_id = f"e025-layer-089-sub-{index:02d}"
            return wait_for_worker(
                client=client,
                ledger=ledger,
                run_id=RUN_ID,
                worker_id=worker_id,
                role="SUB_LAYER_WORKER",
                layer=89,
                worker_index=index,
                offer=selected[worker_id],
                instance_id=FRAGMENT_INSTANCES[index],
                credential=credential,
                certificate=certificate,
                image_digest=image_digest,
                deadline_epoch=readiness_deadline,
                abort_event=abort_readiness,
            )

        def await_parent() -> LiveWorker:
            assert parent_instance_id is not None and parent_offer is not None
            return wait_for_worker(
                client=client,
                ledger=ledger,
                run_id=RUN_ID,
                worker_id="e025-stage-089-parent",
                role="SUB_LAYER_PARENT",
                layer=89,
                worker_index=None,
                offer=parent_offer,
                instance_id=parent_instance_id,
                credential=credential,
                certificate=certificate,
                image_digest=image_digest,
                deadline_epoch=readiness_deadline,
                abort_event=abort_readiness,
            )

        with ThreadPoolExecutor(max_workers=5) as pool:
            future_roles = {
                pool.submit(await_fragment, index): "fragment" for index in range(4)
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
        fragments.sort(key=lambda worker: int(worker.worker_index or 0))
        if parent is None or len(fragments) != 4:
            raise RuntimeError("Stage 2 continuation did not reach five-worker readiness")
        result = run_physical_sub_layer_canary(
            parent=parent,
            fragments=fragments,
            checkpoint=Path(r"F:\models\Kimi-K3").resolve(),
            oracle_trace=Path(
                "artifacts/experiment-014/oracle-full-93-idot0/hidden-trace.f32"
            ).resolve(),
            oracle_routes=Path(
                "artifacts/experiment-014/oracle-full-93-idot0/routes.txt"
            ).resolve(),
            physical_placement=RUN_ROOT / "preflight" / "physical-placement.json",
            credential_path=Path(material["credential_path"]),
            certificate=certificate,
            output_path=STAGE_ROOT / "physical-sub-layer-canary.json",
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
            _retrieve_logs(client, workers, STAGE_ROOT / "logs")
        cleanup = destroy_all_from_ledger(
            ledger_path=ledger_path,
            run_id=RUN_ID,
            reason="physical-sub-layer-canary-progress-aware-complete-or-aborted",
            executable="vastai",
            attempts=6,
        )
        costs = _cost_summary(ledger_path, RUN_ID)
        atomic_write_json(STAGE_ROOT / "rental-cost-summary.json", costs)
        atomic_write_json(STAGE_ROOT / "progress-cleanup-verification.json", cleanup)
        if cleanup.get("zero_live_e025_instances") is True:
            atomic_write_json(
                STAGE_ROOT / "PROGRESS_WATCHDOG_STOP",
                {"timestamp": utc_now(), "reason": "cleanup proven"},
            )
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
        "run_id": RUN_ID,
        "image_reference": image_reference,
        "image_digest": image_digest,
        "parent": parent.endpoint() if parent else None,
        "fragments": [worker.endpoint() for worker in fragments],
        "result": result,
        "failure": failure,
        "cleanup": cleanup,
        "rental_costs": costs,
        "continuation": {
            "retained_fragment_instance_ids": list(FRAGMENT_INSTANCES.values()),
            "new_fragment_rentals_created": False,
            "parent_replacement": replacement,
        },
    }
    atomic_write_json(RUN_ROOT / "correctness" / "sub-layer-canary.json", payload)
    atomic_write_json(STAGE_ROOT / "sub-layer-canary-result.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
